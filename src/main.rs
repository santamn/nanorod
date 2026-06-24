mod config;
mod gpu;
mod model;

use std::collections::VecDeque;
use std::fs::{File, create_dir, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use config::{
    Cli, RunMode, SimParams, production_parameter_combinations, smoke_parameter_combination,
    validate_devices,
};
use gpu::{ComboOutput, ComboWork, GpuProgress, GpuRunConfig};
use model::{ProgressRow, SummaryRow, aggregate_summaries};

/// GPU worker から main thread へ送るメッセージ。
enum WorkerMessage {
    Progress(ProgressRow),
    ComboFinished(ComboOutput),
    WorkerDone { device_id: usize },
    WorkerError { device_id: usize, message: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    model::validate_static_geometry()?;
    validate_devices(&cli.devices, cli.allow_device_zero)?;

    let include_paths = gpu::cuda_include_paths(&cli.cuda_include_path);
    let trial_count = gpu::trial_count_for_mode(cli.mode, cli.trials);
    // 0 step や 0 trial はGPU kernelの進捗判定を壊すため、CLI入口で弾く。
    anyhow::ensure!(trial_count > 0, "--trials must be greater than zero");
    anyhow::ensure!(cli.max_steps > 0, "--max-steps must be greater than zero");
    anyhow::ensure!(
        cli.steps_per_launch > 0,
        "--steps-per-launch must be greater than zero"
    );

    match cli.mode {
        RunMode::Smoke => run_smoke(&cli, trial_count, include_paths),
        RunMode::Production => run_production(&cli, trial_count, include_paths),
    }
}

fn create_new_output_dir(path: &Path) -> Result<()> {
    if path.exists() {
        anyhow::bail!(
            "refusing to write to existing output directory {}; choose a new --output-dir to avoid overwriting records",
            path.display()
        );
    }

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory {}", parent.display()))?;
    }

    create_dir(path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(())
}

/// 1つのパラメータ組み合わせを少数 trial で確認する。
///
/// trial 詳細を出力し、GPU 1,2,3 へ trial を分割して multi-GPU 経路も確認する。
fn run_smoke(cli: &Cli, trial_count: usize, include_paths: Vec<String>) -> Result<()> {
    let params = smoke_parameter_combination(
        cli.smoke_combo_id,
        cli.smoke_m,
        cli.smoke_beta_qe,
        cli.smoke_delta_alpha_e_over_ql,
        cli.smoke_f,
    )?;
    let splits = split_trials(trial_count, cli.devices.len());
    create_new_output_dir(&cli.output_dir)?;

    let smoke_trials_path = cli.output_dir.join("smoke_trials.csv");
    let smoke_summary_path = cli.output_dir.join("smoke_summary.json");
    let progress_path = cli.output_dir.join("progress.jsonl");
    let mut trial_writer = csv::Writer::from_path(&smoke_trials_path)?;
    let mut progress_writer = BufWriter::new(File::create(&progress_path)?);

    let (tx, rx) = mpsc::channel::<WorkerMessage>();
    let mut active_workers = 0usize;
    let mut trial_offset = 0_u64;

    // smoke では同じ combo の trial を GPU ごとに分割する。
    for (&device_id, count) in cli.devices.iter().zip(splits) {
        if count == 0 {
            continue;
        }
        active_workers += 1;
        let worker_tx = tx.clone();
        let config = worker_config(cli, device_id, true, include_paths.clone(), RunMode::Smoke);
        let work = ComboWork {
            params,
            trial_count: count,
            trial_offset,
        };
        trial_offset += count as u64;

        thread::spawn(move || {
            let result = gpu::run_combo(&config, work, |progress| {
                let _ = worker_tx.send(WorkerMessage::Progress(progress_row(
                    &config, &progress, 0, 1,
                )));
            });
            send_worker_result(worker_tx, device_id, result);
        });
    }
    drop(tx);

    let mut completed_workers = 0usize;
    let mut partial_summaries = Vec::<SummaryRow>::new();
    let mut errors = Vec::<String>::new();

    // main thread が writer を担当し、複数 worker から同じファイルへ直接書かない。
    while completed_workers < active_workers {
        match rx.recv()? {
            WorkerMessage::Progress(row) => {
                eprintln!(
                    "[{}] gpu={} combo={} trials={}/{} steps={} status={}",
                    row.run_mode,
                    row.device_id,
                    row.combo_id,
                    row.completed_trials,
                    row.total_trials,
                    row.current_steps,
                    row.status
                );
                write_progress(&mut progress_writer, &row)?;
            }
            WorkerMessage::ComboFinished(output) => {
                // smoke は各 trial の初期状態・終了状態・初通過時間を保存する。
                for trial in &output.trials {
                    trial_writer.serialize(trial.to_csv_row())?;
                }
                trial_writer.flush()?;
                partial_summaries.push(output.summary);
            }
            WorkerMessage::WorkerDone { device_id } => {
                completed_workers += 1;
                eprintln!("[smoke] gpu={device_id} done");
            }
            WorkerMessage::WorkerError { device_id, message } => {
                completed_workers += 1;
                errors.push(format!("gpu {device_id}: {message}"));
            }
        }
    }

    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }

    // GPU ごとの部分 summary を、smoke 全体の1行 summary にまとめる。
    let aggregate = aggregate_summaries(-1, params, cli.seed, &partial_summaries);
    write_json_object(&smoke_summary_path, &aggregate)?;

    eprintln!(
        "[smoke] wrote {} and {}",
        smoke_trials_path.display(),
        smoke_summary_path.display()
    );
    Ok(())
}

/// 全パラメータ組み合わせを処理する production run。
///
/// trial 詳細は保存せず、combo ごとの summary だけを逐次 flush する。
fn run_production(cli: &Cli, trial_count: usize, include_paths: Vec<String>) -> Result<()> {
    let combos = production_parameter_combinations(cli.combo_start, cli.combo_limit)?;
    let total_combos = combos.len();
    create_new_output_dir(&cli.output_dir)?;

    let summary_path = cli.output_dir.join("summary.json");
    let progress_path = cli.output_dir.join("progress.jsonl");
    let mut summary_writer = JsonArrayWriter::create(&summary_path)?;
    let mut progress_writer = BufWriter::new(File::create(&progress_path)?);

    let queue = Arc::new(Mutex::new(VecDeque::<SimParams>::from(combos)));
    let completed_combos = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<WorkerMessage>();

    // 各GPU workerが共有キューから次のcomboを取り出す。
    for &device_id in &cli.devices {
        let worker_tx = tx.clone();
        let worker_queue = Arc::clone(&queue);
        let worker_completed = Arc::clone(&completed_combos);
        let config = worker_config(
            cli,
            device_id,
            false,
            include_paths.clone(),
            RunMode::Production,
        );

        thread::spawn(move || {
            // 1 GPU 上で num_streams 本の combo を同時並行に処理し、queue を空にする。
            let result = gpu::run_device_streamed(
                &config,
                trial_count,
                || next_work(&worker_queue),
                |progress| {
                    let _ = worker_tx.send(WorkerMessage::Progress(progress_row(
                        &config,
                        &progress,
                        worker_completed.load(Ordering::Relaxed),
                        total_combos,
                    )));
                },
                |output| {
                    // 完了順に summary を返す。JSON 側では combo_id で対応できる。
                    worker_completed.fetch_add(1, Ordering::Relaxed);
                    let _ = worker_tx.send(WorkerMessage::ComboFinished(output));
                },
            );

            match result {
                Ok(()) => {
                    let _ = worker_tx.send(WorkerMessage::WorkerDone { device_id });
                }
                Err(error) => {
                    let _ = worker_tx.send(WorkerMessage::WorkerError {
                        device_id,
                        message: format!("{error:#}"),
                    });
                }
            }
        });
    }
    drop(tx);

    let mut completed_workers = 0usize;
    let mut errors = Vec::<String>::new();

    // production でも writer はmain threadに集約し、1行ごとにflushする。
    while completed_workers < cli.devices.len() {
        match rx.recv()? {
            WorkerMessage::Progress(row) => {
                eprintln!(
                    "[{}] gpu={} combo={} combos={}/{} trials={}/{} steps={} status={}",
                    row.run_mode,
                    row.device_id,
                    row.combo_id,
                    row.completed_combos,
                    row.total_combos,
                    row.completed_trials,
                    row.total_trials,
                    row.current_steps,
                    row.status
                );
                write_progress(&mut progress_writer, &row)?;
            }
            WorkerMessage::ComboFinished(output) => {
                summary_writer.write_element(&output.summary)?;
            }
            WorkerMessage::WorkerDone { device_id } => {
                completed_workers += 1;
                eprintln!("[production] gpu={device_id} done");
            }
            WorkerMessage::WorkerError { device_id, message } => {
                completed_workers += 1;
                errors.push(format!("gpu {device_id}: {message}"));
            }
        }
    }

    summary_writer.finish()?;

    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }

    eprintln!("[production] wrote {}", summary_path.display());
    Ok(())
}

/// CLI設定から GPU worker 用の実行設定を作る。
fn worker_config(
    cli: &Cli,
    device_id: usize,
    capture_trials: bool,
    include_paths: Vec<String>,
    mode: RunMode,
) -> GpuRunConfig {
    GpuRunConfig {
        device_id,
        seed: cli.seed,
        max_steps: cli.max_steps,
        steps_per_launch: cli.steps_per_launch,
        num_streams: cli.streams,
        cuda_include_paths: include_paths,
        cuda_arch: cli.cuda_arch.clone(),
        capture_trials,
        progress_interval: Duration::from_secs(cli.progress_interval_sec),
        run_mode_label: run_mode_label(mode).to_string(),
    }
}

/// smoke worker の結果を、成功・失敗どちらでも main thread へ返す。
fn send_worker_result(
    tx: mpsc::Sender<WorkerMessage>,
    device_id: usize,
    result: Result<ComboOutput>,
) {
    match result {
        Ok(output) => {
            if tx.send(WorkerMessage::ComboFinished(output)).is_ok() {
                let _ = tx.send(WorkerMessage::WorkerDone { device_id });
            }
        }
        Err(error) => {
            let _ = tx.send(WorkerMessage::WorkerError {
                device_id,
                message: format!("{error:#}"),
            });
        }
    }
}

/// GPU worker の進捗を `progress.jsonl` 用の行へ変換する。
fn progress_row(
    config: &GpuRunConfig,
    progress: &GpuProgress,
    completed_combos: usize,
    total_combos: usize,
) -> ProgressRow {
    ProgressRow {
        timestamp_ms: timestamp_ms(),
        run_mode: config.run_mode_label.clone(),
        device_id: config.device_id,
        combo_id: progress.combo_id,
        completed_combos,
        total_combos,
        completed_trials: progress.completed_trials,
        total_trials: progress.total_trials,
        current_steps: progress.current_steps,
        max_steps: config.max_steps,
        status: progress.status.clone(),
    }
}

/// production の共有キューから次の combo を取り出す。
fn next_work(queue: &Arc<Mutex<VecDeque<SimParams>>>) -> Option<SimParams> {
    queue.lock().expect("work queue poisoned").pop_front()
}

/// smoke run で trial 数をGPU数にできるだけ均等に分割する。
fn split_trials(total: usize, parts: usize) -> Vec<usize> {
    let base = total / parts;
    let remainder = total % parts;
    (0..parts)
        .map(|idx| base + usize::from(idx < remainder))
        .collect()
}

/// ファイル出力やログに使う実行モード名。
fn run_mode_label(mode: RunMode) -> &'static str {
    match mode {
        RunMode::Smoke => "smoke",
        RunMode::Production => "production",
    }
}

/// 外部依存を増やさず、progress用にUnix epochミリ秒を作る。
fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// progress.jsonl に1イベントをJSON 1行として書き、即座にflushする。
fn write_progress(writer: &mut BufWriter<File>, row: &ProgressRow) -> Result<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_json_object<T: serde::Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

struct JsonArrayWriter {
    writer: BufWriter<File>,
    wrote_any: bool,
    finished: bool,
}

impl JsonArrayWriter {
    fn create(path: &std::path::Path) -> Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"[\n")?;
        writer.flush()?;
        Ok(Self {
            writer,
            wrote_any: false,
            finished: false,
        })
    }

    fn write_element<T: serde::Serialize>(&mut self, value: &T) -> Result<()> {
        if self.wrote_any {
            self.writer.write_all(b",\n")?;
        } else {
            self.wrote_any = true;
        }

        self.writer.write_all(b"  ")?;
        serde_json::to_writer(&mut self.writer, value)?;
        self.writer.flush()?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }

        if self.wrote_any {
            self.writer.write_all(b"\n")?;
        }
        self.writer.write_all(b"]\n")?;
        self.writer.flush()?;
        self.finished = true;
        Ok(())
    }
}
