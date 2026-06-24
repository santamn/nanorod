mod config;
mod gpu;
mod model;

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, create_dir, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use config::{
    Cli, HIST_PHI_BINS, HIST_X_BINS, HIST_Y_BINS, HIST_Y_MAX, SimParams, combo_dir_name,
    parameter_combinations, validate_devices,
};
use gpu::{ComboOutput, GpuProgress, GpuRunConfig};
use model::{ProgressRow, TrialResult};

/// GPU worker から main thread へ送るメッセージ。
enum WorkerMessage {
    Progress(ProgressRow),
    /// 完了した trial の確定行。1 メッセージは1 combo 分。
    TrialsReady(Vec<TrialResult>),
    ComboFinished(ComboOutput),
    WorkerDone { device_id: usize },
    WorkerError { device_id: usize, message: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    model::validate_static_geometry()?;
    validate_devices(&cli.devices)?;

    let include_paths = gpu::cuda_include_paths(&cli.cuda_include_path);
    // 0 trial / 0 step はGPU kernelの進捗判定を壊すため、CLI入口で弾く。
    anyhow::ensure!(cli.trials > 0, "--trials must be greater than zero");
    anyhow::ensure!(cli.max_steps > 0, "--max-steps must be greater than zero");
    anyhow::ensure!(
        cli.steps_per_launch > 0,
        "--steps-per-launch must be greater than zero"
    );

    let combos = parameter_combinations(&cli.m, &cli.beta_pe, &cli.abs_delta_alpha_e_over_p, &cli.f)?;
    run(&cli, combos, include_paths)
}

/// 上書き事故を避けるため、まだ存在しない出力ディレクトリだけを新規作成する。
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

/// 全パラメータ組み合わせを GPU で処理し、combo ごとのフォルダへ結果を保存する。
///
/// 各 GPU worker が共有キューから combo を取り出して並行処理し、完了した combo の
/// trial 詳細と summary を main thread が `output_dir/<combo>/` へ逐次書き出す。
fn run(cli: &Cli, combos: Vec<SimParams>, include_paths: Vec<String>) -> Result<()> {
    let trial_count = cli.trials;
    let total_combos = combos.len();

    create_new_output_dir(&cli.output_dir)?;

    // combo ごとの出力フォルダを先に作り、combo_id から引けるようにしておく。
    let mut combo_dirs = HashMap::<u32, PathBuf>::with_capacity(total_combos);
    for params in &combos {
        let dir = cli.output_dir.join(combo_dir_name(params));
        create_dir(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        combo_dirs.insert(params.combo_id, dir);
    }

    let progress_path = cli.output_dir.join("progress.jsonl");
    let mut progress_writer = BufWriter::new(File::create(&progress_path)?);

    let queue = Arc::new(Mutex::new(VecDeque::<SimParams>::from(combos)));
    let completed_combos = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<WorkerMessage>();

    // 各 GPU worker が共有キューから次の combo を取り出す。
    for &device_id in &cli.devices {
        let worker_tx = tx.clone();
        let worker_queue = Arc::clone(&queue);
        let worker_completed = Arc::clone(&completed_combos);
        let config = worker_config(cli, device_id, include_paths.clone());

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
                |trials| {
                    let _ = worker_tx.send(WorkerMessage::TrialsReady(trials));
                },
                |output| {
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

    // combo ごとに trials.csv を開いたまま保持し、完了した trial を逐次追記する。
    let mut trial_writers = HashMap::<u32, csv::Writer<File>>::new();

    // 出力は main thread に集約し、複数 worker から同じファイルへ直接書かない。
    while completed_workers < cli.devices.len() {
        match rx.recv()? {
            WorkerMessage::Progress(row) => {
                eprintln!(
                    "gpu={} combo={} combos={}/{} trials={}/{} steps={} status={}",
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
            WorkerMessage::TrialsReady(trials) => {
                append_trials(&mut trial_writers, &combo_dirs, &trials)?;
            }
            WorkerMessage::ComboFinished(output) => {
                // 逐次追記してきた trials.csv を flush して閉じてから、残りを書き出す。
                if let Some(mut writer) = trial_writers.remove(&output.summary.combo_id) {
                    writer.flush()?;
                }
                write_combo_output(&combo_dirs, &output)?;
            }
            WorkerMessage::WorkerDone { device_id } => {
                completed_workers += 1;
                eprintln!("gpu={device_id} done");
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

    eprintln!(
        "wrote {} combo result(s) under {}",
        total_combos,
        cli.output_dir.display()
    );
    Ok(())
}

/// 完了した trial の確定行を、その combo の trials.csv へ追記する。
///
/// 1 メッセージは1 combo 分なので、対応する writer を開いたまま使い回し（ヘッダは初回のみ）、
/// バッチごとに flush して完了した trial を逐次ディスクへ反映する。
fn append_trials(
    writers: &mut HashMap<u32, csv::Writer<File>>,
    combo_dirs: &HashMap<u32, PathBuf>,
    trials: &[TrialResult],
) -> Result<()> {
    for trial in trials {
        let writer = match writers.entry(trial.combo_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let dir = combo_dirs
                    .get(&trial.combo_id)
                    .with_context(|| format!("no output directory for combo {}", trial.combo_id))?;
                entry.insert(csv::Writer::from_path(dir.join("trials.csv"))?)
            }
        };
        writer.serialize(trial.to_csv_row())?;
    }

    // バッチは1 combo 分なので、その writer を flush すれば完了ぶんが即ディスクへ出る。
    if let Some(writer) = trials.first().and_then(|first| writers.get_mut(&first.combo_id)) {
        writer.flush()?;
    }
    Ok(())
}

/// 1 combo の summary とヒストグラムを、その combo 専用フォルダへ書き出す。
///
/// trial 詳細（trials.csv）は完了ごとに `append_trials` で逐次追記済みなので、ここでは扱わない。
fn write_combo_output(combo_dirs: &HashMap<u32, PathBuf>, output: &ComboOutput) -> Result<()> {
    let dir = combo_dirs
        .get(&output.summary.combo_id)
        .with_context(|| format!("no output directory for combo {}", output.summary.combo_id))?;

    write_json_object(&dir.join("summary.json"), &output.summary)?;

    // x×φ と x×y の2Dヒストグラムを、解析しやすい tidy 形式の CSV で残す。
    write_histogram_csv(&dir.join("angle_hist.csv"), "phi", &output.hist_phi, HIST_PHI_BINS, |bin| {
        (bin as f64 + 0.5) * 2.0 * std::f64::consts::PI / HIST_PHI_BINS as f64
    })?;
    write_histogram_csv(&dir.join("y_hist.csv"), "y", &output.hist_y, HIST_Y_BINS, |bin| {
        -HIST_Y_MAX + (bin as f64 + 0.5) * 2.0 * HIST_Y_MAX / HIST_Y_BINS as f64
    })?;
    Ok(())
}

/// x×値 の2Dヒストグラムを `x, <value_label>, count` の tidy 形式 CSV へ書き出す。
///
/// `counts` は x を最外として行優先で平坦化されており、`value_center` は2軸目の
/// bin 番号から bin 中心の値を返す。
fn write_histogram_csv(
    path: &Path,
    value_label: &str,
    counts: &[u64],
    value_bins: usize,
    value_center: impl Fn(usize) -> f64,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["x", value_label, "count"])?;
    for x_bin in 0..HIST_X_BINS {
        let x_center = (x_bin as f64 + 0.5) / HIST_X_BINS as f64;
        for value_bin in 0..value_bins {
            writer.write_record([
                x_center.to_string(),
                value_center(value_bin).to_string(),
                counts[x_bin * value_bins + value_bin].to_string(),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// CLI設定から GPU worker 用の実行設定を作る。
fn worker_config(cli: &Cli, device_id: usize, include_paths: Vec<String>) -> GpuRunConfig {
    GpuRunConfig {
        device_id,
        seed: cli.seed,
        max_steps: cli.max_steps,
        steps_per_launch: cli.steps_per_launch,
        hist_stride: cli.hist_stride,
        num_streams: cli.streams,
        cuda_include_paths: include_paths,
        cuda_arch: cli.cuda_arch.clone(),
        progress_interval: Duration::from_secs(cli.progress_interval_sec),
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

/// 共有キューから次の combo を取り出す。
fn next_work(queue: &Arc<Mutex<VecDeque<SimParams>>>) -> Option<SimParams> {
    queue.lock().expect("work queue poisoned").pop_front()
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

/// 任意の serializable 値を、整形済み JSON としてファイルへ書き出す。
fn write_json_object<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
