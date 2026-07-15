//! 全ケースの実行と結果の書き出し。
//!
//! ケースを共有キューに積み、バックエンド（GPU または CPU）のワーカーが取り出して
//! 実行する。結果の書き出しは main thread に集約し、複数ワーカーから同じファイルへ
//! 直接書かない。

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, create_dir, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};

use crate::config::{Case, Config, HIST_PHI_BINS, HIST_X_BINS, HIST_Y_BINS, HIST_Y_MAX};
use crate::model::{CaseOutput, ProgressRow, TrialResult};

/// ワーカーから main thread へ送るメッセージ。
enum WorkerMessage {
    Progress(ProgressRow),
    /// 完了した trial の確定行。1 メッセージは1ケース分。
    TrialsReady(Vec<TrialResult>),
    CaseFinished(CaseOutput),
    WorkerDone {
        worker: String,
    },
    // CPU バックエンドのワーカーは失敗し得ないため、CPU ビルドでは構築されない。
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    WorkerError {
        worker: String,
        message: String,
    },
}

/// 設定された全ケースを実行し、ケースごとのフォルダへ結果を保存する。
///
/// 再現できるように、使用した設定ファイルのコピーも出力フォルダへ残す。
pub fn run_all(config: &Config, config_path: &Path) -> Result<()> {
    let physics = config.physics()?;
    let cases = config.cases()?;
    let total_cases = cases.len();

    create_new_output_dir(&config.output_dir)?;
    std::fs::copy(config_path, config.output_dir.join("config.toml"))
        .with_context(|| format!("設定ファイル {} をコピーできません", config_path.display()))?;

    // ケースごとの出力フォルダを先に作り、case_id から引けるようにしておく。
    let mut case_dirs = HashMap::<u32, PathBuf>::with_capacity(total_cases);
    for case in &cases {
        let dir = config.output_dir.join(case.dir_name());
        create_dir(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        case_dirs.insert(case.case_id, dir);
    }

    let progress_path = config.output_dir.join("progress.jsonl");
    let mut progress_writer = BufWriter::new(File::create(&progress_path)?);

    let queue = Arc::new(Mutex::new(VecDeque::<Case>::from(cases)));
    let completed_cases = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<WorkerMessage>();

    let worker_count = backend::spawn_workers(
        config,
        physics,
        total_cases,
        Arc::clone(&queue),
        Arc::clone(&completed_cases),
        tx,
    )?;

    let mut completed_workers = 0usize;
    let mut errors = Vec::<String>::new();

    // ケースごとに trials.csv を開いたまま保持し、完了した trial を逐次追記する。
    let mut trial_writers = HashMap::<u32, csv::Writer<File>>::new();

    while completed_workers < worker_count {
        match rx.recv()? {
            WorkerMessage::Progress(row) => {
                eprintln!(
                    "worker={} case={} cases={}/{} trials={}/{} steps={} status={}",
                    worker_label(row.gpu_id),
                    row.case_id,
                    row.completed_cases,
                    row.total_cases,
                    row.completed_trials,
                    row.total_trials,
                    row.current_steps,
                    row.status
                );
                write_progress(&mut progress_writer, &row)?;
            }
            WorkerMessage::TrialsReady(trials) => {
                append_trials(&mut trial_writers, &case_dirs, &trials)?;
            }
            WorkerMessage::CaseFinished(output) => {
                // 逐次追記してきた trials.csv を flush して閉じてから、残りを書き出す。
                if let Some(mut writer) = trial_writers.remove(&output.summary.case_id) {
                    writer.flush()?;
                }
                write_case_output(&case_dirs, &output)?;
            }
            WorkerMessage::WorkerDone { worker } => {
                completed_workers += 1;
                eprintln!("worker={worker} done");
            }
            WorkerMessage::WorkerError { worker, message } => {
                completed_workers += 1;
                errors.push(format!("worker {worker}: {message}"));
            }
        }
    }

    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }

    eprintln!(
        "wrote {} case result(s) under {}",
        total_cases,
        config.output_dir.display()
    );
    Ok(())
}

/// GPU の ID または CPU を、進捗表示用の短いラベルにする。
fn worker_label(gpu_id: Option<usize>) -> String {
    match gpu_id {
        Some(id) => format!("gpu:{id}"),
        None => "cpu".to_string(),
    }
}

/// 上書き事故を避けるため、まだ存在しない出力ディレクトリだけを新規作成する。
fn create_new_output_dir(path: &Path) -> Result<()> {
    if path.exists() {
        anyhow::bail!(
            "出力フォルダ {} は既に存在します。上書きを避けるため、別のフォルダを output_dir に指定するか、削除してから実行してください",
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

/// 完了した trial の確定行を、そのケースの trials.csv へ追記する。
///
/// 1 メッセージは1ケース分なので、対応する writer を開いたまま使い回し（ヘッダは初回のみ）、
/// バッチごとに flush して完了した trial を逐次ディスクへ反映する。
fn append_trials(
    writers: &mut HashMap<u32, csv::Writer<File>>,
    case_dirs: &HashMap<u32, PathBuf>,
    trials: &[TrialResult],
) -> Result<()> {
    for trial in trials {
        let writer = match writers.entry(trial.case_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let dir = case_dirs
                    .get(&trial.case_id)
                    .with_context(|| format!("no output directory for case {}", trial.case_id))?;
                entry.insert(csv::Writer::from_path(dir.join("trials.csv"))?)
            }
        };
        writer.serialize(trial.to_csv_row())?;
    }

    // バッチは1ケース分なので、その writer を flush すれば完了ぶんが即ディスクへ出る。
    if let Some(writer) = trials
        .first()
        .and_then(|first| writers.get_mut(&first.case_id))
    {
        writer.flush()?;
    }
    Ok(())
}

/// 1ケースの summary とヒストグラムを、そのケース専用フォルダへ書き出す。
///
/// trial 詳細（trials.csv）は完了ごとに `append_trials` で逐次追記済みなので、ここでは扱わない。
fn write_case_output(case_dirs: &HashMap<u32, PathBuf>, output: &CaseOutput) -> Result<()> {
    let dir = case_dirs
        .get(&output.summary.case_id)
        .with_context(|| format!("no output directory for case {}", output.summary.case_id))?;

    write_json_object(&dir.join("summary.json"), &output.summary)?;

    // x×φ と x×y の2Dヒストグラムを、解析しやすい tidy 形式の CSV で残す。
    write_histogram_csv(
        &dir.join("angle_hist.csv"),
        "phi",
        &output.hist_phi,
        HIST_PHI_BINS,
        |bin| (bin as f64 + 0.5) * 2.0 * std::f64::consts::PI / HIST_PHI_BINS as f64,
    )?;
    write_histogram_csv(
        &dir.join("y_hist.csv"),
        "y",
        &output.hist_y,
        HIST_Y_BINS,
        |bin| -HIST_Y_MAX + (bin as f64 + 0.5) * 2.0 * HIST_Y_MAX / HIST_Y_BINS as f64,
    )?;
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

/// 共有キューから次のケースを取り出す。
fn next_case(queue: &Arc<Mutex<VecDeque<Case>>>) -> Option<Case> {
    queue.lock().expect("work queue poisoned").pop_front()
}

/// 外部依存を増やさず、progress 用に Unix epoch ミリ秒を作る。
fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// progress.jsonl に1イベントを JSON 1行として書き、即座に flush する。
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

/// GPU バックエンド: GPU ごとにワーカースレッドを立て、複数ケースを CUDA stream で
/// 同時並行に処理する。
#[cfg(feature = "gpu")]
mod backend {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use anyhow::{Result, ensure};

    use super::{WorkerMessage, next_case, timestamp_ms};
    use crate::config::{Case, Config, Physics};
    use crate::gpu::{self, GpuProgress, GpuRunConfig};
    use crate::model::ProgressRow;

    /// 1回の kernel 起動で進めるステップ数。完了判定・進捗更新の粒度で、
    /// 1ケースの実行時間(数十秒〜)に対して起動オーバーヘッドが埋もれる大きさにする。
    const STEPS_PER_LAUNCH: u32 = 10_000;
    /// progress.jsonl へ進捗を書き出す最短間隔。
    const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
    /// 1 GPU を飽和させるために狙う同時実行 trial 数。
    ///
    /// A100 での実測では、同時に走る trial 数(ensemble_size × 同時実行ケース数)が
    /// 10万程度に達するまでスループットはほぼ線形に伸び、その先で飽和する。
    /// 飽和点を確実に越えるよう、実測の飽和点の2倍を目標にする。
    const TARGET_CONCURRENT_TRIALS: usize = 200_000;
    /// GPU あたりの同時実行ケース数(= CUDA stream 数)の上限。
    /// 小さいアンサンブルで stream とバッファが際限なく増えるのを防ぐ
    /// (A100 実測ではこの規模まで改善が確認されている)。
    const MAX_TASKS_PER_GPU: usize = 96;

    /// GPU あたりの同時実行ケース数を、実測に基づいて自動調整する。
    ///
    /// 同時実行 trial 数(ensemble_size × ケース数)が飽和目標に達する最小の
    /// ケース数を選ぶ。ケース総数を GPU 台数で分け合った数より多くは載せない
    /// (余った stream は最初から遊ぶだけなので)。
    fn auto_tasks_per_gpu(ensemble_size: usize, total_cases: usize, gpu_count: usize) -> usize {
        let saturating_tasks = TARGET_CONCURRENT_TRIALS
            .div_ceil(ensemble_size.max(1))
            .min(MAX_TASKS_PER_GPU);
        let cases_per_gpu = total_cases.div_ceil(gpu_count.max(1));
        saturating_tasks.min(cases_per_gpu).max(1)
    }

    /// GPU ごとに1本のワーカースレッドを起動し、その数を返す。
    pub fn spawn_workers(
        config: &Config,
        physics: Physics,
        total_cases: usize,
        queue: Arc<Mutex<VecDeque<Case>>>,
        completed_cases: Arc<AtomicUsize>,
        tx: Sender<WorkerMessage>,
    ) -> Result<usize> {
        let gpu_ids = match &config.gpu.ids {
            Some(ids) => ids.clone(),
            None => (0..gpu::device_count()?).collect(),
        };
        ensure!(!gpu_ids.is_empty(), "使用可能な GPU がありません");
        let mut unique_ids = gpu_ids.clone();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        ensure!(
            unique_ids.len() == gpu_ids.len(),
            "gpu.ids に同じ GPU が複数回指定されています"
        );

        let tasks_per_gpu = auto_tasks_per_gpu(config.ensemble_size, total_cases, gpu_ids.len());
        eprintln!(
            "tasks_per_gpu={tasks_per_gpu} (ensemble_size={} とケース数 {} から自動調整)",
            config.ensemble_size, total_cases
        );

        let trial_count = config.ensemble_size;
        let hist_stride = config.hist_stride;

        for &device_id in &gpu_ids {
            let worker_tx = tx.clone();
            let worker_queue = Arc::clone(&queue);
            let worker_completed = Arc::clone(&completed_cases);
            let run_config = GpuRunConfig {
                device_id,
                max_steps: physics.max_steps,
                steps_per_launch: STEPS_PER_LAUNCH,
                hist_stride,
                tasks_per_gpu,
                progress_interval: PROGRESS_INTERVAL,
            };

            thread::spawn(move || {
                // 1 GPU 上で tasks_per_gpu 本のケースを同時並行に処理し、queue を空にする。
                let result = gpu::run_device_streamed(
                    &run_config,
                    &physics,
                    trial_count,
                    || next_case(&worker_queue),
                    |progress| {
                        let _ = worker_tx.send(WorkerMessage::Progress(progress_row(
                            &run_config,
                            &progress,
                            worker_completed.load(Ordering::Relaxed),
                            total_cases,
                        )));
                    },
                    |trials| {
                        let _ = worker_tx.send(WorkerMessage::TrialsReady(trials));
                    },
                    |output| {
                        worker_completed.fetch_add(1, Ordering::Relaxed);
                        let _ = worker_tx.send(WorkerMessage::CaseFinished(output));
                    },
                );

                match result {
                    Ok(()) => {
                        let _ = worker_tx.send(WorkerMessage::WorkerDone {
                            worker: format!("gpu:{device_id}"),
                        });
                    }
                    Err(error) => {
                        let _ = worker_tx.send(WorkerMessage::WorkerError {
                            worker: format!("gpu:{device_id}"),
                            message: format!("{error:#}"),
                        });
                    }
                }
            });
        }

        Ok(gpu_ids.len())
    }

    /// GPU worker の進捗を `progress.jsonl` 用の行へ変換する。
    fn progress_row(
        config: &GpuRunConfig,
        progress: &GpuProgress,
        completed_cases: usize,
        total_cases: usize,
    ) -> ProgressRow {
        ProgressRow {
            timestamp_ms: timestamp_ms(),
            gpu_id: Some(config.device_id),
            case_id: progress.case_id,
            completed_cases,
            total_cases,
            completed_trials: progress.completed_trials,
            total_trials: progress.total_trials,
            current_steps: progress.current_steps,
            max_steps: config.max_steps,
            status: progress.status.clone(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::auto_tasks_per_gpu;

        /// アンサンブルが小さいほど同時実行ケース数を増やし、上限とケース数で頭打ちに
        /// なることを確認する。
        #[test]
        fn auto_tasks_per_gpu_scales_with_ensemble_and_caps() {
            // 標準アンサンブル(3万 trial): 20万 trial 目標 → 7 ケース同時。
            assert_eq!(auto_tasks_per_gpu(30_000, 1000, 4), 7);
            // 小さいアンサンブルでは stream 数の上限で頭打ち。
            assert_eq!(auto_tasks_per_gpu(1_000, 1000, 4), 96);
            // 1ケースで既に飽和目標を超えるなら増やさない。
            assert_eq!(auto_tasks_per_gpu(300_000, 1000, 4), 1);
            // ケース総数を GPU 台数で分け合った数より多くは載せない。
            assert_eq!(auto_tasks_per_gpu(1_000, 6, 3), 2);
        }
    }
}

/// CPU バックエンド: GPU なしでも動作確認できるよう、rayon で trial を並列計算する。
/// GPU 版に比べ桁違いに遅いため、小規模な検証用。
#[cfg(not(feature = "gpu"))]
mod backend {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};
    use std::thread;

    use anyhow::Result;
    use rayon::prelude::*;

    use super::{WorkerMessage, next_case, timestamp_ms};
    use crate::config::{
        Case, Config, HIST_PHI_BINS, HIST_X_BINS, HIST_Y_BINS, HIST_Y_MAX, Mode, Physics,
    };
    use crate::model::{
        CaseOutput, ProgressRow, TrialResult, summarize_first_passage, summarize_fixed_time,
    };
    use crate::simulation::run_trial;

    /// 1ケース分の trial 結果とヒストグラムを rayon の fold/reduce で貯める入れ物。
    struct CaseAccumulator {
        trials: Vec<TrialResult>,
        hist_phi: Vec<u64>,
        hist_y: Vec<u64>,
    }

    impl CaseAccumulator {
        fn new() -> Self {
            Self {
                trials: Vec::new(),
                hist_phi: vec![0; HIST_X_BINS * HIST_PHI_BINS],
                hist_y: vec![0; HIST_X_BINS * HIST_Y_BINS],
            }
        }

        /// 2つの部分集計を1つへまとめる。
        fn merge(mut self, other: Self) -> Self {
            self.trials.extend(other.trials);
            for (into, from) in self.hist_phi.iter_mut().zip(&other.hist_phi) {
                *into += from;
            }
            for (into, from) in self.hist_y.iter_mut().zip(&other.hist_y) {
                *into += from;
            }
            self
        }
    }

    /// CPU ワーカーを1本起動し、その数（常に 1）を返す。
    ///
    /// ケースは順番に処理し、ケース内部の trial を rayon で並列化する。
    pub fn spawn_workers(
        config: &Config,
        physics: Physics,
        total_cases: usize,
        queue: Arc<Mutex<VecDeque<Case>>>,
        completed_cases: Arc<AtomicUsize>,
        tx: Sender<WorkerMessage>,
    ) -> Result<usize> {
        let trial_count = config.ensemble_size;
        let hist_stride = config.hist_stride;

        thread::spawn(move || {
            while let Some(case) = next_case(&queue) {
                let output = run_case(case, physics, trial_count, hist_stride);
                let completed = completed_cases.fetch_add(1, Ordering::Relaxed) + 1;

                let _ = tx.send(WorkerMessage::Progress(ProgressRow {
                    timestamp_ms: timestamp_ms(),
                    gpu_id: None,
                    case_id: case.case_id,
                    completed_cases: completed,
                    total_cases,
                    completed_trials: trial_count,
                    total_trials: trial_count,
                    current_steps: output.max_trial_steps,
                    max_steps: physics.max_steps,
                    status: "completed".to_string(),
                }));
                let _ = tx.send(WorkerMessage::TrialsReady(output.trials));
                let _ = tx.send(WorkerMessage::CaseFinished(output.case_output));
            }

            let _ = tx.send(WorkerMessage::WorkerDone {
                worker: "cpu".to_string(),
            });
        });

        Ok(1)
    }

    /// 1ケース分の実行結果一式。
    struct CpuCaseResult {
        trials: Vec<TrialResult>,
        case_output: CaseOutput,
        max_trial_steps: u64,
    }

    /// 1ケースの全 trial を rayon で並列実行し、summary とヒストグラムまで集計する。
    fn run_case(
        case: Case,
        physics: Physics,
        trial_count: usize,
        hist_stride: u32,
    ) -> CpuCaseResult {
        let case_seed = case.seed();

        let mut accum = (0..trial_count as u64)
            .into_par_iter()
            .fold(CaseAccumulator::new, |mut accum, trial_id| {
                // GPU 側の「ケースシード + trial 系列」に対応する、trial ごとの独立シード。
                let trial_seed = case_seed.wrapping_add(trial_id);
                let mut trial = run_trial(case, physics, trial_seed, |steps, x, y, phi| {
                    if hist_stride > 0 && steps % u64::from(hist_stride) == 0 {
                        accumulate_histograms(x, y, phi, &mut accum.hist_phi, &mut accum.hist_y);
                    }
                });
                trial.trial_id = trial_id;
                accum.trials.push(trial);
                accum
            })
            .reduce(CaseAccumulator::new, CaseAccumulator::merge);

        // reduce の順序に依らず、出力を trial_id 順に揃える。
        accum.trials.sort_by_key(|trial| trial.trial_id);

        let max_trial_steps = accum
            .trials
            .iter()
            .map(|trial| trial.steps)
            .max()
            .unwrap_or(0);

        let summary = match physics.mode {
            Mode::FirstPassage => {
                let times: Vec<f64> = accum.trials.iter().map(|trial| trial.t).collect();
                let statuses: Vec<i32> = accum.trials.iter().map(|trial| trial.status).collect();
                let pass_directions: Vec<i32> = accum
                    .trials
                    .iter()
                    .map(|trial| trial.pass_direction)
                    .collect();
                summarize_first_passage(
                    None,
                    case,
                    trial_count,
                    &physics,
                    &times,
                    &statuses,
                    &pass_directions,
                )
            }
            // fixed_time モードの統計は初期位置と終了位置の変位から取る。
            Mode::FixedTime => {
                let x0s: Vec<f64> = accum.trials.iter().map(|trial| trial.x0).collect();
                let x_ends: Vec<f64> = accum.trials.iter().map(|trial| trial.x_end).collect();
                summarize_fixed_time(None, case, trial_count, &physics, &x0s, &x_ends)
            }
        };

        CpuCaseResult {
            trials: accum.trials,
            case_output: CaseOutput {
                summary,
                hist_phi: accum.hist_phi,
                hist_y: accum.hist_y,
            },
            max_trial_steps,
        }
    }

    /// 通過途中の棒の (x, φ) と (x, y) を2つの2Dヒストグラムへ1票ずつ加算する。
    ///
    /// GPU カーネルの `accumulate_histograms` と同じ binning（x は1周期 [0,1) に畳み、
    /// φ は [0,2π) に巻き戻し、y は [-y_max, y_max] に収める）を CPU 側でも使う。
    fn accumulate_histograms(x: f64, y: f64, phi: f64, hist_phi: &mut [u64], hist_y: &mut [u64]) {
        let x_fraction = x - x.floor();
        let x_bin = ((x_fraction * HIST_X_BINS as f64) as usize).min(HIST_X_BINS - 1);

        let two_pi = 2.0 * std::f64::consts::PI;
        let phi_fraction = phi - two_pi * (phi / two_pi).floor();
        let phi_bin =
            ((phi_fraction / two_pi * HIST_PHI_BINS as f64) as usize).min(HIST_PHI_BINS - 1);
        hist_phi[x_bin * HIST_PHI_BINS + phi_bin] += 1;

        let y_clamped = y.clamp(-HIST_Y_MAX, HIST_Y_MAX);
        let y_fraction = (y_clamped + HIST_Y_MAX) / (2.0 * HIST_Y_MAX);
        let y_bin = ((y_fraction * HIST_Y_BINS as f64) as usize).min(HIST_Y_BINS - 1);
        hist_y[x_bin * HIST_Y_BINS + y_bin] += 1;
    }
}
