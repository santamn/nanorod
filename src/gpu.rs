//! 物理モデルの GPU（CUDA）バックエンド。
//!
//! CUDA C ソース（kernels/simulation.cu）を実行時に NVRTC でコンパイルし、
//! 1 GPU 上で複数ケースを CUDA stream により同時並行に処理する。

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::config::{
    BOUNDARY_REFLECTION_LIMIT, CHANNEL_NECK_PHASE, Case, HIST_PHI_BINS, HIST_X_BINS, HIST_Y_BINS,
    HIST_Y_MAX, MAX_WALL_REPULSION_FORCE, Physics, WALL_K,
};
use crate::model::{
    CaseOutput, STATUS_RUNNING, TrialResult, diffusion_for_length, summarize_trials, wall_y_samples,
};

const KERNEL_SRC: &str = include_str!("kernels/simulation.cu");
/// CUDA kernel のブロックあたりスレッド数。
/// A100 での実測では 128 との差が ±3% 以内で一貫した優位がないため、256 のままにする。
const CUDA_THREADS_PER_BLOCK: u32 = 256;
/// cuRAND Philox state 1個分として確保するバイト数（実サイズは kernel 側で検査する）。
const RNG_STATE_BYTES: usize = 256;

/// CUDA kernel に値渡しするパラメータ。
///
/// `#[repr(C)]` と CUDA 側の `KernelParams` のフィールド順を一致させる必要がある。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct KernelParams {
    m: i32,
    n_wall: i32,
    wall_k: i32,
    boundary_reflection_limit: i32,
    l: f64,
    gamma: f64,
    delta: f64,
    force: f64,
    d_parallel: f64,
    d_perp: f64,
    d_r: f64,
    /// 並進・回転ノイズの係数 sqrt(2 D Δt)。ケース内で不変なのでホスト側で事前計算する。
    trans_noise_parallel: f64,
    trans_noise_perp: f64,
    rot_noise: f64,
    dt: f64,
    sigma: f64,
    epsilon: f64,
    wall_dx: f64,
    particle_dx: f64,
    rc2: f64,
    max_wall_repulsion_force: f64,
    channel_neck_phase: f64,
    hist_x_bins: i32,
    hist_phi_bins: i32,
    hist_y_bins: i32,
    hist_y_max: f64,
}

unsafe impl DeviceRepr for KernelParams {}

impl KernelParams {
    /// 共通の物理定数とケース固有のパラメータから、CUDA kernel で直接使う値へ展開する。
    fn new(physics: &Physics, case: Case) -> Self {
        let diffusion = diffusion_for_length(case.l, physics.diffusion_reference_length);

        Self {
            m: case.m,
            n_wall: physics.n_wall as i32,
            wall_k: WALL_K,
            boundary_reflection_limit: BOUNDARY_REFLECTION_LIMIT as i32,
            l: case.l,
            gamma: case.gamma,
            delta: case.delta,
            force: case.f,
            d_parallel: diffusion.d_parallel,
            d_perp: diffusion.d_perp,
            d_r: diffusion.d_r,
            trans_noise_parallel: (2.0 * diffusion.d_parallel * physics.delta_t).sqrt(),
            trans_noise_perp: (2.0 * diffusion.d_perp * physics.delta_t).sqrt(),
            rot_noise: (2.0 * diffusion.d_r * physics.delta_t).sqrt(),
            dt: physics.delta_t,
            sigma: physics.sigma,
            epsilon: physics.epsilon,
            wall_dx: physics.wall_dx,
            particle_dx: physics.particle_dx,
            rc2: physics.rc2(),
            max_wall_repulsion_force: MAX_WALL_REPULSION_FORCE,
            channel_neck_phase: CHANNEL_NECK_PHASE,
            hist_x_bins: HIST_X_BINS as i32,
            hist_phi_bins: HIST_PHI_BINS as i32,
            hist_y_bins: HIST_Y_BINS as i32,
            hist_y_max: HIST_Y_MAX,
        }
    }
}

/// GPU worker 1つに渡す実行設定。
#[derive(Clone, Debug)]
pub struct GpuRunConfig {
    pub device_id: usize,
    pub max_steps: u64,
    pub steps_per_launch: u32,
    /// 何ステップごとに角度・y分布ヒストグラムへ加算するか。0 で記録を無効にする。
    pub hist_stride: u32,
    /// 1 GPU が同時並行に走らせるケース数（= CUDA stream 数）。
    pub tasks_per_gpu: usize,
    pub progress_interval: Duration,
}

/// kernel chunk ごとに host 側へ返す進捗。
#[derive(Clone, Debug)]
pub struct GpuProgress {
    pub case_id: u32,
    pub completed_trials: usize,
    pub total_trials: usize,
    pub current_steps: u64,
    pub status: String,
}

/// 搭載されている GPU の台数を返す。
pub fn device_count() -> Result<usize> {
    let count = CudaContext::device_count().context("GPU の台数を取得できません")?;
    Ok(count as usize)
}

/// NVRTC が `curand_kernel.h` を見つけるための CUDA include path 候補を作る。
///
/// `/usr/include` は glibc ヘッダを NVRTC が追いかけて失敗しやすいため入れない。
fn cuda_include_paths() -> Vec<String> {
    let mut paths = Vec::new();

    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        paths.push(format!("{cuda_home}/include"));
    }

    for candidate in ["/usr/local/cuda/include", "/usr/local/cuda-12.2/include"] {
        if std::path::Path::new(candidate).exists() {
            paths.push(candidate.to_string());
        }
    }

    paths.sort();
    paths.dedup();
    paths
}

/// 全 slot（stream）で共有する、コンパイル済み kernel と共通の起動引数一式。
struct KernelLaunch {
    setup_rng: CudaFunction,
    init_trials: CudaFunction,
    simulate: CudaFunction,
    /// 全ケース共通・読み取り専用の上壁 y 座標サンプル。
    wall_y_dev: CudaSlice<f64>,
    cfg: LaunchConfig,
    rng_stride: i32,
    n_trials: i32,
}

/// 1つの CUDA stream が処理中のケースの進行状況。
struct SlotJob {
    case: Case,
    kparams: KernelParams,
    launched_steps: u64,
    last_progress: Instant,
}

/// 1 stream 分の再利用バッファと、現在割り当てられているケース。
///
/// ケースをまたいで同じ device バッファを使い回し、`start_job` で乱数・初期状態だけ
/// 作り直す。`job` が `None` の slot は空き。
struct StreamSlot {
    stream: Arc<CudaStream>,
    rng_states: CudaSlice<u8>,
    x0: CudaSlice<f64>,
    y0: CudaSlice<f64>,
    phi0: CudaSlice<f64>,
    x: CudaSlice<f64>,
    y: CudaSlice<f64>,
    phi: CudaSlice<f64>,
    target_right_x: CudaSlice<f64>,
    target_left_x: CudaSlice<f64>,
    times: CudaSlice<f64>,
    steps: CudaSlice<u64>,
    statuses: CudaSlice<i32>,
    pass_directions: CudaSlice<i32>,
    counters: CudaSlice<u64>,
    hist_phi: CudaSlice<u64>,
    hist_y: CudaSlice<u64>,
    job: Option<SlotJob>,
    /// trial index ごとに、確定結果を既に host へ流したか。ケースごとに `start_job` で戻す。
    emitted: Vec<bool>,
    /// `emitted` のうち true の数。新たに完了した trial の有無を counters と比べて判定する。
    emitted_count: usize,
}

impl StreamSlot {
    /// NonBlocking stream を1本作り、trial_count 分の device バッファを確保する。
    fn new(ctx: &Arc<CudaContext>, trial_count: usize) -> Result<Self> {
        let stream = ctx.new_stream()?;
        let rng_states = stream.alloc_zeros::<u8>(trial_count * RNG_STATE_BYTES)?;
        let x0 = stream.alloc_zeros::<f64>(trial_count)?;
        let y0 = stream.alloc_zeros::<f64>(trial_count)?;
        let phi0 = stream.alloc_zeros::<f64>(trial_count)?;
        let x = stream.alloc_zeros::<f64>(trial_count)?;
        let y = stream.alloc_zeros::<f64>(trial_count)?;
        let phi = stream.alloc_zeros::<f64>(trial_count)?;
        let target_right_x = stream.alloc_zeros::<f64>(trial_count)?;
        let target_left_x = stream.alloc_zeros::<f64>(trial_count)?;
        let times = stream.alloc_zeros::<f64>(trial_count)?;
        let steps = stream.alloc_zeros::<u64>(trial_count)?;
        let statuses = stream.alloc_zeros::<i32>(trial_count)?;
        let pass_directions = stream.alloc_zeros::<i32>(trial_count)?;
        let counters = stream.alloc_zeros::<u64>(2)?;
        // ケース全体（全trial・全ステップ）で共有して票を貯める2Dヒストグラム。
        let hist_phi = stream.alloc_zeros::<u64>(HIST_X_BINS * HIST_PHI_BINS)?;
        let hist_y = stream.alloc_zeros::<u64>(HIST_X_BINS * HIST_Y_BINS)?;
        Ok(Self {
            stream,
            rng_states,
            x0,
            y0,
            phi0,
            x,
            y,
            phi,
            target_right_x,
            target_left_x,
            times,
            steps,
            statuses,
            pass_directions,
            counters,
            hist_phi,
            hist_y,
            job: None,
            emitted: vec![false; trial_count],
            emitted_count: 0,
        })
    }

    /// この slot へ新しいケースを割り当て、cuRAND state と初期状態を作り直す。
    ///
    /// 乱数シードはケースのパラメータから決定論的に導出されるため、GPU への
    /// 割り当て順に依らず、同じ設定からは常に同じ結果になる。
    fn start_job(&mut self, case: Case, physics: &Physics, launch: &KernelLaunch) -> Result<()> {
        let kparams = KernelParams::new(physics, case);
        let seed = case.seed();
        let stream = self.stream.clone();

        // 前のケースの完了カウンタとヒストグラムを0に戻してから再初期化する。
        stream.memset_zeros(&mut self.counters)?;
        stream.memset_zeros(&mut self.hist_phi)?;
        stream.memset_zeros(&mut self.hist_y)?;

        // 前のケースの「出力済み」印も戻し、新しいケースを最初から数え直す。
        self.emitted.fill(false);
        self.emitted_count = 0;

        {
            let mut setup = stream.launch_builder(&launch.setup_rng);
            setup.arg(&mut self.rng_states);
            setup.arg(&launch.rng_stride);
            setup.arg(&seed);
            setup.arg(&launch.n_trials);
            unsafe { setup.launch(launch.cfg) }?;
        }

        {
            let mut init = stream.launch_builder(&launch.init_trials);
            init.arg(&kparams);
            init.arg(&mut self.rng_states);
            init.arg(&launch.rng_stride);
            init.arg(&mut self.x0);
            init.arg(&mut self.y0);
            init.arg(&mut self.phi0);
            init.arg(&mut self.x);
            init.arg(&mut self.y);
            init.arg(&mut self.phi);
            init.arg(&mut self.target_right_x);
            init.arg(&mut self.target_left_x);
            init.arg(&mut self.times);
            init.arg(&mut self.steps);
            init.arg(&mut self.statuses);
            init.arg(&mut self.pass_directions);
            init.arg(&launch.n_trials);
            unsafe { init.launch(launch.cfg) }?;
        }

        self.job = Some(SlotJob {
            case,
            kparams,
            launched_steps: 0,
            last_progress: Instant::now(),
        });
        Ok(())
    }

    /// 現在のケースを `steps_per_launch` だけ非同期に進める。
    fn launch_sim(&mut self, launch: &KernelLaunch, config: &GpuRunConfig) -> Result<()> {
        let kparams = self.job.as_ref().expect("launch_sim on idle slot").kparams;
        let stream = self.stream.clone();
        {
            let mut step = stream.launch_builder(&launch.simulate);
            step.arg(&kparams);
            step.arg(&mut self.rng_states);
            step.arg(&launch.rng_stride);
            step.arg(&launch.wall_y_dev);
            step.arg(&mut self.x);
            step.arg(&mut self.y);
            step.arg(&mut self.phi);
            step.arg(&self.target_right_x);
            step.arg(&self.target_left_x);
            step.arg(&mut self.times);
            step.arg(&mut self.steps);
            step.arg(&mut self.statuses);
            step.arg(&mut self.pass_directions);
            step.arg(&mut self.counters);
            step.arg(&mut self.hist_phi);
            step.arg(&mut self.hist_y);
            step.arg(&launch.n_trials);
            step.arg(&config.steps_per_launch);
            step.arg(&config.hist_stride);
            step.arg(&config.max_steps);
            unsafe { step.launch(launch.cfg) }?;
        }
        let job = self.job.as_mut().expect("launch_sim on idle slot");
        job.launched_steps = job
            .launched_steps
            .saturating_add(u64::from(config.steps_per_launch));
        Ok(())
    }

    /// まだ出力していない trial のうち、完了したもの（`emit_all` 時は残り全て）の確定結果を
    /// host へ回収し、`emitted` に印を付けて返す。
    ///
    /// 完了した trial は kernel が以降スキップするため、初期状態・終了状態とも確定済みで、
    /// 途中の chunk で読み出しても値は変わらない。`times`・`statuses`・`pass_directions` は
    /// 呼び出し側が既に回収済みのものを使い回す。
    fn collect_unemitted(
        &mut self,
        device_id: usize,
        case: Case,
        emit_all: bool,
        host_times: &[f64],
        host_statuses: &[i32],
        host_pass_directions: &[i32],
    ) -> Result<Vec<TrialResult>> {
        let host_x0 = self.stream.clone_dtoh(&self.x0)?;
        let host_y0 = self.stream.clone_dtoh(&self.y0)?;
        let host_phi0 = self.stream.clone_dtoh(&self.phi0)?;
        let host_x = self.stream.clone_dtoh(&self.x)?;
        let host_y = self.stream.clone_dtoh(&self.y)?;
        let host_phi = self.stream.clone_dtoh(&self.phi)?;
        let host_steps = self.stream.clone_dtoh(&self.steps)?;

        let mut rows = Vec::new();
        for idx in 0..host_statuses.len() {
            // 既に流した trial と、（最終回でなければ）まだ走行中の trial は飛ばす。
            if self.emitted[idx] || (!emit_all && host_statuses[idx] == STATUS_RUNNING) {
                continue;
            }
            self.emitted[idx] = true;
            self.emitted_count += 1;
            rows.push(TrialResult {
                case_id: case.case_id,
                trial_id: idx as u64,
                gpu_id: Some(device_id),
                case,
                x0: host_x0[idx],
                y0: host_y0[idx],
                phi0: host_phi0[idx],
                x_end: host_x[idx],
                y_end: host_y[idx],
                phi_end: host_phi[idx],
                t: host_times[idx],
                steps: host_steps[idx],
                status: host_statuses[idx],
                pass_direction: host_pass_directions[idx],
            });
        }
        Ok(rows)
    }
}

/// 1 GPU 上で複数ケースを CUDA stream で同時並行に処理し、queue を空にする。
///
/// 1ケースだけでは A100 の一部の SM しか埋まらないため、`tasks_per_gpu` 本の stream に
/// 別々のケースを載せて占有率を上げる。各 trial の計算は1スレッド=1 trial のまま変えない。
/// ケースが完了した stream はすぐ次のケースを queue から取り直すので、遅いケースが
/// GPU 全体を遊ばせ続けることもない。
///
/// 完了した trial は chunk ごとに `on_trials` で逐次流し、ケース全体の集計（summary・
/// ヒストグラム）は完了時に `on_finished` で1度だけ流す。
///
/// `next_case` が `None` を返したら新規割り当てを止め、実行中のケースを全て見送って終了する。
pub fn run_device_streamed<FNext, FProgress, FTrials, FFinished>(
    config: &GpuRunConfig,
    physics: &Physics,
    trial_count: usize,
    mut next_case: FNext,
    mut on_progress: FProgress,
    mut on_trials: FTrials,
    mut on_finished: FFinished,
) -> Result<()>
where
    FNext: FnMut() -> Option<Case>,
    FProgress: FnMut(GpuProgress),
    FTrials: FnMut(Vec<TrialResult>),
    FFinished: FnMut(CaseOutput),
{
    anyhow::ensure!(trial_count > 0, "trial_count must be positive");

    let ctx = CudaContext::new(config.device_id)
        .with_context(|| format!("GPU {} の CUDA context を作成できません", config.device_id))?;
    let ptx = compile_kernel(&ctx).context("CUDA kernel の NVRTC コンパイルに失敗しました")?;
    let module = ctx
        .load_module(ptx)
        .context("CUDA module を読み込めません")?;
    // wall_y は全ケース共通・読み取り専用。1度だけ確保し、同期してから全 stream で共有する。
    let setup_stream = ctx.default_stream();
    let wall_y = wall_y_samples(physics);
    let wall_y_dev = setup_stream.clone_htod(&wall_y)?;
    setup_stream.synchronize()?;

    let launch = KernelLaunch {
        setup_rng: module.load_function("setup_rng_kernel")?,
        init_trials: module.load_function("init_trials_kernel")?,
        simulate: module.load_function("simulate_kernel")?,
        wall_y_dev,
        cfg: launch_config_for_trials(trial_count),
        rng_stride: i32::try_from(RNG_STATE_BYTES).expect("RNG_STATE_BYTES fits in i32"),
        n_trials: i32::try_from(trial_count).context("too many trials for i32")?,
    };

    let tasks_per_gpu = config.tasks_per_gpu.max(1);
    let mut slots: Vec<StreamSlot> = Vec::with_capacity(tasks_per_gpu);
    for _ in 0..tasks_per_gpu {
        slots.push(StreamSlot::new(&ctx, trial_count)?);
    }

    // 最初のケースを各 stream へ割り当てる。ケース数が stream 数より少なくてもよい。
    for slot in &mut slots {
        let Some(case) = next_case() else {
            break;
        };
        slot.start_job(case, physics, &launch)?;
    }

    loop {
        // Phase A: アクティブな全 stream へ1 chunk 分を非同期投入し、GPU 上で重ねて走らせる。
        let mut any_active = false;
        for slot in &mut slots {
            if slot.job.is_some() {
                slot.launch_sim(&launch, config)?;
                any_active = true;
            }
        }
        if !any_active {
            break;
        }

        // Phase B: 各 stream を待って完了判定し、終わったケースを集計して次を補充する。
        for slot in &mut slots {
            if slot.job.is_none() {
                continue;
            }
            slot.stream.synchronize()?;

            let host_counters = slot.stream.clone_dtoh(&slot.counters)?;
            let completed = (host_counters[0] + host_counters[1]) as usize;
            let launched = slot.job.as_ref().expect("active slot").launched_steps;
            let done = completed == trial_count || launched >= config.max_steps;

            let elapsed = slot
                .job
                .as_ref()
                .expect("active slot")
                .last_progress
                .elapsed();
            if done || elapsed >= config.progress_interval {
                let case_id = slot.job.as_ref().expect("active slot").case.case_id;
                on_progress(GpuProgress {
                    case_id,
                    completed_trials: completed,
                    total_trials: trial_count,
                    current_steps: launched.min(config.max_steps),
                    status: if done { "completed" } else { "running" }.to_string(),
                });
                slot.job.as_mut().expect("active slot").last_progress = Instant::now();
            }

            // この chunk で新たに完了した trial があれば、確定した行をすぐ回収して流す。
            // counters は完了 trial 数なので、出力済み数を超えたぶんだけ読み出せばよい。
            if !done && completed > slot.emitted_count {
                let host_times = slot.stream.clone_dtoh(&slot.times)?;
                let host_statuses = slot.stream.clone_dtoh(&slot.statuses)?;
                let host_pass_directions = slot.stream.clone_dtoh(&slot.pass_directions)?;
                let case = slot.job.as_ref().expect("active slot").case;
                let rows = slot.collect_unemitted(
                    config.device_id,
                    case,
                    false,
                    &host_times,
                    &host_statuses,
                    &host_pass_directions,
                )?;
                if !rows.is_empty() {
                    on_trials(rows);
                }
            }

            if done {
                let host_times = slot.stream.clone_dtoh(&slot.times)?;
                let host_statuses = slot.stream.clone_dtoh(&slot.statuses)?;
                let host_pass_directions = slot.stream.clone_dtoh(&slot.pass_directions)?;
                let case = slot.job.as_ref().expect("active slot").case;

                // max_steps で未通過のまま終わった trial など、まだ出していない行を出し切る。
                // summary より先に流して、集計確定までに全 trial 行が書き終わるようにする。
                let rows = slot.collect_unemitted(
                    config.device_id,
                    case,
                    true,
                    &host_times,
                    &host_statuses,
                    &host_pass_directions,
                )?;
                if !rows.is_empty() {
                    on_trials(rows);
                }

                let summary = summarize_trials(
                    Some(config.device_id),
                    case,
                    trial_count,
                    physics,
                    &host_times,
                    &host_statuses,
                    &host_pass_directions,
                );
                let hist_phi = slot.stream.clone_dtoh(&slot.hist_phi)?;
                let hist_y = slot.stream.clone_dtoh(&slot.hist_y)?;
                on_finished(CaseOutput {
                    summary,
                    hist_phi,
                    hist_y,
                });

                // 同じバッファを使い回して次のケースを載せる。queue が空なら idle に戻す。
                match next_case() {
                    Some(case) => {
                        slot.start_job(case, physics, &launch)?;
                    }
                    None => slot.job = None,
                }
            }
        }
    }

    Ok(())
}

/// CUDA C ソースを実行時に PTX へコンパイルする。
///
/// `--gpu-architecture` は実行する GPU の compute capability から自動的に決める。
fn compile_kernel(ctx: &CudaContext) -> Result<cudarc::nvrtc::Ptx> {
    let (major, minor) = ctx
        .compute_capability()
        .context("GPU の compute capability を取得できません")?;
    let options = vec![
        format!("--gpu-architecture=compute_{major}{minor}"),
        "--std=c++14".to_string(),
    ];

    let compile_options = CompileOptions {
        include_paths: cuda_include_paths(),
        options,
        name: Some("simulation.cu".to_string()),
        ..Default::default()
    };

    Ok(compile_ptx_with_opts(KERNEL_SRC, compile_options)?)
}

fn launch_config_for_trials(trial_count: usize) -> LaunchConfig {
    let n = u32::try_from(trial_count).expect("trial count fits in u32");
    LaunchConfig {
        grid_dim: (n.div_ceil(CUDA_THREADS_PER_BLOCK), 1, 1),
        block_dim: (CUDA_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
