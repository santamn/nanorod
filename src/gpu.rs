use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::config::{
    BOUNDARY_REFLECTION_LIMIT, CHANNEL_NECK_PHASE, DT, EPSILON, HIST_PHI_BINS, HIST_X_BINS,
    HIST_Y_BINS, HIST_Y_MAX, MAX_WALL_REPULSION_FORCE, RNG_STATE_BYTES, SIGMA, SimParams, WALL_DX,
    WALL_K,
};
use crate::model::{
    SummaryRow, TrialResult, diffusion_for_length, summarize_trials, wall_y_samples,
};

const KERNEL_SRC: &str = include_str!("kernels/simulation.cu");
const CUDA_THREADS_PER_BLOCK: u32 = 256;

/// CUDA kernel に値渡しするパラメータ。
///
/// `#[repr(C)]` と CUDA 側の `KernelParams` のフィールド順を一致させる必要がある。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct KernelParams {
    combo_id: i32,
    m: i32,
    n_wall: i32,
    wall_k: i32,
    boundary_reflection_limit: i32,
    l: f64,
    beta_qe: f64,
    delta_alpha_e_over_ql: f64,
    force: f64,
    d_parallel: f64,
    d_perp: f64,
    d_r: f64,
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
    /// Rust 側の物理パラメータから、CUDA kernel で直接使う値へ展開する。
    fn from_params(params: SimParams) -> Self {
        let diffusion = diffusion_for_length(params.l);
        let rc = 2.0_f64.powf(1.0 / 6.0) * SIGMA;

        Self {
            combo_id: params.combo_id as i32,
            m: params.m,
            n_wall: crate::config::N_WALL as i32,
            wall_k: WALL_K,
            boundary_reflection_limit: BOUNDARY_REFLECTION_LIMIT as i32,
            l: params.l,
            beta_qe: params.beta_qe,
            delta_alpha_e_over_ql: params.delta_alpha_e_over_ql,
            force: params.force,
            d_parallel: diffusion.d_parallel,
            d_perp: diffusion.d_perp,
            d_r: diffusion.d_r,
            dt: DT,
            sigma: SIGMA,
            epsilon: EPSILON,
            wall_dx: WALL_DX,
            particle_dx: crate::config::PARTICLE_DX,
            rc2: rc * rc,
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
    pub seed: u64,
    pub max_steps: u64,
    pub steps_per_launch: u32,
    /// 何ステップごとに角度・y分布ヒストグラムへ加算するか。0 で記録を無効にする。
    pub hist_stride: u32,
    /// production で1 GPU が同時並行に走らせる combo 数（CUDA stream 数）。
    pub num_streams: usize,
    pub cuda_include_paths: Vec<String>,
    pub cuda_arch: String,
    pub progress_interval: Duration,
}

/// 1 combo の GPU 実行結果。
///
/// `hist_phi` は (x × φ)、`hist_y` は (x × y) の2Dヒストグラムを、x を最外として
/// 行優先で平坦化したもの。
#[derive(Debug)]
pub struct ComboOutput {
    pub summary: SummaryRow,
    pub trials: Vec<TrialResult>,
    pub hist_phi: Vec<u64>,
    pub hist_y: Vec<u64>,
}

/// kernel chunk ごとに host 側へ返す進捗。
#[derive(Clone, Debug)]
pub struct GpuProgress {
    pub combo_id: u32,
    pub completed_trials: usize,
    pub total_trials: usize,
    pub current_steps: u64,
    pub status: String,
}

/// NVRTC が `curand_kernel.h` を見つけるための CUDA include path 候補を作る。
///
/// `/usr/include` は glibc ヘッダを NVRTC が追いかけて失敗しやすいため入れない。
pub fn cuda_include_paths(user_paths: &[String]) -> Vec<String> {
    let mut paths = user_paths.to_vec();

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

/// 1つの CUDA stream が処理中の combo の進行状況。
struct SlotJob {
    params: SimParams,
    kparams: KernelParams,
    launched_steps: u64,
    last_progress: Instant,
}

/// 1 stream 分の再利用バッファと、現在割り当てられている combo。
///
/// production では combo をまたいで同じ device バッファを使い回し、`start_job` で
/// 乱数・初期状態だけ作り直す。`job` が `None` の slot は空き。
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
        // combo 全体（全trial・全ステップ）で共有して票を貯める2Dヒストグラム。
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
        })
    }

    /// この slot へ新しい combo を割り当て、cuRAND state と初期状態を作り直す。
    ///
    /// combo ごとに `combo_id` から決まる `sequence_offset` を使うので、同じ seed なら
    /// 割り当て順に依らずビット単位で同じ結果になる。
    fn start_job(
        &mut self,
        params: SimParams,
        seed: u64,
        rng_stride: i32,
        n_trials: i32,
        cfg: LaunchConfig,
        setup_rng: &CudaFunction,
        init_trials: &CudaFunction,
    ) -> Result<()> {
        let kparams = KernelParams::from_params(params);
        let sequence_offset = u64::from(params.combo_id) * 1_000_000_000;
        let stream = self.stream.clone();

        // 前の combo の完了カウンタとヒストグラムを0に戻してから再初期化する。
        stream.memset_zeros(&mut self.counters)?;
        stream.memset_zeros(&mut self.hist_phi)?;
        stream.memset_zeros(&mut self.hist_y)?;

        {
            let mut setup = stream.launch_builder(setup_rng);
            setup.arg(&mut self.rng_states);
            setup.arg(&rng_stride);
            setup.arg(&seed);
            setup.arg(&sequence_offset);
            setup.arg(&n_trials);
            unsafe { setup.launch(cfg) }?;
        }

        {
            let mut init = stream.launch_builder(init_trials);
            init.arg(&kparams);
            init.arg(&mut self.rng_states);
            init.arg(&rng_stride);
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
            init.arg(&n_trials);
            unsafe { init.launch(cfg) }?;
        }

        self.job = Some(SlotJob {
            params,
            kparams,
            launched_steps: 0,
            last_progress: Instant::now(),
        });
        Ok(())
    }

    /// 現在の combo を `steps_per_launch` だけ非同期に進める。
    fn launch_sim(
        &mut self,
        simulate: &CudaFunction,
        wall_y_dev: &CudaSlice<f64>,
        cfg: LaunchConfig,
        rng_stride: i32,
        n_trials: i32,
        steps_per_launch: u32,
        hist_stride: u32,
        max_steps: u64,
    ) -> Result<()> {
        let kparams = self.job.as_ref().expect("launch_sim on idle slot").kparams;
        let stream = self.stream.clone();
        {
            let mut step = stream.launch_builder(simulate);
            step.arg(&kparams);
            step.arg(&mut self.rng_states);
            step.arg(&rng_stride);
            step.arg(wall_y_dev);
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
            step.arg(&n_trials);
            step.arg(&steps_per_launch);
            step.arg(&hist_stride);
            step.arg(&max_steps);
            unsafe { step.launch(cfg) }?;
        }
        let job = self.job.as_mut().expect("launch_sim on idle slot");
        job.launched_steps = job.launched_steps.saturating_add(u64::from(steps_per_launch));
        Ok(())
    }
}

/// 1 GPU 上で複数 combo を CUDA stream で同時並行に処理し、queue を空にする。
///
/// N=1000 の1 combo は A100 の数 SM しか埋めないため、`num_streams` 本の stream に
/// 別々の combo を載せて占有率を上げる。各 trial の計算は1スレッド=1 trial のまま変えない。
/// combo が完了した stream はすぐ次の combo を queue から取り直すので、遅い combo が
/// GPU 全体を遊ばせ続けることもない。
///
/// `next_params` が `None` を返したら新規割り当てを止め、実行中の combo を全て見送って終了する。
pub fn run_device_streamed<FNext, FProgress, FFinished>(
    config: &GpuRunConfig,
    trial_count: usize,
    mut next_params: FNext,
    mut on_progress: FProgress,
    mut on_finished: FFinished,
) -> Result<()>
where
    FNext: FnMut() -> Option<SimParams>,
    FProgress: FnMut(GpuProgress),
    FFinished: FnMut(ComboOutput),
{
    anyhow::ensure!(trial_count > 0, "trial_count must be positive");

    let ptx = compile_kernel(config).context("failed to compile CUDA kernel with NVRTC")?;
    let ctx = CudaContext::new(config.device_id).with_context(|| {
        format!(
            "failed to create CUDA context for device {}",
            config.device_id
        )
    })?;
    let module = ctx.load_module(ptx).context("failed to load CUDA module")?;
    let setup_rng = module.load_function("setup_rng_kernel")?;
    let init_trials = module.load_function("init_trials_kernel")?;
    let simulate = module.load_function("simulate_kernel")?;

    let rng_stride = i32::try_from(RNG_STATE_BYTES).expect("RNG_STATE_BYTES fits in i32");
    let n_trials_i32 = i32::try_from(trial_count).context("too many trials for i32")?;
    let cfg = launch_config_for_trials(trial_count);

    // wall_y は全 combo 共通・読み取り専用。1度だけ確保し、同期してから全 stream で共有する。
    let setup_stream = ctx.default_stream();
    let wall_y = wall_y_samples();
    let wall_y_dev = setup_stream.clone_htod(&wall_y)?;
    setup_stream.synchronize()?;

    let num_streams = config.num_streams.max(1);
    let mut slots: Vec<StreamSlot> = Vec::with_capacity(num_streams);
    for _ in 0..num_streams {
        slots.push(StreamSlot::new(&ctx, trial_count)?);
    }

    // 最初の combo を各 stream へ割り当てる。combo 数が stream 数より少なくてもよい。
    for slot in &mut slots {
        let Some(params) = next_params() else {
            break;
        };
        slot.start_job(
            params,
            config.seed,
            rng_stride,
            n_trials_i32,
            cfg,
            &setup_rng,
            &init_trials,
        )?;
    }

    loop {
        // Phase A: アクティブな全 stream へ1 chunk 分を非同期投入し、GPU 上で重ねて走らせる。
        let mut any_active = false;
        for slot in &mut slots {
            if slot.job.is_some() {
                slot.launch_sim(
                    &simulate,
                    &wall_y_dev,
                    cfg,
                    rng_stride,
                    n_trials_i32,
                    config.steps_per_launch,
                    config.hist_stride,
                    config.max_steps,
                )?;
                any_active = true;
            }
        }
        if !any_active {
            break;
        }

        // Phase B: 各 stream を待って完了判定し、終わった combo を集計して次を補充する。
        for slot in &mut slots {
            if slot.job.is_none() {
                continue;
            }
            slot.stream.synchronize()?;

            let host_counters = slot.stream.clone_dtoh(&slot.counters)?;
            let completed = (host_counters[0] + host_counters[1]) as usize;
            let launched = slot.job.as_ref().expect("active slot").launched_steps;
            let done = completed == trial_count || launched >= config.max_steps;

            let elapsed = slot.job.as_ref().expect("active slot").last_progress.elapsed();
            if done || elapsed >= config.progress_interval {
                let combo_id = slot.job.as_ref().expect("active slot").params.combo_id;
                on_progress(GpuProgress {
                    combo_id,
                    completed_trials: completed,
                    total_trials: trial_count,
                    current_steps: launched.min(config.max_steps),
                    status: if done { "completed" } else { "running" }.to_string(),
                });
                slot.job.as_mut().expect("active slot").last_progress = Instant::now();
            }

            if done {
                let host_times = slot.stream.clone_dtoh(&slot.times)?;
                let host_statuses = slot.stream.clone_dtoh(&slot.statuses)?;
                let host_pass_directions = slot.stream.clone_dtoh(&slot.pass_directions)?;
                let params = slot.job.as_ref().expect("active slot").params;
                let summary = summarize_trials(
                    config.device_id,
                    params,
                    trial_count,
                    config.seed,
                    &host_times,
                    &host_statuses,
                    &host_pass_directions,
                );
                let trials = capture_trials(
                    slot,
                    config.device_id,
                    params,
                    trial_count,
                    &host_times,
                    &host_statuses,
                    &host_pass_directions,
                )?;
                let hist_phi = slot.stream.clone_dtoh(&slot.hist_phi)?;
                let hist_y = slot.stream.clone_dtoh(&slot.hist_y)?;
                on_finished(ComboOutput {
                    summary,
                    trials,
                    hist_phi,
                    hist_y,
                });

                // 同じバッファを使い回して次の combo を載せる。queue が空なら idle に戻す。
                match next_params() {
                    Some(params) => {
                        slot.start_job(
                            params,
                            config.seed,
                            rng_stride,
                            n_trials_i32,
                            cfg,
                            &setup_rng,
                            &init_trials,
                        )?;
                    }
                    None => slot.job = None,
                }
            }
        }
    }

    Ok(())
}

/// 完了した combo の各 trial の初期状態・終了状態を host へ回収し、詳細結果にまとめる。
///
/// `times`・`statuses`・`pass_directions` は呼び出し側が既に回収済みのものを使い回す。
fn capture_trials(
    slot: &StreamSlot,
    device_id: usize,
    params: SimParams,
    trial_count: usize,
    host_times: &[f64],
    host_statuses: &[i32],
    host_pass_directions: &[i32],
) -> Result<Vec<TrialResult>> {
    let host_x0 = slot.stream.clone_dtoh(&slot.x0)?;
    let host_y0 = slot.stream.clone_dtoh(&slot.y0)?;
    let host_phi0 = slot.stream.clone_dtoh(&slot.phi0)?;
    let host_x = slot.stream.clone_dtoh(&slot.x)?;
    let host_y = slot.stream.clone_dtoh(&slot.y)?;
    let host_phi = slot.stream.clone_dtoh(&slot.phi)?;
    let host_steps = slot.stream.clone_dtoh(&slot.steps)?;

    Ok((0..trial_count)
        .map(|idx| TrialResult {
            combo_id: params.combo_id,
            trial_id: idx as u64,
            device_id,
            params,
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
        })
        .collect())
}

/// CUDA C ソースを実行時に PTX へコンパイルする。
fn compile_kernel(config: &GpuRunConfig) -> Result<cudarc::nvrtc::Ptx> {
    let options = vec![
        format!("--gpu-architecture={}", config.cuda_arch),
        "--std=c++14".to_string(),
    ];

    let compile_options = CompileOptions {
        include_paths: config.cuda_include_paths.clone(),
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
