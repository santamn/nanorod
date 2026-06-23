use std::path::PathBuf;

use clap::{Parser, ValueEnum};

// シミュレーション全体で共有する無次元化済みの物理定数。
pub const L_PERIOD: f64 = 1.0;
pub const EPSILON: f64 = 2.0;
pub const SIGMA: f64 = 8.0e-3;
pub const DT: f64 = 4.0e-7;
pub const WALL_DX: f64 = 0.25 * SIGMA;
pub const PARTICLE_DX: f64 = 0.8 * SIGMA;
/// 全ての棒長で同じ時間スケールを使うため、D_0 の計算にだけ使う基準棒長。
pub const DIFFUSION_REFERENCE_LENGTH: f64 = 6.0 * PARTICLE_DX;
pub const N_WALL: usize = (L_PERIOD / WALL_DX) as usize;
pub const WALL_K: i32 = 5;
pub const MAX_WALL_REPULSION_FORCE: f64 = 2.5e4;
pub const BOUNDARY_REFLECTION_LIMIT: usize = 32;
pub const CHANNEL_NECK_PHASE: f64 = 0.809_640_837_312_333_2;
pub const RNG_STATE_BYTES: usize = 256;
pub const DEFAULT_MAX_STEPS: u64 = 250_000_000; // 2.5 × 10^8
pub const DEFAULT_PRODUCTION_TRIALS: usize = 1000;
pub const DEFAULT_SMOKE_TRIALS: usize = 96;

// rod_simulation.md にある全パラメータ掃引の値。
pub const M_VALUES: [i32; 5] = [1, 4, 8, 15, 30];
pub const BETA_QE_VALUES: [f64; 6] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
pub const DELTA_ALPHA_E_OVER_QL_VALUES: [f64; 6] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
pub const FORCE_VALUES: [f64; 19] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
    90.0, 100.0,
];

/// 実行モード。smoke は詳細出力付きの小規模確認、production は全体集計用。
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RunMode {
    Smoke,
    Production,
}

/// コマンドライン引数。
///
/// デフォルトでは GPU 1,2,3 を使う smoke run にしておき、本番実行は明示的な
/// `--mode production` が必要になるようにしている。
#[derive(Debug, Parser)]
#[command(
    version,
    about = "GPU simulation for Brownian motion of rod-like particles"
)]
pub struct Cli {
    #[arg(long, value_enum, default_value_t = RunMode::Smoke)]
    pub mode: RunMode,

    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,

    #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
    pub devices: Vec<usize>,

    #[arg(long)]
    pub allow_device_zero: bool,

    #[arg(long)]
    pub trials: Option<usize>,

    #[arg(long, default_value_t = 0)]
    pub smoke_combo_id: usize,

    #[arg(long)]
    pub smoke_m: Option<i32>,

    #[arg(long)]
    pub smoke_beta_qe: Option<f64>,

    #[arg(long)]
    pub smoke_delta_alpha_e_over_ql: Option<f64>,

    #[arg(long = "smoke-f", alias = "smoke-force")]
    pub smoke_f: Option<f64>,

    #[arg(long)]
    pub combo_limit: Option<usize>,

    #[arg(long, default_value_t = 0)]
    pub combo_start: usize,

    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    pub max_steps: u64,

    #[arg(long, default_value_t = 10_000)]
    pub steps_per_launch: u32,

    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    #[arg(long, default_value_t = 5)]
    pub progress_interval_sec: u64,

    #[arg(long)]
    pub cuda_include_path: Vec<String>,

    #[arg(long, default_value = "compute_80")]
    pub cuda_arch: String,
}

/// 1つのパラメータ組み合わせ。
///
/// `m` は棒の片側代表点数で、実際の棒長は `l = 2m * 0.8sigma`。
#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    pub combo_id: u32,
    pub m: i32,
    pub l: f64,
    pub beta_qe: f64,
    pub delta_alpha_e_over_ql: f64,
    pub force: f64,
}

/// 代表点間隔 `0.8sigma` と `m` から棒長 `l` を求める。
pub fn particle_length(m: i32) -> f64 {
    2.0 * f64::from(m) * PARTICLE_DX
}

/// 仕様書の全パラメータ組み合わせを決定的な順序で列挙する。
pub fn all_parameter_combinations() -> Vec<SimParams> {
    let mut combos = Vec::with_capacity(
        M_VALUES.len()
            * BETA_QE_VALUES.len()
            * DELTA_ALPHA_E_OVER_QL_VALUES.len()
            * FORCE_VALUES.len(),
    );

    for &m in &M_VALUES {
        for &beta_qe in &BETA_QE_VALUES {
            for &delta_alpha_e_over_ql in &DELTA_ALPHA_E_OVER_QL_VALUES {
                for &force in &FORCE_VALUES {
                    combos.push(SimParams {
                        combo_id: combos.len() as u32,
                        m,
                        l: particle_length(m),
                        beta_qe,
                        delta_alpha_e_over_ql,
                        force,
                    });
                }
            }
        }
    }

    combos
}

/// production run で処理するパラメータ組み合わせだけを取り出す。
pub fn production_parameter_combinations(
    combo_start: usize,
    combo_limit: Option<usize>,
) -> anyhow::Result<Vec<SimParams>> {
    let mut combos = all_parameter_combinations();
    if let Some(limit) = combo_limit {
        combos.truncate(limit);
    }

    anyhow::ensure!(
        combo_start < combos.len(),
        "--combo-start {} leaves no parameter combinations to run after applying --combo-limit ({} combo(s) selected)",
        combo_start,
        combos.len()
    );

    if combo_start > 0 {
        combos.drain(..combo_start);
    }

    Ok(combos)
}

/// smoke run で使う1つのパラメータ組み合わせを選び、必要なら物理量を差し替える。
pub fn smoke_parameter_combination(
    smoke_combo_id: usize,
    smoke_m: Option<i32>,
    smoke_beta_qe: Option<f64>,
    smoke_delta_alpha_e_over_ql: Option<f64>,
    smoke_f: Option<f64>,
) -> anyhow::Result<SimParams> {
    let combos = all_parameter_combinations();
    let Some(mut params) = combos.get(smoke_combo_id).copied() else {
        anyhow::bail!("smoke combo id {smoke_combo_id} is out of range");
    };

    if let Some(m) = smoke_m {
        anyhow::ensure!((1..=30).contains(&m), "--smoke-m must be between 1 and 30");
        params.m = m;
        params.l = particle_length(m);
    }

    if let Some(beta_qe) = smoke_beta_qe {
        ensure_finite_smoke_override("smoke-beta-qe", beta_qe)?;
        params.beta_qe = beta_qe;
    }

    if let Some(delta_alpha_e_over_ql) = smoke_delta_alpha_e_over_ql {
        ensure_finite_smoke_override("smoke-delta-alpha-e-over-ql", delta_alpha_e_over_ql)?;
        params.delta_alpha_e_over_ql = delta_alpha_e_over_ql;
    }

    if let Some(force) = smoke_f {
        ensure_finite_smoke_override("smoke-f", force)?;
        params.force = force;
    }

    Ok(params)
}

/// smoke 専用上書き値に NaN や無限大が混ざらないことを確認する。
fn ensure_finite_smoke_override(name: &str, value: f64) -> anyhow::Result<()> {
    anyhow::ensure!(value.is_finite(), "--{} must be finite", name);
    Ok(())
}

/// architecture.md の制約に従い、GPU 0 を誤って使わないように検証する。
pub fn validate_devices(devices: &[usize], allow_device_zero: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        !devices.is_empty(),
        "at least one GPU device must be specified"
    );

    if !allow_device_zero {
        anyhow::ensure!(
            !devices.contains(&0),
            "GPU device 0 is disabled by architecture.md; pass --allow-device-zero only if you really intend to use it"
        );
    }

    Ok(())
}

/// 実行モードごとの既定 trial 数。
pub fn default_trial_count(mode: RunMode) -> usize {
    match mode {
        RunMode::Smoke => DEFAULT_SMOKE_TRIALS,
        RunMode::Production => DEFAULT_PRODUCTION_TRIALS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_matches_spec() {
        assert_eq!(all_parameter_combinations().len(), 3420);
    }

    #[test]
    fn m_values_map_to_expected_point_counts_and_lengths() {
        let expected_points = [3, 9, 17, 31, 61];
        for (&m, &points) in M_VALUES.iter().zip(&expected_points) {
            assert_eq!(2 * m + 1, points);
            assert_eq!(particle_length(m), 2.0 * f64::from(m) * 0.8 * SIGMA);
        }
    }

    /// D_0 の基準棒長が、全棒長で共有する 6 点間隔ぶんの長さになっていることを確認する。
    #[test]
    fn diffusion_reference_length_matches_channel_neck_scale() {
        assert_eq!(DIFFUSION_REFERENCE_LENGTH, 6.0 * 0.8 * SIGMA);
        assert!((DIFFUSION_REFERENCE_LENGTH - 0.0384).abs() < 1.0e-15);
    }

    /// 境界補正と反発力上限に使う数値安全用の共通定数を確認する。
    #[test]
    fn numerical_safety_constants_match_shared_model() {
        assert_eq!(MAX_WALL_REPULSION_FORCE, 2.5e4);
        assert_eq!(BOUNDARY_REFLECTION_LIMIT, 32);
        assert!((CHANNEL_NECK_PHASE - 0.809_640_837_312_333_2).abs() < 1.0e-15);
    }

    #[test]
    fn production_range_keeps_original_combo_ids() {
        let combos = production_parameter_combinations(495, Some(684)).unwrap();
        assert_eq!(combos.first().unwrap().combo_id, 495);
        assert_eq!(combos.last().unwrap().combo_id, 683);
        assert!(combos.iter().all(|params| params.m == 1));
        assert_eq!(combos.len(), 189);
    }

    #[test]
    fn production_range_rejects_empty_selection() {
        let error = production_parameter_combinations(684, Some(684)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("leaves no parameter combinations")
        );
    }

    /// smoke 専用の `m` 上書きが、本番掃引の組み合わせを増やさず棒長だけ変えることを確認する。
    #[test]
    fn smoke_parameters_can_override_m_without_changing_sweep() {
        let params = smoke_parameter_combination(0, Some(3), None, None, None).unwrap();

        assert_eq!(params.combo_id, 0);
        assert_eq!(params.m, 3);
        assert_eq!(params.l, particle_length(3));
        assert_eq!(all_parameter_combinations().len(), 3420);
    }

    /// smoke 専用の物理量上書きが、0を含む任意の有限値を受け付けることを確認する。
    #[test]
    fn smoke_parameters_can_override_physical_values() {
        let params =
            smoke_parameter_combination(0, Some(3), Some(1.0), Some(0.0), Some(0.0)).unwrap();

        assert_eq!(params.combo_id, 0);
        assert_eq!(params.m, 3);
        assert_eq!(params.l, particle_length(3));
        assert_eq!(params.beta_qe, 1.0);
        assert_eq!(params.delta_alpha_e_over_ql, 0.0);
        assert_eq!(params.force, 0.0);
        assert_eq!(all_parameter_combinations().len(), 3420);
    }

    /// smoke 専用の物理量上書きで非有限値を弾けることを確認する。
    #[test]
    fn smoke_parameters_reject_non_finite_overrides() {
        let error = smoke_parameter_combination(0, None, Some(f64::NAN), None, None).unwrap_err();

        assert!(error.to_string().contains("--smoke-beta-qe must be finite"));
    }
}
