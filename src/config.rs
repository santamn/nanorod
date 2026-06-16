use std::path::PathBuf;

use clap::{Parser, ValueEnum};

// シミュレーション全体で共有する無次元化済みの物理定数。
pub const L_PERIOD: f64 = 1.0;
pub const EPSILON: f64 = 2.0;
pub const SIGMA: f64 = 8.0e-3;
pub const DT: f64 = 4.0e-7;
pub const WALL_DX: f64 = 0.25 * SIGMA;
pub const PARTICLE_DX: f64 = 0.8 * SIGMA;
pub const N_WALL: usize = (L_PERIOD / WALL_DX) as usize;
pub const WALL_K: i32 = 5;
pub const RNG_STATE_BYTES: usize = 256;
pub const DEFAULT_MAX_STEPS: u64 = 25_000_000;
pub const DEFAULT_PRODUCTION_TRIALS: usize = 30_000;
pub const DEFAULT_SMOKE_TRIALS: usize = 96;

// rod_simulation.md にある全パラメータ掃引の値。
pub const M_VALUES: [i32; 5] = [1, 4, 8, 15, 30];
pub const BETA_PE_VALUES: [f64; 6] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
pub const DELTA_ALPHA_E_OVER_P_VALUES: [f64; 6] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
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
    pub combo_limit: Option<usize>,

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
    pub beta_pe: f64,
    pub delta_alpha_e_over_p: f64,
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
            * BETA_PE_VALUES.len()
            * DELTA_ALPHA_E_OVER_P_VALUES.len()
            * FORCE_VALUES.len(),
    );

    for &m in &M_VALUES {
        for &beta_pe in &BETA_PE_VALUES {
            for &delta_alpha_e_over_p in &DELTA_ALPHA_E_OVER_P_VALUES {
                for &force in &FORCE_VALUES {
                    combos.push(SimParams {
                        combo_id: combos.len() as u32,
                        m,
                        l: particle_length(m),
                        beta_pe,
                        delta_alpha_e_over_p,
                        force,
                    });
                }
            }
        }
    }

    combos
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
}
