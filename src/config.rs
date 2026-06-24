use std::path::PathBuf;

use clap::Parser;

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
pub const DEFAULT_TRIALS: usize = 1000;

// 角度・y分布ヒストグラムの解像度と範囲。GPU kernel と CSV 出力で共有する。
// x は1周期 [0,1) に畳み、φ は [0,2π) に巻き戻し、y は [-Y_MAX, Y_MAX] に収める。
pub const HIST_X_BINS: usize = 100;
pub const HIST_PHI_BINS: usize = 36;
pub const HIST_Y_BINS: usize = 64;
/// y ヒストグラムの片側範囲。流路半幅 omega(x) の最大値（≈2.18）を覆う値にしておく。
pub const HIST_Y_MAX: f64 = 2.3;
/// 何ステップごとにヒストグラムへ加算するかの既定値。
pub const DEFAULT_HIST_STRIDE: u32 = 100;

/// コマンドライン引数。
///
/// 掃引するパラメータ（m, f, βqE, ΔαE/(qL)）はリストで与え、それらの直積を
/// combo として実行する。各 combo の結果は `output_dir` の下の専用フォルダへ保存する。
#[derive(Debug, Parser)]
#[command(
    version,
    about = "GPU simulation for Brownian motion of rod-like particles"
)]
pub struct Cli {
    /// 結果を書き出すルートディレクトリ。combo ごとにこの下へサブフォルダを作る。
    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,

    /// 使用する GPU デバイス番号（カンマ区切り）。
    #[arg(long, value_delimiter = ',', default_value = "0,1,2")]
    pub devices: Vec<usize>,

    /// 1 combo あたりの試行回数。
    #[arg(long, default_value_t = DEFAULT_TRIALS)]
    pub trials: usize,

    /// 棒の片側代表点数 m のリスト（カンマ区切り）。
    #[arg(long, value_delimiter = ',', required = true)]
    pub m: Vec<i32>,

    /// 駆動力 f のリスト。`1/3` のような分数も受け付ける。
    #[arg(long = "f", value_delimiter = ',', value_parser = parse_ratio, required = true)]
    pub f: Vec<f64>,

    /// βqE のリスト。`1/3` のような分数も受け付ける。
    #[arg(long = "beta-qe", value_delimiter = ',', value_parser = parse_ratio, required = true)]
    pub beta_qe: Vec<f64>,

    /// ΔαE/(qL) のリスト。`1/3` のような分数も受け付ける。
    #[arg(long = "delta-alpha-e-over-ql", value_delimiter = ',', value_parser = parse_ratio, required = true)]
    pub delta_alpha_e_over_ql: Vec<f64>,

    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    pub max_steps: u64,

    #[arg(long, default_value_t = 10_000)]
    pub steps_per_launch: u32,

    /// 角度・y分布ヒストグラムへ加算するステップ間隔。0 で記録を無効にする。
    #[arg(long, default_value_t = DEFAULT_HIST_STRIDE)]
    pub hist_stride: u32,

    /// 各 GPU が同時並行に処理する combo 数（=CUDA stream 数）。
    ///
    /// N=1000 では1 combo が A100 の数 SM しか使わないため、複数 combo を
    /// 並行させて占有率を上げる。約16でA100が飽和する（measured knee）。
    #[arg(long, default_value_t = 16)]
    pub streams: usize,

    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    #[arg(long, default_value_t = 5)]
    pub progress_interval_sec: u64,

    #[arg(long)]
    pub cuda_include_path: Vec<String>,

    #[arg(long, default_value = "compute_80")]
    pub cuda_arch: String,
}

/// `a/b` の分数表記と通常の小数表記の両方を f64 として解釈する。
///
/// `1/3` のように割り切れない値もコマンドラインから直接与えられるようにする。
fn parse_ratio(text: &str) -> Result<f64, String> {
    let text = text.trim();

    let value = if let Some((numerator, denominator)) = text.split_once('/') {
        let numerator: f64 = numerator
            .trim()
            .parse()
            .map_err(|_| format!("invalid numerator in `{text}`"))?;
        let denominator: f64 = denominator
            .trim()
            .parse()
            .map_err(|_| format!("invalid denominator in `{text}`"))?;
        if denominator == 0.0 {
            return Err(format!("division by zero in `{text}`"));
        }
        numerator / denominator
    } else {
        text.parse()
            .map_err(|_| format!("`{text}` is not a number"))?
    };

    if !value.is_finite() {
        return Err(format!("`{text}` is not a finite number"));
    }
    Ok(value)
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

/// CLI で与えられた各パラメータのリストから、その直積を combo として列挙する。
///
/// 列挙順は m → βqE → ΔαE/(qL) → f の入れ子で、`combo_id` を 0 から振り直す。
/// 同じ値が重複して与えられても、combo が二重にならないよう各リストを重複排除する。
pub fn parameter_combinations(
    m_values: &[i32],
    beta_qe_values: &[f64],
    delta_alpha_e_over_ql_values: &[f64],
    force_values: &[f64],
) -> anyhow::Result<Vec<SimParams>> {
    for &m in m_values {
        anyhow::ensure!(m >= 1, "m must be at least 1, got {m}");
    }

    let m_values = dedup_in_order_i32(m_values);
    let beta_qe_values = dedup_in_order_f64(beta_qe_values);
    let delta_values = dedup_in_order_f64(delta_alpha_e_over_ql_values);
    let force_values = dedup_in_order_f64(force_values);

    let mut combos = Vec::with_capacity(
        m_values.len() * beta_qe_values.len() * delta_values.len() * force_values.len(),
    );

    for &m in &m_values {
        for &beta_qe in &beta_qe_values {
            for &delta_alpha_e_over_ql in &delta_values {
                for &force in &force_values {
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

    anyhow::ensure!(!combos.is_empty(), "no parameter combinations to run");
    Ok(combos)
}

/// combo の物理パラメータから、結果を保存するサブフォルダ名を作る。
pub fn combo_dir_name(params: &SimParams) -> String {
    format!(
        "m{}_f{}_beta{}_delta{}",
        params.m,
        format_param(params.force),
        format_param(params.beta_qe),
        format_param(params.delta_alpha_e_over_ql),
    )
}

/// フォルダ名用に f64 を、末尾の余分なゼロを落とした短い文字列へ整形する。
fn format_param(value: f64) -> String {
    let text = format!("{value:.6}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

/// i32 のリストを、出現順を保ったまま重複排除する。
fn dedup_in_order_i32(values: &[i32]) -> Vec<i32> {
    let mut unique = Vec::with_capacity(values.len());
    for &value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

/// f64 のリストを、ビット表現で同一視して出現順を保ったまま重複排除する。
fn dedup_in_order_f64(values: &[f64]) -> Vec<f64> {
    let mut unique: Vec<f64> = Vec::with_capacity(values.len());
    for &value in values {
        if !unique.iter().any(|&kept| kept.to_bits() == value.to_bits()) {
            unique.push(value);
        }
    }
    unique
}

/// 少なくとも1つの GPU デバイスが指定されていることを確認する。
pub fn validate_devices(devices: &[usize]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !devices.is_empty(),
        "at least one GPU device must be specified"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_combinations_form_cartesian_product() {
        let combos =
            parameter_combinations(&[1, 4], &[0.25, 0.5], &[1.0], &[1.0, 2.0, 3.0]).unwrap();

        #[allow(clippy::identity_op)]
        {
            assert_eq!(combos.len(), 2 * 2 * 1 * 3);
        }
        // combo_id は 0 から連番で振られる。
        assert_eq!(combos.first().unwrap().combo_id, 0);
        assert_eq!(combos.last().unwrap().combo_id, 11);
        // 入れ子の最内は f なので、先頭2要素は f だけが変わる。
        assert_eq!(combos[0].m, 1);
        assert_eq!(combos[0].force, 1.0);
        assert_eq!(combos[1].force, 2.0);
    }

    #[test]
    fn parameter_combinations_dedup_repeated_values() {
        let combos = parameter_combinations(&[1, 1, 4], &[0.5, 0.5], &[1.0], &[2.0]).unwrap();
        assert_eq!(combos.len(), 2);
    }

    #[test]
    fn parameter_combinations_reject_non_positive_m() {
        let error = parameter_combinations(&[0], &[0.5], &[1.0], &[2.0]).unwrap_err();
        assert!(error.to_string().contains("m must be at least 1"));
    }

    #[test]
    fn parse_ratio_accepts_fractions_and_decimals() {
        assert!((parse_ratio("1/3").unwrap() - 1.0 / 3.0).abs() < 1.0e-15);
        assert_eq!(parse_ratio("0.5").unwrap(), 0.5);
        assert_eq!(parse_ratio(" 3 / 4 ").unwrap(), 0.75);
        assert!(parse_ratio("1/0").is_err());
        assert!(parse_ratio("abc").is_err());
    }

    #[test]
    fn combo_dir_name_trims_trailing_zeros() {
        let params = SimParams {
            combo_id: 0,
            m: 3,
            l: particle_length(3),
            beta_qe: 1.0,
            delta_alpha_e_over_ql: 0.5,
            force: 0.0,
        };
        assert_eq!(combo_dir_name(&params), "m3_f0_beta1_delta0.5");
    }

    #[test]
    fn m_values_map_to_expected_point_counts_and_lengths() {
        let expected_points = [(1, 3), (4, 9), (8, 17), (15, 31), (30, 61)];
        for (m, points) in expected_points {
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
}
