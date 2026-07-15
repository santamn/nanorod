//! GPU/CPU 両バックエンドとアニメーションで共有する物理モデルの数式と集計。
//!
//! 流路形状 omega(x)・境界の鏡像反射・Tirado の拡散係数・試行結果の集計など、
//! 実装バックエンドに依存しない純粋な計算をまとめる。

use serde::Serialize;

use crate::config::{BOUNDARY_REFLECTION_LIMIT, CHANNEL_NECK_PHASE, Case, L_PERIOD, Physics};

// CUDA kernel と共有する trial の状態コード。
pub const STATUS_RUNNING: i32 = 0;
pub const STATUS_OK: i32 = 1;
pub const STATUS_MAX_STEPS: i32 = 2;

// CUDA kernel と共有する1周期通過方向のコード。
pub const PASS_DIRECTION_NONE: i32 = 0;
pub const PASS_DIRECTION_RIGHT: i32 = 1;
pub const PASS_DIRECTION_LEFT: i32 = -1;

/// 基準長で固定した D_0 によって無次元化された拡散係数。
///
/// `d_perp` は仕様書の簡略化に従い、Tirado の値ではなく `0.5 * d_parallel` を使う。
#[derive(Clone, Copy, Debug)]
pub struct Diffusion {
    pub d_parallel: f64,
    pub d_perp: f64,
    pub d_r: f64,
}

/// Tirado and Garcia de la Torre の式で使うアスペクト比依存の補正項。
#[derive(Clone, Copy, Debug)]
struct TiradoTerms {
    log_p: f64,
    nu_parallel: f64,
    delta_perp: f64,
    denominator: f64,
}

/// バックエンドから回収した1試行分の詳細結果。
///
/// 各ケースの `trials.csv` に1行として書き出される。
#[derive(Clone, Debug)]
pub struct TrialResult {
    pub case_id: u32,
    pub trial_id: u64,
    /// 計算した GPU の ID。CPU バックエンドでは None。
    pub gpu_id: Option<usize>,
    pub case: Case,
    pub x0: f64,
    pub y0: f64,
    pub phi0: f64,
    pub x_end: f64,
    pub y_end: f64,
    pub phi_end: f64,
    pub t: f64,
    pub steps: u64,
    pub status: i32,
    pub pass_direction: i32,
}

/// trials.csv の1行。
#[derive(Debug, Serialize)]
pub struct TrialCsvRow {
    pub case_id: u32,
    pub trial_id: u64,
    pub gpu_id: Option<usize>,
    pub m: i32,
    pub l: f64,
    pub gamma: f64,
    pub delta: f64,
    pub f: f64,
    pub x0: f64,
    pub y0: f64,
    pub phi0: f64,
    pub x_end: f64,
    pub y_end: f64,
    pub phi_end: f64,
    #[serde(rename = "T")]
    pub t: f64,
    pub steps: u64,
    pub status: &'static str,
    pub pass_direction: &'static str,
}

/// 計測モードごとに意味が異なる統計値。
///
/// summary.json では `"mode"` タグ（`first_passage` / `fixed_time`）付きで
/// SummaryRow に平坦化して埋め込まれる。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ModeStats {
    /// first_passage モード: 1周期を初通過するまでの時間の統計。
    FirstPassage {
        /// 1周期を通過した試行数。T1・T2 の平均はこの試行だけで取る。
        n_ok: usize,
        n_right_passes: usize,
        n_left_passes: usize,
        /// 時間 T までに未通過だった試行数（平均の計算から除外）。
        n_max_steps: usize,
        passage_fraction: f64,
        /// 平均初通過時間 T₁。
        #[serde(rename = "T1")]
        t1: f64,
        /// 初通過時間の2乗平均 T₂。
        #[serde(rename = "T2")]
        t2: f64,
    },
    /// fixed_time モード: 時間 T まで走らせた変位 Δx = x(T) − x₀ の統計。
    FixedTime {
        /// 全試行を走らせた計測時間 T = max_steps × Δt。
        #[serde(rename = "T")]
        time: f64,
        /// 変位の平均 ⟨Δx⟩。
        mean_dx: f64,
        /// 変位の分散 ⟨Δx²⟩ − ⟨Δx⟩²。
        var_dx: f64,
        /// 平均速度 ⟨v⟩ = ⟨Δx⟩ / T。
        v_mean: f64,
        /// 1周期を進む平均時間 L/|⟨v⟩|（平均初通過時間の推定）。
        #[serde(rename = "T1")]
        t1: f64,
    },
}

/// 1ケース分の集計結果（summary.json の中身）。
#[derive(Clone, Debug, Serialize)]
pub struct SummaryRow {
    pub case_id: u32,
    /// 計算した GPU の ID。CPU バックエンドでは null。
    pub gpu_id: Option<usize>,
    pub m: i32,
    pub l: f64,
    pub gamma: f64,
    pub delta: f64,
    pub f: f64,
    pub n_total: usize,
    /// 計測モードとモード固有の統計値。
    #[serde(flatten)]
    pub stats: ModeStats,
    /// 非線形移動度 μ = v/f。f=0 では定義できず NaN（JSON では null）。
    pub mu: f64,
    #[serde(rename = "D_eff")]
    pub d_eff: f64,
    pub dt: f64,
    pub sigma: f64,
    pub epsilon: f64,
    pub seed: u64,
}

/// 1ケース分の集計結果と占有ヒストグラム。
///
/// trial 詳細は完了ごとに別経路で流すため、ここには含めない。
/// `hist_phi` は (x × φ)、`hist_y` は (x × y) の2Dヒストグラムを、x を最外として
/// 行優先で平坦化したもの。
#[derive(Debug)]
pub struct CaseOutput {
    pub summary: SummaryRow,
    pub hist_phi: Vec<u64>,
    pub hist_y: Vec<u64>,
}

/// 実行中に `progress.jsonl` へ書き出す進捗イベント。
#[derive(Debug, Serialize)]
pub struct ProgressRow {
    pub timestamp_ms: u128,
    /// 計算している GPU の ID。CPU バックエンドでは null。
    pub gpu_id: Option<usize>,
    pub case_id: u32,
    pub completed_cases: usize,
    pub total_cases: usize,
    pub completed_trials: usize,
    pub total_trials: usize,
    pub current_steps: u64,
    pub max_steps: u64,
    pub status: String,
}

impl TrialResult {
    /// 内部状態コードを読みやすい文字列へ変換して CSV 行にする。
    pub fn to_csv_row(&self) -> TrialCsvRow {
        TrialCsvRow {
            case_id: self.case_id,
            trial_id: self.trial_id,
            gpu_id: self.gpu_id,
            m: self.case.m,
            l: self.case.l,
            gamma: self.case.gamma,
            delta: self.case.delta,
            f: self.case.f,
            x0: self.x0,
            y0: self.y0,
            phi0: self.phi0,
            x_end: self.x_end,
            y_end: self.y_end,
            phi_end: self.phi_end,
            t: self.t,
            steps: self.steps,
            status: status_label(self.status),
            pass_direction: pass_direction_label(self.pass_direction),
        }
    }
}

/// CUDA kernel と同じ status 数値を、人間が読めるラベルへ変換する。
pub fn status_label(status: i32) -> &'static str {
    match status {
        STATUS_OK => "ok",
        STATUS_MAX_STEPS => "max_steps",
        STATUS_RUNNING => "running",
        _ => "unknown",
    }
}

/// CUDA kernel と同じ通過方向コードを、人間が読めるラベルへ変換する。
pub fn pass_direction_label(direction: i32) -> &'static str {
    match direction {
        PASS_DIRECTION_RIGHT => "right",
        PASS_DIRECTION_LEFT => "left",
        PASS_DIRECTION_NONE => "not_passed",
        _ => "unknown",
    }
}

/// 流路の半幅 `omega(x)`。
pub fn omega(x: f64) -> f64 {
    let two_pi_x = 2.0 * std::f64::consts::PI * x;
    two_pi_x.sin() + 0.25 * (2.0 * two_pi_x).sin() + 1.12
}

/// 境界条件を満たすように鏡像反射した棒の重心状態。
#[derive(Clone, Copy, Debug)]
pub struct BoundaryCorrectedState {
    pub x: f64,
    pub y: f64,
    pub phi: f64,
    /// 補正が行われたかの診断用フラグ（テストで境界処理の発火を検証するために残す）。
    #[allow(dead_code)]
    pub reflected: bool,
}

/// 流路の半幅 `omega(x)` の一階微分。
pub fn omega_derivative(x: f64) -> f64 {
    let two_pi_x = 2.0 * std::f64::consts::PI * x;
    2.0 * std::f64::consts::PI * two_pi_x.cos() + std::f64::consts::PI * (2.0 * two_pi_x).cos()
}

/// 流路の半幅 `omega(x)` の二階微分。
fn omega_second_derivative(x: f64) -> f64 {
    let two_pi_x = 2.0 * std::f64::consts::PI * x;
    let four_pi_sq = 4.0 * std::f64::consts::PI * std::f64::consts::PI;
    -four_pi_sq * two_pi_x.sin() - four_pi_sq * (2.0 * two_pi_x).sin()
}

/// 境界外に出た棒状態を、現在位置側の境界の接線に対して鏡像反射で流路内へ戻す。
pub fn reflect_state_into_channel(
    x: f64,
    y: f64,
    phi: f64,
    anchor_x: f64,
) -> BoundaryCorrectedState {
    let mut state = BoundaryCorrectedState {
        x,
        y,
        phi,
        reflected: false,
    };

    for _ in 0..BOUNDARY_REFLECTION_LIMIT {
        let Some(sign) = boundary_wall_sign(state.x, state.y) else {
            return state;
        };

        state = reflect_state_at_wall(state, sign, anchor_x);
    }

    state
}

/// 流路外にある場合は、越えている上壁または下壁の符号を返す。
fn boundary_wall_sign(x: f64, y: f64) -> Option<f64> {
    let width = omega(x);

    if y > width {
        Some(1.0)
    } else if y < -width {
        Some(-1.0)
    } else {
        None
    }
}

/// 上壁または下壁に対して重心位置と棒の向きを 1 回だけ鏡像反射する。
fn reflect_state_at_wall(
    state: BoundaryCorrectedState,
    sign: f64,
    anchor_x: f64,
) -> BoundaryCorrectedState {
    let foot_x = perpendicular_foot_x(state.x, state.y, sign, anchor_x);
    let foot_y = sign * omega(foot_x);
    let slope = sign * omega_derivative(foot_x);

    BoundaryCorrectedState {
        x: 2.0 * foot_x - state.x,
        y: 2.0 * foot_y - state.y,
        phi: reflect_angle_across_tangent(state.phi, slope),
        reflected: true,
    }
}

/// 境界の接線方向へ棒の向きベクトルを鏡映した角度を返す。
fn reflect_angle_across_tangent(phi: f64, slope: f64) -> f64 {
    2.0 * slope.atan() - phi
}

/// 点から上壁または下壁へ下ろした垂線の足の x 座標を Newton 法で求める。
fn perpendicular_foot_x(px: f64, py: f64, sign: f64, initial_x: f64) -> f64 {
    const EPSILON: f64 = 1.0e-10;
    const NEWTON_STEPS: usize = 5;

    let (section_min_x, section_max_x) = channel_section_bounds(initial_x);
    let mut x = initial_x.clamp(section_min_x, section_max_x);
    for _ in 0..NEWTON_STEPS {
        let d = wall_foot_newton_delta(px, py, sign, x);
        if d.abs() > EPSILON {
            x = (x - d).clamp(section_min_x, section_max_x);
        } else {
            break;
        }
    }
    x
}

/// 現在位置が属する流路の膨らみ区間を、左右のくびれの x 座標で返す。
fn channel_section_bounds(anchor_x: f64) -> (f64, f64) {
    let section_min_x = (anchor_x - CHANNEL_NECK_PHASE).floor() + CHANNEL_NECK_PHASE;
    (section_min_x, section_min_x + L_PERIOD)
}

/// 壁への垂線の足を求める Newton 法の 1 step 分の補正量を返す。
fn wall_foot_newton_delta(px: f64, py: f64, sign: f64, x: f64) -> f64 {
    let w_sub = omega(x) - sign * py;
    let w_p = omega_derivative(x);
    let w_pp = omega_second_derivative(x);

    (w_p * w_sub + x - px) / (w_pp * w_sub + w_p * w_p + 1.0)
}

/// 1周期分の上壁 y 座標を事前サンプリングする。
///
/// 下壁は符号を反転して参照する。
pub fn wall_y_samples(physics: &Physics) -> Vec<f64> {
    (0..physics.n_wall)
        .map(|k| omega(k as f64 * physics.wall_dx))
        .collect()
}

/// 棒長から Tirado and Garcia de la Torre の補正項を計算する。
fn tirado_terms_for_length(l: f64) -> TiradoTerms {
    let p = 60.0 * l;
    let log_p = p.ln();
    let inv_p = 1.0 / p;
    let inv_p2 = inv_p * inv_p;

    let nu_parallel = -0.207 + 0.980 * inv_p - 0.133 * inv_p2;
    let nu_perp = 0.839 + 0.185 * inv_p + 0.233 * inv_p2;
    let delta_perp = -0.662 + 0.917 * inv_p - 0.050 * inv_p2;
    let denominator = 3.0 * log_p + 2.0 * nu_parallel + nu_perp;

    TiradoTerms {
        log_p,
        nu_parallel,
        delta_perp,
        denominator,
    }
}

/// Tirado and Garcia de la Torre の式から無次元化済み拡散係数を計算する。
///
/// `reference_length` は D_0 を定める基準棒長で、全ての棒長がこの共通の D_0 で
/// 無次元化される（棒長ごとに時間スケールが変わらないようにするため）。
pub fn diffusion_for_length(l: f64, reference_length: f64) -> Diffusion {
    let terms = tirado_terms_for_length(l);
    let reference_terms = tirado_terms_for_length(reference_length);
    let translational_scale = reference_length / l;

    let d_parallel =
        4.0 * translational_scale * (terms.log_p + terms.nu_parallel) / reference_terms.denominator;
    let d_perp = 0.5 * d_parallel;
    let d_r = 24.0 * reference_length * (terms.log_p + terms.delta_perp)
        / (l * l * l * reference_terms.denominator);

    Diffusion {
        d_parallel,
        d_perp,
        d_r,
    }
}

/// first_passage モード: バックエンドから戻した初通過時間と status から summary を作る。
///
/// `max_steps` に到達した trial は `n_max_steps` に数え、平均値の計算からは除外する。
pub fn summarize_first_passage(
    gpu_id: Option<usize>,
    case: Case,
    n_total: usize,
    physics: &Physics,
    times: &[f64],
    statuses: &[i32],
    pass_directions: &[i32],
) -> SummaryRow {
    debug_assert_eq!(times.len(), statuses.len());
    debug_assert_eq!(statuses.len(), pass_directions.len());

    let mut n_ok = 0usize;
    let mut n_right_passes = 0usize;
    let mut n_left_passes = 0usize;
    let mut n_max_steps = 0usize;
    let mut sum_t = 0.0f64;
    let mut sum_t2 = 0.0f64;

    for ((&time, &status), &pass_direction) in times.iter().zip(statuses).zip(pass_directions) {
        match status {
            STATUS_OK => {
                n_ok += 1;
                sum_t += time;
                sum_t2 += time * time;
                match pass_direction {
                    PASS_DIRECTION_RIGHT => n_right_passes += 1,
                    PASS_DIRECTION_LEFT => n_left_passes += 1,
                    _ => {}
                }
            }
            STATUS_MAX_STEPS => n_max_steps += 1,
            _ => {}
        }
    }

    // 初通過した trial だけで平均初通過時間・移動度・有効拡散係数を計算する。
    let (t1, t2, mu, d_eff) = if n_ok > 0 {
        let n_ok_f = n_ok as f64;
        let t1 = sum_t / n_ok_f;
        let t2 = sum_t2 / n_ok_f;
        // μ = v/f（v = L/T1）。f=0 では移動度が定義できないので NaN を返す。
        let v = L_PERIOD / t1;
        let mu = if case.f != 0.0 { v / case.f } else { f64::NAN };
        let d_eff = 0.5 * L_PERIOD * L_PERIOD * (t2 - t1 * t1) / (t1 * t1 * t1);
        (t1, t2, mu, d_eff)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    let stats = ModeStats::FirstPassage {
        n_ok,
        n_right_passes,
        n_left_passes,
        n_max_steps,
        passage_fraction: n_ok as f64 / n_total as f64,
        t1,
        t2,
    };
    summary_row(gpu_id, case, n_total, physics, stats, mu, d_eff)
}

/// fixed_time モード: 時間 T まで走らせた全試行の変位 Δx = x_end − x₀ の統計から、
/// 平均速度・非線形移動度・有効拡散係数と、1周期を進む平均時間を直接算出する。
pub fn summarize_fixed_time(
    gpu_id: Option<usize>,
    case: Case,
    n_total: usize,
    physics: &Physics,
    x0s: &[f64],
    x_ends: &[f64],
) -> SummaryRow {
    debug_assert_eq!(x0s.len(), x_ends.len());

    // 全試行が同じステップ数だけ進むため、計測時間は一律 T = max_steps × Δt。
    let time = physics.max_steps as f64 * physics.delta_t;
    let n = x0s.len() as f64;
    let mut sum_dx = 0.0f64;
    let mut sum_dx2 = 0.0f64;
    for (&x0, &x_end) in x0s.iter().zip(x_ends) {
        let dx = x_end - x0;
        sum_dx += dx;
        sum_dx2 += dx * dx;
    }
    let mean_dx = sum_dx / n;
    // 分散 ⟨Δx²⟩ − ⟨Δx⟩²。丸め誤差でわずかに負へ落ちた場合は 0 に切り上げる。
    let var_dx = (sum_dx2 / n - mean_dx * mean_dx).max(0.0);

    let v_mean = mean_dx / time;
    // μ = ⟨v⟩/f。f=0 では移動度が定義できないので NaN を返す。
    let mu = if case.f != 0.0 {
        v_mean / case.f
    } else {
        f64::NAN
    };
    let d_eff = var_dx / (2.0 * time);
    // 1周期 L を平均速度で進むのにかかる時間。⟨v⟩=0 では無限大（JSON では null）。
    let t1 = L_PERIOD / v_mean.abs();

    let stats = ModeStats::FixedTime {
        time,
        mean_dx,
        var_dx,
        v_mean,
        t1,
    };
    summary_row(gpu_id, case, n_total, physics, stats, mu, d_eff)
}

/// モード固有の統計値と共通のメタデータから SummaryRow を組み立てる。
fn summary_row(
    gpu_id: Option<usize>,
    case: Case,
    n_total: usize,
    physics: &Physics,
    stats: ModeStats,
    mu: f64,
    d_eff: f64,
) -> SummaryRow {
    SummaryRow {
        case_id: case.case_id,
        gpu_id,
        m: case.m,
        l: case.l,
        gamma: case.gamma,
        delta: case.delta,
        f: case.f,
        n_total,
        stats,
        mu,
        d_eff,
        dt: physics.delta_t,
        sigma: physics.sigma,
        epsilon: physics.epsilon,
        seed: case.seed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Mode;

    /// σ = 8e-3, Δt = 4e-7, T = 100 の標準設定に対応する Physics を返す。
    fn test_physics() -> Physics {
        Physics {
            delta_t: 4.0e-7,
            mode: Mode::FirstPassage,
            max_steps: 250_000_000,
            sigma: 8.0e-3,
            epsilon: 2.0,
            wall_dx: 0.25 * 8.0e-3,
            particle_dx: 0.8 * 8.0e-3,
            n_wall: 500,
            diffusion_reference_length: 6.0 * 0.8 * 8.0e-3,
        }
    }

    #[test]
    fn omega_is_periodic() {
        for x in [-2.3, -0.1, 0.0, 0.25, 0.7, 3.2] {
            assert!((omega(x) - omega(x + 1.0)).abs() < 1e-12);
        }
    }

    /// 上壁の外側に出た予測点が境界の内側へ鏡映されることを確認する。
    #[test]
    fn boundary_correction_reflects_upper_wall_prediction() {
        let foot_x = 0.31;
        let foot_y = omega(foot_x);
        let slope = omega_derivative(foot_x);
        let normal_len = (slope * slope + 1.0).sqrt();
        let outside_distance = 0.08;
        let outside_x = foot_x - slope / normal_len * outside_distance;
        let outside_y = foot_y + 1.0 / normal_len * outside_distance;
        let corrected = reflect_state_into_channel(outside_x, outside_y, 0.5, outside_x);

        assert!(corrected.reflected);
        assert!((corrected.x - (foot_x + slope / normal_len * outside_distance)).abs() < 1e-9);
        assert!((corrected.y - (foot_y - 1.0 / normal_len * outside_distance)).abs() < 1e-9);
    }

    /// 上壁の接線に対して棒の角度が鏡映されることを確認する。
    #[test]
    fn boundary_correction_reflects_angle_across_upper_tangent() {
        let foot_x = 0.31;
        let foot_y = omega(foot_x);
        let slope = omega_derivative(foot_x);
        let normal_len = (slope * slope + 1.0).sqrt();
        let outside_x = foot_x - slope / normal_len * 0.08;
        let outside_y = foot_y + 1.0 / normal_len * 0.08;
        let phi = 0.5;
        let corrected = reflect_state_into_channel(outside_x, outside_y, phi, outside_x);

        assert!(corrected.reflected);
        assert!((corrected.phi - (2.0 * slope.atan() - phi)).abs() < 1e-9);
    }

    /// 流路内の予測点は補正されず、そのまま修正子段階へ渡されることを確認する。
    #[test]
    fn boundary_correction_keeps_inside_prediction() {
        let corrected = reflect_state_into_channel(0.25, 0.0, 1.2, 0.25);

        assert!(!corrected.reflected);
        assert!((corrected.x - 0.25).abs() < 1e-12);
        assert!(corrected.y.abs() < 1e-12);
        assert!((corrected.phi - 1.2).abs() < 1e-12);
    }

    /// くびれを越えた予測点でも現在位置側から垂線の足を探索することを確認する。
    #[test]
    fn boundary_correction_starts_wall_foot_search_from_current_x() {
        let current_x = 0.70;
        let predicted_x = 0.84;
        let predicted_y = omega(predicted_x) + 0.03;
        let corrected = reflect_state_into_channel(predicted_x, predicted_y, 0.0, current_x);

        assert!(corrected.reflected);
        assert!(corrected.x < CHANNEL_NECK_PHASE);
    }

    /// くびれ付近で大きく壁外に出た状態も、複数回の鏡像反射で流路内へ戻ることを確認する。
    #[test]
    fn boundary_reflection_repeats_until_state_is_inside_channel() {
        let neck_x = CHANNEL_NECK_PHASE;
        let corrected = reflect_state_into_channel(neck_x, omega(neck_x) + 0.46, 0.0, neck_x);

        assert!(corrected.reflected);
        assert!(corrected.y.abs() <= omega(corrected.x) + 1.0e-12);
    }

    #[test]
    fn wall_sampling_follows_physics_geometry() {
        let physics = test_physics();
        let samples = wall_y_samples(&physics);

        assert_eq!(samples.len(), 500);
        assert!((samples[0] - omega(0.0)).abs() < 1e-15);
        assert!((samples[499] - omega(499.0 * physics.wall_dx)).abs() < 1e-15);
    }

    #[test]
    fn diffusion_is_positive_and_finite_for_all_lengths() {
        let physics = test_physics();
        for m in [1, 4, 8, 15, 30] {
            let diffusion = diffusion_for_length(
                physics.particle_length(m),
                physics.diffusion_reference_length,
            );
            assert!(diffusion.d_parallel.is_finite() && diffusion.d_parallel > 0.0);
            assert!(diffusion.d_perp.is_finite() && diffusion.d_perp > 0.0);
            assert!(diffusion.d_r.is_finite() && diffusion.d_r > 0.0);
        }
    }

    /// 初通過した trial だけを平均しつつ、左右どちらへ通過したかを数えることを確認する。
    #[test]
    fn summarize_first_passage_counts_pass_directions() {
        let physics = test_physics();
        let case = Case {
            case_id: 7,
            m: 3,
            l: physics.particle_length(3),
            gamma: 0.25,
            delta: 0.5,
            f: 1.0,
        };
        let times = [1.0, 2.0, 0.0];
        let statuses = [STATUS_OK, STATUS_OK, STATUS_MAX_STEPS];
        let pass_directions = [
            PASS_DIRECTION_RIGHT,
            PASS_DIRECTION_LEFT,
            PASS_DIRECTION_NONE,
        ];

        let summary = summarize_first_passage(
            Some(2),
            case,
            3,
            &physics,
            &times,
            &statuses,
            &pass_directions,
        );

        let ModeStats::FirstPassage {
            n_ok,
            n_right_passes,
            n_left_passes,
            n_max_steps,
            passage_fraction,
            t1,
            t2,
        } = summary.stats
        else {
            panic!("first_passage の統計であるべきです");
        };
        assert_eq!(n_ok, 2);
        assert_eq!(n_right_passes, 1);
        assert_eq!(n_left_passes, 1);
        assert_eq!(n_max_steps, 1);
        assert!((passage_fraction - 2.0 / 3.0).abs() < 1.0e-12);
        assert!((t1 - 1.5).abs() < 1.0e-12);
        assert!((t2 - 2.5).abs() < 1.0e-12);
        assert_eq!(summary.seed, case.seed());
        assert_eq!(summary.gpu_id, Some(2));
    }

    /// 変位の統計から μ・D_eff・1周期を進む平均時間を直接算出することを確認する。
    #[test]
    fn summarize_fixed_time_computes_displacement_statistics() {
        // max_steps × Δt = 100 が計測時間 T になる。
        let physics = Physics {
            mode: Mode::FixedTime,
            ..test_physics()
        };
        let case = Case {
            case_id: 1,
            m: 3,
            l: physics.particle_length(3),
            gamma: 0.25,
            delta: 0.5,
            f: 2.0,
        };
        // Δx = 4, 6 → ⟨Δx⟩ = 5, ⟨Δx²⟩ = 26, 分散 = 1。
        let x0s = [0.0, 0.5];
        let x_ends = [4.0, 6.5];

        let summary = summarize_fixed_time(None, case, 2, &physics, &x0s, &x_ends);

        let ModeStats::FixedTime {
            time,
            mean_dx,
            var_dx,
            v_mean,
            t1,
        } = summary.stats
        else {
            panic!("fixed_time の統計であるべきです");
        };
        assert!((time - 100.0).abs() < 1.0e-9);
        assert!((mean_dx - 5.0).abs() < 1.0e-12);
        assert!((var_dx - 1.0).abs() < 1.0e-9);
        assert!((v_mean - 0.05).abs() < 1.0e-15);
        // T1 = L/|⟨v⟩| = 1/0.05、μ = ⟨v⟩/f = 0.05/2、D_eff = 分散/(2T) = 1/200。
        assert!((t1 - 20.0).abs() < 1.0e-9);
        assert!((summary.mu - 0.025).abs() < 1.0e-15);
        assert!((summary.d_eff - 0.005).abs() < 1.0e-12);
        assert_eq!(summary.n_total, 2);
    }

    /// fixed_time モードで f = 0 のとき、μ は定義できず NaN になることを確認する。
    #[test]
    fn summarize_fixed_time_returns_nan_mu_for_zero_force() {
        let physics = Physics {
            mode: Mode::FixedTime,
            ..test_physics()
        };
        let case = Case {
            case_id: 0,
            m: 1,
            l: physics.particle_length(1),
            gamma: 0.0,
            delta: 0.0,
            f: 0.0,
        };
        let summary = summarize_fixed_time(None, case, 2, &physics, &[0.0, 0.0], &[1.0, -1.0]);

        assert!(summary.mu.is_nan());
        // 変位が打ち消し合うと ⟨v⟩ = 0 になり、T1 は無限大（JSON では null）になる。
        let ModeStats::FixedTime { v_mean, t1, .. } = summary.stats else {
            panic!("fixed_time の統計であるべきです");
        };
        assert_eq!(v_mean, 0.0);
        assert!(t1.is_infinite());
    }

    /// 基準長そのものでは、固定 D_0 の式が従来の同一棒長正規化と一致することを確認する。
    #[test]
    fn diffusion_at_reference_length_matches_same_length_normalization() {
        let reference_length = test_physics().diffusion_reference_length;
        let terms = tirado_terms_for_length(reference_length);
        let diffusion = diffusion_for_length(reference_length, reference_length);

        let expected_parallel = 4.0 * (terms.log_p + terms.nu_parallel) / terms.denominator;
        let expected_rotation = 24.0 / (reference_length * reference_length)
            * (terms.log_p + terms.delta_perp)
            / terms.denominator;

        assert!((diffusion.d_parallel - expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_perp - 0.5 * expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_r - expected_rotation).abs() < 1.0e-9);
    }

    /// 基準長以外の棒でも、D_0 の分母だけは共通の基準棒長から取ることを確認する。
    #[test]
    fn diffusion_uses_shared_reference_length_for_non_reference_rods() {
        let physics = test_physics();
        let reference_length = physics.diffusion_reference_length;
        let l = physics.particle_length(1);
        let terms = tirado_terms_for_length(l);
        let reference_terms = tirado_terms_for_length(reference_length);
        let diffusion = diffusion_for_length(l, reference_length);

        let expected_parallel = 4.0 * (reference_length / l) * (terms.log_p + terms.nu_parallel)
            / reference_terms.denominator;
        let expected_rotation = 24.0 * reference_length * (terms.log_p + terms.delta_perp)
            / (l * l * l * reference_terms.denominator);

        assert!((diffusion.d_parallel - expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_perp - 0.5 * expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_r - expected_rotation).abs() < 1.0e-9);
    }
}
