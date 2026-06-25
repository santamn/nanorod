use serde::Serialize;

use crate::config::{
    BOUNDARY_REFLECTION_LIMIT, CHANNEL_NECK_PHASE, DIFFUSION_REFERENCE_LENGTH, EPSILON, L_PERIOD,
    N_WALL, SIGMA, SimParams, WALL_DX, WALL_K, particle_length,
};

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

/// GPU から回収した1試行分の詳細結果。
///
/// smoke run の `smoke_trials.csv` にだけ展開される。
#[derive(Clone, Debug)]
pub struct TrialResult {
    pub combo_id: u32,
    pub trial_id: u64,
    pub device_id: usize,
    pub params: SimParams,
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

/// smoke run で出力する trial 詳細 CSV の1行。
#[derive(Debug, Serialize)]
pub struct TrialCsvRow {
    pub combo_id: u32,
    pub trial_id: u64,
    pub device_id: usize,
    pub m: i32,
    pub l: f64,
    pub beta_pe: f64,
    pub abs_delta_alpha_e_over_p: f64,
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

/// パラメータ組み合わせごとの集計 JSON の1要素。
///
/// production run ではこの行だけを出力し、trial 詳細は保存しない。
#[derive(Clone, Debug, Serialize)]
pub struct SummaryRow {
    pub combo_id: u32,
    pub device_id: usize,
    pub m: i32,
    pub l: f64,
    pub beta_pe: f64,
    pub abs_delta_alpha_e_over_p: f64,
    pub f: f64,
    pub n_total: usize,
    pub n_ok: usize,
    pub n_right_passes: usize,
    pub n_left_passes: usize,
    pub n_max_steps: usize,
    pub passage_fraction: f64,
    #[serde(rename = "T1")]
    pub t1: f64,
    #[serde(rename = "T2")]
    pub t2: f64,
    /// 移動度 μ = v/f（v = L/T1 は1周期あたりの平均速度）。f=0 では定義できず NaN。
    pub mu: f64,
    #[serde(rename = "D_eff")]
    pub d_eff: f64,
    pub dt: f64,
    pub sigma: f64,
    pub epsilon: f64,
    pub seed: u64,
}

/// 実行中に `progress.jsonl` へ書き出す進捗イベント。
#[derive(Debug, Serialize)]
pub struct ProgressRow {
    pub timestamp_ms: u128,
    pub device_id: usize,
    pub combo_id: u32,
    pub completed_combos: usize,
    pub total_combos: usize,
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
            combo_id: self.combo_id,
            trial_id: self.trial_id,
            device_id: self.device_id,
            m: self.params.m,
            l: self.params.l,
            beta_pe: self.params.beta_pe,
            abs_delta_alpha_e_over_p: self.params.abs_delta_alpha_e_over_p,
            f: self.params.force,
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
/// 下壁は CUDA kernel 側で符号を反転して参照する。
pub fn wall_y_samples() -> Vec<f64> {
    (0..N_WALL).map(|k| omega(k as f64 * WALL_DX)).collect()
}

/// 棒長から Tirado and Garcia de la Torre の補正項を計算する。
fn tirado_terms_for_length(l: f64) -> TiradoTerms {
    let p = 60.0 * l;
    let log_p = p.ln();
    let inv_p = 1.0 / p;
    let inv_p2 = inv_p * inv_p;

    let nu_parallel = -0.207 + 0.980 * inv_p - 0.133 * inv_p2;
    let nu_perp = 0.839 + 0.185 * inv_p + 0.233 * inv_p2;
    let delta_perp = -0.630 + 0.917 * inv_p - 0.050 * inv_p2;
    let denominator = 3.0 * log_p + 2.0 * nu_parallel + nu_perp;

    TiradoTerms {
        log_p,
        nu_parallel,
        delta_perp,
        denominator,
    }
}

/// Tirado and Garcia de la Torre の式から無次元化済み拡散係数を計算する。
pub fn diffusion_for_length(l: f64) -> Diffusion {
    let terms = tirado_terms_for_length(l);
    let reference_terms = tirado_terms_for_length(DIFFUSION_REFERENCE_LENGTH);
    let translational_scale = DIFFUSION_REFERENCE_LENGTH / l;

    let d_parallel =
        4.0 * translational_scale * (terms.log_p + terms.nu_parallel) / reference_terms.denominator;
    let d_perp = 0.5 * d_parallel;
    let d_r = 24.0 * DIFFUSION_REFERENCE_LENGTH * (terms.log_p + terms.delta_perp)
        / (l * l * l * reference_terms.denominator);

    Diffusion {
        d_parallel,
        d_perp,
        d_r,
    }
}

/// GPU から戻した初通過時間と status から、1 combo 分の summary を作る。
///
/// `max_steps` に到達した trial は `n_max_steps` に数え、平均値の計算からは除外する。
pub fn summarize_trials(
    device_id: usize,
    params: SimParams,
    n_total: usize,
    seed: u64,
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
        let mu = if params.force != 0.0 {
            v / params.force
        } else {
            f64::NAN
        };
        let d_eff = 0.5 * L_PERIOD * L_PERIOD * (t2 - t1 * t1) / (t1 * t1 * t1);
        (t1, t2, mu, d_eff)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    SummaryRow {
        combo_id: params.combo_id,
        device_id,
        m: params.m,
        l: params.l,
        beta_pe: params.beta_pe,
        abs_delta_alpha_e_over_p: params.abs_delta_alpha_e_over_p,
        f: params.force,
        n_total,
        n_ok,
        n_right_passes,
        n_left_passes,
        n_max_steps,
        passage_fraction: n_ok as f64 / n_total as f64,
        t1,
        t2,
        mu,
        d_eff,
        dt: crate::config::DT,
        sigma: SIGMA,
        epsilon: EPSILON,
        seed,
    }
}

/// 仕様書から決まる静的な幾何パラメータが、実装上も一致しているか確認する。
pub fn validate_static_geometry() -> anyhow::Result<()> {
    anyhow::ensure!(N_WALL == 500, "expected 500 wall samples, got {N_WALL}");
    anyhow::ensure!(WALL_K == 5, "expected wall neighbor K=5, got {WALL_K}");
    anyhow::ensure!((particle_length(1) - 2.0 * 0.8 * SIGMA).abs() < 1e-15);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(corrected.x < 0.8096408373123332);
    }

    /// くびれ付近で大きく壁外に出た状態も、複数回の鏡像反射で流路内へ戻ることを確認する。
    #[test]
    fn boundary_reflection_repeats_until_state_is_inside_channel() {
        let neck_x = 0.8096408373123332;
        let corrected = reflect_state_into_channel(neck_x, omega(neck_x) + 0.46, 0.0, neck_x);

        assert!(corrected.reflected);
        assert!(corrected.y.abs() <= omega(corrected.x) + 1.0e-12);
    }

    #[test]
    fn wall_sampling_constants_match_spec() {
        assert_eq!(N_WALL, 500);
        assert_eq!(WALL_K, 5);
        assert_eq!(wall_y_samples().len(), 500);
    }

    #[test]
    fn diffusion_is_positive_and_finite_for_all_lengths() {
        for m in [1, 4, 8, 15, 30] {
            let diffusion = diffusion_for_length(particle_length(m));
            assert!(diffusion.d_parallel.is_finite() && diffusion.d_parallel > 0.0);
            assert!(diffusion.d_perp.is_finite() && diffusion.d_perp > 0.0);
            assert!(diffusion.d_r.is_finite() && diffusion.d_r > 0.0);
        }
    }

    /// 初通過した trial だけを平均しつつ、左右どちらへ通過したかを数えることを確認する。
    #[test]
    fn summarize_trials_counts_pass_directions() {
        let params = SimParams {
            combo_id: 7,
            m: 3,
            l: particle_length(3),
            beta_pe: 0.25,
            abs_delta_alpha_e_over_p: 0.5,
            force: 1.0,
        };
        let times = [1.0, 2.0, 0.0];
        let statuses = [STATUS_OK, STATUS_OK, STATUS_MAX_STEPS];
        let pass_directions = [
            PASS_DIRECTION_RIGHT,
            PASS_DIRECTION_LEFT,
            PASS_DIRECTION_NONE,
        ];

        let summary = summarize_trials(2, params, 3, 11, &times, &statuses, &pass_directions);

        assert_eq!(summary.n_ok, 2);
        assert_eq!(summary.n_right_passes, 1);
        assert_eq!(summary.n_left_passes, 1);
        assert_eq!(summary.n_max_steps, 1);
        assert!((summary.passage_fraction - 2.0 / 3.0).abs() < 1.0e-12);
        assert!((summary.t1 - 1.5).abs() < 1.0e-12);
        assert!((summary.t2 - 2.5).abs() < 1.0e-12);
    }

    /// 基準長そのものでは、固定 D_0 の式が従来の同一棒長正規化と一致することを確認する。
    #[test]
    fn diffusion_at_reference_length_matches_same_length_normalization() {
        let terms = tirado_terms_for_length(DIFFUSION_REFERENCE_LENGTH);
        let diffusion = diffusion_for_length(DIFFUSION_REFERENCE_LENGTH);

        let expected_parallel = 4.0 * (terms.log_p + terms.nu_parallel) / terms.denominator;
        let expected_rotation = 24.0 / (DIFFUSION_REFERENCE_LENGTH * DIFFUSION_REFERENCE_LENGTH)
            * (terms.log_p + terms.delta_perp)
            / terms.denominator;

        assert!((diffusion.d_parallel - expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_perp - 0.5 * expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_r - expected_rotation).abs() < 1.0e-9);
    }

    /// 基準長以外の棒でも、D_0 の分母だけは共通の基準棒長から取ることを確認する。
    #[test]
    fn diffusion_uses_shared_reference_length_for_non_reference_rods() {
        let l = particle_length(1);
        let terms = tirado_terms_for_length(l);
        let reference_terms = tirado_terms_for_length(DIFFUSION_REFERENCE_LENGTH);
        let diffusion = diffusion_for_length(l);

        let expected_parallel =
            4.0 * (DIFFUSION_REFERENCE_LENGTH / l) * (terms.log_p + terms.nu_parallel)
                / reference_terms.denominator;
        let expected_rotation =
            24.0 * DIFFUSION_REFERENCE_LENGTH * (terms.log_p + terms.delta_perp)
                / (l * l * l * reference_terms.denominator);

        assert!((diffusion.d_parallel - expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_perp - 0.5 * expected_parallel).abs() < 1.0e-12);
        assert!((diffusion.d_r - expected_rotation).abs() < 1.0e-9);
    }
}
