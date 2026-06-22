use std::f64::consts::PI;

use crate::config::{
    DEFAULT_MAX_STEPS, DT, EPSILON, MAX_WALL_REPULSION_FORCE, PARTICLE_DX, SIGMA, SimParams,
    WALL_DX, WALL_K, particle_length,
};
use crate::model::{
    Diffusion, PASS_DIRECTION_LEFT, PASS_DIRECTION_NONE, PASS_DIRECTION_RIGHT,
    diffusion_for_length, omega, reflect_state_into_channel, wall_y_samples,
};

/// アニメーションで操作できるシミュレーションパラメータ。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisualParams {
    pub seed: u64,
    pub m: i32,
    pub force: f64,
    pub beta_pe: f64,
    pub delta_alpha_e_over_p: f64,
}

impl Default for VisualParams {
    /// UI 起動時に使う見やすさ優先の既定値を返す。
    fn default() -> Self {
        Self {
            seed: 1,
            m: 8,
            force: 0.0,
            beta_pe: 0.0,
            delta_alpha_e_over_p: 0.0,
        }
    }
}

impl VisualParams {
    /// GPU/CPU 共通の物理パラメータ構造へ変換する。
    pub fn sim_params(self) -> SimParams {
        SimParams {
            combo_id: 0,
            m: self.m,
            l: particle_length(self.m),
            beta_pe: self.beta_pe,
            delta_alpha_e_over_p: self.delta_alpha_e_over_p,
            force: self.force,
        }
    }
}

/// 棒上の 1 つの代表点または端点の座標。
#[derive(Clone, Copy, Debug)]
pub struct RodPoint {
    pub x: f64,
    pub y: f64,
}

/// 棒の重心並進力と角度方向トルクをまとめた一般化力。
#[derive(Clone, Copy, Debug)]
struct GeneralizedForce {
    force_x: f64,
    force_y: f64,
    torque: f64,
}

/// 角度から決まる実験室系の拡散テンソルと並進ノイズ。
#[derive(Clone, Copy, Debug)]
struct LabTransport {
    dxx: f64,
    dxy: f64,
    dyy: f64,
    noise_x: f64,
    noise_y: f64,
}

/// 1 step 内で重心位置と角度へ加える増分。
#[derive(Clone, Copy, Debug)]
struct StepIncrement {
    dx: f64,
    dy: f64,
    dphi: f64,
}

/// アニメーション用に 1 粒子だけを CPU で逐次計算する状態。
#[derive(Clone, Debug)]
pub struct VisualSimulation {
    pub params: VisualParams,
    pub diffusion: Diffusion,
    pub x0: f64,
    pub y0: f64,
    pub phi0: f64,
    pub x: f64,
    pub y: f64,
    pub phi: f64,
    pub target_right_x: f64,
    pub target_left_x: f64,
    pub t: f64,
    pub steps: u64,
    pub first_passed: bool,
    pub pass_direction: i32,
    pub completed: bool,
    rng: SplitMixRng,
    wall_y: Vec<f64>,
}

impl VisualSimulation {
    /// seed と可視化パラメータから初期状態を決定的に生成する。
    pub fn new(params: VisualParams) -> Self {
        let mut rng = SplitMixRng::new(params.seed);
        let sim_params = params.sim_params();
        let x0 = -0.1 + 0.8 * rng.uniform_open01();
        let half_l = 0.5 * sim_params.l;
        let width = omega(x0);
        let y_min = -width + half_l;
        let y_max = width - half_l;
        let y0 = if y_max > y_min {
            y_min + (y_max - y_min) * rng.uniform_open01()
        } else {
            0.0
        };
        let phi0 = 2.0 * PI * rng.uniform_open01();

        Self {
            params,
            diffusion: diffusion_for_length(sim_params.l),
            x0,
            y0,
            phi0,
            x: x0,
            y: y0,
            phi: phi0,
            target_right_x: x0 + 1.0,
            target_left_x: x0 - 1.0,
            t: 0.0,
            steps: 0,
            first_passed: false,
            pass_direction: PASS_DIRECTION_NONE,
            completed: false,
            rng,
            wall_y: wall_y_samples(),
        }
    }

    /// パラメータ変更時に乱数系列も含めて初期状態を作り直す。
    pub fn reset(&mut self, params: VisualParams) {
        *self = Self::new(params);
    }

    /// 指定ステップだけ進めるが、最大ステップ到達後は進めない。
    pub fn step_many(&mut self, step_count: usize) {
        if self.completed {
            return;
        }

        if self.steps >= DEFAULT_MAX_STEPS {
            self.completed = true;
            return;
        }

        for _ in 0..step_count {
            self.step();
            if self.completed {
                break;
            }
        }
    }

    /// 描画用に棒の両端だけを返す。
    pub fn rod_endpoints(&self) -> (RodPoint, RodPoint) {
        let (s, c) = self.phi.sin_cos();
        let half_l = f64::from(self.params.m) * PARTICLE_DX;
        (
            RodPoint {
                x: self.x - half_l * c,
                y: self.y - half_l * s,
            },
            RodPoint {
                x: self.x + half_l * c,
                y: self.y + half_l * s,
            },
        )
    }

    /// 予測子・修正子法で 1 時間刻みだけ状態を更新する。
    fn step(&mut self) {
        if self.steps >= DEFAULT_MAX_STEPS {
            self.completed = true;
            return;
        }

        let normal_tx = self.rng.normal();
        let normal_ty = self.rng.normal();
        let normal_r = self.rng.normal();
        let noise_phi = (2.0 * self.diffusion.d_r * DT).sqrt() * normal_r;

        let predictor_transport = self.lab_transport_at(self.phi, normal_tx, normal_ty);
        let predictor_force = self.generalized_force_at(self.x, self.y, self.phi);
        let predictor_increment =
            self.step_increment(predictor_transport, predictor_force, noise_phi);
        let anchor_x = self.x;

        // 予測子で境界外へ出た状態は、修正子評価の前に流路内へ鏡像反射する。
        let predicted = reflect_state_into_channel(
            self.x + predictor_increment.dx,
            self.y + predictor_increment.dy,
            self.phi + predictor_increment.dphi,
            anchor_x,
        );

        let corrector_transport = self.lab_transport_at(predicted.phi, normal_tx, normal_ty);
        let corrector_force = self.generalized_force_at(predicted.x, predicted.y, predicted.phi);
        let corrector_increment =
            self.step_increment(corrector_transport, corrector_force, noise_phi);

        // 修正子で得た最終状態も、保存前に同じ境界条件で鏡像反射する。
        let corrected = reflect_state_into_channel(
            self.x + corrector_increment.dx,
            self.y + corrector_increment.dy,
            self.phi + corrector_increment.dphi,
            anchor_x,
        );
        self.x = corrected.x;
        self.y = corrected.y;
        self.phi = corrected.phi;
        self.steps += 1;
        self.t = self.steps as f64 * DT;

        if !self.first_passed {
            self.record_first_passage();
        }
        if self.steps >= DEFAULT_MAX_STEPS {
            self.completed = true;
        }
    }

    /// 現在の重心位置が x_0 ± L のどちらへ1周期分通過したかを記録する。
    fn record_first_passage(&mut self) {
        if self.x > self.target_right_x {
            self.first_passed = true;
            self.pass_direction = PASS_DIRECTION_RIGHT;
        } else if self.x < self.target_left_x {
            self.first_passed = true;
            self.pass_direction = PASS_DIRECTION_LEFT;
        }
    }

    /// 実験室系の輸送係数と一般化力から、1 step 分の状態増分を作る。
    fn step_increment(
        &self,
        transport: LabTransport,
        force: GeneralizedForce,
        noise_phi: f64,
    ) -> StepIncrement {
        let drift_x = (transport.dxx * force.force_x + transport.dxy * force.force_y) * DT;
        let drift_y = (transport.dxy * force.force_x + transport.dyy * force.force_y) * DT;
        let dphi = self.diffusion.d_r * force.torque * DT + noise_phi;

        StepIncrement {
            dx: drift_x + transport.noise_x,
            dy: drift_y + transport.noise_y,
            dphi,
        }
    }

    /// 指定角度における実験室系の拡散テンソルと同じ乱数からの並進ノイズを返す。
    fn lab_transport_at(&self, phi: f64, normal_tx: f64, normal_ty: f64) -> LabTransport {
        let (s, c) = phi.sin_cos();
        let (sin2, cos2) = (2.0 * phi).sin_cos();
        let d_scale = 0.25 * self.diffusion.d_parallel;
        let noise_body_x = (2.0 * self.diffusion.d_parallel * DT).sqrt() * normal_tx;
        let noise_body_y = (2.0 * self.diffusion.d_perp * DT).sqrt() * normal_ty;

        LabTransport {
            dxx: d_scale * (3.0 + cos2),
            dxy: d_scale * sin2,
            dyy: d_scale * (3.0 - cos2),
            noise_x: c * noise_body_x - s * noise_body_y,
            noise_y: s * noise_body_x + c * noise_body_y,
        }
    }

    /// 指定された状態で、壁反発・粒子全体への外力・電場トルクを合成した一般化力を返す。
    fn generalized_force_at(&self, x: f64, y: f64, phi: f64) -> GeneralizedForce {
        let (s, c) = phi.sin_cos();
        let mut rep_sum_x = 0.0;
        let mut rep_sum_y = 0.0;
        let mut torque_sum = 0.0;

        for j in -self.params.m..=self.params.m {
            let offset = f64::from(j) * PARTICLE_DX;
            let rep_x = x + offset * c;
            let rep_y = y + offset * s;
            let (force_x, force_y) = self.wall_force(rep_x, rep_y);

            rep_sum_x += force_x;
            rep_sum_y += force_y;
            torque_sum += offset * (c * force_y - s * force_x);
        }

        let tau_e = self.params.beta_pe * c * (1.0 + self.params.delta_alpha_e_over_p * s);

        GeneralizedForce {
            force_x: self.params.force + rep_sum_x,
            force_y: rep_sum_y,
            torque: torque_sum + tau_e,
        }
    }

    /// 近傍の壁サンプル点だけから WCA 反発力を合算する。
    fn wall_force(&self, rep_x: f64, rep_y: f64) -> (f64, f64) {
        let mut force_x = 0.0;
        let mut force_y = 0.0;
        let k0 = round_to_i64(rep_x / WALL_DX);

        for q in -WALL_K..=WALL_K {
            let k = k0 + i64::from(q);
            let k_mod = positive_mod(k, self.wall_y.len() as i64) as usize;
            let wall_x = k as f64 * WALL_DX;
            let upper_y = self.wall_y[k_mod];

            add_wca_force(rep_x, rep_y, wall_x, upper_y, &mut force_x, &mut force_y);
            add_wca_force(rep_x, rep_y, wall_x, -upper_y, &mut force_x, &mut force_y);
        }

        (force_x, force_y)
    }
}

/// 1 つの粒子代表点と 1 つの壁点から WCA 反発力を加算する。
fn add_wca_force(
    rep_x: f64,
    rep_y: f64,
    wall_x: f64,
    wall_y: f64,
    force_x: &mut f64,
    force_y: &mut f64,
) {
    let dx = rep_x - wall_x;
    let dy = rep_y - wall_y;
    let r2 = dx * dx + dy * dy;
    let sigma2 = SIGMA * SIGMA;
    let rc2 = 2.0_f64.powf(1.0 / 3.0) * sigma2;

    if r2 < rc2 {
        let s2 = sigma2 / r2;
        let s6 = s2 * s2 * s2;
        let coeff = 24.0 * EPSILON / r2 * s6 * (2.0 * s6 - 1.0);
        let (pair_force_x, pair_force_y) = capped_wall_force(coeff * dx, coeff * dy);
        *force_x += pair_force_x;
        *force_y += pair_force_y;
    }
}

/// 1 つの壁点から受ける反発力の大きさを上限値に収める。
fn capped_wall_force(force_x: f64, force_y: f64) -> (f64, f64) {
    let force2 = force_x * force_x + force_y * force_y;
    let force = force2.sqrt();

    if force > MAX_WALL_REPULSION_FORCE {
        let scale = MAX_WALL_REPULSION_FORCE / force;
        (force_x * scale, force_y * scale)
    } else {
        (force_x, force_y)
    }
}

/// 負のインデックスも周期境界へ正しく折り返す剰余を返す。
fn positive_mod(value: i64, modulus: i64) -> i64 {
    let r = value % modulus;
    if r < 0 { r + modulus } else { r }
}

/// 壁サンプル点の最近傍インデックスを整数へ丸める。
fn round_to_i64(value: f64) -> i64 {
    if value >= 0.0 {
        (value + 0.5) as i64
    } else {
        (value - 0.5) as i64
    }
}

/// GPU 側と同じ seed から再現性のある乱数列を作る簡易 RNG。
#[derive(Clone, Debug)]
struct SplitMixRng {
    state: u64,
    cached_normal: Option<f64>,
}

impl SplitMixRng {
    /// 指定 seed で乱数生成器を初期化する。
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    /// SplitMix64 の次の 64bit 値を生成する。
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 0 と 1 を含まない一様乱数を返す。
    fn uniform_open01(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * SCALE
    }

    /// Box-Muller 法で標準正規乱数を返す。
    fn normal(&mut self) -> f64 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let (s, c) = theta.sin_cos();
        self.cached_normal = Some(r * s);
        r * c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 短時間の更新で有限な状態が保たれることを確認する。
    #[test]
    fn visual_simulation_advances_with_finite_state() {
        let mut sim = VisualSimulation::new(VisualParams::default());
        sim.step_many(128);

        assert!(sim.x.is_finite());
        assert!(sim.y.is_finite());
        assert!(sim.phi.is_finite());
        assert_eq!(sim.steps, 128);
    }

    /// アニメーション相当の短い実行で発散しないことを確認する。
    #[test]
    fn visual_simulation_stays_bounded_during_short_animation_run() {
        let mut sim = VisualSimulation::new(VisualParams::default());
        sim.step_many(50_000);

        assert!(sim.x.is_finite());
        assert!(sim.y.is_finite());
        assert!(sim.phi.is_finite());
        assert!(sim.x.abs() < 10.0);
        assert!(sim.y.abs() < 10.0);
    }

    /// 強い外力条件でも、各 step 後の重心が流路内へ鏡像反射されることを確認する。
    #[test]
    fn visual_simulation_reflects_high_force_state_inside_channel() {
        let mut sim = VisualSimulation::new(VisualParams {
            m: 20,
            force: 100.0,
            beta_pe: 0.0,
            delta_alpha_e_over_p: 0.0,
            ..VisualParams::default()
        });

        for _ in 0..70_000 {
            sim.step();
            assert!(sim.y.abs() <= omega(sim.x) + 1.0e-9);
        }
    }

    /// x_0 + L 通過後も最大ステップ到達までは完了扱いにしないことを確認する。
    #[test]
    fn visual_simulation_records_right_first_passage() {
        let mut sim = VisualSimulation::new(VisualParams::default());
        sim.x = sim.target_right_x + 0.01;
        sim.step_many(1);

        assert!(sim.first_passed);
        assert_eq!(sim.pass_direction, PASS_DIRECTION_RIGHT);
        assert!(!sim.completed);
        assert_eq!(sim.steps, 1);
    }

    /// x_0 - L 側へ通過した場合も初通過として方向を保存することを確認する。
    #[test]
    fn visual_simulation_records_left_first_passage() {
        let mut sim = VisualSimulation::new(VisualParams::default());
        sim.x = sim.target_left_x - 0.01;
        sim.step_many(1);

        assert!(sim.first_passed);
        assert_eq!(sim.pass_direction, PASS_DIRECTION_LEFT);
        assert!(!sim.completed);
        assert_eq!(sim.steps, 1);
    }

    /// `DEFAULT_MAX_STEPS` でアニメーションが完了扱いになることを確認する。
    #[test]
    fn visual_simulation_completes_at_default_max_steps() {
        let mut sim = VisualSimulation::new(VisualParams::default());
        sim.steps = DEFAULT_MAX_STEPS - 1;
        sim.t = sim.steps as f64 * DT;

        sim.step_many(8);

        assert_eq!(sim.steps, DEFAULT_MAX_STEPS);
        assert!(sim.completed);
    }

    /// x 方向外力は代表点ごとの力ではなく、粒子全体への合力として扱うことを確認する。
    #[test]
    fn external_force_is_not_scaled_by_representative_points() {
        for m in [1, 30] {
            let sim = VisualSimulation::new(VisualParams {
                m,
                force: 7.0,
                beta_pe: 0.0,
                delta_alpha_e_over_p: 0.0,
                ..VisualParams::default()
            });
            let force = sim.generalized_force_at(0.25, 0.0, 0.0);

            assert!((force.force_x - 7.0).abs() < 1.0e-12);
            assert!(force.force_y.abs() < 1.0e-12);
        }
    }

    /// 壁反発力の大きさが上限値そのものに正規化されることを確認する。
    #[test]
    fn capped_wall_force_normalizes_to_max_force() {
        let input_x = 3.0 * MAX_WALL_REPULSION_FORCE;
        let input_y = 4.0 * MAX_WALL_REPULSION_FORCE;
        let (force_x, force_y) = capped_wall_force(input_x, input_y);
        let magnitude = (force_x * force_x + force_y * force_y).sqrt();

        assert!((magnitude - MAX_WALL_REPULSION_FORCE).abs() < 1.0e-9);
    }
}
