use std::f64::consts::PI;

use crate::config::{DT, EPSILON, PARTICLE_DX, SIGMA, SimParams, WALL_DX, WALL_K, particle_length};
use crate::model::{Diffusion, diffusion_for_length, omega, wall_y_samples};

const MAX_VISUAL_WCA_COEFF: f64 = 2.5e6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisualParams {
    pub seed: u64,
    pub m: i32,
    pub force: f64,
    pub beta_pe: f64,
    pub delta_alpha_e_over_p: f64,
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            seed: 1,
            m: 8,
            force: 10.0,
            beta_pe: 1.0,
            delta_alpha_e_over_p: 1.0,
        }
    }
}

impl VisualParams {
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

    pub fn point_count(self) -> i32 {
        2 * self.m + 1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RodPoint {
    pub x: f64,
    pub y: f64,
}

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
    pub target_x: f64,
    pub t: f64,
    pub steps: u64,
    pub completed: bool,
    rng: SplitMixRng,
    wall_y: Vec<f64>,
}

impl VisualSimulation {
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
            target_x: x0 + 1.0,
            t: 0.0,
            steps: 0,
            completed: false,
            rng,
            wall_y: wall_y_samples(),
        }
    }

    pub fn reset(&mut self, params: VisualParams) {
        *self = Self::new(params);
    }

    pub fn step_many(&mut self, step_count: usize) {
        if self.completed {
            return;
        }

        for _ in 0..step_count {
            self.step();
            if self.completed {
                break;
            }
        }
    }

    pub fn representative_points(&self) -> Vec<RodPoint> {
        let (s, c) = self.phi.sin_cos();
        (-self.params.m..=self.params.m)
            .map(|j| {
                let offset = f64::from(j) * PARTICLE_DX;
                RodPoint {
                    x: self.x + offset * c,
                    y: self.y + offset * s,
                }
            })
            .collect()
    }

    fn step(&mut self) {
        let (s, c) = self.phi.sin_cos();
        let mut rep_sum_x = 0.0;
        let mut rep_sum_y = 0.0;
        let mut torque_sum = 0.0;

        for j in -self.params.m..=self.params.m {
            let offset = f64::from(j) * PARTICLE_DX;
            let rep_x = self.x + offset * c;
            let rep_y = self.y + offset * s;
            let (force_x, force_y) = self.wall_force(rep_x, rep_y);

            rep_sum_x += force_x;
            rep_sum_y += force_y;
            torque_sum += offset * (c * force_y - s * force_x);
        }

        let inv_points = 1.0 / f64::from(self.params.point_count());
        let total_force_x = self.params.force + rep_sum_x * inv_points;
        let total_force_y = rep_sum_y * inv_points;

        let (sin2, cos2) = (2.0 * self.phi).sin_cos();
        let d_scale = 0.25 * self.diffusion.d_parallel;
        let dxx = d_scale * (3.0 + cos2);
        let dxy = d_scale * sin2;
        let dyy = d_scale * (3.0 - cos2);

        let drift_x = (dxx * total_force_x + dxy * total_force_y) * DT;
        let drift_y = (dxy * total_force_x + dyy * total_force_y) * DT;

        let normal_tx = self.rng.normal();
        let normal_ty = self.rng.normal();
        let normal_r = self.rng.normal();
        let noise_body_x = (2.0 * self.diffusion.d_parallel * DT).sqrt() * normal_tx;
        let noise_body_y = (2.0 * self.diffusion.d_perp * DT).sqrt() * normal_ty;
        let noise_x = c * noise_body_x - s * noise_body_y;
        let noise_y = s * noise_body_x + c * noise_body_y;

        let tau_e = self.params.beta_pe * c * (1.0 + self.params.delta_alpha_e_over_p * s);
        let dphi = self.diffusion.d_r * (torque_sum + tau_e) * DT
            + (2.0 * self.diffusion.d_r * DT).sqrt() * normal_r;

        self.x += drift_x + noise_x;
        self.y += drift_y + noise_y;
        self.phi += dphi;
        self.steps += 1;
        self.t = self.steps as f64 * DT;

        if self.x > self.target_x {
            self.completed = true;
        }
    }

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
    let rc = 2.0_f64.powf(1.0 / 6.0) * SIGMA;
    let rc2 = rc * rc;

    if r2 > 0.0 && r2 < rc2 {
        let sigma2 = SIGMA * SIGMA;
        let s2 = sigma2 / r2;
        let s6 = s2 * s2 * s2;
        let coeff = (24.0 * EPSILON / r2 * s6 * (2.0 * s6 - 1.0))
            .clamp(-MAX_VISUAL_WCA_COEFF, MAX_VISUAL_WCA_COEFF);
        *force_x += coeff * dx;
        *force_y += coeff * dy;
    }
}

fn positive_mod(value: i64, modulus: i64) -> i64 {
    let r = value % modulus;
    if r < 0 { r + modulus } else { r }
}

fn round_to_i64(value: f64) -> i64 {
    if value >= 0.0 {
        (value + 0.5) as i64
    } else {
        (value - 0.5) as i64
    }
}

#[derive(Clone, Debug)]
struct SplitMixRng {
    state: u64,
    cached_normal: Option<f64>,
}

impl SplitMixRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform_open01(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * SCALE
    }

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

    #[test]
    fn visual_simulation_advances_with_finite_state() {
        let mut sim = VisualSimulation::new(VisualParams::default());
        sim.step_many(128);

        assert!(sim.x.is_finite());
        assert!(sim.y.is_finite());
        assert!(sim.phi.is_finite());
        assert_eq!(sim.steps, 128);
    }

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

    #[test]
    fn representative_point_count_matches_m() {
        let sim = VisualSimulation::new(VisualParams {
            m: 4,
            ..VisualParams::default()
        });

        assert_eq!(sim.representative_points().len(), 9);
    }
}
