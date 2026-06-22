#include <curand_kernel.h>
#include <math_constants.h>

// Rust側ではcuRAND stateの具体サイズを直接知らないため、trialごとに256 bytesを確保する。
static_assert(
    sizeof(curandStatePhilox4_32_10_t) <= 256,
    "RNG_STATE_BYTES must be at least sizeof(curandStatePhilox4_32_10_t)");

// Rust側の #[repr(C)] KernelParams と同じ順序・型にする。
struct KernelParams
{
  int combo_id;
  int m;
  int n_wall;
  int wall_k;
  int boundary_reflection_limit;
  double l;
  double beta_pe;
  double delta_alpha_e_over_p;
  double force;
  double d_parallel;
  double d_perp;
  double d_r;
  double dt;
  double sigma;
  double epsilon;
  double wall_dx;
  double particle_dx;
  double rc2;
  double max_wall_repulsion_force;
  double channel_neck_phase;
};

// 並進力と角度方向トルクをまとめ、予測点と修正点で同じ力計算を使い回す。
struct GeneralizedForce
{
  double force_x;
  double force_y;
  double torque;
};

// 角度から決まる実験室系の拡散テンソルと並進ノイズをまとめる。
struct LabTransport
{
  double dxx;
  double dxy;
  double dyy;
  double noise_x;
  double noise_y;
};

// 境界条件を満たすように鏡像反射した棒の重心状態。
struct RodState
{
  double x;
  double y;
  double phi;
};

// byte buffer上に確保したcuRAND state領域から、trial i のstateポインタを取り出す。
__device__ curandStatePhilox4_32_10_t *rng_state_at(
    unsigned char *rng_states,
    int rng_stride,
    int idx)
{
  return reinterpret_cast<curandStatePhilox4_32_10_t *>(
      rng_states + static_cast<size_t>(idx) * static_cast<size_t>(rng_stride));
}

// 初期位置のy範囲を決めるため、壁の半幅 omega(x) をGPU側でも計算する。
// omega(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx) + 0.5sin(2πx)cos(2πx) + 1.12
__device__ double omega(double x)
{
  double s, c;
  sincospi(2.0 * x, &s, &c);
  return s + 0.5 * s * c + 1.12;
}

// 流路の半幅 omega(x) の一階微分を返す。
__device__ double omega_derivative(double x)
{
  double s, c;
  sincospi(2.0 * x, &s, &c);
  return fma(c, fma(2.0 * CUDART_PI, c, 2.0 * CUDART_PI), -CUDART_PI);
}

// 壁への垂線の足を求めるNewton法の1 step分の補正量を返す。
__device__ double wall_foot_newton_delta(
    double px,
    double py,
    double sign,
    double x)
{
  constexpr double MINUS_FOUR_PI_SQ = -4.0 * CUDART_PI * CUDART_PI;
  constexpr double MINUS_EIGHT_PI_SQ = -8.0 * CUDART_PI * CUDART_PI;

  double s, c;
  sincospi(2.0 * x, &s, &c);

  double offset = 1.12 - sign * py;
  double w_sub = fma(s, fma(0.5, c, 1.0), offset);
  double w_p = fma(c, fma(2.0 * CUDART_PI, c, 2.0 * CUDART_PI), -CUDART_PI);
  double w_pp = s * fma(MINUS_EIGHT_PI_SQ, c, MINUS_FOUR_PI_SQ);

  return fma(w_p, w_sub, x - px) / fma(w_pp, w_sub, fma(w_p, w_p, 1.0));
}

// 指定値を閉区間に収める。
__device__ double clamp_double(double value, double min_value, double max_value)
{
  return fmin(fmax(value, min_value), max_value);
}

// 現在位置が属する流路の膨らみ区間を、左右のくびれのx座標で返す。
__device__ void channel_section_bounds(
    double anchor_x,
    const KernelParams params,
    double *section_min_x,
    double *section_max_x)
{
  *section_min_x = floor(anchor_x - params.channel_neck_phase) + params.channel_neck_phase;
  *section_max_x = *section_min_x + 1.0;
}

// 点から上壁または下壁へ下ろした垂線の足のx座標をNewton法で求める。
__device__ double perpendicular_foot_x(
    double px,
    double py,
    double sign,
    double initial_x,
    const KernelParams params)
{
  constexpr double EPSILON = 1.0e-10;
  double section_min_x, section_max_x;
  channel_section_bounds(initial_x, params, &section_min_x, &section_max_x);
  double x = clamp_double(initial_x, section_min_x, section_max_x);

  for (int i = 0; i < 5; ++i)
  {
    double d = wall_foot_newton_delta(px, py, sign, x);
    if (fabs(d) > EPSILON)
    {
      x = clamp_double(x - d, section_min_x, section_max_x);
    }
    else
    {
      break;
    }
  }

  return x;
}

// 境界の接線方向へ棒の向きベクトルを鏡映した角度を返す。
__device__ double reflect_angle_across_tangent(double phii, double slope)
{
  return 2.0 * atan(slope) - phii;
}

// 流路外にある場合は、越えている上壁または下壁の符号を返す。
__device__ double boundary_wall_sign(double xi, double yi)
{
  double width = omega(xi);
  if (yi > width)
  {
    return 1.0;
  }
  else if (yi < -width)
  {
    return -1.0;
  }
  else
  {
    return 0.0;
  }
}

// 上壁または下壁に対して重心位置と棒の向きを1回だけ鏡像反射する。
__device__ RodState reflect_state_at_wall(
    RodState state,
    double sign,
    double anchor_x,
    const KernelParams params)
{
  double foot_x = perpendicular_foot_x(state.x, state.y, sign, anchor_x, params);
  double foot_y = sign * omega(foot_x);
  double slope = sign * omega_derivative(foot_x);

  RodState result;
  result.x = 2.0 * foot_x - state.x;
  result.y = 2.0 * foot_y - state.y;
  result.phi = reflect_angle_across_tangent(state.phi, slope);
  return result;
}

// 境界外に出た棒状態を、現在位置側の境界の接線に対して鏡像反射で流路内へ戻す。
__device__ RodState reflect_state_into_channel(
    double xi,
    double yi,
    double phii,
    double anchor_x,
    const KernelParams params)
{
  RodState result;
  result.x = xi;
  result.y = yi;
  result.phi = phii;

  for (int i = 0; i < params.boundary_reflection_limit; ++i)
  {
    double sign = boundary_wall_sign(result.x, result.y);
    if (sign == 0.0)
    {
      return result;
    }

    result = reflect_state_at_wall(result, sign, anchor_x, params);
  }

  return result;
}

// 負の周期インデックスも壁配列へ正しく写すための剰余。
__device__ int positive_mod_ll(long long value, int modulus)
{
  long long r = value % static_cast<long long>(modulus);
  return static_cast<int>(r < 0 ? r + modulus : r);
}

// NVRTCで標準Cヘッダに依存しないよう、round相当を簡単に実装する。
__device__ long long round_to_ll(double value)
{
  return value >= 0.0
             ? static_cast<long long>(value + 0.5)
             : static_cast<long long>(value - 0.5);
}

// 1つの代表点と1つの壁点のWCA反発力を加算する。
__device__ void add_wca_force(
    double rep_x,
    double rep_y,
    double wall_x,
    double wall_y,
    const KernelParams params,
    double *force_x,
    double *force_y)
{
  double dx = rep_x - wall_x;
  double dy = rep_y - wall_y;
  double r2 = dx * dx + dy * dy;

  if (r2 > 0.0 && r2 < params.rc2)
  {
    // 仕様書の平方根を使わない形: s2 = sigma^2 / r^2, s6 = s2^3。
    double sigma2 = params.sigma * params.sigma;
    double s2 = sigma2 / r2;
    double s6 = s2 * s2 * s2;
    double coeff = 24.0 * params.epsilon / r2 * s6 * (2.0 * s6 - 1.0);
    double pair_force_x = coeff * dx;
    double pair_force_y = coeff * dy;
    double pair_force2 = pair_force_x * pair_force_x + pair_force_y * pair_force_y;
    double pair_force = sqrt(pair_force2);

    // 近接時の特異的な壁反発だけを上限値に収め、反発方向は保つ。
    if (pair_force > params.max_wall_repulsion_force)
    {
      double scale = params.max_wall_repulsion_force / pair_force;
      pair_force_x *= scale;
      pair_force_y *= scale;
    }

    *force_x += pair_force_x;
    *force_y += pair_force_y;
  }
}

// 指定された棒状態で、壁反発・粒子全体への外力・電場トルクを合成した一般化力を返す。
__device__ GeneralizedForce generalized_force_at(
    double xi,
    double yi,
    double phii,
    const double *wall_y,
    const KernelParams params)
{
  double s, c;
  sincos(phii, &s, &c);
  double rep_sum_x = 0.0;
  double rep_sum_y = 0.0;
  double torque_sum = 0.0;

  // 棒上の 2m+1 個の代表点について、近傍壁点からの反発力を合計する。
  for (int j = -params.m; j <= params.m; ++j)
  {
    double offset = static_cast<double>(j) * params.particle_dx;
    double rep_x = xi + offset * c;
    double rep_y = yi + offset * s;
    double force_x = 0.0;
    double force_y = 0.0;
    long long k0 = round_to_ll(rep_x / params.wall_dx);

    // 壁点はx方向インデックスだけで近傍を絞る。上下壁を同時に評価する。
    for (int q = -params.wall_k; q <= params.wall_k; ++q)
    {
      long long k = k0 + static_cast<long long>(q);
      int k_mod = positive_mod_ll(k, params.n_wall);
      double wall_x = static_cast<double>(k) * params.wall_dx;
      double upper_y = wall_y[k_mod];

      add_wca_force(rep_x, rep_y, wall_x, upper_y, params, &force_x, &force_y);
      add_wca_force(rep_x, rep_y, wall_x, -upper_y, params, &force_x, &force_y);
    }

    rep_sum_x += force_x;
    rep_sum_y += force_y;
    // 2次元外積 n x (offset * f_rep) が壁反発由来のトルクになる。
    torque_sum += offset * (c * force_y - s * force_x);
  }

  double tau_e = params.beta_pe * c * (1.0 + params.delta_alpha_e_over_p * s);

  GeneralizedForce result;
  result.force_x = params.force + rep_sum_x;
  result.force_y = rep_sum_y;
  result.torque = torque_sum + tau_e;
  return result;
}

// 指定角度における実験室系の拡散テンソルと同じ乱数からの並進ノイズを返す。
__device__ LabTransport lab_transport_at(
    double phii,
    double2 normal_t,
    const KernelParams params)
{
  double s, c;
  sincos(phii, &s, &c);

  double cos2, sin2;
  sincos(2.0 * phii, &sin2, &cos2);
  double d_scale = 0.25 * params.d_parallel;

  double noise_body_x = sqrt(2.0 * params.d_parallel * params.dt) * normal_t.x;
  double noise_body_y = sqrt(2.0 * params.d_perp * params.dt) * normal_t.y;

  LabTransport result;
  result.dxx = d_scale * (3.0 + cos2);
  result.dxy = d_scale * sin2;
  result.dyy = d_scale * (3.0 - cos2);
  result.noise_x = c * noise_body_x - s * noise_body_y;
  result.noise_y = s * noise_body_x + c * noise_body_y;
  return result;
}

// 各trialに独立したPhilox系cuRAND stateを初期化する。
extern "C" __global__ void setup_rng_kernel(
    unsigned char *rng_states,
    int rng_stride,
    unsigned long long seed,
    unsigned long long sequence_offset,
    int n_trials)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_trials)
  {
    return;
  }

  if (rng_stride < static_cast<int>(sizeof(curandStatePhilox4_32_10_t)))
  {
    return;
  }

  curandStatePhilox4_32_10_t *state = rng_state_at(rng_states, rng_stride, i);
  curand_init(seed, sequence_offset + static_cast<unsigned long long>(i), 0, state);
}

// 初期位置・角度を乱数で決め、状態配列を初期化する。
extern "C" __global__ void init_trials_kernel(
    KernelParams params,
    unsigned char *rng_states,
    int rng_stride,
    double *x0_out,
    double *y0_out,
    double *phi0_out,
    double *x,
    double *y,
    double *phi,
    double *target_right_x,
    double *target_left_x,
    double *times,
    unsigned long long *steps,
    int *statuses,
    int *pass_directions,
    int n_trials)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_trials)
  {
    return;
  }

  curandStatePhilox4_32_10_t *state = rng_state_at(rng_states, rng_stride, i);
  double ux = curand_uniform_double(state);
  double uy = curand_uniform_double(state);
  double uphi = curand_uniform_double(state);

  double x0 = -0.1 + 0.8 * ux;
  // 棒全体が上下壁の内側に入るよう、中心yの範囲を l/2 だけ狭める。
  double half_l = 0.5 * params.l;
  double width = omega(x0);
  double y_min = -width + half_l;
  double y_max = width - half_l;
  double y0 = (y_max > y_min) ? (y_min + (y_max - y_min) * uy) : 0.0;
  double phi0 = 2.0 * CUDART_PI * uphi;

  x0_out[i] = x0;
  y0_out[i] = y0;
  phi0_out[i] = phi0;
  x[i] = x0;
  y[i] = y0;
  phi[i] = phi0;
  target_right_x[i] = x0 + 1.0;
  target_left_x[i] = x0 - 1.0;
  times[i] = 0.0;
  steps[i] = 0ULL;
  statuses[i] = 0;
  pass_directions[i] = 0;
}

// 未完了trialを固定ステップ数だけ進め、初通過またはmax_steps到達を記録する。
extern "C" __global__ void simulate_kernel(
    KernelParams params,
    unsigned char *rng_states,
    int rng_stride,
    const double *wall_y,
    double *x,
    double *y,
    double *phi,
    const double *target_right_x,
    const double *target_left_x,
    double *times,
    unsigned long long *steps,
    int *statuses,
    int *pass_directions,
    unsigned long long *counters,
    int n_trials,
    unsigned int steps_per_launch,
    unsigned long long max_steps)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_trials || statuses[i] != 0)
  {
    return;
  }

  curandStatePhilox4_32_10_t *state = rng_state_at(rng_states, rng_stride, i);
  double xi = x[i];
  double yi = y[i];
  double phii = phi[i];
  unsigned long long step_count = steps[i];
  int status = 0;

  for (unsigned int local_step = 0; local_step < steps_per_launch; ++local_step)
  {
    if (step_count >= max_steps)
    {
      status = 2;
      break;
    }

    // 並進に2個、回転に1個の標準正規乱数を使う。
    double2 normal_t = curand_normal2_double(state);
    double normal_r = curand_normal_double(state);
    double noise_phi = sqrt(2.0 * params.d_r * params.dt) * normal_r;

    LabTransport predictor_transport = lab_transport_at(phii, normal_t, params);
    GeneralizedForce predictor_force = generalized_force_at(xi, yi, phii, wall_y, params);
    double predictor_drift_x =
        (predictor_transport.dxx * predictor_force.force_x +
         predictor_transport.dxy * predictor_force.force_y) *
        params.dt;
    double predictor_drift_y =
        (predictor_transport.dxy * predictor_force.force_x +
         predictor_transport.dyy * predictor_force.force_y) *
        params.dt;
    double predictor_dphi = params.d_r * predictor_force.torque * params.dt + noise_phi;

    double predicted_x = xi + predictor_drift_x + predictor_transport.noise_x;
    double predicted_y = yi + predictor_drift_y + predictor_transport.noise_y;
    double predicted_phi = phii + predictor_dphi;

    // 予測子で境界外へ出た状態は、修正子評価の前に流路内へ鏡像反射する。
    RodState predicted =
        reflect_state_into_channel(predicted_x, predicted_y, predicted_phi, xi, params);

    // 論文の修正子段階に合わせ、力・拡散テンソル・並進ノイズを補正後の予測角度で再評価する。
    LabTransport corrector_transport =
        lab_transport_at(predicted.phi, normal_t, params);
    GeneralizedForce corrector_force =
        generalized_force_at(
            predicted.x,
            predicted.y,
            predicted.phi,
            wall_y,
            params);
    double corrector_drift_x =
        (corrector_transport.dxx * corrector_force.force_x +
         corrector_transport.dxy * corrector_force.force_y) *
        params.dt;
    double corrector_drift_y =
        (corrector_transport.dxy * corrector_force.force_x +
         corrector_transport.dyy * corrector_force.force_y) *
        params.dt;
    double corrector_dphi = params.d_r * corrector_force.torque * params.dt + noise_phi;

    // 修正子で得た最終状態も、保存前に同じ境界条件で鏡像反射する。
    RodState corrected =
        reflect_state_into_channel(
            xi + corrector_drift_x + corrector_transport.noise_x,
            yi + corrector_drift_y + corrector_transport.noise_y,
            phii + corrector_dphi,
            xi,
            params);
    xi = corrected.x;
    yi = corrected.y;
    phii = corrected.phi;
    step_count += 1ULL;
    double ti = static_cast<double>(step_count) * params.dt;

    // 補間はせず、初めて x0 ± 1 のどちらかへ到達したステップ末の状態を記録する。
    if (xi > target_right_x[i])
    {
      status = 1;
      pass_directions[i] = 1;
      times[i] = ti;
      break;
    }
    if (xi < target_left_x[i])
    {
      status = 1;
      pass_directions[i] = -1;
      times[i] = ti;
      break;
    }

    // 最大ステップに到達したtrialはsummaryで別カウントにする。
    if (step_count >= max_steps)
    {
      status = 2;
      times[i] = ti;
      break;
    }
  }

  x[i] = xi;
  y[i] = yi;
  phi[i] = phii;
  steps[i] = step_count;

  if (status != 0)
  {
    statuses[i] = status;
    // host側が進捗を見るための完了カウンタ。0: ok, 1: max_steps。
    if (status == 1)
    {
      atomicAdd(&counters[0], 1ULL);
    }
    else
    {
      atomicAdd(&counters[1], 1ULL);
    }
  }
}
