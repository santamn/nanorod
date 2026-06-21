#include <curand_kernel.h>
#include <math_constants.h>

// 1つの壁点から受けるWCA反発力ベクトルの最大値。
#define MAX_WALL_REPULSION_FORCE 2.5e6

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
  int point_count;
  int _pad0;
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
    if (pair_force > MAX_WALL_REPULSION_FORCE)
    {
      double scale = MAX_WALL_REPULSION_FORCE / pair_force;
      pair_force_x *= scale;
      pair_force_y *= scale;
    }

    *force_x += pair_force_x;
    *force_y += pair_force_y;
  }
}

// 指定された棒状態で、壁反発・外力・電場トルクを合成した一般化力を返す。
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
  result.force_x = static_cast<double>(params.point_count) * params.force + rep_sum_x;
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
    double *target_x,
    double *times,
    unsigned long long *steps,
    int *statuses,
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
  target_x[i] = x0 + 1.0;
  times[i] = 0.0;
  steps[i] = 0ULL;
  statuses[i] = 0;
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
    const double *target_x,
    double *times,
    unsigned long long *steps,
    int *statuses,
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

    // 論文の修正子段階に合わせ、力・拡散テンソル・並進ノイズを予測角度で再評価する。
    LabTransport corrector_transport = lab_transport_at(predicted_phi, normal_t, params);
    GeneralizedForce corrector_force =
        generalized_force_at(predicted_x, predicted_y, predicted_phi, wall_y, params);
    double corrector_drift_x =
        (corrector_transport.dxx * corrector_force.force_x +
         corrector_transport.dxy * corrector_force.force_y) *
        params.dt;
    double corrector_drift_y =
        (corrector_transport.dxy * corrector_force.force_x +
         corrector_transport.dyy * corrector_force.force_y) *
        params.dt;
    double corrector_dphi = params.d_r * corrector_force.torque * params.dt + noise_phi;

    // 予測子・修正子法で1ステップ進める。phiは毎step正規化しない。
    xi += corrector_drift_x + corrector_transport.noise_x;
    yi += corrector_drift_y + corrector_transport.noise_y;
    phii += corrector_dphi;
    step_count += 1ULL;
    double ti = static_cast<double>(step_count) * params.dt;

    // 補間はせず、初めて x > x0 + 1 になったステップ末の状態を記録する。
    if (xi > target_x[i])
    {
      status = 1;
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
