# シミュレーション実装計画

## 目的

`docs/specification.md` と `docs/rod_simulation.md` に従い、Rust + CUDA + cudarc で棒状粒子のブラウン運動を GPU 上で多数並列に計算する。まず代表的な 1 パラメータ組み合わせの小さな試行で初期状態・終了状態・初通過時間を確認し、その後、ユーザーから明示的に依頼されたときだけ全パラメータ組み合わせの production simulation を実行する。production simulation では各試行の詳細は出力せず、組み合わせごとの `T1`, `T2`, 平均速度 `v`, 有効拡散係数 `D_eff` を `summary.csv` に出力する。

## 実装範囲

- 無次元化後の確率微分方程式を Euler-Maruyama 法で時間発展する。
- 壁形状は `omega(x) = sin(2*pi*x) + 0.25*sin(4*pi*x) + 1.12`、周期 `L = 1` とする。
- WCA 反発力は、仕様書通り `epsilon = 2`, `sigma = 8e-3`, `dt = 4e-7` を使う。
- 壁点は `0.25*sigma = 0.002` 間隔で 1 周期 500 点、近傍探索は x 方向インデックスのみで `K = 5` の固定長候補に絞る。
- 粒子長は `l = 2m * 0.8 * sigma` とし、`m = 1, 4, 8, 15, 30` の 5 通りを使う。
- パラメータは `l` 5 通り、`beta*pE` 6 通り、`Delta alpha*E/p` 6 通り、外力 `f` 19 通りの合計 3420 組とする。
- 各組み合わせにつき `N = 30000` 試行、合計 102,600,000 試行を処理する。
- 1 試行あたりの最大ステップ数は `2.5e7` とする。`dt = 4e-7` なので最大シミュレーション時刻は `10` である。
- CPU フォールバックは実装しない。GPU 初期化またはカーネル実行に失敗したら、その理由を出して停止する。

## 主要な設計方針

### 1. GPU デバイス選択と並列実行

`docs/architecture.md` により GPU ID 0 は使わない。CLI の既定値は `--devices 1,2,3` とし、GPU 1, 2, 3 をすべて使う。`--devices` に `0` が含まれる場合は、明示的な危険オプションなしでは拒否する。A100 上で動かす前提なので、過剰な互換フォールバックは入れない。

production simulation では、CPU 側に共有ワークキューを置き、各 GPU worker thread が次の未処理 `combo_id` を取得して処理する。各 worker は自分の CUDA context、stream、device memory を持つ。GPU 間で device memory を共有しないため、実装を単純に保てる。

出力は writer thread に集約する。GPU worker が完了した summary row と進捗イベントを channel へ送り、writer thread が `summary.csv` と `progress.jsonl` を更新して頻繁に flush する。これにより、複数 GPU から同じ CSV へ同時書き込みする複雑さを避ける。

### 2. Rust 側の構成

コードベースの複雑さを抑えるため、最初は以下の最小構成にする。

- `src/main.rs`: CLI 解析、全体 orchestration。
- `src/config.rs`: 定数、CLI オプション、パラメータ掃引定義。
- `src/model.rs`: 拡散係数、`m` と `l` の変換、壁サンプリング配列などの CPU 側 pure function。
- `src/gpu.rs`: cudarc による CUDA context、stream、PTX compile、kernel launch、GPU worker。
- `src/kernels/simulation.cu`: `include_str!` で読み込み、NVRTC で PTX にコンパイルする CUDA カーネル。

`output.rs` は writer が肥大化した場合だけ分離する。初期実装では `main.rs` 内の writer function に留める。

### 3. 依存クレート

2026-06-16 時点で `cargo search` / `cargo info` により、`cudarc` は `0.19.7`、`clap` は `4.6.1`、`serde` は `1.0.228`、`csv` は `1.4.0`、`anyhow` は `1.0.102` が現行版であることを確認した。

CUDA 12.2 との整合性については、`cudarc 0.19.7` が `cuda-12020` feature を持つことを確認した。そのため `cuda-version-from-build-system` による自動推定ではなく、CUDA 12.2 固定で `cuda-12020` を指定する。NVIDIA の CUDA 12.2 Release Notes では、CUDA 12.2 の NVRTC は `12.2.91`、cuRAND は `10.3.3.53`、Linux の CUDA 12.2 GA 対応 driver は `>=535.54.03`、minor version compatibility の最小 driver は `>=525.60.13` とされている。ホスト側に CUDA Toolkit 12.2 の headers と NVRTC が存在する前提で実装する。

```toml
[dependencies]
anyhow = "1.0"
clap = { version = "4.6.1", features = ["derive"] }
csv = "1.4.0"
serde = { version = "1.0.228", features = ["derive"] }
cudarc = { version = "0.19.7", default-features = false, features = [
  "std",
  "driver",
  "nvrtc",
  "dynamic-loading",
  "cuda-12020",
] }
```

`cudarc` は `driver` API でデバイスメモリと kernel launch を扱い、`nvrtc` で CUDA C カーネルを PTX にコンパイルする。乱数は CUDA の cuRAND device API を使い、`curand_kernel.h` を CUDA kernel 側で include する。Rust から cuRAND host API を直接呼ばない限り、cudarc の `curand` feature は追加しない。

### 4. データ型

数値安定性を優先して、状態量と集計値は `f64` を使う。A100 は FP64 性能が高く、`dt = 4e-7` の長時間積分では単精度より安全である。性能測定後に必要なら `f32` 版を別 feature として検討する。

### 5. バッチ処理

1 パラメータ組み合わせあたり 30000 試行なので、基本は 1 組み合わせを 1 GPU worker の 1 バッチとして載せる。production simulation では 3 GPU が別々の組み合わせを同時に処理する。smoke run では 1 組み合わせの少数 trial を GPU 1, 2, 3 に分割し、multi-GPU 経路そのものも最初に確認する。

## 数値実装の詳細

### 1. 代表点

粒子長 `l` を直接列挙せず、代表点間隔 `0.8*sigma` と half index `m` で指定する。

```text
m_values = [1, 4, 8, 15, 30]
l = 2 * m * 0.8 * sigma
代表点 j = -m, ..., m
point_count = 2m + 1
offset_j = j * 0.8 * sigma
r_j = X + offset_j * n
```

これにより、`m = 1, 4, 8, 15, 30` はそれぞれ代表点数 `3, 9, 17, 31, 61` に対応する。トルク計算では `offset_j` と各代表点の反発力を使い、2 次元外積 `n_x * f_y - n_y * f_x` を計算する。

### 2. 拡散係数

各 `l` について `p = 40*l` とし、Tirado and Garcia de la Torre の式で `D_parallel` と `D_r` を計算する。仕様書の簡略化に従い、並進ノイズと `D_lab` では `D_perp = 0.5 * D_parallel` を使う。

```text
D_lab(phi) = D_parallel / 4 * [[3 + cos(2phi), sin(2phi)],
                               [sin(2phi), 3 - cos(2phi)]]
```

### 3. 1 ステップ更新

各スレッドが 1 試行を担当し、未通過の試行だけを更新する。

```text
F = (f, 0) + average_j(f_rep_j)
tau_rep = n cross sum_j(offset_j * f_rep_j)
tau_E = beta_pE * cos(phi) * (1 + delta_alpha_E_over_p * sin(phi))

dX = D_lab(phi) * F * dt
   + R(phi) * [sqrt(2*D_parallel*dt)*N1, sqrt(2*D_perp*dt)*N2]

dphi = D_r * (tau_rep + tau_E) * dt
     + sqrt(2*D_r*dt) * N3
```

`x` は壁参照には周期化するが、初通過判定では絶対座標を保持する。`phi` は step ごとには正規化しない。毎 step の `fmod` / `remainder` は割り算系の重い演算になり、GPU 上で全 trial、全 step に入れるコストが大きい。一方で、`sin` / `cos` は内部で引数処理を行い、`max_steps = 2.5e7`, `dt = 4e-7` の範囲では通常 `phi` が数値的に破綻するほど巨大になるリターンは小さい。出力時に読みやすい角度が必要な場合だけ host 側で正規化する。もし検証で `phi` が極端に増大するケースが見つかった場合は、step ごとではなく chunk 境界でだけ条件付き正規化する。

### 4. WCA 反発力

各代表点で

```text
k0 = round(x_j / wall_dx)
for q in -K..=K:
  k = k0 + q
  k_mod = ((k % N_wall) + N_wall) % N_wall
  x_wall = k * wall_dx
  y_wall = +/- wall_y[k_mod]
```

として上下壁を調べる。`0 < r2 < rc2` のときだけ

```text
s2 = sigma2 / r2
s6 = s2 * s2 * s2
f += 24 * epsilon / r2 * s6 * (2*s6 - 1) * d
```

を加算する。平方根は使わない。

### 5. 乱数

乱数は独自実装せず、CUDA の cuRAND device API を使う。kernel 側で `curand_kernel.h` を include し、各 trial に `curandStatePhilox4_32_10_t` を 1 つ持たせる。

初期化は simulation kernel と分け、setup kernel で

```text
curand_init(seed, global_trial_id, 0, &rng_state[trial_id])
```

を呼ぶ。複数 chunk にまたがる simulation では RNG state を device global memory に保存し、次の launch で続きから使う。これは cuRAND の performance note に沿った方針で、毎 launch で `curand_init` を呼び直すより単純かつ速い。

各 step では標準正規乱数が 3 個必要なので、double precision の場合は `curand_normal2_double` を 2 回呼び、4 個のうち 3 個を使う。single precision へ切り替える実験をする場合は、Philox 用の `curand_normal4` が候補になる。

### 6. 初通過判定

初期化時に `target_x = x0 + 1` を保存する。各 step の更新後に

```text
x > target_x
```

となった最初の step を初通過とする。線形補間は行わず、その step 末の時刻、座標、状態をそのまま記録する。

```text
T = (step + 1) * dt
x_end = x
y_end = y
phi_end = phi
```

同一 trial では最初に `x > target_x` となった時点で `status = ok` とし、その後の更新は行わない。

### 7. 長時間試行の扱い

1 回の kernel launch 内で固定ステップ数 `steps_per_launch` だけ進め、host 側で未完了数を確認して再 launch する。これにより、極端に長い試行があっても進捗表示、停止、結果 flush が可能になる。

既定の `--max-steps` は `25_000_000` とする。到達した trial は `status = max_steps` とし、production summary では `n_ok` と `n_max_steps` を分けて記録する。`T1`, `T2`, `v`, `D_eff` は原則として `ok` trial だけから計算し、`n_max_steps > 0` の組み合わせは後で分かるように summary に残す。

## 出力計画

出力先は `--output-dir` で指定する。smoke run と production simulation で出力を分ける。

### Smoke run

実装後、production simulation の前に必ず 1 パラメータ組み合わせの小さな試行を実行する。既定では `m`, `beta*pE`, `Delta alpha*E/p`, `f` から 1 つずつ選んだ 1 組み合わせを使い、少数 trial を GPU 1, 2, 3 に分割する。最終 production simulation はユーザーが依頼するまで実行しない。

`smoke_trials.csv` は各試行 1 行。

```text
combo_id,trial_id,device_id,m,l,beta_pE,delta_alpha_E_over_p,f,
x0,y0,phi0,x_end,y_end,phi_end,T,steps,status
```

ここでは仕様確認のため、初期状態、終了状態、各 trial の初通過時間 `T` を必ず出力する。writer は少なくとも 100 trial ごと、または GPU worker から batch result を受け取るたびに flush する。

`smoke_summary.csv` は同じ組み合わせの集計を 1 行で出力する。

### Production simulation

production simulation の主出力は `summary.csv` のみとする。各 trial の初期状態、終了状態、`T` は出力しない。

各パラメータ組み合わせ 1 行。

```text
combo_id,device_id,m,l,beta_pE,delta_alpha_E_over_p,f,
n_total,n_ok,n_max_steps,T1,T2,v,D_eff,dt,sigma,epsilon,seed
```

`T1 = mean(T)`, `T2 = mean(T*T)`, `v = 1/T1`, `D_eff = 0.5 * (T2 - T1*T1) / (T1*T1*T1)` とする。

`summary.csv` は 1 行書くたびに flush する。production では trial 詳細を出さないため、出力サイズと I/O 律速を抑えられる。

### 進捗出力

実行中に状態を見られるよう、標準エラーと `progress.jsonl` の両方へ進捗を出す。

`progress.jsonl` は 1 イベント 1 行。

```text
timestamp,run_mode,device_id,combo_id,completed_combos,total_combos,
completed_trials,total_trials,current_steps,max_steps,status
```

writer は progress event ごとに flush する。progress event の既定頻度は 5 秒に 1 回とし、CLI で `--progress-interval-sec` を変更できるようにする。

## 実装順序

1. CLI、定数、`m` ベースのパラメータ掃引、smoke / production の出力スキーマを実装する。
2. CPU 側で壁点配列、拡散係数、`m -> l` 変換を計算する pure function を実装し、単体テストを追加する。
3. cudarc の context/stream 初期化、GPU ID 0 拒否、`--devices 1,2,3` の worker 起動、NVRTC compile、空 kernel launch の疎通を確認する。
4. CUDA kernel に cuRAND setup kernel と試行初期化処理を実装し、`x0`, `y0`, `phi0` が指定範囲に入ることを小規模テストで確認する。
5. WCA 反発力の device 関数を実装し、CPU 参照実装と数点で一致することを確認する。
6. 1 ステップ更新 kernel を実装し、反発なし・ノイズなしの簡単な条件で解析的に期待される移動方向と一致することを確認する。
7. `max_steps = 25_000_000` と chunked simulation を実装し、未完了 trial を `max_steps` として停止できるようにする。
8. 最初に smoke run を実行し、`smoke_trials.csv` に初期状態、終了状態、各 trial の `T` が出ること、`smoke_summary.csv` と `progress.jsonl` が頻繁に flush されることを確認する。
9. 少数 trial の production-mode dry run を実行し、`summary.csv` のみが出ること、3 GPU worker の進捗が見えることを確認する。
10. 全 3420 組み合わせ、各 `N = 30000` の production 実行に備え、進捗ログ、flush、resume 用の最小限の仕組みを追加する。ただし、本実行はユーザーから依頼されるまで実施しない。
11. release build とホスト実行手順を README または実行メモにまとめる。

## 検証計画

- `cargo fmt` と `cargo clippy` を通す。
- pure function の単体テスト:
  - `omega(x)` の周期性。
  - `N_wall = 500`, `K = 5`。
  - `m = 1, 4, 8, 15, 30` が代表点数 `3, 9, 17, 31, 61` と `l = 2m * 0.8sigma` に対応すること。
  - 拡散係数が正で有限であること。
- GPU smoke test:
  - `--mode smoke --devices 1,2,3 --trials ...` で完走すること。
  - `smoke_trials.csv` に `x0`, `y0`, `phi0`, `x_end`, `y_end`, `phi_end`, `T` が出ること。
  - `smoke_summary.csv` の `n_ok + n_max_steps` が `n_total` と一致すること。
- 数値 sanity check:
  - 外力 `f` を大きくすると平均初通過時間 `T1` が小さくなる傾向を確認する。
  - ノイズを固定 seed にして同じ結果が再現されること。
  - WCA 反発力のカットオフ外で力がゼロ、カットオフ内で壁から離れる向きになること。
- 性能確認:
  - 1 組み合わせあたりの実行時間、kernel occupancy、host-device コピー時間、CSV 書き込み時間を測る。
  - cuRAND が律速になる場合は `curand_normal2_double` の呼び方、`steps_per_launch`、状態配列の読み書き頻度を見直す。

## 未確定事項

- `D_perp = 0.5*D_parallel` は仕様書の最終的な簡略式を優先する。Tirado の `D_perp` をそのまま使う解釈に変更する場合は、`D_lab` とノイズ項を同時に変更する。
- `phi` は毎 step 正規化しない。もし smoke run で極端な角度増大による三角関数の精度問題が疑われる場合のみ、chunk 境界での条件付き正規化を追加する。
- production simulation では trial 詳細を出力しない。将来、特定 combo の詳細が必要になった場合は smoke/debug mode で対象 combo だけ trial CSV を出す。

## 参照

- `docs/specification.md`
- `docs/rod_simulation.md`
- `docs/architecture.md`
- cudarc crates.io: https://crates.io/crates/cudarc/0.19.7
- cudarc docs.rs: https://docs.rs/cudarc/latest/cudarc/
- cudarc feature flags: https://docs.rs/crate/cudarc/latest/features
- cudarc driver module: https://docs.rs/cudarc/latest/cudarc/driver/
- cudarc nvrtc module: https://docs.rs/cudarc/latest/cudarc/nvrtc/
- NVIDIA CUDA 12.2 Release Notes: https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html
- NVIDIA cuRAND Device API: https://docs.nvidia.com/cuda/archive/12.2.0/curand/group__DEVICE.html
- clap docs.rs: https://docs.rs/clap/latest/clap/
- serde docs.rs: https://docs.rs/serde/latest/serde/
- csv docs.rs: https://docs.rs/csv/latest/csv/
- anyhow docs.rs: https://docs.rs/anyhow/latest/anyhow/
