# nanorod

周期的に幅が変化する2次元チャネル内で、電気双極子の性質を持つ棒状粒子が行うブラウン運動を数値シミュレーションするプログラムです。一定の外力と一定の電場のもとで、粒子の平均初通過時間・非線形移動度・有効拡散係数を、GPU(CUDA)を用いた大規模アンサンブル計算で求めます。計測には、各試行を1周期の初通過で打ち切る `first_passage` モードと、全試行を一定時間走らせて変位の統計から直接算出する `fixed_time` モードの2つがあります。

物理モデルの詳細は [docs/rod_simulation.md](docs/rod_simulation.md) を、境界における補正については [docs/boundary_reflection.md](docs/boundary_reflection.md) を、計算機環境は [docs/architecture.md](docs/architecture.md) を参照してください。

## 必要環境

- Rust (2024 edition)
- CUDA Toolkit 12.x と NVIDIA GPU
  - カーネルは実行時に NVRTC で GPU の世代に合わせてコンパイルされるため、`nvcc` は不要です
- GPU がない環境では CPU 版としてビルド可能(後述)

用意されている [Dev Container](.devcontainer/devcontainer.json) を使うと、上記の環境がすぐに整います。

## 使い方

### 1. 設定ファイルを書く

シミュレーションの定数はすべて TOML ファイルで指定します。リポジトリ直下の [config.toml](config.toml) が設定例です。

```toml
delta_t = 4e-7          # 時間刻み幅 Δt
time = 100.0            # 1試行あたりのシミュレーション時間 T
sigma = 8e-3            # WCA ポテンシャルの σ(壁点間隔 0.25σ・代表点間隔 0.8σ の基準)
epsilon = 2.0           # WCA ポテンシャルの ε
ensemble_size = 1000    # 1ケースあたりの試行数(アンサンブルサイズ)
output_dir = "example/" # 全ケースの結果をまとめて出力するフォルダ

# 計測モード(省略時 "fixed_time"。詳細は「計測モード」の節を参照)
mode = "fixed_time"

# 以下の4つはリストで指定し、その全組み合わせ(直積)が実行される
# gamma, delta, f は "1/3" のような分数文字列でも指定可能
gamma = [1.0, 2.0]  # 電場トルクの永久双極子成分 γ = βpE
delta = ["1/3", 3.0] # 電場トルクの異方性成分と永久双極子成分の比 δ = |Δα|E/p
f = [5.0, 10.0]      # 一定外力(x方向)
m = [3, 6]           # 棒の片側代表点数(棒長 l = 2m × 0.8σ)

hist_stride = 100  # 角度・y分布ヒストグラムへ加算するステップ間隔(0で無効、省略時100)

[gpu]                  # 省略可
# ids = [0, 1, 2, 3]  # 使用するGPUのID(省略時は全GPU)
```

#### 計測モード

物理モデルは両モードで共通で、試行の打ち切り方と μ・D_eff の算出方法だけが変わります。

- **`fixed_time`(既定)**: 全試行を時間 T まで走らせ、各試行の変位 Δx = x(T) − x₀ の統計から次の量を直接算出します。
  - 平均速度 ⟨v⟩ = ⟨Δx⟩ / T
  - 非線形移動度 μ = ⟨v⟩ / f
  - 有効拡散係数 D_eff = (⟨Δx²⟩ − ⟨Δx⟩²) / 2T
  - 1周期を進む平均時間(平均初通過時間の推定) T₁ = L / |⟨v⟩|
- **`first_passage`**: 各試行を、重心が初期位置から1周期分(x₀ ± L)進んだ時点で打ち切ります。初通過時間の平均 T₁ とその2乗平均 T₂ から μ と D_eff を求めます。時間 T までに未通過だった試行は打ち切られ、平均の計算から除外されます。

### 2. シミュレーションを実行する

```sh
cargo run --release --features gpu                          # ./config.toml を使って全ケースを実行
cargo run --release --features gpu -- --config sweep.toml   # 設定ファイルを指定して実行
```

ケースは全GPUのワーカーに自動的に振り分けられ、終わったものから順に結果が書き出されます。進捗は標準エラー出力と `progress.jsonl` で確認できます。

### 3. 結果を見る

`output_dir` に次の構造で出力されます。

```
example/
├── config.toml                    # 使用した設定ファイルのコピー(再現用)
├── progress.jsonl                 # 進捗ログ(1行1イベントのJSON)
├── m1_f1_gamma0.25_delta0.5/      # ケースごとのフォルダ(m{m}_f{f}_gamma{γ}_delta{δ})
│   ├── summary.json               # このケースの集計結果
│   ├── trials.csv                 # 全試行の初期状態・終了状態・所要時間
│   ├── angle_hist.csv             # (x × φ) の角度分布ヒストグラム
│   └── y_hist.csv                 # (x × y) の y 分布ヒストグラム
├── m1_f1_gamma0.25_delta1/
│   └── ...
└── ...
```

`summary.json` の主な項目(両モード共通):

| 項目 | 意味 |
| --- | --- |
| `mode` | 計測モード(`first_passage` / `fixed_time`) |
| `n_total` | 総試行数 |
| `mu` | 非線形移動度 μ(f = 0 では `null`) |
| `D_eff` | 有効拡散係数 |

`mode = "first_passage"` のとき(μ = v/f、v = L/T₁、D_eff = L²(T₂ − T₁²) / 2T₁³):

| 項目 | 意味 |
| --- | --- |
| `n_ok` | 1周期を通過した試行数 |
| `n_right_passes` / `n_left_passes` | 右 / 左方向へ通過した試行数 |
| `n_max_steps` | 時間 T までに未通過だった試行数(平均の計算から除外) |
| `T1` / `T2` | 平均初通過時間 T₁ とその2乗平均 T₂ |

`mode = "fixed_time"` のとき(μ = ⟨v⟩/f、D_eff = (⟨Δx²⟩ − ⟨Δx⟩²) / 2T):

| 項目 | 意味 |
| --- | --- |
| `T` | 全試行を走らせた計測時間 |
| `mean_dx` / `var_dx` | 変位の平均 ⟨Δx⟩ と分散 ⟨Δx²⟩ − ⟨Δx⟩² |
| `v_mean` | 平均速度 ⟨v⟩ = ⟨Δx⟩ / T |
| `T1` | 1周期を進む平均時間 L / \|⟨v⟩\|(平均初通過時間の推定) |

各ケースの乱数シードはパラメータの組から決定論的に導出されるため、同じ設定ファイルからは GPU への割り当て順序に依らず、常にビット単位で同じ結果が得られます。

`angle_hist.csv` / `y_hist.csv` は、通過途中の棒の状態を `hist_stride` ステップごとに集計した占有時間重み付きの2次元ヒストグラムで、`x, phi, count`(または `x, y, count`)の tidy 形式です。`x` で pivot すると x 座標ごとの角度分布・y 分布が得られます。y は各 x の壁位置 ±ω(x) で正規化すると、y 方向に一様に達しているかを確認できます。

### 4. アニメーションを見る

```sh
cargo run --release -- animate
```

1粒子の運動をリアルタイムに描画します。初期パラメータには設定ファイルの各リスト(m, f, gamma, delta)の先頭値が使われます。

- **seed / m / f / γ / δ は GUI で変更可能**: f, γ, δ は実行中の粒子に即座に反映され、m は変更した瞬間に、seed は「リセット」ボタンを押したときに初期状態から反映されます
- 粒子(棒)・重心の軌跡・初期位置(青線)・左右の通過判定位置(緑線・赤線)が表示されます

## CPU版としてのビルド

GPU がないマシンでは、同じ物理モデルの CPU 実装(rayon 並列)で動かせます。GPU 版に比べ桁違いに遅いため、小規模な動作確認・検証用です。`gpu` フィーチャーを付けずにビルドすると CPU 版になります。

```sh
cargo run --release
```

乱数生成器だけが GPU 版(cuRAND Philox)と異なるため、結果は統計的には同等ですがビット単位では一致しません。

## プロジェクト構成

```
src/
├── main.rs           # CLI(run / animate サブコマンド)
├── config.rs         # TOML設定の読み込みとケース(直積)への展開
├── model.rs          # 共有の物理数式と集計(流路形状・境界反射・拡散係数)
├── simulation.rs     # 物理モデルのCPU実装(アニメーションとCPU版で使用)
├── runner.rs         # 全ケースの実行と結果の書き出し(GPU/CPUバックエンド)
├── renderer.rs       # アニメーション表示(egui/eframe)
├── gpu.rs            # GPUバックエンド(NVRTC・CUDA stream 管理)
└── kernels/
    └── simulation.cu # 物理モデルのGPU実装(CUDAカーネル)
docs/
├── rod_simulation.md      # 物理モデルの導出と定式化
├── boundary_reflection.md # 境界外状態の鏡映補正
└── architecture.md        # 実行マシンのハードウェア構成
```

## 性能メモ(NVIDIA A100 での実測)

- 1 step の計算コストは棒の代表点数 2m+1 にほぼ比例し、飽和スループットは A100 1枚あたり約 **2.6〜2.7×10⁹ 代表点ステップ/s**(m=1 で約 9×10⁸ steps/s、m=30 で約 4.3×10⁷ steps/s)
- 1ケースだけでは GPU を使い切れないため、**複数ケースを複数の CUDA stream で同時実行**して占有率を上げています。同時実行数は、同時に走る試行数が飽和スループットに達するよう `ensemble_size` とケース数から自動調整されるため、ユーザーが指定する項目はありません
