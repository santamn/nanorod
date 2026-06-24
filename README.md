# nanorod

流路（周期的にくびれたチャネル）中を進む棒状粒子のブラウン運動を GPU 上で数値シミュレーションし、1周期の初通過時間や移動度などを集計するプログラム。

物理モデルの詳細は [docs/rod_simulation.md](docs/rod_simulation.md) を参照。

## ビルド

GPU シミュレーション本体は `gpu` feature と CUDA toolkit（NVRTC / cuRAND）が必要。

```bash
cargo build --release --features gpu
```

実行時に NVRTC が `curand_kernel.h` などを見つけられるよう、必要に応じて環境変数 `CUDA_HOME` か `--cuda-include-path` を指定する。

## 実行

実行したいパラメータをリストで与えると、それらの全ての組み合わせが combo として実行される。

```bash
cargo run --release --features gpu --bin nanorod -- \
  --m 1,4,8 \
  --f 1,2,5,10 \
  --beta-pe 0.25,0.5,1.0 \
  --abs-delta-alpha-e-over-p 0.5,2.0
```

上の例では `3 × 4 × 3 × 2 = 72` combo を実行する。

### パラメータ

これらは必須で、それぞれカンマ区切りのリストとして与える。
`f`・`beta-pe`・`abs-delta-alpha-e-over-p` は `1/3` のような分数表記も使える。

| オプション | 意味 |
| :--- | :--- |
| `--m <M,...>` | 棒の片側代表点数 m のリスト（棒長は `l = 2m * 0.8σ`）。m ≥ 1 の整数。 |
| `--f <F,...>` | 駆動力 f のリスト。例: `--f 1,2,1/3` |
| `--beta-pe <B,...>` | βpE（双極子モーメント p = ql を含む電場結合の大きさ）のリスト。電場トルクに直接掛かる係数で、棒長 l に依らない。例: `--beta-pe 0.5,1/3` |
| `--abs-delta-alpha-e-over-p <D,...>` | \|Δα\|E/p（電場トルクの異方性成分と永久双極子成分の比）のリスト。トルクは補足資料 式(8) の `τ_E = βpE·cosφ·(1 − (\|Δα\|E/p)·sinφ)` で、正の値ほど棒を y 軸から横へ倒す効果が強くなる（1 を超えると傾いた配向が安定）。例: `--abs-delta-alpha-e-over-p 1,2,3` |

同じ値を重複して与えても、combo は重複しないよう自動的に1つにまとめられる。

### 実行制御オプション

| オプション | 既定値 | 意味 |
| :--- | :--- | :--- |
| `--output-dir <DIR>` | `output` | 結果を書き出すルートディレクトリ。既存のディレクトリには上書きせず、新規でなければエラーになる。 |
| `--devices <ID,...>` | `0,1,2` | 使用する GPU デバイス番号。 |
| `--trials <N>` | `1000` | 1 combo あたりの試行回数。 |
| `--streams <N>` | `16` | 各 GPU が同時並行に処理する combo 数（= CUDA stream 数）。占有率を上げるための値で、A100 では約16で飽和する。 |
| `--max-steps <N>` | `250000000` | 1 試行あたりの最大ステップ数。これを超えた試行は未通過として集計から除外する。 |
| `--steps-per-launch <N>` | `10000` | 1 回の kernel 起動で進めるステップ数。 |
| `--hist-stride <N>` | `100` | 角度・y分布ヒストグラムへ加算するステップ間隔。`0` で記録を無効にする。 |
| `--seed <N>` | `1` | 乱数シード。combo ごとに決まる系列オフセットと組み合わせて再現性を保つ。 |
| `--progress-interval-sec <N>` | `5` | 進捗を `progress.jsonl` に書き出す間隔（秒）。 |
| `--cuda-include-path <PATH>` | （なし） | NVRTC に渡す追加 include パス。複数指定可。 |
| `--cuda-arch <ARCH>` | `compute_80` | NVRTC の `--gpu-architecture`。 |

## 出力

`--output-dir` の下に、combo ごとのサブフォルダが作られる。フォルダ名はパラメータから決まる:

```
output/
├── progress.jsonl                 # 全 combo 共通の進捗ログ（1 行 1 イベント）
├── m1_f1_beta0.25_delta0.5/
│   ├── summary.json               # この combo の集計結果
│   ├── trials.csv                 # 各試行の初期状態・終了状態・初通過時間
│   ├── angle_hist.csv             # (x × φ) の角度分布ヒストグラム
│   └── y_hist.csv                 # (x × y) の y 分布ヒストグラム
├── m1_f2_beta0.25_delta0.5/
│   └── ...
└── ...
```

フォルダ名は `m{m}_f{f}_beta{beta}_delta{delta}` の形式で、数値は末尾の余分なゼロを落として表記する（例: `1/3` は `0.333333`、`0` は `0`）。

### `summary.json` の主な項目

| 項目 | 意味 |
| :--- | :--- |
| `n_total` / `n_ok` | 総試行数 / 初通過した試行数 |
| `n_right_passes` / `n_left_passes` | 右方向 / 左方向へ1周期通過した試行数 |
| `n_max_steps` | 最大ステップ数に達して未通過だった試行数 |
| `passage_fraction` | `n_ok / n_total` |
| `T1` / `T2` | 平均初通過時間 / その2乗平均 |
| `mu` | 移動度 μ = v/f（v = L/T1 は1周期あたりの平均速度）。`f = 0` では定義できず `NaN`。 |
| `D_eff` | 有効拡散係数 |

### 角度・y分布ヒストグラム

`angle_hist.csv` と `y_hist.csv` は、通過途中の棒の状態を `--hist-stride` ステップごとにGPU上で集計した2次元ヒストグラム。combo 内の全試行・全ステップにわたって占有時間で重み付けされた分布で、`x, phi, count`（または `x, y, count`）のtidy 形式で出力される。

- `x` は1周期 `[0, 1)` に畳んだ位置の bin 中心（くびれは既知の位相に対応）
- `phi` は `[0, 2π)` に巻き戻した角度の bin 中心
- `y` は `[-2.3, 2.3]` を等分した bin 中心。
  - 各 `x` における壁位置は `±omega(x)` で計算できるので、`y` を壁位置で正規化すれば「y方向に一様（平行）に達しているか」を確認できる。

解析時は `x` で pivot すれば、x 座標ごとの角度分布・y分布が得られる。
