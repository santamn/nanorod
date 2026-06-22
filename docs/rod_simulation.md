# 棒状粒子のブラウン運動の数値シミュレーション

周期的に幅が変化するような2次元空間において、電気双極子の性質を持つ棒状粒子が、一定の外力と一定の電場のもとで行うブラウン運動を数値的にシミュレートし、粒子の平均初通過時間を求める。

## 数値シミュレーションの確率微分方程式

### ランジュバン方程式

棒状粒子のシミュレーションを行うにあって、棒状粒子を中心点とそこから左右対称な位置に等間隔にとった $2m$ 個の点の計 $2m+1$ 個の点によって代表する。代表点は壁からの反発力とそのトルクを計算するために使い、$x$ 軸方向に一定にかかる外力は粒子全体に作用する合力として扱う。これから考えるようなブラウン運動では粘性抵抗が十分大きいので、系は過減衰状態であると仮定すると、粒子の重心の位置ベクトル $\bm{X}$ と向きを表す角度 $\phi$ は次のようなランジュバン方程式に従うと考えられる。

$$
\begin{cases}
  \dfrac{d\bm{X}}{dt} &= \mathbb{R}(\phi) \beta \mathbb{D} \mathbb{R}(\phi)^{-1} F + \mathbb{R}(\phi) \bm{\xi}(t) \\
  \dfrac{d\phi}{dt} &= \beta D_r \tau + \xi_r(t)
\end{cases}
$$

ただし、$\bm{F}$ は並進運動を駆動する力、$\tau$ は粒子に働くトルク、$\mathbb{R}(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}$ は角度 $\phi$ に対応する回転行列、$\mathbb{D} = \begin{pmatrix} D_{\|} & 0 \\ 0 & D_\perp \end{pmatrix}$ は粒子の座標系における拡散係数行列（ $D_{\|}$ は棒の長軸方向の拡散係数、 $D_\perp$ は棒の単軸方向の拡散係数）、$D_r$ は回転の拡散係数を表す。

今、棒の中心点とそこから左右対称な位置に等間隔にとった $2m$ 個の点の計 $2m+1$ 個の点 $j \in \{-m, -m+1, \ldots, m\}$ には壁からの反発力 $\bm{f^{\text{rep}}_j}$ が働く。一方で、一定の外力 $\bm{f}$ は粒子全体にかかる合力なので、各代表点に同じ外力を与えて合算しない。したがって、並進運動を駆動する力 $\bm{F}$ は次のように表される。

$$
\bm{F} = \bm{f} + \sum_{j=-m}^m \bm{f}^{\text{rep}}_j
$$

また、トルク $\tau$ は電気双極子が電場から受ける作用によって生まれるトルク $\tau_{\bm{E}}(\phi)$ と、[壁からの反発力によって生まれるトルク](#電場によるトルクの計算) $\tau^{\text{rep}}$ の和に分けられる。ここで、 $l$ は粒子の長軸方向の長さである。

$$
\tau = \tau^{\text{rep}} + \tau_{\bm{E}}(\phi) = \dfrac{l}{2m}\bm{n} \times \left( \sum_{k=1}^m k\left(\bm{f}_k^{\text{rep}} - \bm{f}_{-k}^{\text{rep}} \right)\right) + pE\cos\phi(1+\frac{\Delta\alpha E}{p}\sin\phi)
$$

また、$\bm{\xi}(t)$ と $\xi_r(t)$ はそれぞれ位置と回転に対するガウス白色雑音である。これらの雑音は平均がゼロであると仮定されており、その共分散は次のように表される。

$$\begin{aligned}
\langle \bm{\xi}(t) \rangle &= \bm{0}, 
& \langle \bm{\xi}^\top(t)\bm{\xi}(t') \rangle &= \begin{pmatrix} 
  2D_\| & 0 \\
  0    & 2D_\perp
\end{pmatrix} \delta(t-t') \\
\langle \xi_r(t) \rangle &= 0, 
& \langle \xi_r(t)\xi_r(t') \rangle &= 2D_r \delta(t-t')
\end{aligned}$$

### 無次元化

粒子の流路の周期長を $L$ とすると、位置ベクトル $\bm{X}$ を $\widetilde{\bm{X}} = \frac{\bm{X}}{L}$ と無次元化することができる。また、時間 $t$ を標準的な拡散係数 $D_0$ を使って $\widetilde{t} = \frac{D_0}{L^2}t$ と無次元化することもできる。これらの無次元化を行うと、ランジュバン方程式は以下のように書き換えられる。

$$
\begin{cases}
  \frac{d\widetilde{\bm{X}}}{d\widetilde{t}} &= \mathbb{R}(\phi) (\mathbb{D}/D_0) \mathbb{R}(\phi)^{-1} (\beta L\bm{F}) + \mathbb{R}(\phi) \left\{\frac{L}{D_0}{\bm{\xi}}(\frac{L^2}{D_0}\widetilde{t})\right\} \\
  \frac{d\phi}{d\widetilde{t}} &= \left(D_r/\frac{D_0}{L^2}\right) (\beta\tau_{\text{rep}} + \beta\tau_E)+ \left\{ \frac{L^2}{D_0}{\xi}_r(\frac{L^2}{D_0}\widetilde{t}) \right\}
\end{cases}
$$

ここで、

$$\begin{aligned}
\tilde{l} = \dfrac{l}{L},\ \widetilde{\bm{F}} = \beta L\bm{F},&\ \widetilde{\bm{f}}^{\text{rep}} = \beta L\bm{f}^{\text{rep}}\\
\widetilde{\mathbb{D}} = \dfrac{\mathbb{D}}{D_0},& \ 
\widetilde{D_r} = \dfrac{D_r}{\frac{D_0}{L^2}} \\
\widetilde{\tau}_{\text{rep}} = \beta\tau_{\text{rep}} &= \dfrac{\tilde{l}}{2m}\bm{n} \times \left(\sum_{k=1}^m k\left(\widetilde{\bm{f}}_k^{\text{rep}} - \widetilde{\bm{f}}_{-k}^{\text{rep}} \right)\right) \\
\widetilde{\tau}_E = \beta\tau_E &= \beta pE\cos\phi(1+\Delta\alpha\frac{E}{p}\sin\phi)\\
\bm{\widetilde{\xi}}(t) = \frac{L}{D_0}{\bm{\xi}}(t),& \quad
\widetilde{\xi}_r(t) = \frac{L^2}{D_0}{\xi}_r(t)
\end{aligned}$$

とおくと、無次元化されたランジュバン方程式は以下のように表される。

$$
\begin{cases}
  \frac{d\widetilde{\bm{X}}}{d\widetilde{t}} &= \mathbb{R}(\phi) \widetilde{\mathbb{D}} \mathbb{R}(\phi)^{-1} \widetilde{\bm{F}} + \mathbb{R}(\phi) \bm{\widetilde{\xi}}(\widetilde{t}) \\
  \frac{d\phi}{d\widetilde{t}} &= \widetilde{D_r} (\widetilde{\tau}_{\text{rep}} + \widetilde{\tau}_E)+ \widetilde{\xi}_r(\widetilde{t})
\end{cases}
$$

これを2次元と1次元の(標準)Wiener過程の微小増分 $d\bm{W}, dW_r$ を用いた確率微分方程式に直すと以下のように書ける。

$$
\begin{cases}
  d\widetilde{\bm{X}} &= \mathbb{R}(\phi) \widetilde{\mathbb{D}} \mathbb{R}(\phi)^{-1} \widetilde{\bm{F}} d\tilde{t} + \mathbb{R}(\phi) \sqrt{2\tilde{\mathbb{D}}} \, d\bm{W}(\tilde{t}) \\
  d\phi &= \widetilde{D}_r (\widetilde{\tau}_{\text{rep}} + \widetilde{\tau}_E) d\tilde{t} + \sqrt{2\widetilde{D}_r} \, dW_r(\tilde{t})
\end{cases}
$$

### 拡散係数

Tirado and Garcia de la Torre の式を用いて、$\widetilde{\mathbb{D}}$ と $\widetilde{D}_r$ を次のように表すことができる。ここで、標準となる拡散係数 $D_0$ は棒長ごとに変えず、基準長

$$
\tilde{l}_0 = 6 \times 0.8\sigma = 6 \times 0.8 \times 8\times 10^{-3} = 0.0384
$$

の棒について

$$
D_0=\frac{k_B T}{8 \pi \eta L\tilde{l}_0}\left(3 \ln p_0+2 \nu_{\|,0}+\nu_{\perp,0}\right),\qquad p_0=40\tilde{l}_0
$$

として定義する。この共通の $D_0$ で全ての棒長を無次元化することで、棒長ごとに時間スケールが変わらないようにする。

$$
\tilde{D}_{\|}(\tilde{l})=\frac{D_{\|}(\tilde{l})}{D_0}
=\frac{4\tilde{l}_0}{\tilde{l}}\frac{\ln p+\nu_{\|}}{3 \ln p_0+2 \nu_{\|,0}+\nu_{\perp,0}}
$$

$$
\tilde{D}_{\perp}(\tilde{l})=\frac{D_{\perp}(\tilde{l})}{D_0}
=\frac{2\tilde{l}_0}{\tilde{l}}\frac{\ln p+\nu_{\perp}}{3 \ln p_0+2 \nu_{\|,0}+\nu_{\perp,0}}
$$

$$
\begin{gathered}
\widetilde{D}_r(\tilde{l})
= D_r(\tilde{l}) \times \frac{L^2}{D_0}
&= \frac{24\tilde{l}_0}{\tilde{l}^3}
\frac{\ln p+\delta_{\perp}}{3 \ln p_0+2 \nu_{\|,0}+\nu_{\perp,0}}
\end{gathered}
$$

ただし、$p = \dfrac{l}{d}$ は棒の長さと太さの比を表す無次元量である。また、$\nu_{\|}, \nu_{\perp}, \delta_{\perp}$ はアスペクト比 $p$ の関数であり、$\nu_{\|,0}, \nu_{\perp,0}$ は $p_0$ における値である。これらは次のように定義される。

$$\begin{aligned}
\nu_{\|} &= -0.207 + \frac{0.980}{p} - \frac{0.133}{p^2} \\
\nu_{\perp} &= 0.839 + \frac{0.185}{p} + \frac{0.233}{p^2} \\
\delta_{\perp} &= -0.630 + \frac{0.917}{p} - \frac{0.050}{p^2}
\end{aligned}$$

X. Yang (2019) の論文に合わせるため、 $p = 40\tilde{l}$ とする。

このとき、

$$\begin{align*}
\mathbb{R}(\phi) \mathbb{D} \mathbb{R}^{-1}(\phi) &= \mathbb{R}(\phi) \mathbb{D} \mathbb{R}(-\phi) \\
&= \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} D_{\parallel} & 0 \\ 0 & D_{\perp} \end{pmatrix} \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} \\
&= \begin{pmatrix} D_{\parallel}\cos^2\phi + D_{\perp}\sin^2\phi & (D_{\parallel} - D_{\perp})\sin\phi\cos\phi \\ (D_{\parallel} - D_{\perp})\sin\phi\cos\phi & D_{\parallel}\sin^2\phi + D_{\perp}\cos^2\phi \end{pmatrix}
\end{align*}$$

X. Yang (2019) の論文では $\dfrac{D_{\|}}{D_\perp} \approx 2$ の関係があると書かれているので、このシミュレーションでは $D_\perp = \frac{1}{2}D_{\|}$ とする。すると、上の式は次のように簡略化される。

$$\begin{align*}
\mathbb{D}_{\text{lab}}(\phi) = \mathbb{R}(\phi) \mathbb{D} \mathbb{R}^{-1}(\phi)&= \begin{pmatrix} \frac{3}{4}D_{\parallel} + \frac{1}{4}D_{\parallel}\cos 2\phi & \frac{1}{4}D_{\parallel}\sin 2\phi \\ \frac{1}{4}D_{\parallel}\sin 2\phi & \frac{3}{4}D_{\parallel} - \frac{1}{4}D_{\parallel}\cos 2\phi \end{pmatrix} \\
&= \frac{D_{\parallel}}{4} \begin{pmatrix} 3 + \cos 2\phi & \sin 2\phi \\ \sin 2\phi & 3 - \cos 2\phi \end{pmatrix}
\end{align*}$$

### 確率微分方程式

シミュレーションで用いる確率微分方程式は以下の通り。ただし、無次元化を表すチルダは省略している。

$$\begin{cases}
  d\bm{X} &= \mathbb{D}_{\text{lab}}(\phi) \bm{F} dt + \mathbb{R}(\phi) \sqrt{2\mathbb{D}} \, d\bm{W}(t) \\
  d\phi &= D_r (\tau_{\text{rep}} + \tau_E) dt + \sqrt{2D_r} \, dW_r(t)
\end{cases}$$

### 数値解法

先行研究の Supporting Information に合わせ、Euler-Maruyama 法ではなく予測子・修正子法で 1 step を進める。このシミュレーションでは $D_{\|}$, $D_\perp$, $D_r$ は粒子長だけで決まるが、実験室系の拡散テンソル $\mathbb{D}_{\text{lab}}(\phi)$ と並進ノイズの回転 $\mathbb{R}(\phi)$ は角度 $\phi$ に依存する。そのため、修正子段階では棒に働く一般化力

$$
\bm{G}(\bm{X}, \phi) = (F_x, F_y, \tau_{\text{rep}} + \tau_E)
$$

に加えて、$\mathbb{D}_{\text{lab}}(\phi)$ と $\mathbb{R}(\phi)\sqrt{2\mathbb{D}\Delta t}$ も予測角度 $\phi^\*$ で再評価する。各 step で同じガウス乱数を使い、まず現在状態 $s_n=(\bm{X}_n,\phi_n)$ から予測点 $s^\*$ を作る。

$$\begin{aligned}
\bm{X}^\* &= \bm{X}_n + \mathbb{D}_{\text{lab}}(\phi_n)\bm{F}(s_n)\Delta t
  + \mathbb{R}(\phi_n)\sqrt{2\mathbb{D}\Delta t}\,\bm{\eta}_t,\\
\phi^\* &= \phi_n + D_r\tau(s_n)\Delta t + \sqrt{2D_r\Delta t}\,\eta_r.
\end{aligned}$$

次に予測点で一般化力、実験室系の拡散テンソル、並進ノイズの回転を再評価し、最終的な更新を

$$\begin{aligned}
\bm{X}_{n+1} &= \bm{X}_n + \mathbb{D}_{\text{lab}}(\phi^\*)\bm{F}(s^\*)\Delta t
  + \mathbb{R}(\phi^\*)\sqrt{2\mathbb{D}\Delta t}\,\bm{\eta}_t,\\
\phi_{n+1} &= \phi_n + D_r\tau(s^\*)\Delta t + \sqrt{2D_r\Delta t}\,\eta_r
\end{aligned}$$

で求める。ここで $\bm{\eta}_t$ と $\eta_r$ は予測子と修正子で共通の標準正規乱数である。

ただし、強い $x$ 方向外力と反発力上限の組み合わせでは、予測点や修正子で得た最終候補状態の重心が流路外に出る場合がある。その場合は、重心位置と棒の角度を現在位置側の境界に対して鏡映した状態へ補正する。予測点は修正子段階の評価前に補正し、最終候補状態は保存前に補正する。補正の幾何と実装上の扱いは [境界外状態の鏡映補正](./boundary_reflection.md) にまとめる。

## 壁面の処理

### 反発力の計算

WCAポテンシャルを用いて、粒子が壁に接触したときの反発力をモデル化する。WCAポテンシャルは、次のように定義される。

$$U_{\text{WCA}}(r) = \begin{cases}
4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right] + \epsilon & r < 2^{1/6}\sigma \\
0 & r \geq 2^{1/6}\sigma
\end{cases}$$

数値シミュレーション上では、壁に一定の間隔で点を配置し、粒子の各代表点（二つの端点と重心）と壁の代表点との距離 $r$ によって壁からの反発力を計算する。

$$
\bm{f}^{\text{rep}}_j = \sum_{|r_k - r_i| < \sqrt[6]{2}\sigma} -\nabla U_{\text{WCA}}(|r_k - r_j|) = \sum_{|r_k - r_j| < \sqrt[6]{2}\sigma} 
\frac{24\epsilon}{\sigma}\left(\frac{\sigma}{r}\right)^7\left[2\left(\frac{\sigma}{r}\right)^6 - 1\right] \bm{e}_{k \to j}
$$

ここで、$r_k$ は壁の代表点の位置、$r_j \ (j \in \{-m,\ldots,m\})$ は粒子の代表点の位置、$\bm{e}_{k \to j}$ は壁の代表点から粒子の代表点への単位ベクトルである。ここで、

$$
\bm{d}_{jk} = \bm{r}_j - \bm{r}_k,\
r_{jk}^2 = |\bm{d}|^2,\
s^2 = \frac{\sigma^2}{r^2},\
s^6 = \left(\frac{\sigma^2}{r^2}\right)^3
$$

とすると、より効率的に計算できる形に書き換えられる。

$$
\bm{f}^{\text{rep}}_j = \sum_{r_{jk}^2 < \sqrt[3]{2}\sigma^2} \frac{24\epsilon}{r_{jk}^2}s^6\left(2s^6 - 1\right) \bm{d}_{jk}
$$

X. Yang (2019) の論文にならって、壁からの反発力を計算するためのサンプリング点は $0.25\sigma$ ごとに配置する。周期 $L = 1$ のとき、$0.25\sigma = 0.002$ なので、1周期につき $500$ 個のサンプリング点が配置される。また、粒子上には $0.8\sigma$ ごとに代表点を配置する。各種定数は次の通り。

$$
\epsilon = 2,\ 
\sigma = 8\times 10^{-3},\ 
dt = 4\times 10^{-7}
$$

### 計算の効率化

WCA ポテンシャルはカットオフ半径 $r_c=2^{1/6}\sigma$ より遠い壁点からの寄与がゼロであるため、全ての粒子代表点と全ての壁代表点の組を調べてはならない。

本シミュレーションの壁は任意の2次元点群ではなく、 $y=\pm\omega(x)$ で与えられる固定された周期曲線なので、2次元グリッドや動的セルリストは使わず、壁点の $x$ 方向インデックスだけを用いて近傍候補を絞る。

壁点間隔を $\Delta x_{\text{wall}}=0.25\sigma$ とし、1周期あたりの壁点数を
$$
N_{\text{wall}}=\frac{L}{\Delta x_{\text{wall}}}
$$
とする。上壁・下壁の $y$ 座標
$$
y_k^+=\omega(k\Delta x_{\text{wall}}),\qquad
y_k^-=-\omega(k\Delta x_{\text{wall}})
$$
をあらかじめ配列として GPU メモリ上に保持する。

各粒子代表点 $\bm r_j=(x_j,y_j)$ について、
$$
k_0=\mathrm{round}(x_j/\Delta x_{\text{wall}})
$$
を求め、
$$
K=\left\lceil \frac{r_c}{\Delta x_{\text{wall}}} \right\rceil
$$
として、$k=k_0-K,\ldots,k_0+K$ の壁点だけを調べる。今回の値では
$$
r_c=2^{1/6}\sigma,\qquad \Delta x_{\text{wall}}=0.25\sigma
$$
なので $K=5$ であり、各粒子代表点について上下壁あわせて最大 $2(2K+1)=22$ 点だけを調べればよい。

周期境界は、壁形状の参照には
$$
k_{\mathrm{mod}}=((k\bmod N_{\text{wall}})+N_{\text{wall}})\bmod N_{\text{wall}}
$$
を使い、壁点の $x$ 座標には
$$
x_{\text{wall}}=k\Delta x_{\text{wall}}
$$
を使う。これにより、粒子が複数周期分進んでも、最近傍の周期壁点との距離を正しく計算できる。

各候補壁点について
$$
\bm d=\bm r_j-\bm r_k,\qquad r^2=|\bm d|^2
$$
を計算し、
$$
0<r^2<r_c^2
$$
のときのみ WCA 反発力を加える。平方根は使わず、
$$
s^2=\frac{\sigma^2}{r^2},\qquad s^6=(s^2)^3
$$
として
$$
\bm f \mathrel{+}=24\epsilon\frac{1}{r^2}s^6(2s^6-1)\bm d
$$
で計算する。

ただし、LJ 型反発力は $r \to 0$ で特異的に大きくなるため、1 つの粒子代表点と 1 つの壁点から受ける反発力ベクトルの大きさには

$$
|\bm f_{jk}^{\text{rep}}| \leq 2.5\times 10^4
$$

の上限を設ける。この値は反発力の大きさそのものの上限であり、二乗値ではない。上限を超える場合は、反発力の向きを保ったまま大きさだけを $2.5\times 10^4$ に切り詰めてから合算する。

GPU 実装では、各スレッドが1つの粒子、または1つの粒子代表点を担当し、$q=-K,\ldots,K$ と上下壁に対する固定長ループを回す。候補数が固定されるため、可変長セルリストよりも分岐と間接メモリアクセスが少なく、GPU 上で効率的に実行できる。

## 数値計測

$y = \pm \omega(x)\ \text{where } \omega(x) = \sin(2\pi x) + 0.25\sin(4\pi x) + 1.12$ に囲まれた領域中を上の[確率微分方程式](#確率微分方程式)に従って運動する棒状粒子のシミュレーションを行い、左右どちらかへ1周期分進むまでの1次及び2次の平均初通過時間 $T_1(x_0 \to x_0 \pm L), T_2(x_0 \to x_0 \pm L)$ を数値的に求める。

具体的には次の手順で行う:

1. 粒子の重心の初期位置 $x_0 \in (-0.1, 0.7),\ y_0 \in (-\omega(x_0)+\frac{l}{2},\omega(x_0)-\frac{l}{2})$ と角度 $\phi_0 \in (0,2\pi)$ をランダムに決定する
2. 初期状態から粒子が $x = x_0 + L = x_0 + 1$ または $x = x_0 - L = x_0 - 1$ ( $\omega(x)$ の周期は $L=1$ )を初めて通過するまでの時間を計測し、左右どちらへ通過したかも記録する
3. 1と2を $N = 3\times 10^4$ 回行う
4. 3で得られた $T$ の平均 $T_1$ と 2乗の平均 $T_2$ を求める。
5. $T_1$ と $T_2$ から次の値を求める
  $$\begin{aligned}
  &\text{平均速度 } v = \frac{L}{T_1} \\
  &\text{有効拡散係数 } D_{\text{eff}} = \frac{L^2}{2}\left(\frac{T_2 - T_1^2}{T_1^3} \right)
   \end{aligned}$$

シミュレーションは、以下のパラメータの組み合わせを全て試す形で行う。

$$\begin{align*}
l = 2\times 0.8\sigma \ (代表点3個, m = 1) &,\ 8\times 0.8\sigma \ (代表点9個, m = 4),\\
16\times 0.8\sigma \ (代表点17個, m = 8) &,\ 30\times 0.8\sigma \ (代表点31個, m = 15),\\
60\times 0.8\sigma \ (代表点61個, m = 30) &
\end{align*}$$

$$
\beta pE = \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1, \frac{3}{2}, 2
$$

$$
\Delta\alpha\frac{E}{p} = \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1, \frac{3}{2}, 2
$$

各シミュレーションにおいて、 $x$ 軸方向へ一定にかける粒子全体への合力 $\bm{f} = f\bm{e}_x$ は次のパターンで変化させる。

$$
f = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
$$


この時、必ず以下を記録に残すこと。

- 定数の値 $(l, \beta pE, \Delta\alpha\frac{E}{p}, f)$
- 各定数の組み合わせに対する $T$ の平均 $T_1$ と 2乗の平均 $T_2$、及び $T_1$ と $T_2$ から求めた平均速度 $v$ と有効拡散係数 $D_{\text{eff}}$

## 資料

### 電場によるトルクの計算

トルク $\tau_{\bm{E}}$ は電気双極子が電場から受ける作用によって生まれるものであるから、電場 $\bm{E}$ のもとでの電気双極子のエネルギー $U_{\bm{E}}$ を用いて、$\tau_{\bm{E}} = -\frac{\partial}{\partial \Phi}U_{\bm{E}}$ と表される。今考えている粒子の永久双極子モーメントを $\bm{p}$ 、分極テンソルを $\hat{\alpha}$ とすると、電気双極子のエネルギーは

$$ U_{\bm{E}} = -\bm{p}\cdot\bm{E} - \frac{1}{2}\bm{E}^\top\hat{\alpha}\bm{E} $$

と表される。ここで、分極テンソル $\hat{\alpha}$ は棒の向きに沿った成分 $\alpha_{\parallel}$ と、棒の向きに垂直な成分 $\alpha_{\perp}$ を持つ対称なテンソルであると仮定する。すなわち、$\hat{\alpha} = \alpha_{\parallel}\bm{n}\bm{n}^\top + \alpha_{\perp}(\bm{I} - \bm{n}\bm{n}^\top)$ であるとすると、$U_{\bm{E}}$ は次のように表せる。

$$ U_{\bm{E}} = -\bm{p}\cdot\bm{E} - \frac{1}{2}\alpha_{\perp}|\bm{E}|^2 - \frac{1}{2}(\alpha_{\parallel} - \alpha_{\perp})(\bm{n}\cdot\bm{E})^2 $$

$\bm{n} = \left(\begin{array}{l}\cos\phi \\ \sin\phi\end{array}\right),\ \bm{E} = \left(\begin{array}{l}0 \\ E\end{array}\right)$ と設定すると

$$ U_{\bm{E}}(\phi) = -pE\sin\phi - \frac{E^2}{2}\alpha_{\perp} - \frac{E^2}{2}(\alpha_{\parallel} - \alpha_{\perp})\sin^2\phi $$

より

$$\begin{aligned}
\tau_{\bm{E}}(\phi) &= -\frac{\partial}{\partial \phi}U_{\bm{E}}(\phi) \\
&= pE\cos\phi + \Delta\alpha E^2\sin \phi\cos \phi \\
&= pE\cos\phi(1+\Delta\alpha\frac{E}{p}\sin\phi)
\end{aligned}$$

ここで $\Delta\alpha \coloneqq \alpha_{\parallel} - \alpha_{\perp},\ \bm{p} = p\bm{n}$ である。
