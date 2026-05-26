# トルクの表式の修正

## 1. トルクの定義から導出

過減衰 Langevin の保存力部分は free-energy 勾配ですから、本質的に次が成り立つ必要があります:

$$F_\theta = -\frac{\partial U_\text{total}}{\partial \theta}$$

ロッドが body-X 軸上の 1D 粒子列 (粒子 $k$ の body 位置 $(X_k, 0)$) で、lab 系の粒子位置は

$$r_k(\theta) = r + R(\theta)\begin{pmatrix}X_k \\ 0\end{pmatrix}$$

これを $\theta$ で偏微分します。$R'(θ)\cdot(X_k, 0)$ を body 系に戻すと body 成分 $(0, X_k)$ になるので

$$\frac{\partial r_k}{\partial \theta}\Bigg|_{\text{body}} = \begin{pmatrix} 0 \\ X_k\end{pmatrix}$$

連鎖律より

$$\frac{\partial U}{\partial \theta} = \sum_{k,l} \nabla_{r_k} U \cdot \frac{\partial r_k}{\partial \theta} = \sum_{k,l} X_k \cdot \nabla_Y U(|r_l - r_k|)$$

したがって

$$\boxed{F_\theta = -\sum_{k,l} X_k \cdot \nabla_Y U(|r_l - r_k|)}$$

ここで $X_k$ は 符号付きの body-X 座標 (ロッド左端で負, 右端で正)。SI の $|r_l - r|$ (壁粒子とロッド中心の lab 系距離の絶対値) とは別物です。

## 2. パリティ対称性によるテスト (SI 式が物理的に成立しない決定的根拠)

ロッドを body-X 軸に沿わせ、壁粒子を 1 個だけ body 座標 $(X_l, Y_l)$ に置きます。これを body-Y 軸まわりでミラー反転 すると $X_l \to -X_l$ ($Y_l$ 不変)。

物理的に期待される振る舞い:

- 反転前: 壁が右端側にあり、$Y_l$ 方向の力がロッドを「右端を引き上げる」向きに回す
- 反転後: 壁が左端側、同じ $Y_l$ 方向の力は「左端を引き上げる」向きに回す
トルクは符号反転しなければならない

| | 標準式 $\sum X_k \cdot\nabla_Y U$ | SI 式 $\sum \|r_l - r\|\cdot\nabla_Y U$ |
|---|---|---|
| 最近接 $X_k$ | $+a \to -a$ | (該当なし) |
| $\|r_l - r\|$ | $\sqrt{a^2 + Y_l^2}$ (不変) | 不変 |
| $\nabla_Y U$ | $\frac{Y_l - Y_k}{r}\cdot U'$ で $X_k \to -X_k$ に対し最近接粒子の $Y_k = 0$ なので不変 | 同上、不変 |
| トルク | 符号反転 ✓ | 不変 ✗ |

つまり SI の式を文字通り計算すると、ロッドの左右どちら側に壁が来ても 同じ符号のトルク を返してしまい、ロッドを壁と平行に向ける復元トルクが生じません。これは Fig. 2(a) や Fig. 3(b) で観測されている "neck 内でロッドが $θ=0$ 付近に揃う" という現象 (式 (3) の解析の前提でもある) と矛盾します。

## 3. 表記揺れの最尤解釈

LJ のカットオフが $1.12\sigma$ と非常に短いため、各壁粒子 $l$ に対し有意な力を及ぼすロッド粒子 $k$ はほぼ唯一に決まります。その $k$ は近似的に

$$X_k \approx (r_l - r) \cdot \hat{x}_\text{body} = (r_l - r)_X^\text{body}$$

つまり「壁から見たロッド中心ベクトルの body-X 成分 (符号付き)」です。SI が

$(r_l - r)_X^\text{body}$ または $(r_l - r)\cdot\hat{x}$ のような符号付き投影を意図して書いたのに、組版時に $|\cdot|$ (絶対値) として印字されてしまった、と解釈するのが整合的です。$|r_l - r|$ (3次元距離) と $(r_l - r)_X^\text{body}$ (符号付き X 投影) は LaTeX ソース上 $|\vec r_l - \vec r|$ と $(\vec r_l - \vec r)_X$ の差で、誤植としてはあり得るパターンです。
