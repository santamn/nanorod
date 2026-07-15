//! TOML 設定ファイルの読み込みと、シミュレーションケースへの展開。
//!
//! シミュレーションの設定値はすべて TOML ファイルで指定する。掃引するパラメータ
//! （γ, δ, f, m）はリストで与え、その直積を「ケース」として展開する。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer};

use crate::model::diffusion_for_length;

// ---- モデルの形そのものから決まる、設定ファイルでは変更しない定数 ----

/// 流路の周期長。全ての長さはこの周期で無次元化されているため常に 1。
pub const L_PERIOD: f64 = 1.0;
/// 1つの粒子代表点が1つの壁点から受ける反発力の大きさの上限（仕様で固定）。
pub const MAX_WALL_REPULSION_FORCE: f64 = 2.5e4;
/// 境界外に出た状態を鏡像反射で流路内へ戻す最大反復回数。
pub const BOUNDARY_REFLECTION_LIMIT: usize = 32;
/// 流路のくびれ（omega(x) が最小になる点）の位相。omega の形だけで決まる。
pub const CHANNEL_NECK_PHASE: f64 = 0.809_640_837_312_333_2;
/// 壁近傍探索で左右に見る壁点数 K = ceil(r_c / Δx_wall) = ceil(4·2^(1/6)) = 5。
/// r_c と Δx_wall はどちらも σ に比例するため、σ の値に依らない定数になる。
pub const WALL_K: i32 = 5;

// 角度・y分布ヒストグラムの解像度と範囲。GPU kernel と CSV 出力で共有する。
// x は1周期 [0,1) に畳み、φ は [0,2π) に巻き戻し、y は [-Y_MAX, Y_MAX] に収める。
pub const HIST_X_BINS: usize = 100;
pub const HIST_PHI_BINS: usize = 36;
pub const HIST_Y_BINS: usize = 64;
/// y ヒストグラムの片側範囲。流路半幅 omega(x) の最大値（≈2.18）を覆う値にしておく。
pub const HIST_Y_MAX: f64 = 2.3;

/// `hist_stride` を省略したときの既定値（100 ステップごとに1票）。
fn default_hist_stride() -> u32 {
    100
}

/// 計測モード。物理モデルは両モードで共通で、試行の打ち切り方と
/// 移動度・有効拡散係数の算出方法だけが変わる。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// 各試行を1周期の初通過で打ち切り、初通過時間の統計から μ と D_eff を求める。
    FirstPassage,
    /// 全試行を時間 T まで走らせ、変位 Δx の統計から μ と D_eff を直接算出する（既定）。
    #[default]
    FixedTime,
}

/// TOML 形式の設定ファイルに対応する構造体。
///
/// `gamma`, `delta`, `f`, `m` はリストで指定し、その全組み合わせ（直積）が
/// ケースとしてシミュレーションされる。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// 時間刻み幅 Δt。
    pub delta_t: f64,
    /// 1試行あたりのシミュレーション時間 T。first_passage モードでは未通過試行の
    /// 打ち切り時刻、fixed_time モードでは全試行に共通の計測時間になる。
    pub time: f64,
    /// 計測モード（省略時は fixed_time）。
    #[serde(default)]
    pub mode: Mode,
    /// 1ケースあたりの試行数（アンサンブルサイズ）。
    pub ensemble_size: usize,
    /// 全ケースの結果をまとめて出力するフォルダ。
    pub output_dir: PathBuf,
    /// WCA ポテンシャルの σ。壁点間隔 0.25σ・代表点間隔 0.8σ の基準にもなる。
    pub sigma: f64,
    /// WCA ポテンシャルの ε。
    pub epsilon: f64,
    /// 電場トルクの永久双極子成分の大きさ γ = βpE のリスト（"1/3" のような分数表記も可）。
    #[serde(deserialize_with = "deserialize_frac_vec")]
    pub gamma: Vec<f64>,
    /// 電場トルクの異方性成分と永久双極子成分の比 δ = |Δα|E/p のリスト（分数表記も可）。
    #[serde(deserialize_with = "deserialize_frac_vec")]
    pub delta: Vec<f64>,
    /// x 方向の一定外力 f のリスト（分数表記も可）。
    #[serde(deserialize_with = "deserialize_frac_vec")]
    pub f: Vec<f64>,
    /// 棒の片側代表点数 m のリスト。棒長は l = 2m × 0.8σ。
    pub m: Vec<i32>,
    /// 角度・y分布ヒストグラムへ加算するステップ間隔。0 で記録を無効化（省略時 100）。
    #[serde(default = "default_hist_stride")]
    pub hist_stride: u32,
    /// GPU 実行に関する設定（省略可）。
    #[serde(default)]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // CPU ビルドでは参照されない
    pub gpu: GpuConfig,
}

/// GPU 実行に関する任意設定。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // CPU ビルドでは参照されない
pub struct GpuConfig {
    /// 使用する GPU の ID（省略時は搭載されている全 GPU）。
    ///
    /// GPU あたりの同時実行ケース数は ensemble_size とケース数から自動調整される
    /// ため、設定項目はない（runner::backend::auto_tasks_per_gpu を参照）。
    pub ids: Option<Vec<usize>>,
}

impl Config {
    /// TOML ファイルを読み込み、値の妥当性を検証する。
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("設定ファイル {} を読み込めません", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("設定ファイル {} の形式が不正です", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// 各設定値の範囲と、σ・m から導かれる幾何・拡散係数の健全性を検証する。
    fn validate(&self) -> Result<()> {
        ensure!(
            self.delta_t.is_finite() && self.delta_t > 0.0,
            "delta_t は正の値にしてください"
        );
        ensure!(
            self.time.is_finite() && self.time > 0.0,
            "time は正の値にしてください"
        );
        ensure!(
            self.time / self.delta_t >= 1.0,
            "time / delta_t が1ステップに満たない設定です"
        );
        ensure!(
            self.ensemble_size >= 1 && self.ensemble_size <= i32::MAX as usize,
            "ensemble_size は 1 以上 {} 以下にしてください",
            i32::MAX
        );
        ensure!(
            self.sigma.is_finite() && self.sigma > 0.0,
            "sigma は正の値にしてください"
        );
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "epsilon は正の値にしてください"
        );

        for (name, list) in [
            ("gamma", &self.gamma),
            ("delta", &self.delta),
            ("f", &self.f),
        ] {
            ensure!(!list.is_empty(), "{name} には1つ以上の値を指定してください");
            ensure!(
                list.iter().all(|v| v.is_finite()),
                "{name} に有限でない値が含まれています"
            );
        }
        ensure!(!self.m.is_empty(), "m には1つ以上の値を指定してください");
        for &m in &self.m {
            ensure!(m >= 1, "m は1以上にしてください（{m} が指定されました）");
        }

        // σ から壁点数が整数として定まり、近傍探索窓が1周期に収まることを確認する。
        let physics = self.physics()?;

        // Tirado の式は棒のアスペクト比が小さすぎると負の拡散係数を返すため、
        // 全ての m で拡散係数が正になることを事前に確認する。
        for &m in &self.m {
            let l = physics.particle_length(m);
            let diffusion = diffusion_for_length(l, physics.diffusion_reference_length);
            ensure!(
                diffusion.d_parallel > 0.0 && diffusion.d_perp > 0.0 && diffusion.d_r > 0.0,
                "m = {m}（棒長 l = {l:.6}）は拡散係数の式（Tirado）の適用範囲外です。\
                 sigma または m を見直してください"
            );
        }
        Ok(())
    }

    /// 設定から全ケース共通の物理定数一式を導出する。
    pub fn physics(&self) -> Result<Physics> {
        let wall_dx = 0.25 * self.sigma;
        let n_wall_exact = L_PERIOD / wall_dx;
        let n_wall = n_wall_exact.round();
        // 壁点は k mod N_wall で周期参照するため、1周期がちょうど N_wall 分割でなければならない。
        ensure!(
            (n_wall_exact - n_wall).abs() <= 1.0e-6 * n_wall,
            "1周期が壁点間隔 0.25σ で割り切れません（L / 0.25σ = {n_wall_exact}）。\
             sigma には 1 / (0.25σ) が整数になる値を指定してください"
        );
        let min_wall = f64::from(2 * WALL_K + 1);
        ensure!(
            n_wall >= min_wall && n_wall <= f64::from(i32::MAX),
            "壁点数 {n_wall} が近傍探索窓（{min_wall} 点）に対して少なすぎるか多すぎます"
        );

        let particle_dx = 0.8 * self.sigma;
        Ok(Physics {
            delta_t: self.delta_t,
            mode: self.mode,
            max_steps: (self.time / self.delta_t).round() as u64,
            sigma: self.sigma,
            epsilon: self.epsilon,
            wall_dx,
            particle_dx,
            n_wall: n_wall as usize,
            diffusion_reference_length: 6.0 * particle_dx,
        })
    }

    /// パラメータリストの全組み合わせ（直積）をケースとして展開する。
    ///
    /// 列挙順は m → γ → δ → f の入れ子で、`case_id` を 0 から振る。
    /// 同じ値が重複して与えられても、ケース（と出力フォルダ）が二重にならないよう
    /// 各リストを出現順を保ったまま重複排除する。
    pub fn cases(&self) -> Result<Vec<Case>> {
        let physics = self.physics()?;
        let m_values = dedup_in_order_i32(&self.m);
        let gamma_values = dedup_in_order_f64(&self.gamma);
        let delta_values = dedup_in_order_f64(&self.delta);
        let f_values = dedup_in_order_f64(&self.f);

        let mut cases = Vec::with_capacity(
            m_values.len() * gamma_values.len() * delta_values.len() * f_values.len(),
        );
        for &m in &m_values {
            for &gamma in &gamma_values {
                for &delta in &delta_values {
                    for &f in &f_values {
                        cases.push(Case {
                            case_id: cases.len() as u32,
                            m,
                            l: physics.particle_length(m),
                            gamma,
                            delta,
                            f,
                        });
                    }
                }
            }
        }

        // フォルダ名は小数点以下6桁へ丸めるため、それより近い値同士は出力先が衝突する。
        // 実行後に気づくと結果が失われかねないので、展開時点で検出して止める。
        let mut names = std::collections::HashSet::with_capacity(cases.len());
        for case in &cases {
            ensure!(
                names.insert(case.dir_name()),
                "ケースの出力フォルダ名 {} が重複します。パラメータ値の差が小さすぎて \
                 小数点以下6桁の表記で区別できません",
                case.dir_name()
            );
        }
        Ok(cases)
    }
}

/// 設定から導出した、全ケース共通の物理定数と数値パラメータ。
#[derive(Clone, Copy, Debug)]
pub struct Physics {
    /// 時間刻み幅 Δt。
    pub delta_t: f64,
    /// 計測モード。first_passage は初通過で打ち切り、fixed_time は max_steps まで走り切る。
    pub mode: Mode,
    /// 1試行のステップ数上限 round(time / Δt)。first_passage モードでは未通過試行の
    /// 打ち切り、fixed_time モードでは全試行に共通の計測時間 T になる。
    pub max_steps: u64,
    /// WCA ポテンシャルの σ。
    pub sigma: f64,
    /// WCA ポテンシャルの ε。
    pub epsilon: f64,
    /// 壁のサンプリング点間隔 0.25σ。
    pub wall_dx: f64,
    /// 棒の代表点間隔 0.8σ。
    pub particle_dx: f64,
    /// 1周期あたりの壁サンプリング点数 L / (0.25σ)。
    pub n_wall: usize,
    /// D_0 の計算にだけ使う基準棒長 6 × 0.8σ。全棒長で同じ時間スケールを使うための固定値。
    pub diffusion_reference_length: f64,
}

impl Physics {
    /// WCA カットオフ半径の2乗 r_c² = (2^(1/6)σ)² = 2^(1/3)σ²。
    pub fn rc2(&self) -> f64 {
        2.0_f64.cbrt() * self.sigma * self.sigma
    }

    /// 代表点間隔と m から棒長 l = 2m × 0.8σ を求める。
    pub fn particle_length(&self, m: i32) -> f64 {
        2.0 * f64::from(m) * self.particle_dx
    }
}

/// 1回のシミュレーションに対応するパラメータの組。
#[derive(Clone, Copy, Debug)]
pub struct Case {
    /// 展開順に 0 から振られる通し番号。出力の対応付けに使う。
    pub case_id: u32,
    /// 棒の片側代表点数。代表点は全体で 2m+1 個。
    pub m: i32,
    /// 棒長 l = 2m × 0.8σ。
    pub l: f64,
    /// 電場トルクの永久双極子成分の大きさ γ = βpE（p = ql を含む結合の大きさそのもの）。
    pub gamma: f64,
    /// 電場トルクの異方性成分と永久双極子成分の比 δ = |Δα|E/p。正で棒を横へ倒す効果を生む。
    pub delta: f64,
    /// x 方向の一定外力 f。
    pub f: f64,
}

impl Case {
    /// 結果を保存するサブフォルダ名（例: `m3_f10_gamma1_delta0.333333`）。
    pub fn dir_name(&self) -> String {
        format!(
            "m{}_f{}_gamma{}_delta{}",
            self.m,
            format_param(self.f),
            format_param(self.gamma),
            format_param(self.delta),
        )
    }

    /// パラメータの組から再現性のある乱数シードを導出する（FNV-1a ハッシュ）。
    ///
    /// 実行順序や GPU への割り当てに依存せず、同じパラメータには常に同じシードが
    /// 与えられるため、同じ設定ファイルからは常に同じ結果が得られる。
    pub fn seed(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for bits in [
            self.m as u64,
            self.gamma.to_bits(),
            self.delta.to_bits(),
            self.f.to_bits(),
        ] {
            hash ^= bits;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
}

/// 数値または "1/3" のような分数文字列のリストを f64 のリストにデシリアライズする。
fn deserialize_frac_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FracValue {
        Number(f64),
        Text(String),
    }

    Vec::<FracValue>::deserialize(deserializer)?
        .into_iter()
        .map(|value| match value {
            FracValue::Number(number) => Ok(number),
            FracValue::Text(text) => parse_fraction(&text).map_err(serde::de::Error::custom),
        })
        .collect()
}

/// "分子/分母" 形式、または通常の数値文字列を f64 に変換する。
fn parse_fraction(text: &str) -> Result<f64, String> {
    let value = match text.split_once('/') {
        Some((numerator, denominator)) => {
            let numerator: f64 = numerator
                .trim()
                .parse()
                .map_err(|_| format!("不正な分数表記です: \"{text}\""))?;
            let denominator: f64 = denominator
                .trim()
                .parse()
                .map_err(|_| format!("不正な分数表記です: \"{text}\""))?;
            if denominator == 0.0 {
                return Err(format!("分母が0の分数です: \"{text}\""));
            }
            numerator / denominator
        }
        None => text
            .trim()
            .parse()
            .map_err(|_| format!("不正な数値です: \"{text}\""))?,
    };

    if !f64::is_finite(value) {
        return Err(format!("有限の数値ではありません: \"{text}\""));
    }
    Ok(value)
}

/// フォルダ名用に f64 を、小数点以下6桁へ丸めて末尾の余分な0を落とした短い文字列へ整形する。
fn format_param(value: f64) -> String {
    let text = format!("{value:.6}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// i32 のリストを、出現順を保ったまま重複排除する。
fn dedup_in_order_i32(values: &[i32]) -> Vec<i32> {
    let mut unique = Vec::with_capacity(values.len());
    for &value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

/// f64 のリストを、ビット表現で同一視して出現順を保ったまま重複排除する。
fn dedup_in_order_f64(values: &[f64]) -> Vec<f64> {
    let mut unique: Vec<f64> = Vec::with_capacity(values.len());
    for &value in values {
        if !unique.iter().any(|&kept| kept.to_bits() == value.to_bits()) {
            unique.push(value);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の最小設定を返す。
    fn minimal_config_text() -> &'static str {
        r#"
delta_t = 4e-7
time = 100.0
ensemble_size = 1000
output_dir = "example/"
sigma = 8e-3
epsilon = 2.0
gamma = [0.25, "1/2"]
delta = ["1/3"]
f = [1.0, 2.0, 3.0]
m = [1, 4]
"#
    }

    #[test]
    fn config_parses_fractions_and_defaults() {
        let config: Config = toml::from_str(minimal_config_text()).unwrap();
        config.validate().unwrap();

        assert_eq!(config.gamma, vec![0.25, 0.5]);
        assert!((config.delta[0] - 1.0 / 3.0).abs() < 1.0e-15);
        assert_eq!(config.hist_stride, 100);
        assert_eq!(config.mode, Mode::FixedTime);
        assert!(config.gpu.ids.is_none());
    }

    #[test]
    fn config_parses_first_passage_mode() {
        let text = format!("{}\nmode = \"first_passage\"\n", minimal_config_text());
        let config: Config = toml::from_str(&text).unwrap();
        assert_eq!(config.mode, Mode::FirstPassage);
    }

    #[test]
    fn config_rejects_unknown_mode() {
        let text = format!("{}\nmode = \"both\"\n", minimal_config_text());
        assert!(toml::from_str::<Config>(&text).is_err());
    }

    #[test]
    fn config_rejects_unknown_keys() {
        let text = format!("{}\nunknown_key = 1.0\n", minimal_config_text());
        assert!(toml::from_str::<Config>(&text).is_err());
    }

    /// tasks_per_gpu は自動調整に置き換えたため、指定されたらエラーで知らせる。
    #[test]
    fn config_rejects_removed_tasks_per_gpu() {
        let text = format!("{}\n[gpu]\ntasks_per_gpu = 4\n", minimal_config_text());
        assert!(toml::from_str::<Config>(&text).is_err());
    }

    #[test]
    fn physics_derives_wall_geometry_from_sigma() {
        let config: Config = toml::from_str(minimal_config_text()).unwrap();
        let physics = config.physics().unwrap();

        assert_eq!(physics.n_wall, 500);
        assert!((physics.wall_dx - 0.002).abs() < 1.0e-15);
        assert!((physics.particle_dx - 0.0064).abs() < 1.0e-15);
        assert!((physics.diffusion_reference_length - 0.0384).abs() < 1.0e-15);
        assert_eq!(physics.max_steps, 250_000_000);
        assert!((physics.rc2() - 2.0_f64.cbrt() * 8.0e-3 * 8.0e-3).abs() < 1.0e-18);
    }

    #[test]
    fn physics_rejects_sigma_with_non_integer_wall_count() {
        let text = minimal_config_text().replace("sigma = 8e-3", "sigma = 7.7e-3");
        let config: Config = toml::from_str(&text).unwrap();
        assert!(config.physics().is_err());
    }

    #[test]
    fn cases_form_cartesian_product_in_declared_order() {
        let config: Config = toml::from_str(minimal_config_text()).unwrap();
        let cases = config.cases().unwrap();

        // m(2) × γ(2) × δ(1) × f(3) の直積。
        assert_eq!(cases.len(), 2 * 2 * 3);
        assert_eq!(cases.first().unwrap().case_id, 0);
        assert_eq!(cases.last().unwrap().case_id, 11);
        // 入れ子の最内は f なので、先頭2要素は f だけが変わる。
        assert_eq!(cases[0].m, 1);
        assert_eq!(cases[0].f, 1.0);
        assert_eq!(cases[1].f, 2.0);
        // l は m と σ から導出される。
        assert!((cases[0].l - 2.0 * 0.8 * 8.0e-3).abs() < 1.0e-15);
    }

    #[test]
    fn cases_dedup_repeated_values() {
        let text = minimal_config_text().replace("m = [1, 4]", "m = [1, 1, 4]");
        let config: Config = toml::from_str(&text).unwrap();
        assert_eq!(config.cases().unwrap().len(), 2 * 2 * 3);
    }

    #[test]
    fn cases_reject_colliding_dir_names() {
        // 6桁丸めでは区別できない 2 値は、出力フォルダが衝突するため展開時に弾く。
        let text = minimal_config_text().replace("f = [1.0, 2.0, 3.0]", "f = [1e-9, 2e-9]");
        let config: Config = toml::from_str(&text).unwrap();
        let error = config.cases().unwrap_err();
        assert!(error.to_string().contains("重複"));
    }

    #[test]
    fn validate_rejects_non_positive_m() {
        let text = minimal_config_text().replace("m = [1, 4]", "m = [0]");
        let config: Config = toml::from_str(&text).unwrap();
        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("m は1以上"));
    }

    #[test]
    fn parse_fraction_accepts_fractions_and_decimals() {
        assert!((parse_fraction("1/3").unwrap() - 1.0 / 3.0).abs() < 1.0e-15);
        assert_eq!(parse_fraction("0.5").unwrap(), 0.5);
        assert_eq!(parse_fraction(" 3 / 4 ").unwrap(), 0.75);
        assert!(parse_fraction("1/0").is_err());
        assert!(parse_fraction("abc").is_err());
    }

    #[test]
    fn case_dir_name_trims_trailing_zeros() {
        let case = Case {
            case_id: 0,
            m: 3,
            l: 0.0,
            gamma: 1.0,
            delta: 1.0 / 3.0,
            f: 0.0,
        };
        assert_eq!(case.dir_name(), "m3_f0_gamma1_delta0.333333");
    }

    #[test]
    fn case_seed_is_deterministic_and_distinct() {
        let base = Case {
            case_id: 0,
            m: 3,
            l: 0.0384,
            gamma: 1.0,
            delta: 0.5,
            f: 2.0,
        };
        let same = Case { case_id: 9, ..base };
        let different = Case { f: 3.0, ..base };

        // case_id には依存せず、物理パラメータだけで決まる。
        assert_eq!(base.seed(), same.seed());
        assert_ne!(base.seed(), different.seed());
    }
}
