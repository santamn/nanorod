//! 1粒子アニメーションの GUI 表示（egui/eframe）。
//!
//! シミュレーション本体と同じ物理モデル（simulation.rs / model.rs）を使い、
//! 初期パラメータには設定ファイルの各リストの先頭値を使う。

use std::{path::Path, sync::Arc};

use anyhow::{Result, anyhow};
use eframe::egui;
use egui::{
    Color32, FontData, FontDefinitions, FontFamily, Pos2, Rect, Sense, Shape, Stroke, pos2, vec2,
};

use crate::config::{Config, Physics};
use crate::model::{self, PASS_DIRECTION_LEFT, PASS_DIRECTION_RIGHT};
use crate::simulation::{VisualParams, VisualSimulation};

const TRAIL_LIMIT: usize = 900; // 重心の軌跡を保持する最大点数
const CHANNEL_Y_SPAN: f64 = 5.25;
const STEPS_PER_FRAME: usize = 1_000; // 滑らかな描画を保つために毎フレーム進める固定ステップ数
const CAMERA_FOLLOW_RATE: f64 = 8.0; // 粒子の横揺れを画面へ直接伝えないための追従速度
const CAMERA_MAX_LAG: f64 = 0.8; // 強い外力でも粒子を画面中央付近に保つ最大遅れ
const CAMERA_MAX_DT: f64 = 0.1; // フレーム停止後にカメラが急に飛ばないようにする時間刻み上限

const JAPANESE_FONT_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "C:\\Windows\\Fonts\\meiryo.ttc",
    "C:\\Windows\\Fonts\\YuGothM.ttc",
];

/// native window と egui アプリケーションを起動する。
///
/// 初期パラメータは設定ファイルの各リスト（m, f, gamma, delta）の先頭値。
pub fn run_animation(config: &Config) -> Result<()> {
    let physics = config.physics()?;
    let controls = VisualParams {
        seed: 1,
        m: config.m[0],
        f: config.f[0],
        gamma: config.gamma[0],
        delta: config.delta[0],
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Nanorod Animation")
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([900.0, 560.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Nanorod Animation",
        options,
        Box::new(move |cc| Ok(Box::new(AnimationApp::new(cc, controls, physics)))),
    )
    .map_err(|error| anyhow!("アニメーションを起動できません: {error}"))
}

/// アニメーション全体の状態と UI 操作をまとめる。
struct AnimationApp {
    controls: VisualParams,
    simulation: VisualSimulation,
    camera: CameraFollow,
    running: bool,
    trail: Vec<Pos2>,
}

impl AnimationApp {
    /// 日本語フォントと描画スタイルを整えて初期状態を作る。
    fn new(cc: &eframe::CreationContext<'_>, controls: VisualParams, physics: Physics) -> Self {
        install_system_japanese_font(&cc.egui_ctx);
        configure_style(&cc.egui_ctx);

        let simulation = VisualSimulation::new(controls, physics);
        let camera = CameraFollow::new(simulation.x);
        let trail = vec![pos2(simulation.x as f32, simulation.y as f32)];

        Self {
            controls,
            simulation,
            camera,
            running: false,
            trail,
        }
    }

    /// 現在の操作パラメータ（seed を含む）で粒子と軌跡を初期状態へ戻す。
    fn reset(&mut self) {
        self.simulation.reset(self.controls);
        self.camera.reset(self.simulation.x);
        self.trail.clear();
        self.trail
            .push(pos2(self.simulation.x as f32, self.simulation.y as f32));
    }

    /// 描画フレームごとに固定ステップだけ進めて、表示の滑らかさを安定させる。
    fn advance(&mut self, ctx: &egui::Context) {
        if !self.running {
            return;
        }

        if self.simulation.completed {
            self.running = false;
            return;
        }

        self.simulation.step_many(STEPS_PER_FRAME);
        self.push_trail_point();
        self.camera
            .advance_toward(self.simulation.x, animation_dt(ctx));

        if self.simulation.completed {
            self.running = false;
        } else {
            ctx.request_repaint();
        }
    }

    /// 重心の軌跡を一定数だけ保持して、長時間実行時の描画負荷を抑える。
    fn push_trail_point(&mut self) {
        self.trail
            .push(pos2(self.simulation.x as f32, self.simulation.y as f32));
        if self.trail.len() > TRAIL_LIMIT {
            let overflow = self.trail.len() - TRAIL_LIMIT;
            self.trail.drain(0..overflow);
        }
    }
}

impl eframe::App for AnimationApp {
    /// 毎フレーム、描画前にシミュレーション状態を進める。
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.advance(ctx);
    }

    /// 操作パネルとアニメーションキャンバスを描画する。
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(300.0)
            .show(ui, |ui| {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let label = if self.running { "停止" } else { "開始" };
                    if ui
                        .add_sized(vec2(64.0, 30.0), egui::Button::new(label))
                        .clicked()
                    {
                        if self.simulation.completed {
                            self.reset();
                        }
                        self.running = !self.running;
                        if self.running {
                            ui.ctx().request_repaint();
                        }
                    }

                    if ui
                        .add_sized(vec2(82.0, 30.0), egui::Button::new("リセット"))
                        .clicked()
                    {
                        self.running = false;
                        self.reset();
                    }
                });

                ui.separator();

                // seed は Reset を押したとき、m は変更した瞬間に初期状態から反映される。
                // f, γ, δ は力の項にしか現れないため、実行中の粒子へ即座に反映する。
                let mut reset_needed = false;
                let mut force_changed = false;
                let point_count_label =
                    point_count_label(&self.simulation.physics, self.controls.m);
                ui.add(
                    egui::DragValue::new(&mut self.controls.seed)
                        .speed(1.0)
                        .prefix("seed : "),
                );
                reset_needed |= ui
                    .add(egui::Slider::new(&mut self.controls.m, 1..=30).text(point_count_label))
                    .changed();
                force_changed |= ui
                    .add(egui::Slider::new(&mut self.controls.f, 0.0..=100.0).text("f"))
                    .changed();
                force_changed |= ui
                    .add(egui::Slider::new(&mut self.controls.gamma, -5.0..=5.0).text("γ = βpE"))
                    .changed();
                force_changed |= ui
                    .add(egui::Slider::new(&mut self.controls.delta, 0.0..=5.0).text("δ = |Δα|E/p"))
                    .changed();

                if reset_needed {
                    self.running = false;
                    self.reset();
                } else if force_changed {
                    self.simulation.apply_force_params(
                        self.controls.f,
                        self.controls.gamma,
                        self.controls.delta,
                    );
                }

                ui.separator();
                egui::Grid::new("stats").spacing([8.0, 4.0]).show(ui, |ui| {
                    ui.label("シミュレーション時刻");
                    ui.label(format!("{:.6}", self.simulation.t));
                    ui.end_row();

                    ui.label("ステップ数");
                    ui.label(format!(
                        "{} / {}",
                        self.simulation.steps, self.simulation.physics.max_steps
                    ));
                    ui.end_row();

                    ui.label("x_0 ± Lを通過");
                    ui.label(pass_direction_display(self.simulation.pass_direction));
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.heading("初期状態");
                        ui.label(format!("x_0 : {:.4}", self.simulation.x0));
                        ui.label(format!("y_0 : {:.4}", self.simulation.y0));
                        ui.label(format!("φ_0 : {:.3}", self.simulation.phi0));
                    });

                    ui.add_space(12.0);

                    ui.vertical(|ui| {
                        ui.heading("現在状態");
                        ui.label(format!("x : {:.4}", self.simulation.x));
                        ui.label(format!("y : {:.4}", self.simulation.y));
                        ui.label(format!("φ : {:.3}", display_angle(self.simulation.phi)));
                    });
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            let available = ui.available_size();
            let (response, painter) = ui.allocate_painter(available, Sense::hover());
            let painter = painter.with_clip_rect(response.rect);
            draw_simulation(
                &painter,
                response.rect,
                self.camera.x(),
                &self.simulation,
                &self.trail,
            );
        });
    }
}

/// 横方向だけを粒子へゆっくり追従させるカメラ状態。
#[derive(Clone, Copy, Debug)]
struct CameraFollow {
    x: f64,
}

impl CameraFollow {
    /// 初期粒子位置を画面中心としてカメラを作る。
    fn new(x: f64) -> Self {
        Self { x }
    }

    /// パラメータ変更やリセット時に、残った遅れを消してカメラを初期位置へ戻す。
    fn reset(&mut self, x: f64) {
        self.x = x;
    }

    /// 指数平滑で目標位置へ近づき、急な横揺れを画面全体の揺れにしない。
    fn advance_toward(&mut self, target_x: f64, dt: f64) {
        let dt = dt.clamp(0.0, CAMERA_MAX_DT);
        let lag = target_x - self.x;
        if lag.abs() > CAMERA_MAX_LAG {
            self.x = target_x - lag.signum() * CAMERA_MAX_LAG;
        }

        let alpha = 1.0 - (-CAMERA_FOLLOW_RATE * dt).exp();
        self.x += (target_x - self.x) * alpha;
    }

    /// 描画に使う現在のカメラ中心 x 座標を返す。
    fn x(&self) -> f64 {
        self.x
    }
}

/// OS にある日本語フォントを egui に登録する。
fn install_system_japanese_font(ctx: &egui::Context) {
    let Some((name, data)) = load_first_existing_font(JAPANESE_FONT_CANDIDATES) else {
        return;
    };

    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert(name.clone(), Arc::new(FontData::from_owned(data)));

    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push(name.clone());
    }

    ctx.set_fonts(fonts);
}

/// 候補パスから最初に読めるフォントを探す。
fn load_first_existing_font(paths: &[&str]) -> Option<(String, Vec<u8>)> {
    paths.iter().find_map(|path| {
        let bytes = std::fs::read(path).ok()?;
        let name = Path::new(path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("system_japanese")
            .to_owned();
        Some((name, bytes))
    })
}

/// 文字・ボタン・背景のコントラストを上げて読み取りやすい見た目にする。
fn configure_style(ctx: &egui::Context) {
    ctx.global_style_mut(|style| {
        let text = Color32::from_rgb(24, 33, 37);
        style.visuals.window_fill = Color32::from_rgb(252, 253, 250);
        style.visuals.panel_fill = Color32::from_rgb(252, 253, 250);
        style.visuals.extreme_bg_color = Color32::from_rgb(232, 237, 236);
        style.visuals.override_text_color = Some(text);
        style.visuals.widgets.noninteractive.fg_stroke.color = text;
        style.visuals.widgets.inactive.fg_stroke.color = text;
        style.visuals.widgets.hovered.fg_stroke.color = text;
        style.visuals.widgets.active.fg_stroke.color = text;
        style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(225, 232, 232);
        style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(211, 222, 222);
        style.visuals.widgets.active.weak_bg_fill = Color32::from_rgb(196, 211, 212);
        style.spacing.item_spacing = vec2(8.0, 10.0);
        style.spacing.button_padding = vec2(10.0, 6.0);
    });
}

/// egui の安定化されたフレーム間隔を、カメラ平滑化で扱いやすい秒単位へ変換する。
fn animation_dt(ctx: &egui::Context) -> f64 {
    f64::from(ctx.input(|input| input.stable_dt)).min(CAMERA_MAX_DT)
}

/// `m` の隣に棒長 `l` を表示する短いラベルを作る。
fn point_count_label(physics: &Physics, m: i32) -> String {
    let length = physics.particle_length(m);
    format!("m ( l = {length:.4} )")
}

/// 角度を画面表示用に 0 から 2π の範囲へ丸める。
fn display_angle(phi: f64) -> f64 {
    phi.rem_euclid(std::f64::consts::TAU)
}

/// 初通過方向を操作パネルへ表示する短い日本語ラベルへ変換する。
fn pass_direction_display(direction: i32) -> &'static str {
    match direction {
        PASS_DIRECTION_RIGHT => "右",
        PASS_DIRECTION_LEFT => "左",
        _ => "未",
    }
}

/// 現在の粒子状態と流路をキャンバスへ描画する。
fn draw_simulation(
    painter: &egui::Painter,
    rect: Rect,
    camera_x: f64,
    simulation: &VisualSimulation,
    trail: &[Pos2],
) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(246, 250, 248));

    let world = WorldView::new(rect, camera_x, 0.0, CHANNEL_Y_SPAN);

    draw_channel(painter, &world);
    draw_markers(painter, &world, simulation);
    draw_trail(painter, &world, trail);
    draw_rod(painter, &world, simulation);
}

/// 周期流路の上端・下端・中心線・周期グリッドを描く。
fn draw_channel(painter: &egui::Painter, world: &WorldView) {
    let mut top = Vec::with_capacity(260);
    let mut bottom = Vec::with_capacity(260);
    let samples = ((world.x_max - world.x_min) * 160.0).ceil() as usize;

    for idx in 0..=samples {
        let x = world.x_min + (world.x_max - world.x_min) * idx as f64 / samples as f64;
        let y = model::omega(x);
        top.push(world.to_screen(x, y));
        bottom.push(world.to_screen(x, -y));
    }

    painter.add(Shape::line(
        top.clone(),
        Stroke::new(2.6, Color32::from_rgb(0, 93, 104)),
    ));
    painter.add(Shape::line(
        bottom.clone(),
        Stroke::new(2.6, Color32::from_rgb(0, 93, 104)),
    ));

    let mut centerline = Vec::with_capacity(80);
    let centerline_samples = ((world.x_max - world.x_min) * 40.0).ceil() as usize;
    for idx in 0..=centerline_samples {
        let x = world.x_min + (world.x_max - world.x_min) * idx as f64 / centerline_samples as f64;
        centerline.push(world.to_screen(x, 0.0));
    }
    painter.add(Shape::line(
        centerline,
        Stroke::new(1.2, Color32::from_rgba_premultiplied(49, 68, 72, 120)),
    ));

    for period in (world.x_min.floor() as i32 - 1)..=(world.x_max.ceil() as i32 + 1) {
        let x = f64::from(period);
        if x >= world.x_min && x <= world.x_max {
            let a = world.to_screen(x, world.y_min);
            let b = world.to_screen(x, world.y_max);
            painter.line_segment(
                [a, b],
                Stroke::new(1.0, Color32::from_rgba_premultiplied(49, 68, 72, 80)),
            );
        }
    }
}

/// 初期位置と x_0 ± L 通過判定位置を縦線で示す。
fn draw_markers(painter: &egui::Painter, world: &WorldView, simulation: &VisualSimulation) {
    let start_a = world.to_screen(simulation.x0, world.y_min);
    let start_b = world.to_screen(simulation.x0, world.y_max);
    painter.line_segment(
        [start_a, start_b],
        Stroke::new(1.8, Color32::from_rgba_premultiplied(44, 105, 185, 190)),
    );

    let left_target_a = world.to_screen(simulation.target_left_x, world.y_min);
    let left_target_b = world.to_screen(simulation.target_left_x, world.y_max);
    painter.line_segment(
        [left_target_a, left_target_b],
        Stroke::new(2.2, Color32::from_rgba_premultiplied(58, 130, 83, 220)),
    );

    let target_a = world.to_screen(simulation.target_right_x, world.y_min);
    let target_b = world.to_screen(simulation.target_right_x, world.y_max);
    painter.line_segment(
        [target_a, target_b],
        Stroke::new(2.2, Color32::from_rgba_premultiplied(215, 72, 43, 220)),
    );
}

/// 重心の直近の移動履歴を表示する。
fn draw_trail(painter: &egui::Painter, world: &WorldView, trail: &[Pos2]) {
    if trail.len() < 2 {
        return;
    }

    let points = trail
        .iter()
        .map(|point| world.to_screen(point.x as f64, point.y as f64))
        .collect::<Vec<_>>();
    painter.add(Shape::line(
        points,
        Stroke::new(2.0, Color32::from_rgba_premultiplied(82, 100, 136, 170)),
    ));
}

/// 棒状粒子は代表点を省き、棒本体と重心だけを描く。
fn draw_rod(painter: &egui::Painter, world: &WorldView, simulation: &VisualSimulation) {
    let (first, last) = simulation.rod_endpoints();

    let start = world.to_screen(first.x, first.y);
    let end = world.to_screen(last.x, last.y);
    painter.line_segment(
        [start, end],
        Stroke::new(6.0, Color32::from_rgb(20, 36, 49)),
    );

    let center = world.to_screen(simulation.x, simulation.y);
    painter.circle_filled(center, 2.6, Color32::from_rgb(246, 184, 51));
    painter.circle_stroke(center, 2.6, Stroke::new(1.1, Color32::from_rgb(20, 36, 49)));
}

/// 世界座標と画面座標の対応を保持する。
struct WorldView {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    scale: f64,
    origin: Pos2,
}

impl WorldView {
    /// y 範囲は固定し、x 方向だけ粒子に追従するビューを作る。
    fn new(rect: Rect, center_x: f64, center_y: f64, y_span: f64) -> Self {
        let pad = 14.0;
        let drawable = rect.shrink(pad);
        let aspect = (drawable.width() / drawable.height().max(1.0)).max(1.0) as f64;
        let x_span = y_span * aspect;
        let x_min = center_x - 0.5 * x_span;
        let x_max = center_x + 0.5 * x_span;
        let y_min = center_y - 0.5 * y_span;
        let y_max = center_y + 0.5 * y_span;
        let scale_x = drawable.width() as f64 / x_span;
        let scale_y = drawable.height() as f64 / y_span;
        let scale = scale_x.min(scale_y);
        let world_width_px = x_span * scale;
        let world_height_px = y_span * scale;
        let origin = pos2(
            drawable.center().x - 0.5 * world_width_px as f32,
            drawable.center().y + 0.5 * world_height_px as f32,
        );

        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            scale,
            origin,
        }
    }

    /// 物理空間の点を egui の画面座標へ変換する。
    fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        let sx = self.origin.x + ((x - self.x_min) * self.scale) as f32;
        let sy = self.origin.y - ((y - self.y_min) * self.scale) as f32;
        pos2(sx, sy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 追従カメラが粒子位置へ一気に飛ばず、途中までだけ近づくことを確認する。
    #[test]
    fn camera_follow_smooths_small_horizontal_motion() {
        let mut camera = CameraFollow::new(0.0);

        camera.advance_toward(1.0, 1.0 / 60.0);

        assert!(camera.x() > 0.0);
        assert!(camera.x() < 1.0);
    }

    /// 追従カメラが大きな移動でも粒子を見失わない範囲に遅れを制限することを確認する。
    #[test]
    fn camera_follow_limits_large_horizontal_lag() {
        let mut camera = CameraFollow::new(0.0);

        camera.advance_toward(10.0, 1.0 / 60.0);

        assert!((10.0 - camera.x()).abs() <= CAMERA_MAX_LAG);
    }
}
