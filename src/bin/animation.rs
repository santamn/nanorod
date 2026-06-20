use std::time::Instant;

#[allow(dead_code)]
#[path = "../config.rs"]
mod config;
#[path = "../cpu_sim.rs"]
mod cpu_sim;
#[allow(dead_code)]
#[path = "../model.rs"]
mod model;

use cpu_sim::{VisualParams, VisualSimulation};
use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, pos2, vec2};

const TRAIL_LIMIT: usize = 900;

fn main() -> eframe::Result {
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
        Box::new(|cc| Ok(Box::new(AnimationApp::new(cc)))),
    )
}

struct AnimationApp {
    controls: VisualParams,
    simulation: VisualSimulation,
    running: bool,
    speed_scale: f64,
    step_remainder: f64,
    last_frame: Instant,
    trail: Vec<Pos2>,
}

impl AnimationApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);

        let controls = VisualParams::default();
        let simulation = VisualSimulation::new(controls);
        let trail = vec![pos2(simulation.x as f32, simulation.y as f32)];

        Self {
            controls,
            simulation,
            running: false,
            speed_scale: 1.0,
            step_remainder: 0.0,
            last_frame: Instant::now(),
            trail,
        }
    }

    fn reset(&mut self) {
        self.simulation.reset(self.controls);
        self.step_remainder = 0.0;
        self.trail.clear();
        self.trail
            .push(pos2(self.simulation.x as f32, self.simulation.y as f32));
    }

    fn advance(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;

        if !self.running {
            return;
        }

        if self.simulation.completed {
            self.running = false;
            return;
        }

        self.step_remainder += elapsed * self.speed_scale / config::DT;
        let step_cap = adaptive_step_cap(self.controls.point_count());
        let mut steps = self.step_remainder.floor() as usize;
        if steps > step_cap {
            steps = step_cap;
            self.step_remainder = 0.0;
        } else {
            self.step_remainder -= steps as f64;
        }

        if steps > 0 {
            self.simulation.step_many(steps);
            self.push_trail_point();
        }

        if self.simulation.completed {
            self.running = false;
        } else {
            ctx.request_repaint();
        }
    }

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
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.advance(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(300.0)
            .show_inside(ui, |ui| {
                ui.add_space(8.0);
                ui.heading("粒子アニメーション");
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    let label = if self.running { "停止" } else { "開始" };
                    if ui.button(label).clicked() {
                        if self.simulation.completed {
                            self.reset();
                        }
                        self.running = !self.running;
                        self.last_frame = Instant::now();
                        if self.running {
                            ui.ctx().request_repaint();
                        }
                    }

                    if ui.button("リセット").clicked() {
                        self.running = false;
                        self.reset();
                    }
                });

                ui.separator();

                let mut changed = false;
                let point_count_label = point_count_label(self.controls.m);
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.controls.seed)
                            .speed(1.0)
                            .prefix("seed "),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.controls.m, 1..=30).text(point_count_label))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.controls.force, 0.0..=100.0).text("f"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.controls.beta_pe, -5.0..=5.0).text("βpE"))
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.controls.delta_alpha_e_over_p, -5.0..=5.0)
                            .text("ΔαE/p"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.speed_scale, 0.01..=2.0).text("速度"))
                    .changed();

                if changed {
                    self.running = false;
                    self.reset();
                }

                ui.separator();
                ui.label(format!("実験時刻 {:.6} s", self.simulation.t));
                ui.label(format!("step {}", self.simulation.steps));
                ui.label(format!("x0 {:.4}", self.simulation.x0));
                ui.label(format!("y0 {:.4}", self.simulation.y0));
                ui.label(format!("φ0 {:.3}", self.simulation.phi0));
                ui.label(format!("x {:.4}", self.simulation.x));
                ui.label(format!("y {:.4}", self.simulation.y));
                ui.label(format!("φ {:.3}", self.simulation.phi));
                ui.label(if self.simulation.completed {
                    "初通過 済"
                } else {
                    "初通過 未"
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let available = ui.available_size();
            let (response, painter) = ui.allocate_painter(available, Sense::hover());
            draw_simulation(&painter, response.rect, &self.simulation, &self.trail);
        });
    }
}

fn configure_style(ctx: &egui::Context) {
    ctx.global_style_mut(|style| {
        style.visuals.window_fill = Color32::from_rgb(248, 249, 247);
        style.visuals.panel_fill = Color32::from_rgb(248, 249, 247);
        style.spacing.item_spacing = vec2(8.0, 10.0);
    });
}

fn adaptive_step_cap(point_count: i32) -> usize {
    let count = point_count.max(1) as usize;
    (96_000 / count).clamp(900, 8_000)
}

fn point_count_label(m: i32) -> String {
    format!("代表点 {}", 2 * m + 1)
}

fn draw_simulation(
    painter: &egui::Painter,
    rect: Rect,
    simulation: &VisualSimulation,
    trail: &[Pos2],
) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(239, 244, 243));

    let camera_x = simulation.x + 0.34;
    let world = WorldView::new(rect, camera_x, 2.35, 5.4);

    draw_channel(painter, &world);
    draw_markers(painter, &world, simulation);
    draw_trail(painter, &world, trail);
    draw_rod(painter, &world, simulation);
}

fn draw_channel(painter: &egui::Painter, world: &WorldView) {
    let mut top = Vec::with_capacity(260);
    let mut bottom = Vec::with_capacity(260);
    let samples = 240;

    for idx in 0..=samples {
        let x = world.x_min + (world.x_max - world.x_min) * idx as f64 / samples as f64;
        let y = model::omega(x);
        top.push(world.to_screen(x, y));
        bottom.push(world.to_screen(x, -y));
    }

    painter.add(Shape::line(
        top.clone(),
        Stroke::new(2.0, Color32::from_rgb(38, 101, 111)),
    ));
    painter.add(Shape::line(
        bottom.clone(),
        Stroke::new(2.0, Color32::from_rgb(38, 101, 111)),
    ));

    let mut centerline = Vec::with_capacity(80);
    for idx in 0..=80 {
        let x = world.x_min + (world.x_max - world.x_min) * idx as f64 / 80.0;
        centerline.push(world.to_screen(x, 0.0));
    }
    painter.add(Shape::line(
        centerline,
        Stroke::new(1.0, Color32::from_rgba_premultiplied(94, 115, 120, 90)),
    ));

    for period in (world.x_min.floor() as i32 - 1)..=(world.x_max.ceil() as i32 + 1) {
        let x = f64::from(period);
        if x >= world.x_min && x <= world.x_max {
            let a = world.to_screen(x, -2.5);
            let b = world.to_screen(x, 2.5);
            painter.line_segment(
                [a, b],
                Stroke::new(1.0, Color32::from_rgba_premultiplied(94, 115, 120, 55)),
            );
        }
    }
}

fn draw_markers(painter: &egui::Painter, world: &WorldView, simulation: &VisualSimulation) {
    let start_a = world.to_screen(simulation.x0, -2.5);
    let start_b = world.to_screen(simulation.x0, 2.5);
    painter.line_segment(
        [start_a, start_b],
        Stroke::new(1.5, Color32::from_rgba_premultiplied(65, 93, 167, 110)),
    );

    let target_a = world.to_screen(simulation.target_x, -2.5);
    let target_b = world.to_screen(simulation.target_x, 2.5);
    painter.line_segment(
        [target_a, target_b],
        Stroke::new(2.0, Color32::from_rgba_premultiplied(199, 80, 57, 165)),
    );
}

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
        Stroke::new(1.8, Color32::from_rgba_premultiplied(68, 118, 132, 115)),
    ));
}

fn draw_rod(painter: &egui::Painter, world: &WorldView, simulation: &VisualSimulation) {
    let rod_points = simulation.representative_points();
    let Some(first) = rod_points.first() else {
        return;
    };
    let Some(last) = rod_points.last() else {
        return;
    };

    let start = world.to_screen(first.x, first.y);
    let end = world.to_screen(last.x, last.y);
    painter.line_segment(
        [start, end],
        Stroke::new(5.0, Color32::from_rgb(33, 55, 68)),
    );

    for (idx, point) in rod_points.iter().enumerate() {
        let p = world.to_screen(point.x, point.y);
        let color = if idx == rod_points.len() - 1 {
            Color32::from_rgb(223, 95, 70)
        } else {
            Color32::from_rgb(246, 197, 82)
        };
        painter.circle_filled(p, 3.6, color);
        painter.circle_stroke(p, 3.6, Stroke::new(1.0, Color32::from_rgb(33, 55, 68)));
    }

    let center = world.to_screen(simulation.x, simulation.y);
    painter.circle_filled(center, 5.0, Color32::from_rgb(42, 132, 128));
}

struct WorldView {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    scale: f64,
    origin: Pos2,
}

impl WorldView {
    fn new(rect: Rect, center_x: f64, x_span: f64, y_span: f64) -> Self {
        let x_min = center_x - 0.5 * x_span;
        let x_max = center_x + 0.5 * x_span;
        let y_min = -0.5 * y_span;
        let pad = 26.0;
        let drawable = rect.shrink(pad);
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
            scale,
            origin,
        }
    }

    fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        let sx = self.origin.x + ((x - self.x_min) * self.scale) as f32;
        let sy = self.origin.y - ((y - self.y_min) * self.scale) as f32;
        pos2(sx, sy)
    }
}
