use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align, Align2, Color32, Key, Layout, Modifiers, PointerButton, Pos2, Rect, RichText,
    Sense, Vec2,
};
use glam::Vec3;

use crate::{
    camera::Camera,
    history::History,
    mesh::{Mesh, RayHit},
    renderer::{BrushCursor, ViewportRenderer},
    sculpt::{BrushSample, BrushSettings, SculptEngine, SculptTool, SymmetryAxis},
    stl::{ImportReport, load_stl, save_stl_atomic},
};

const INITIAL_BRUSH_RADIUS_POINTS: f32 = 55.0;
const MIN_BRUSH_RADIUS_POINTS: f32 = 4.0;
const MAX_BRUSH_RADIUS_POINTS: f32 = 300.0;

struct MeshDocument {
    mesh: Mesh,
    source_path: PathBuf,
    report: ImportReport,
    dirty: bool,
}

#[derive(Clone)]
enum PendingAction {
    OpenDialog,
    Import(PathBuf),
    Close,
}

pub struct SculptLiteApp {
    renderer: Option<ViewportRenderer>,
    renderer_error: Option<String>,
    camera: Camera,
    document: Option<MeshDocument>,
    history: History,
    sculpt: SculptEngine,
    tool: SculptTool,
    brush: BrushSettings,
    brush_radius_points: f32,
    wireframe: bool,
    stroke_last_pointer: Option<Pos2>,
    stroke_history_saved: bool,
    pending_action: Option<PendingAction>,
    allow_close: bool,
    status: String,
    error: Option<String>,
}

impl SculptLiteApp {
    pub fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        let renderer_result = ViewportRenderer::new(creation_context);
        let (renderer, renderer_error) = match renderer_result {
            Ok(renderer) => (Some(renderer), None),
            Err(error) => (None, Some(error.to_string())),
        };

        creation_context.egui_ctx.set_visuals(egui::Visuals::dark());

        Self {
            renderer,
            renderer_error,
            camera: Camera::default(),
            document: None,
            history: History::default(),
            sculpt: SculptEngine::default(),
            tool: SculptTool::default(),
            brush: BrushSettings::default(),
            brush_radius_points: INITIAL_BRUSH_RADIUS_POINTS,
            wireframe: false,
            stroke_last_pointer: None,
            stroke_history_saved: false,
            pending_action: None,
            allow_close: false,
            status: "Import an STL to begin".to_owned(),
            error: None,
        }
    }

    fn request_action(&mut self, action: PendingAction, context: &egui::Context) {
        if self
            .document
            .as_ref()
            .is_some_and(|document| document.dirty)
        {
            self.pending_action = Some(action);
        } else {
            self.perform_action(action, context);
        }
    }

    fn perform_action(&mut self, action: PendingAction, context: &egui::Context) {
        match action {
            PendingAction::OpenDialog => {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("STL mesh", &["stl"])
                    .set_title("Import STL")
                    .pick_file()
                {
                    self.import_path(&path);
                }
            }
            PendingAction::Import(path) => self.import_path(&path),
            PendingAction::Close => {
                self.allow_close = true;
                context.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn import_path(&mut self, path: &Path) {
        match load_stl(path) {
            Ok((mesh, report)) => {
                let (minimum, maximum) = mesh.bounds().unwrap_or((Vec3::splat(-1.0), Vec3::ONE));
                self.camera.fit(minimum, maximum);
                self.history.clear();
                self.sculpt.reset_for_mesh(&mesh);
                if let Some(renderer) = &self.renderer {
                    renderer.update_mesh(&mesh);
                }
                self.status = if report.has_topology_warnings() {
                    format!("Loaded {} with protected topology regions", path.display())
                } else {
                    format!("Loaded {}", path.display())
                };
                self.document = Some(MeshDocument {
                    mesh,
                    source_path: path.to_path_buf(),
                    report,
                    dirty: false,
                });
                self.stroke_last_pointer = None;
                self.stroke_history_saved = false;
            }
            Err(error) => {
                self.error = Some(format!("Could not import {}\n\n{error}", path.display()));
            }
        }
    }

    fn export_as(&mut self) -> bool {
        let Some(document) = self.document.as_ref() else {
            return false;
        };

        let suggested_name = document
            .source_path
            .file_stem()
            .and_then(|name| name.to_str())
            .map_or_else(
                || "sculpted.stl".to_owned(),
                |name| format!("{name}-sculpted.stl"),
            );
        let Some(mut path) = rfd::FileDialog::new()
            .add_filter("STL mesh", &["stl"])
            .set_title("Export STL As")
            .set_file_name(suggested_name)
            .save_file()
        else {
            return false;
        };
        if path.extension().is_none() {
            path.set_extension("stl");
        }

        match save_stl_atomic(&path, &document.mesh) {
            Ok(()) => {
                if let Some(document) = self.document.as_mut() {
                    document.dirty = false;
                }
                self.status = format!("Exported {}", path.display());
                true
            }
            Err(error) => {
                self.error = Some(format!("Could not export {}\n\n{error}", path.display()));
                false
            }
        }
    }

    fn upload_mesh(&self) {
        if let (Some(renderer), Some(document)) = (&self.renderer, &self.document) {
            renderer.update_mesh(&document.mesh);
        }
    }

    fn upload_vertices(&self) {
        if let (Some(renderer), Some(document)) = (&self.renderer, &self.document) {
            renderer.update_vertices(&document.mesh);
        }
    }

    fn frame_mesh(&mut self) {
        if let Some((minimum, maximum)) = self
            .document
            .as_ref()
            .and_then(|document| document.mesh.bounds())
        {
            self.camera.fit(minimum, maximum);
        }
    }

    fn undo(&mut self) {
        let changed = self
            .document
            .as_mut()
            .is_some_and(|document| self.history.undo(&mut document.mesh));
        if changed {
            if let Some(document) = self.document.as_mut() {
                document.dirty = true;
            }
            self.upload_mesh();
            self.status = "Undo".to_owned();
        }
    }

    fn redo(&mut self) {
        let changed = self
            .document
            .as_mut()
            .is_some_and(|document| self.history.redo(&mut document.mesh));
        if changed {
            if let Some(document) = self.document.as_mut() {
                document.dirty = true;
            }
            self.upload_mesh();
            self.status = "Redo".to_owned();
        }
    }

    fn edit_mask(&mut self, invert: bool) {
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let changed = if invert {
            document.mask_needs_invert()
        } else {
            document.mesh.mask.iter().any(|value| *value != 0.0)
        };
        if !changed {
            return;
        }

        self.history.push_before(&document.mesh);
        if invert {
            for value in &mut document.mesh.mask {
                *value = 1.0 - value.clamp(0.0, 1.0);
            }
            self.status = "Mask inverted".to_owned();
        } else {
            document.mesh.mask.fill(0.0);
            self.status = "Mask cleared".to_owned();
        }
        document.dirty = true;
        self.upload_vertices();
    }

    fn handle_shortcuts(&mut self, context: &egui::Context) {
        if context.egui_wants_keyboard_input() {
            return;
        }

        let open = context.input_mut(|input| input.consume_key(Modifiers::CTRL, Key::O));
        let export = context
            .input_mut(|input| input.consume_key(Modifiers::CTRL | Modifiers::SHIFT, Key::S));
        let undo = context.input_mut(|input| input.consume_key(Modifiers::CTRL, Key::Z));
        let redo = context.input_mut(|input| {
            input.consume_key(Modifiers::CTRL, Key::Y)
                || input.consume_key(Modifiers::CTRL | Modifiers::SHIFT, Key::Z)
        });
        let frame = context.input_mut(|input| input.consume_key(Modifiers::NONE, Key::F));
        let smaller =
            context.input_mut(|input| input.consume_key(Modifiers::NONE, Key::OpenBracket));
        let larger =
            context.input_mut(|input| input.consume_key(Modifiers::NONE, Key::CloseBracket));

        if open {
            self.request_action(PendingAction::OpenDialog, context);
        }
        if export {
            self.export_as();
        }
        if undo {
            self.undo();
        }
        if redo {
            self.redo();
        }
        if frame {
            self.frame_mesh();
        }
        if smaller {
            self.brush_radius_points =
                (self.brush_radius_points / 1.12).max(MIN_BRUSH_RADIUS_POINTS);
        }
        if larger {
            self.brush_radius_points =
                (self.brush_radius_points * 1.12).min(MAX_BRUSH_RADIUS_POINTS);
        }
    }

    fn top_bar(&mut self, root_ui: &mut egui::Ui, context: &egui::Context) {
        egui::Panel::top("top_bar").show(root_ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Import STL").clicked() {
                    self.request_action(PendingAction::OpenDialog, context);
                }
                if ui
                    .add_enabled(self.document.is_some(), egui::Button::new("Export As…"))
                    .clicked()
                {
                    self.export_as();
                }
                ui.separator();
                if ui
                    .add_enabled(self.history.can_undo(), egui::Button::new("Undo"))
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add_enabled(self.history.can_redo(), egui::Button::new("Redo"))
                    .clicked()
                {
                    self.redo();
                }
                ui.separator();
                if ui
                    .add_enabled(self.document.is_some(), egui::Button::new("Frame"))
                    .clicked()
                {
                    self.frame_mesh();
                }
                ui.checkbox(&mut self.wireframe, "Wireframe");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new("SculptLite").strong());
                });
            });
        });
    }

    fn tool_panel(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::left("tools")
            .resizable(false)
            .default_size(228.0)
            .show(root_ui, |ui| {
                ui.heading("Sculpt");
                ui.add_space(4.0);
                ui.add_enabled_ui(self.document.is_some(), |ui| {
                    egui::Grid::new("tool_grid")
                        .num_columns(2)
                        .spacing([6.0, 6.0])
                        .show(ui, |ui| {
                            for (index, tool) in SculptTool::ALL.into_iter().enumerate() {
                                ui.selectable_value(&mut self.tool, tool, tool.label());
                                if index % 2 == 1 {
                                    ui.end_row();
                                }
                            }
                        });

                    ui.separator();
                    ui.label("Radius");
                    ui.add(
                        egui::Slider::new(
                            &mut self.brush_radius_points,
                            MIN_BRUSH_RADIUS_POINTS..=MAX_BRUSH_RADIUS_POINTS,
                        )
                        .suffix(" px")
                        .logarithmic(true),
                    );
                    ui.label("Strength");
                    ui.add(egui::Slider::new(&mut self.brush.strength, 0.01..=1.0));
                    ui.label("Hardness");
                    ui.add(egui::Slider::new(&mut self.brush.falloff, 0.0..=0.95));
                    ui.add_enabled_ui(self.tool != SculptTool::Mask, |ui| {
                        ui.label("Dynamic detail");
                        ui.add(
                            egui::Slider::new(&mut self.brush.detail, 0.03..=0.35)
                                .logarithmic(true),
                        );
                    });
                    ui.checkbox(&mut self.brush.invert, "Invert brush");

                    ui.separator();
                    ui.label("Symmetry");
                    egui::ComboBox::from_id_salt("symmetry")
                        .selected_text(match self.brush.symmetry {
                            None => "Off",
                            Some(SymmetryAxis::X) => "X axis",
                            Some(SymmetryAxis::Y) => "Y axis",
                            Some(SymmetryAxis::Z) => "Z axis",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.brush.symmetry, None, "Off");
                            ui.selectable_value(
                                &mut self.brush.symmetry,
                                Some(SymmetryAxis::X),
                                "X axis",
                            );
                            ui.selectable_value(
                                &mut self.brush.symmetry,
                                Some(SymmetryAxis::Y),
                                "Y axis",
                            );
                            ui.selectable_value(
                                &mut self.brush.symmetry,
                                Some(SymmetryAxis::Z),
                                "Z axis",
                            );
                        });

                    ui.horizontal(|ui| {
                        if ui.button("Clear mask").clicked() {
                            self.edit_mask(false);
                        }
                        if ui.button("Invert mask").clicked() {
                            self.edit_mask(true);
                        }
                    });
                });

                ui.separator();
                if let Some(document) = &self.document {
                    ui.label(RichText::new("Mesh").strong());
                    ui.label(format!(
                        "{} vertices",
                        grouped(document.mesh.positions.len())
                    ));
                    ui.label(format!("{} faces", grouped(document.mesh.triangles.len())));
                    if let Some((minimum, maximum)) = document.mesh.bounds() {
                        let size = maximum - minimum;
                        ui.label(format!(
                            "Size: {:.2} × {:.2} × {:.2} units",
                            size.x, size.y, size.z
                        ));
                    }
                    ui.collapsing("Import report", |ui| {
                        ui.label(document.report.to_string());
                    });
                } else {
                    ui.label("No mesh loaded");
                }

                ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                    let mib = self.history.bytes_used() as f64 / (1024.0 * 1024.0);
                    let budget_mib = self.history.byte_budget() as f64 / (1024.0 * 1024.0);
                    ui.small(format!("Undo memory: {mib:.1} / {budget_mib:.0} MiB"));
                    ui.small("Shift: smooth · Ctrl: invert");
                    ui.small("RMB: orbit · MMB: pan · Wheel: zoom");
                });
            });
    }

    fn status_bar(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::bottom("status_bar").show(root_ui, |ui| {
            ui.horizontal(|ui| {
                ui.small(&self.status);
                if let Some(error) = &self.renderer_error {
                    ui.separator();
                    ui.colored_label(Color32::LIGHT_RED, format!("Renderer unavailable: {error}"));
                }
            });
        });
    }

    fn viewport(&mut self, root_ui: &mut egui::Ui, context: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_rgb(25, 27, 31)))
            .show(root_ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(rect, Sense::click_and_drag());

                if self.document.is_none() {
                    ui.painter().text(
                        rect.center() - Vec2::new(0.0, 18.0),
                        Align2::CENTER_CENTER,
                        "Import an STL mesh",
                        egui::FontId::proportional(24.0),
                        Color32::from_gray(210),
                    );
                    let button_rect = Rect::from_center_size(
                        rect.center() + Vec2::new(0.0, 24.0),
                        Vec2::new(130.0, 34.0),
                    );
                    if ui
                        .put(button_rect, egui::Button::new("Choose STL…"))
                        .clicked()
                    {
                        self.request_action(PendingAction::OpenDialog, context);
                    }
                    return;
                }

                let pointer = response
                    .hover_pos()
                    .or_else(|| context.pointer_latest_pos());
                let hover_hit = pointer.and_then(|position| self.hit_at(position, rect));
                let cursor = pointer
                    .filter(|position| rect.contains(*position))
                    .map(|position| {
                        let mut cursor = BrushCursor::new(position, self.brush_radius_points);
                        cursor.active = self.sculpt.is_stroking();
                        cursor.color = if context.input(|input| input.modifiers.ctrl) {
                            Color32::from_rgb(238, 128, 92)
                        } else {
                            Color32::from_rgb(115, 205, 255)
                        };
                        cursor
                    });

                if let Some(renderer) = &self.renderer {
                    renderer.update_camera(
                        &self.camera,
                        (rect.width() / rect.height().max(1.0)).max(0.001),
                    );
                    renderer.paint(ui, rect, cursor, self.wireframe);
                } else {
                    ui.painter()
                        .rect_filled(rect, 0.0, Color32::from_rgb(20, 22, 25));
                }

                let pointer_delta = context.input(|input| input.pointer.delta());
                if response.dragged_by(PointerButton::Secondary) {
                    self.camera.orbit(pointer_delta);
                } else if response.dragged_by(PointerButton::Middle) {
                    self.camera.pan(pointer_delta, rect.height());
                }
                if response.hovered() {
                    let scroll = context.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll);
                    }
                }

                let mut began_stroke = false;
                if response.drag_started_by(PointerButton::Primary)
                    && let Some(position) = pointer
                    && hover_hit.is_some()
                    && let Some(document) = self.document.as_ref()
                {
                    self.stroke_history_saved = self.history.push_before(&document.mesh);
                    self.sculpt.begin_stroke(&document.mesh);
                    self.stroke_last_pointer = Some(position);
                    self.apply_pointer_sample(context, rect, position, Vec2::ZERO);
                    began_stroke = true;
                }

                if response.dragged_by(PointerButton::Primary)
                    && !began_stroke
                    && self.sculpt.is_stroking()
                    && let Some(position) = pointer
                {
                    self.apply_pointer_segment(context, rect, position);
                }

                if response.drag_stopped_by(PointerButton::Primary) && self.sculpt.is_stroking() {
                    let changed = self
                        .document
                        .as_mut()
                        .is_some_and(|document| self.sculpt.end_stroke(&mut document.mesh));
                    if changed {
                        if let Some(document) = self.document.as_mut() {
                            document.dirty = true;
                        }
                        self.upload_mesh();
                        self.status = if self.stroke_history_saved {
                            format!("{} stroke", self.tool.label())
                        } else {
                            format!("{} stroke (undo memory limit reached)", self.tool.label())
                        };
                    } else {
                        if self.stroke_history_saved {
                            self.history.discard_latest();
                        }
                    }
                    self.stroke_last_pointer = None;
                    self.stroke_history_saved = false;
                }
            });
    }

    fn hit_at(&self, pointer: Pos2, viewport: Rect) -> Option<RayHit> {
        let ray = self.camera.screen_ray(pointer, viewport)?;
        self.document
            .as_ref()?
            .mesh
            .raycast(ray.origin, ray.direction)
    }

    fn apply_pointer_segment(&mut self, context: &egui::Context, viewport: Rect, pointer: Pos2) {
        let previous = self.stroke_last_pointer.unwrap_or(pointer);
        let delta = pointer - previous;
        if delta.length_sq() < 0.25 {
            return;
        }
        let spacing = (self.brush_radius_points * 0.15).max(1.0);
        let steps = (delta.length() / spacing).ceil().clamp(1.0, 8.0) as usize;
        let mut prior = previous;
        for step in 1..=steps {
            let position = previous + delta * (step as f32 / steps as f32);
            self.apply_pointer_sample(context, viewport, position, position - prior);
            prior = position;
        }
        self.stroke_last_pointer = Some(pointer);
    }

    fn apply_pointer_sample(
        &mut self,
        context: &egui::Context,
        viewport: Rect,
        pointer: Pos2,
        pointer_delta: Vec2,
    ) {
        let Some(ray) = self.camera.screen_ray(pointer, viewport) else {
            return;
        };
        let Some(hit) = self
            .document
            .as_ref()
            .and_then(|document| document.mesh.raycast(ray.origin, ray.direction))
        else {
            return;
        };

        let depth_scale = (hit.distance / self.camera.distance.max(1.0e-6)).max(0.05);
        let units_per_point = self.camera.world_units_per_pixel(viewport.height()) * depth_scale;
        let world_drag = self.camera.right() * pointer_delta.x * units_per_point
            - self.camera.up() * pointer_delta.y * units_per_point;
        let modifiers = context.input(|input| input.modifiers);
        let effective_tool = if modifiers.shift && self.tool != SculptTool::Mask {
            SculptTool::Smooth
        } else {
            self.tool
        };
        let mut settings = self.brush;
        settings.radius = self.brush_radius_points * units_per_point;
        let sample = BrushSample::from_hit(&hit, world_drag, ray.direction, 1.0, modifiers.ctrl);

        let changed = self.document.as_mut().is_some_and(|document| {
            self.sculpt
                .apply_sample(&mut document.mesh, effective_tool, &settings, sample)
        });
        if changed {
            self.upload_vertices();
            context.request_repaint();
        }
    }

    fn unsaved_prompt(&mut self, context: &egui::Context) {
        let Some(pending) = self.pending_action.clone() else {
            return;
        };
        egui::Modal::new(egui::Id::new("unsaved_sculpt")).show(context, |ui| {
            ui.heading("Unsaved sculpt");
            ui.label("The current sculpt has changes that have not been exported.");
            ui.label("Export it before continuing?");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Export As…").clicked() && self.export_as() {
                    self.pending_action = None;
                    self.perform_action(pending.clone(), context);
                }
                if ui.button("Discard changes").clicked() {
                    if let Some(document) = self.document.as_mut() {
                        document.dirty = false;
                    }
                    self.pending_action = None;
                    self.perform_action(pending.clone(), context);
                }
                if ui.button("Cancel").clicked() {
                    self.pending_action = None;
                }
            });
        });
    }

    fn error_prompt(&mut self, context: &egui::Context) {
        let Some(message) = self.error.clone() else {
            return;
        };
        egui::Modal::new(egui::Id::new("sculpt_lite_error")).show(context, |ui| {
            ui.heading("SculptLite error");
            ui.label(message);
            if ui.button("Close").clicked() {
                self.error = None;
            }
        });
    }
}

impl eframe::App for SculptLiteApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let context = root_ui.ctx().clone();
        if context.input(|input| input.viewport().close_requested())
            && !self.allow_close
            && self
                .document
                .as_ref()
                .is_some_and(|document| document.dirty)
        {
            context.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            if self.pending_action.is_none() {
                self.pending_action = Some(PendingAction::Close);
            }
        }

        let dropped_stl = context.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .find(|path| {
                    path.extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("stl"))
                })
        });
        if let Some(path) = dropped_stl {
            self.request_action(PendingAction::Import(path), &context);
        }

        self.handle_shortcuts(&context);
        let title = self.document.as_ref().map_or_else(
            || "SculptLite".to_owned(),
            |document| {
                let name = document
                    .source_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Untitled STL");
                format!(
                    "SculptLite — {name}{}",
                    if document.dirty { " *" } else { "" }
                )
            },
        );
        context.send_viewport_cmd(egui::ViewportCommand::Title(title));

        self.top_bar(root_ui, &context);
        self.status_bar(root_ui);
        self.tool_panel(root_ui);
        self.viewport(root_ui, &context);
        if self.error.is_some() {
            self.error_prompt(&context);
        } else {
            self.unsaved_prompt(&context);
        }

        if self.sculpt.is_stroking() {
            context.request_repaint();
        }
    }
}

impl MeshDocument {
    fn mask_needs_invert(&self) -> bool {
        !self.mesh.mask.is_empty()
    }
}

fn grouped(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_formats_face_counts() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }
}
