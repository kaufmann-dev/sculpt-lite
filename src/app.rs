use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Align, Align2, Color32, Key, Layout, Modifiers, PointerButton, Pos2, Rect, RichText,
    Sense, Vec2,
};
use glam::Vec3;

use crate::{
    camera::Camera,
    history::{History, MeshSnapshot},
    mesh::{Mesh, RayHit},
    renderer::{BrushCursor, ViewportRenderer},
    sculpt::{BrushSample, BrushSettings, RemeshRequest, SculptEngine, SculptTool, SymmetryAxis},
    stl::{ImportReport, load_stl, save_stl_atomic},
};

const INITIAL_BRUSH_RADIUS_POINTS: f32 = 55.0;
const MIN_BRUSH_RADIUS_POINTS: f32 = 4.0;
const MAX_BRUSH_RADIUS_POINTS: f32 = 300.0;
const MAX_ADAPTIVE_TOPOLOGY_FACES: usize = 250_000;

struct MeshDocument {
    mesh: Option<Mesh>,
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

enum WorkerJob {
    Import(PathBuf),
    Remesh {
        mesh: Box<Mesh>,
        request: RemeshRequest,
    },
}

enum WorkerResult {
    Import {
        path: PathBuf,
        result: Result<(Mesh, ImportReport), String>,
    },
    RemeshCheckpoint(Arc<MeshSnapshot>),
    Remesh {
        mesh: Mesh,
        error: Option<String>,
    },
}

struct BackgroundWorker {
    sender: Sender<WorkerJob>,
    receiver: Receiver<WorkerResult>,
}

impl BackgroundWorker {
    fn start(context: egui::Context) -> std::io::Result<Self> {
        let (job_sender, job_receiver) = mpsc::channel::<WorkerJob>();
        let (result_sender, result_receiver) = mpsc::channel::<WorkerResult>();
        thread::Builder::new()
            .name("sculptlite-mesh-worker".to_owned())
            .spawn(move || {
                while let Ok(job) = job_receiver.recv() {
                    let result = match job {
                        WorkerJob::Import(path) => {
                            let result = catch_unwind(AssertUnwindSafe(|| load_stl(&path)))
                                .map_err(panic_message)
                                .and_then(|result| result.map_err(|error| error.to_string()));
                            WorkerResult::Import { path, result }
                        }
                        WorkerJob::Remesh { mesh, request } => {
                            let mut mesh = *mesh;
                            let recovery = Arc::new(MeshSnapshot::capture(&mesh));
                            if result_sender
                                .send(WorkerResult::RemeshCheckpoint(Arc::clone(&recovery)))
                                .is_err()
                            {
                                break;
                            }
                            context.request_repaint();
                            let error = catch_unwind(AssertUnwindSafe(|| {
                                let _ = mesh
                                    .remesh_region(&request.affected_vertices, request.settings);
                            }))
                            .err()
                            .map(panic_message);
                            if error.is_some() {
                                recovery.restore(&mut mesh);
                            }
                            WorkerResult::Remesh { mesh, error }
                        }
                    };
                    if result_sender.send(result).is_err() {
                        break;
                    }
                    context.request_repaint();
                }
            })?;
        Ok(Self {
            sender: job_sender,
            receiver: result_receiver,
        })
    }
}

enum BackgroundTask {
    Import {
        path: PathBuf,
        started: Instant,
    },
    Remesh {
        tool: SculptTool,
        history_saved: bool,
        started: Instant,
        recovery: Option<Arc<MeshSnapshot>>,
    },
}

impl BackgroundTask {
    fn is_remesh(&self) -> bool {
        matches!(self, Self::Remesh { .. })
    }

    fn progress_text(&self) -> String {
        let (label, started) = match self {
            Self::Import { path, started } => {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("STL mesh");
                (format!("Importing {name}"), started)
            }
            Self::Remesh { started, .. } => ("Optimizing sculpt topology".to_owned(), started),
        };
        format!("{label}… {:.1}s", started.elapsed().as_secs_f32())
    }
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
    adaptive_topology: bool,
    brush_radius_points: f32,
    wireframe: bool,
    stroke_last_pointer: Option<Pos2>,
    stroke_history_saved: bool,
    pending_action: Option<PendingAction>,
    worker: Option<BackgroundWorker>,
    background_task: Option<BackgroundTask>,
    worker_error: Option<String>,
    allow_close: bool,
    status: String,
    error: Option<String>,
}

impl SculptLiteApp {
    pub fn new(
        creation_context: &eframe::CreationContext<'_>,
        initial_path: Option<PathBuf>,
    ) -> Self {
        let renderer_result = ViewportRenderer::new(creation_context);
        let (renderer, renderer_error) = match renderer_result {
            Ok(renderer) => (Some(renderer), None),
            Err(error) => (None, Some(error.to_string())),
        };

        creation_context.egui_ctx.set_visuals(egui::Visuals::dark());

        let (worker, worker_error) =
            match BackgroundWorker::start(creation_context.egui_ctx.clone()) {
                Ok(worker) => (Some(worker), None),
                Err(error) => (
                    None,
                    Some(format!("Could not start the mesh worker: {error}")),
                ),
            };

        let mut app = Self {
            renderer,
            renderer_error,
            camera: Camera::default(),
            document: None,
            history: History::default(),
            sculpt: SculptEngine::default(),
            tool: SculptTool::default(),
            brush: BrushSettings::default(),
            adaptive_topology: false,
            brush_radius_points: INITIAL_BRUSH_RADIUS_POINTS,
            wireframe: false,
            stroke_last_pointer: None,
            stroke_history_saved: false,
            pending_action: None,
            worker,
            background_task: None,
            worker_error,
            allow_close: false,
            status: "Import an STL to begin".to_owned(),
            error: None,
        };
        if let Some(path) = initial_path {
            app.start_import(path, &creation_context.egui_ctx);
        }
        app
    }

    fn request_action(&mut self, action: PendingAction, context: &egui::Context) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            self.status = "Wait for the current mesh operation to finish".to_owned();
            return;
        }
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
                    self.start_import(path, context);
                }
            }
            PendingAction::Import(path) => self.start_import(path, context),
            PendingAction::Close => {
                self.allow_close = true;
                context.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn start_import(&mut self, path: PathBuf, context: &egui::Context) {
        let Some(worker) = &self.worker else {
            self.error = Some(
                self.worker_error
                    .clone()
                    .unwrap_or_else(|| "The mesh worker is unavailable".to_owned()),
            );
            return;
        };
        if let Err(error) = worker.sender.send(WorkerJob::Import(path.clone())) {
            let WorkerJob::Import(path) = error.0 else {
                unreachable!("the submitted worker job remains an import")
            };
            self.worker_error = Some("The mesh worker stopped unexpectedly".to_owned());
            self.error = Some(format!(
                "Could not begin importing {}\n\nThe mesh worker stopped unexpectedly.",
                path.display()
            ));
            return;
        }
        self.background_task = Some(BackgroundTask::Import {
            path,
            started: Instant::now(),
        });
        self.status = "Reading STL and building mesh topology".to_owned();
        context.request_repaint();
    }

    fn install_import(&mut self, path: PathBuf, mesh: Mesh, report: ImportReport) {
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
            mesh: Some(mesh),
            source_path: path,
            report,
            dirty: false,
        });
        self.stroke_last_pointer = None;
        self.stroke_history_saved = false;
    }

    fn start_remesh(&mut self, request: RemeshRequest, context: &egui::Context) {
        let Some(mesh) = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.take())
        else {
            self.error =
                Some("Could not optimize the stroke because its mesh is missing".to_owned());
            return;
        };
        let Some(worker) = &self.worker else {
            if let Some(document) = self.document.as_mut() {
                document.mesh = Some(mesh);
            }
            self.error = Some(
                self.worker_error
                    .clone()
                    .unwrap_or_else(|| "The mesh worker is unavailable".to_owned()),
            );
            return;
        };

        match worker.sender.send(WorkerJob::Remesh {
            mesh: Box::new(mesh),
            request,
        }) {
            Ok(()) => {
                self.background_task = Some(BackgroundTask::Remesh {
                    tool: self.tool,
                    history_saved: self.stroke_history_saved,
                    started: Instant::now(),
                    recovery: None,
                });
                self.status = "Optimizing sculpt topology".to_owned();
                context.request_repaint();
            }
            Err(error) => {
                let WorkerJob::Remesh { mesh, .. } = error.0 else {
                    unreachable!("the submitted worker job remains a remesh")
                };
                if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(*mesh);
                }
                self.worker_error = Some("The mesh worker stopped unexpectedly".to_owned());
                self.error = Some(
                    "Could not optimize the stroke because the mesh worker stopped unexpectedly"
                        .to_owned(),
                );
            }
        }
    }

    fn poll_background_task(&mut self) {
        if self.background_task.is_none() {
            return;
        }
        let Some(worker) = &self.worker else {
            return;
        };
        let result = match worker.receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                self.worker_error = Some("The mesh worker stopped unexpectedly".to_owned());
                self.error = Some(match self.background_task.take() {
                    Some(BackgroundTask::Remesh {
                        recovery: Some(recovery),
                        ..
                    }) => {
                        self.restore_remesh_recovery(&recovery);
                        "The mesh worker stopped; the sculpt was restored without topology optimization"
                            .to_owned()
                    }
                    Some(BackgroundTask::Remesh { .. }) => {
                        "The mesh worker stopped before it could preserve the sculpted mesh"
                            .to_owned()
                    }
                    _ => "The mesh worker stopped before finishing the import".to_owned(),
                });
                return;
            }
        };
        let Some(task) = self.background_task.take() else {
            return;
        };

        match (task, result) {
            (
                BackgroundTask::Import { .. },
                WorkerResult::Import {
                    path,
                    result: Ok((mesh, report)),
                },
            ) => self.install_import(path, mesh, report),
            (
                BackgroundTask::Import { .. },
                WorkerResult::Import {
                    path,
                    result: Err(error),
                },
            ) => {
                self.status = "Import failed; the current mesh was left unchanged".to_owned();
                self.error = Some(format!("Could not import {}\n\n{error}", path.display()));
            }
            (
                BackgroundTask::Remesh {
                    tool,
                    history_saved,
                    started,
                    ..
                },
                WorkerResult::RemeshCheckpoint(recovery),
            ) => {
                self.background_task = Some(BackgroundTask::Remesh {
                    tool,
                    history_saved,
                    started,
                    recovery: Some(recovery),
                });
            }
            (
                BackgroundTask::Remesh {
                    tool,
                    history_saved,
                    started,
                    ..
                },
                WorkerResult::Remesh { mesh, error },
            ) => {
                if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(mesh);
                    document.dirty = true;
                }
                self.upload_mesh();
                let undo_note = if history_saved {
                    ""
                } else {
                    " (undo memory limit reached)"
                };
                self.status = format!(
                    "{} stroke optimized in {:.1}s{undo_note}",
                    tool.label(),
                    started.elapsed().as_secs_f32()
                );
                if let Some(error) = error {
                    self.error = Some(format!(
                        "The sculpt was kept, but topology optimization failed.\n\n{error}"
                    ));
                }
            }
            (BackgroundTask::Import { .. }, WorkerResult::RemeshCheckpoint(_))
            | (BackgroundTask::Import { .. }, WorkerResult::Remesh { .. }) => {
                self.error = Some("The mesh worker returned an unexpected result".to_owned());
            }
            (
                BackgroundTask::Remesh { recovery, .. },
                WorkerResult::Import {
                    result: Ok((mesh, _)),
                    ..
                },
            ) => {
                if let Some(recovery) = recovery {
                    self.restore_remesh_recovery(&recovery);
                } else if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(mesh);
                    document.dirty = true;
                }
                self.error = Some("The mesh worker returned an unexpected result".to_owned());
            }
            (
                BackgroundTask::Remesh { recovery, .. },
                WorkerResult::Import {
                    result: Err(error), ..
                },
            ) => {
                if let Some(recovery) = recovery {
                    self.restore_remesh_recovery(&recovery);
                }
                self.error = Some(format!(
                    "The mesh worker returned an unexpected error: {error}"
                ));
            }
        }
    }

    fn restore_remesh_recovery(&mut self, recovery: &MeshSnapshot) {
        let mut mesh = Mesh::default();
        recovery.restore(&mut mesh);
        if let Some(document) = self.document.as_mut() {
            document.mesh = Some(mesh);
            document.dirty = true;
        }
        self.upload_mesh();
    }

    fn export_as(&mut self) -> bool {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return false;
        }
        let Some(document) = self.document.as_ref() else {
            return false;
        };
        let Some(mesh) = document.mesh.as_ref() else {
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

        match save_stl_atomic(&path, mesh) {
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
        if let (Some(renderer), Some(mesh)) = (
            &self.renderer,
            self.document
                .as_ref()
                .and_then(|document| document.mesh.as_ref()),
        ) {
            renderer.update_mesh(mesh);
        }
    }

    fn upload_vertices(&self) {
        if let (Some(renderer), Some(mesh)) = (
            &self.renderer,
            self.document
                .as_ref()
                .and_then(|document| document.mesh.as_ref()),
        ) {
            renderer.update_vertices(mesh);
        }
    }

    fn upload_vertices_partial(&self, changed_vertices: &[u32]) {
        if let (Some(renderer), Some(mesh)) = (
            &self.renderer,
            self.document
                .as_ref()
                .and_then(|document| document.mesh.as_ref()),
        ) {
            renderer.update_vertices_partial(mesh, changed_vertices);
        }
    }

    fn frame_mesh(&mut self) {
        if let Some((minimum, maximum)) = self
            .document
            .as_ref()
            .and_then(|document| document.mesh.as_ref())
            .and_then(Mesh::bounds)
        {
            self.camera.fit(minimum, maximum);
        }
    }

    fn undo(&mut self) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return;
        }
        let changed = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.as_mut())
            .is_some_and(|mesh| self.history.undo(mesh));
        if changed {
            if let Some(document) = self.document.as_mut() {
                document.dirty = true;
            }
            self.upload_mesh();
            self.status = "Undo".to_owned();
        }
    }

    fn redo(&mut self) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return;
        }
        let changed = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.as_mut())
            .is_some_and(|mesh| self.history.redo(mesh));
        if changed {
            if let Some(document) = self.document.as_mut() {
                document.dirty = true;
            }
            self.upload_mesh();
            self.status = "Redo".to_owned();
        }
    }

    fn edit_mask(&mut self, invert: bool) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return;
        }
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let Some(mesh) = document.mesh.as_mut() else {
            return;
        };
        let changed = if invert {
            !mesh.mask.is_empty()
        } else {
            mesh.mask.iter().any(|value| *value != 0.0)
        };
        if !changed {
            return;
        }

        self.history.push_before(mesh);
        if invert {
            for value in &mut mesh.mask {
                *value = 1.0 - value.clamp(0.0, 1.0);
            }
            self.status = "Mask inverted".to_owned();
        } else {
            mesh.mask.fill(0.0);
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
            let actions_ready = self.background_task.is_none() && !self.sculpt.is_stroking();
            let mesh_ready = actions_ready
                && self
                    .document
                    .as_ref()
                    .is_some_and(|document| document.mesh.is_some());
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(actions_ready, egui::Button::new("Import STL"))
                    .clicked()
                {
                    self.request_action(PendingAction::OpenDialog, context);
                }
                if ui
                    .add_enabled(mesh_ready, egui::Button::new("Export As…"))
                    .clicked()
                {
                    self.export_as();
                }
                ui.separator();
                if ui
                    .add_enabled(
                        mesh_ready && self.history.can_undo(),
                        egui::Button::new("Undo"),
                    )
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add_enabled(
                        mesh_ready && self.history.can_redo(),
                        egui::Button::new("Redo"),
                    )
                    .clicked()
                {
                    self.redo();
                }
                ui.separator();
                if ui
                    .add_enabled(mesh_ready, egui::Button::new("Frame"))
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
                let mesh_ready = self.background_task.is_none()
                    && !self.sculpt.is_stroking()
                    && self
                        .document
                        .as_ref()
                        .is_some_and(|document| document.mesh.is_some());
                let adaptive_topology_available = self
                    .document
                    .as_ref()
                    .and_then(|document| document.mesh.as_ref())
                    .is_some_and(|mesh| adaptive_topology_available(mesh.triangles.len()));
                ui.add_enabled_ui(mesh_ready, |ui| {
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
                        if adaptive_topology_available {
                            ui.checkbox(&mut self.adaptive_topology, "Adaptive topology");
                            if self.adaptive_topology {
                                ui.label("Detail");
                                ui.add(
                                    egui::Slider::new(&mut self.brush.detail, 0.03..=0.35)
                                        .logarithmic(true),
                                );
                            }
                        } else {
                            let mut effective_off = false;
                            ui.add_enabled(
                                false,
                                egui::Checkbox::new(&mut effective_off, "Adaptive topology"),
                            );
                            ui.small(
                                "Unavailable above 250,000 faces to keep sculpting responsive.",
                            );
                        }
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
                    if let Some(mesh) = &document.mesh {
                        ui.label(format!("{} vertices", grouped(mesh.positions.len())));
                        ui.label(format!("{} faces", grouped(mesh.triangles.len())));
                        if let Some((minimum, maximum)) = mesh.bounds() {
                            let size = maximum - minimum;
                            ui.label(format!(
                                "Size: {:.2} × {:.2} × {:.2} units",
                                size.x, size.y, size.z
                            ));
                        }
                    } else {
                        ui.label("Topology optimization in progress");
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
                if let Some(task) = &self.background_task {
                    ui.spinner();
                    ui.small(task.progress_text());
                } else {
                    ui.small(&self.status);
                }
                if let Some(error) = &self.renderer_error {
                    ui.separator();
                    ui.colored_label(Color32::LIGHT_RED, format!("Renderer unavailable: {error}"));
                }
                if let Some(error) = &self.worker_error {
                    ui.separator();
                    ui.colored_label(Color32::LIGHT_RED, error);
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
                        .put(
                            button_rect,
                            egui::Button::new("Choose STL…").sense(
                                if self.background_task.is_none() {
                                    Sense::click()
                                } else {
                                    Sense::hover()
                                },
                            ),
                        )
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
                if self.background_task.is_none()
                    && response.drag_started_by(PointerButton::Primary)
                    && let Some(position) = pointer
                    && hover_hit.is_some()
                    && let Some(document) = self.document.as_ref()
                    && let Some(mesh) = document.mesh.as_ref()
                {
                    self.stroke_history_saved = self.history.push_before(mesh);
                    self.sculpt.begin_stroke(mesh);
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
                    let outcome = self.sculpt.end_stroke();
                    if outcome.changed {
                        if let Some(document) = self.document.as_mut() {
                            document.dirty = true;
                        }
                        if let Some(request) = outcome.remesh {
                            self.start_remesh(request, context);
                        } else {
                            self.status = if self.stroke_history_saved {
                                format!("{} stroke", self.tool.label())
                            } else {
                                format!("{} stroke (undo memory limit reached)", self.tool.label())
                            };
                        }
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
            .as_ref()?
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
            .and_then(|document| document.mesh.as_ref())
            .and_then(|mesh| mesh.raycast(ray.origin, ray.direction))
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
        let face_count = self
            .document
            .as_ref()
            .and_then(|document| document.mesh.as_ref())
            .map_or(0, |mesh| mesh.triangles.len());
        if effective_tool == SculptTool::Mask
            || !self.adaptive_topology
            || !adaptive_topology_available(face_count)
        {
            settings.detail = 0.0;
        }
        let sample = BrushSample::from_hit(&hit, world_drag, ray.direction, 1.0, modifiers.ctrl);

        let changed = self.document.as_mut().is_some_and(|document| {
            document.mesh.as_mut().is_some_and(|mesh| {
                self.sculpt
                    .apply_sample(mesh, effective_tool, &settings, sample)
            })
        });
        let updated_vertices = self.sculpt.take_updated_vertices();
        if changed {
            if let Some(document) = self.document.as_mut() {
                document.dirty = true;
            }
            self.upload_vertices_partial(&updated_vertices);
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
        self.poll_background_task();
        if context.input(|input| input.viewport().close_requested()) && !self.allow_close {
            if self
                .background_task
                .as_ref()
                .is_some_and(BackgroundTask::is_remesh)
            {
                context.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.status = "Wait for topology optimization before closing".to_owned();
            } else if self
                .document
                .as_ref()
                .is_some_and(|document| document.dirty)
            {
                context.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                if self.pending_action.is_none() {
                    self.pending_action = Some(PendingAction::Close);
                }
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
        if self.background_task.is_some() {
            context.request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "mesh operation panicked".to_owned()
    }
}

fn adaptive_topology_available(face_count: usize) -> bool {
    face_count <= MAX_ADAPTIVE_TOPOLOGY_FACES
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

    #[test]
    fn adaptive_topology_is_limited_to_moderate_meshes() {
        assert!(adaptive_topology_available(250_000));
        assert!(!adaptive_topology_available(250_001));
        assert!(!adaptive_topology_available(1_590_000));
    }
}
