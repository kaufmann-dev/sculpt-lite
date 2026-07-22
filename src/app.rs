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
    camera::{Camera, CameraFrame},
    history::{History, HistoryAction, HistoryEntry, LocalEdit, MaskChange, MeshSnapshot},
    mesh::{Mesh, MeshChangeSet, RayHit},
    renderer::{BrushCursor, MeshGpuPreparer, PreparedMeshUpload, ViewportRenderer},
    sculpt::{BrushSample, BrushSettings, SculptEngine, SculptTool, SymmetryAxis},
    stl::{ImportReport, load_stl, save_stl_atomic},
    stroke::{MAX_DABS_PER_FRAME, StrokeSampler},
};

#[derive(Clone, Copy)]
struct PointerHit {
    hit: RayHit,
    view_direction: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SculptInput {
    camera: CameraFrame,
    tool: SculptTool,
    brush: BrushSettings,
    adaptive_topology: bool,
    adaptive_detail: f32,
    radius_points: f32,
}

#[derive(Debug, Default)]
struct FrameRenderBatch {
    vertices: Vec<u32>,
    changes: MeshChangeSet,
    has_topology: bool,
    changed: bool,
}

impl FrameRenderBatch {
    fn clear(&mut self) {
        self.vertices.clear();
        self.changes.clear();
        self.has_topology = false;
        self.changed = false;
    }

    fn queue(&mut self, updated_vertices: Vec<u32>, mesh_changes: Option<MeshChangeSet>) {
        self.changed = true;
        if let Some(changes) = mesh_changes {
            if !self.has_topology {
                self.changes.include_vertices(self.vertices.drain(..));
                self.has_topology = true;
            }
            self.changes.merge(changes);
        } else if self.has_topology {
            self.changes.include_vertices(updated_vertices);
        } else {
            self.vertices.extend(updated_vertices);
        }
    }
}

const INITIAL_BRUSH_RADIUS_POINTS: f32 = 55.0;
const MIN_BRUSH_RADIUS_POINTS: f32 = 4.0;
const MAX_BRUSH_RADIUS_POINTS: f32 = 300.0;
const DEFAULT_AIRBRUSH_DABS_PER_SECOND: f32 = 10.0;
const MIN_AIRBRUSH_DABS_PER_SECOND: f32 = 2.0;
const MAX_AIRBRUSH_DABS_PER_SECOND: f32 = 30.0;
const BRUSH_SPACING_RADIUS_FRACTION: f32 = 0.15;
const DEFAULT_ADAPTIVE_DETAIL: f32 = 0.12;
const SCULPT_FRAME_BUDGET: Duration = Duration::from_millis(8);

fn adaptive_target_edge_length(radius: f32, detail: f32, units_per_point: f32) -> f32 {
    (radius * detail.clamp(0.03, 0.35)).max(units_per_point)
}

struct MeshDocument {
    mesh: Option<Mesh>,
    bounds: Option<MeshBounds>,
    source_path: PathBuf,
    report: ImportReport,
    dirty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MeshBounds {
    minimum: Vec3,
    maximum: Vec3,
    minimum_vertices: [u32; 3],
    maximum_vertices: [u32; 3],
    exact: bool,
}

impl MeshBounds {
    fn from_mesh(mesh: &Mesh) -> Option<Self> {
        let (&first, positions) = mesh.positions.split_first()?;
        if !first.is_finite() {
            return None;
        }
        let mut bounds = Self {
            minimum: first,
            maximum: first,
            minimum_vertices: [0; 3],
            maximum_vertices: [0; 3],
            exact: true,
        };
        for (index, &position) in positions.iter().enumerate() {
            if !position.is_finite() {
                continue;
            }
            bounds.include(index as u32 + 1, position);
        }
        Some(bounds)
    }

    fn include(&mut self, vertex: u32, position: Vec3) {
        for axis in 0..3 {
            if position[axis] < self.minimum[axis] {
                self.minimum[axis] = position[axis];
                self.minimum_vertices[axis] = vertex;
            }
            if position[axis] > self.maximum[axis] {
                self.maximum[axis] = position[axis];
                self.maximum_vertices[axis] = vertex;
            }
        }
    }

    fn update(&mut self, mesh: &Mesh, changed_vertices: &[u32]) {
        if self
            .minimum_vertices
            .iter()
            .chain(&self.maximum_vertices)
            .any(|&vertex| vertex as usize >= mesh.positions.len())
        {
            self.exact = false;
        }
        let invalidated = changed_vertices.iter().copied().any(|vertex| {
            let Some(&position) = mesh.positions.get(vertex as usize) else {
                return self.minimum_vertices.contains(&vertex)
                    || self.maximum_vertices.contains(&vertex);
            };
            if !position.is_finite() {
                return true;
            }
            (0..3).any(|axis| {
                (self.minimum_vertices[axis] == vertex && position[axis] > self.minimum[axis])
                    || (self.maximum_vertices[axis] == vertex
                        && position[axis] < self.maximum[axis])
            })
        });
        if invalidated {
            self.exact = false;
        }
        for &vertex in changed_vertices {
            if let Some(&position) = mesh.positions.get(vertex as usize)
                && position.is_finite()
            {
                self.include(vertex, position);
            }
        }
    }

    fn min_max(self) -> (Vec3, Vec3) {
        (self.minimum, self.maximum)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NavigationAction {
    Pan,
    Orbit,
}

fn navigation_action(button: PointerButton) -> Option<NavigationAction> {
    match button {
        PointerButton::Secondary => Some(NavigationAction::Pan),
        PointerButton::Middle => Some(NavigationAction::Orbit),
        _ => None,
    }
}

fn viewport_sense() -> Sense {
    Sense::drag()
}

fn effective_tool(tool: SculptTool, modifiers: Modifiers) -> SculptTool {
    if modifiers.shift && tool != SculptTool::Mask {
        SculptTool::Smooth
    } else {
        tool
    }
}

#[derive(Clone)]
enum PendingAction {
    OpenDialog,
    Import(PathBuf),
    Close,
}

enum WorkerJob {
    Import(PathBuf),
    Export { mesh: Box<Mesh>, path: PathBuf },
}

type ImportResult = Result<(Box<Mesh>, ImportReport, Option<Box<PreparedMeshUpload>>), String>;

enum WorkerResult {
    Import {
        path: PathBuf,
        result: ImportResult,
    },
    ExportCheckpoint(Arc<MeshSnapshot>),
    Export {
        path: PathBuf,
        mesh: Box<Mesh>,
        result: Result<(), String>,
    },
}

struct BackgroundWorker {
    sender: Sender<WorkerJob>,
    receiver: Receiver<WorkerResult>,
}

impl BackgroundWorker {
    fn start(
        context: egui::Context,
        mesh_preparer: Option<MeshGpuPreparer>,
    ) -> std::io::Result<Self> {
        let (job_sender, job_receiver) = mpsc::channel::<WorkerJob>();
        let (result_sender, result_receiver) = mpsc::channel::<WorkerResult>();
        thread::Builder::new()
            .name("sculptlite-mesh-worker".to_owned())
            .spawn(move || {
                while let Ok(job) = job_receiver.recv() {
                    let result = match job {
                        WorkerJob::Import(path) => {
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                let (mesh, report) =
                                    load_stl(&path).map_err(|error| error.to_string())?;
                                let upload = mesh_preparer
                                    .as_ref()
                                    .map(|preparer| Box::new(preparer.prepare_mesh(&mesh)));
                                Ok::<_, String>((Box::new(mesh), report, upload))
                            }))
                            .map_err(panic_message)
                            .and_then(|result| result);
                            WorkerResult::Import { path, result }
                        }
                        WorkerJob::Export { mesh, path } => {
                            let recovery = Arc::new(MeshSnapshot::capture(&mesh));
                            if result_sender
                                .send(WorkerResult::ExportCheckpoint(recovery))
                                .is_err()
                            {
                                break;
                            }
                            context.request_repaint();
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                save_stl_atomic(&path, &mesh).map_err(|error| error.to_string())
                            }))
                            .map_err(panic_message)
                            .and_then(|result| result);
                            WorkerResult::Export { path, mesh, result }
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
    Export {
        path: PathBuf,
        started: Instant,
        recovery: Option<Arc<MeshSnapshot>>,
        dirty_before: bool,
        after: Option<PendingAction>,
    },
}

impl BackgroundTask {
    fn owns_mesh(&self) -> bool {
        matches!(self, Self::Export { .. })
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
            Self::Export { path, started, .. } => {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("STL mesh");
                (format!("Exporting {name}"), started)
            }
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
    adaptive_detail: f32,
    airbrush: bool,
    airbrush_dabs_per_second: f32,
    brush_radius_points: f32,
    wireframe: bool,
    stroke_sampler: Option<StrokeSampler<SculptInput>>,
    frame_render: FrameRenderBatch,
    pending_action: Option<PendingAction>,
    worker: Option<BackgroundWorker>,
    background_task: Option<BackgroundTask>,
    worker_error: Option<String>,
    allow_close: bool,
    window_title: String,
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

        let mesh_preparer = renderer.as_ref().map(ViewportRenderer::mesh_preparer);
        let (worker, worker_error) =
            match BackgroundWorker::start(creation_context.egui_ctx.clone(), mesh_preparer) {
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
            adaptive_detail: DEFAULT_ADAPTIVE_DETAIL,
            airbrush: false,
            airbrush_dabs_per_second: DEFAULT_AIRBRUSH_DABS_PER_SECOND,
            brush_radius_points: INITIAL_BRUSH_RADIUS_POINTS,
            wireframe: false,
            stroke_sampler: None,
            frame_render: FrameRenderBatch::default(),
            pending_action: None,
            worker,
            background_task: None,
            worker_error,
            allow_close: false,
            window_title: "SculptLite".to_owned(),
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

    fn install_import(
        &mut self,
        path: PathBuf,
        mesh: Mesh,
        report: ImportReport,
        upload: Option<PreparedMeshUpload>,
    ) {
        let bounds = MeshBounds::from_mesh(&mesh);
        let (minimum, maximum) = bounds
            .map(MeshBounds::min_max)
            .unwrap_or((Vec3::splat(-1.0), Vec3::ONE));
        self.camera.fit(minimum, maximum);
        self.history.clear();
        self.sculpt.reset_for_mesh(&mesh);
        if let (Some(renderer), Some(upload)) = (&self.renderer, upload) {
            renderer.install_prepared_mesh(upload);
        }
        self.status = if report.has_topology_warnings() {
            format!("Loaded {} with protected topology regions", path.display())
        } else {
            format!("Loaded {}", path.display())
        };
        self.document = Some(MeshDocument {
            mesh: Some(mesh),
            bounds,
            source_path: path,
            report,
            dirty: false,
        });
        self.stroke_sampler = None;
        self.frame_render = FrameRenderBatch::default();
    }

    fn poll_background_task(&mut self, context: &egui::Context) {
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
                let message = self.background_task.take().map_or_else(
                    || "The mesh worker stopped".to_owned(),
                    |task| self.recover_background_task(task),
                );
                self.error = Some(message);
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
                    result: Ok((mesh, report, upload)),
                },
            ) => self.install_import(path, *mesh, report, upload.map(|upload| *upload)),
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
                BackgroundTask::Export {
                    path,
                    started,
                    dirty_before,
                    after,
                    ..
                },
                WorkerResult::ExportCheckpoint(recovery),
            ) => {
                self.background_task = Some(BackgroundTask::Export {
                    path,
                    started,
                    recovery: Some(recovery),
                    dirty_before,
                    after,
                });
            }
            (BackgroundTask::Export { after, .. }, WorkerResult::Export { path, mesh, result }) => {
                if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(*mesh);
                }
                match result {
                    Ok(()) => {
                        if let Some(document) = self.document.as_mut() {
                            document.dirty = false;
                        }
                        self.status = format!("Exported {}", path.display());
                        if let Some(action) = after {
                            self.perform_action(action, context);
                        }
                    }
                    Err(error) => {
                        self.pending_action = after;
                        self.error =
                            Some(format!("Could not export {}\n\n{error}", path.display()));
                    }
                }
            }
            (task, _) => {
                let recovery = self.recover_background_task(task);
                self.error = Some(format!(
                    "The mesh worker returned an unexpected result.\n\n{recovery}"
                ));
            }
        }
    }

    fn recover_background_task(&mut self, task: BackgroundTask) -> String {
        match task {
            BackgroundTask::Import { .. } => {
                "The mesh worker stopped before finishing the import".to_owned()
            }
            BackgroundTask::Export {
                recovery,
                dirty_before,
                after,
                ..
            } => {
                if let Some(recovery) = recovery {
                    self.restore_recovery(&recovery, dirty_before);
                }
                self.pending_action = after;
                "The export stopped; the editable mesh was restored".to_owned()
            }
        }
    }

    fn restore_recovery(&mut self, recovery: &MeshSnapshot, dirty: bool) {
        let mut mesh = Mesh::default();
        recovery.restore(&mut mesh);
        if let Some(document) = self.document.as_mut() {
            document.mesh = Some(mesh);
            document.bounds = document.mesh.as_ref().and_then(MeshBounds::from_mesh);
            document.dirty = dirty;
        }
        self.upload_mesh();
    }

    fn export_as(&mut self, after: Option<PendingAction>, context: &egui::Context) -> bool {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return false;
        }
        let Some(document) = self.document.as_ref() else {
            return false;
        };
        if document.mesh.is_none() {
            return false;
        }

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

        self.start_export(path, after, context)
    }

    fn start_export(
        &mut self,
        path: PathBuf,
        after: Option<PendingAction>,
        context: &egui::Context,
    ) -> bool {
        let dirty_before = self
            .document
            .as_ref()
            .is_some_and(|document| document.dirty);
        let Some(mesh) = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.take())
        else {
            return false;
        };
        let Some(worker) = &self.worker else {
            if let Some(document) = self.document.as_mut() {
                document.mesh = Some(mesh);
            }
            self.error = Some("The mesh worker is unavailable".to_owned());
            return false;
        };
        match worker.sender.send(WorkerJob::Export {
            mesh: Box::new(mesh),
            path: path.clone(),
        }) {
            Ok(()) => {
                self.background_task = Some(BackgroundTask::Export {
                    path,
                    started: Instant::now(),
                    recovery: None,
                    dirty_before,
                    after,
                });
                context.request_repaint();
                true
            }
            Err(error) => {
                let WorkerJob::Export { mesh, .. } = error.0 else {
                    unreachable!("the submitted worker job remains an export")
                };
                if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(*mesh);
                }
                self.worker_error = Some("The mesh worker stopped unexpectedly".to_owned());
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

    fn upload_mesh_partial(&self, changes: &MeshChangeSet) {
        if let (Some(renderer), Some(mesh)) = (
            &self.renderer,
            self.document
                .as_ref()
                .and_then(|document| document.mesh.as_ref()),
        ) {
            renderer.update_mesh_partial(mesh, changes);
        }
    }

    fn begin_frame_render_batch(&mut self) {
        self.frame_render.clear();
    }

    fn queue_frame_render_update(
        &mut self,
        updated_vertices: Vec<u32>,
        mesh_changes: Option<MeshChangeSet>,
    ) {
        self.frame_render.queue(updated_vertices, mesh_changes);
    }

    fn flush_frame_render_batch(&mut self, context: &egui::Context) {
        if !self.frame_render.changed {
            return;
        }
        if self.frame_render.has_topology {
            let counts = self
                .document
                .as_ref()
                .and_then(|document| document.mesh.as_ref())
                .map(|mesh| (mesh.positions.len(), mesh.triangles.len()));
            if let Some((vertex_count, face_count)) = counts {
                self.frame_render.changes.finalize(vertex_count, face_count);
                self.upload_mesh_partial(&self.frame_render.changes);
            }
        } else {
            self.frame_render.vertices.sort_unstable();
            self.frame_render.vertices.dedup();
            self.upload_vertices_partial(&self.frame_render.vertices);
        }
        context.request_repaint();
    }

    fn frame_mesh(&mut self) {
        let bounds = self
            .document
            .as_ref()
            .and_then(|document| document.mesh.as_ref())
            .and_then(MeshBounds::from_mesh);
        if let Some(document) = self.document.as_mut() {
            document.bounds = bounds;
        }
        if let Some((minimum, maximum)) = bounds.map(MeshBounds::min_max) {
            self.camera.fit(minimum, maximum);
        }
    }

    fn refresh_document_bounds(&mut self) {
        if let Some(document) = self.document.as_mut() {
            document.bounds = document.mesh.as_ref().and_then(MeshBounds::from_mesh);
        }
    }

    fn update_document_bounds(&mut self, changed_vertices: &[u32]) {
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let Some(mesh) = document.mesh.as_ref() else {
            document.bounds = None;
            return;
        };
        if let Some(bounds) = document.bounds.as_mut() {
            bounds.update(mesh, changed_vertices);
        } else {
            document.bounds = MeshBounds::from_mesh(mesh);
        }
    }

    fn undo(&mut self, _context: &egui::Context) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return;
        }
        let action = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.as_mut())
            .map_or(HistoryAction::Empty, |mesh| self.history.undo(mesh));
        match action {
            HistoryAction::Empty => {}
            HistoryAction::Local { changed_vertices } => {
                if let Some(document) = self.document.as_mut() {
                    document.dirty = true;
                }
                self.upload_vertices_partial(&changed_vertices);
                self.refresh_document_bounds();
                self.status = "Undo".to_owned();
            }
            HistoryAction::Topology { changes } => {
                if let Some(document) = self.document.as_mut() {
                    document.dirty = true;
                }
                self.upload_mesh_partial(&changes);
                self.refresh_document_bounds();
                self.status = "Undo".to_owned();
            }
        }
    }

    fn redo(&mut self, _context: &egui::Context) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return;
        }
        let action = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.as_mut())
            .map_or(HistoryAction::Empty, |mesh| self.history.redo(mesh));
        match action {
            HistoryAction::Empty => {}
            HistoryAction::Local { changed_vertices } => {
                if let Some(document) = self.document.as_mut() {
                    document.dirty = true;
                }
                self.upload_vertices_partial(&changed_vertices);
                self.refresh_document_bounds();
                self.status = "Redo".to_owned();
            }
            HistoryAction::Topology { changes } => {
                if let Some(document) = self.document.as_mut() {
                    document.dirty = true;
                }
                self.upload_mesh_partial(&changes);
                self.refresh_document_bounds();
                self.status = "Redo".to_owned();
            }
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
        let mut changes = Vec::new();
        for (vertex, value) in mesh.mask.iter_mut().enumerate() {
            let before = *value;
            let after = if invert {
                1.0 - before.clamp(0.0, 1.0)
            } else {
                0.0
            };
            if before == after {
                continue;
            }
            *value = after;
            changes.push(MaskChange {
                vertex: vertex as u32,
                before,
                after,
            });
        }
        let edit = LocalEdit::new(Vec::new(), changes);
        if edit.is_empty() {
            return;
        }
        let changed_vertices = edit
            .masks
            .iter()
            .map(|change| change.vertex)
            .collect::<Vec<_>>();
        let history_saved = self.history.record(HistoryEntry::Local(edit));
        self.status = format!(
            "{}{}",
            if invert {
                "Mask inverted"
            } else {
                "Mask cleared"
            },
            if history_saved {
                ""
            } else {
                " (undo memory limit reached)"
            }
        );
        document.dirty = true;
        self.upload_vertices_partial(&changed_vertices);
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
            self.export_as(None, context);
        }
        if undo {
            self.undo(context);
        }
        if redo {
            self.redo(context);
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
                    self.export_as(None, context);
                }
                ui.separator();
                if ui
                    .add_enabled(
                        mesh_ready && self.history.can_undo(),
                        egui::Button::new("Undo"),
                    )
                    .clicked()
                {
                    self.undo(context);
                }
                if ui
                    .add_enabled(
                        mesh_ready && self.history.can_redo(),
                        egui::Button::new("Redo"),
                    )
                    .clicked()
                {
                    self.redo(context);
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
                    ui.add_enabled_ui(self.tool != SculptTool::Grab, |ui| {
                        ui.checkbox(&mut self.airbrush, "Airbrush");
                        if self.airbrush {
                            ui.label("Rate");
                            ui.add(
                                egui::Slider::new(
                                    &mut self.airbrush_dabs_per_second,
                                    MIN_AIRBRUSH_DABS_PER_SECOND..=MAX_AIRBRUSH_DABS_PER_SECOND,
                                )
                                .suffix(" dabs/s"),
                            );
                        }
                    });
                    ui.add_enabled_ui(self.tool != SculptTool::Mask, |ui| {
                        ui.checkbox(&mut self.adaptive_topology, "Adaptive topology");
                        if self.adaptive_topology {
                            ui.small("Updates topology continuously inside the brush region.");
                            ui.label("Detail");
                            ui.add(
                                egui::Slider::new(&mut self.adaptive_detail, 0.03..=0.35)
                                    .logarithmic(true),
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
                        ui.label(format!(
                            "{} vertices · {} faces",
                            grouped(mesh.positions.len()),
                            grouped(mesh.triangles.len())
                        ));
                        if let Some((minimum, maximum)) = document.bounds.map(MeshBounds::min_max) {
                            let size = maximum - minimum;
                            ui.label(format!("{:.1} × {:.1} × {:.1}", size.x, size.y, size.z));
                        }
                    } else {
                        ui.label("Mesh operation in progress");
                    }
                    egui::CollapsingHeader::new("Import details")
                        .default_open(false)
                        .show(ui, |ui| {
                            let report = document.report;
                            let removed = report.removed_invalid_faces
                                + report.removed_degenerate_faces
                                + report.removed_duplicate_faces;
                            let topology =
                                if report.boundary_edges == 0 && report.non_manifold_edges == 0 {
                                    "Closed · manifold".to_owned()
                                } else {
                                    format!(
                                        "{} boundary · {} non-manifold",
                                        grouped(report.boundary_edges),
                                        grouped(report.non_manifold_edges)
                                    )
                                };
                            egui::Grid::new("import_details_grid")
                                .num_columns(2)
                                .spacing([10.0, 3.0])
                                .show(ui, |ui| {
                                    ui.small("Source");
                                    ui.small(format!(
                                        "{} triangles",
                                        grouped(report.source_triangles)
                                    ));
                                    ui.end_row();
                                    ui.small("Welded");
                                    ui.small(format!(
                                        "{} vertices",
                                        grouped(report.welded_vertices)
                                    ));
                                    ui.end_row();
                                    ui.small("Removed");
                                    ui.small(format!("{} faces", grouped(removed)));
                                    ui.end_row();
                                    ui.small("Topology");
                                    ui.small(topology);
                                    ui.end_row();
                                });
                        });
                } else {
                    ui.label("No mesh loaded");
                }

                ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                    let mib = self.history.bytes_used() as f64 / (1024.0 * 1024.0);
                    let budget_mib = self.history.byte_budget() as f64 / (1024.0 * 1024.0);
                    ui.small(format!("Undo: {mib:.1} / {budget_mib:.0} MiB"));
                    ui.small("Shift: smooth · Ctrl: invert");
                    ui.small("RMB: pan · MMB: orbit · Wheel: zoom");
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
        self.begin_frame_render_batch();
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_rgb(25, 27, 31)))
            .show(root_ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(rect, viewport_sense());
                if let Some(sampler) = self.stroke_sampler.as_mut() {
                    sampler.advance_to(Instant::now());
                }

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
                let pointer_delta = context.input(|input| input.pointer.delta());
                let navigation = if response.dragged_by(PointerButton::Secondary) {
                    navigation_action(PointerButton::Secondary)
                } else if response.dragged_by(PointerButton::Middle) {
                    navigation_action(PointerButton::Middle)
                } else {
                    None
                };
                match navigation {
                    Some(NavigationAction::Pan) => {
                        self.camera.pan(pointer_delta, rect.height());
                    }
                    Some(NavigationAction::Orbit) => self.camera.orbit(pointer_delta),
                    None => {}
                }
                if response.hovered() {
                    let scroll = context.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll);
                    }
                }

                let Some(camera_frame) = self.camera.frame(rect) else {
                    return;
                };
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
                    renderer.update_camera(camera_frame);
                    renderer.paint(ui, rect, cursor, self.wireframe);
                } else {
                    ui.painter()
                        .rect_filled(rect, 0.0, Color32::from_rgb(20, 22, 25));
                }

                let mut began_stroke = false;
                if self.background_task.is_none()
                    && response.drag_started_by(PointerButton::Primary)
                    && let Some(position) = pointer
                    && let Some(pointer_hit) = self.hit_at(position, camera_frame)
                    && let Some(document) = self.document.as_ref()
                    && let Some(mesh) = document.mesh.as_ref()
                {
                    self.sculpt.begin_stroke(mesh);
                    let modifiers = context.input(|input| input.modifiers);
                    let sculpt_input = self.sculpt_input(camera_frame);
                    self.stroke_sampler = Some(StrokeSampler::begin(
                        position,
                        modifiers,
                        self.brush_spacing(),
                        sculpt_input,
                        Instant::now(),
                    ));
                    let effective_tool = effective_tool(sculpt_input.tool, modifiers);
                    if effective_tool != SculptTool::Grab {
                        let initial_dab = self
                            .stroke_sampler
                            .as_mut()
                            .and_then(StrokeSampler::take_initial_dab);
                        if let Some(initial_dab) = initial_dab {
                            self.apply_pointer_sample(
                                initial_dab.context,
                                initial_dab.position,
                                Vec2::ZERO,
                                initial_dab.modifiers,
                                Some(pointer_hit),
                            );
                        }
                    }
                    began_stroke = true;
                }

                if !began_stroke && self.sculpt.is_stroking() {
                    let stopped = response.drag_stopped_by(PointerButton::Primary);
                    let primary_down =
                        context.input(|input| input.pointer.button_down(PointerButton::Primary));
                    let modifiers = context.input(|input| input.modifiers);
                    let sculpt_input = self.sculpt_input(camera_frame);
                    let brush_spacing = self.brush_spacing();
                    let effective_tool = effective_tool(sculpt_input.tool, modifiers);
                    if (primary_down || stopped)
                        && let Some(position) = pointer
                    {
                        if effective_tool == SculptTool::Grab {
                            let delta = self
                                .stroke_sampler
                                .as_mut()
                                .and_then(|sampler| sampler.consume_grab_delta(position))
                                .unwrap_or(Vec2::ZERO);
                            if delta.length_sq() > f32::EPSILON {
                                self.apply_pointer_sample(
                                    sculpt_input,
                                    position,
                                    delta,
                                    modifiers,
                                    None,
                                );
                                if let Some(sampler) = self.stroke_sampler.as_mut() {
                                    sampler.record_spatial_dab();
                                }
                            }
                        } else if let Some(sampler) = self.stroke_sampler.as_mut() {
                            sampler.enqueue_pointer(
                                position,
                                modifiers,
                                brush_spacing,
                                sculpt_input,
                            );
                        }
                    }
                    if stopped && let Some(sampler) = self.stroke_sampler.as_mut() {
                        sampler.release();
                    }

                    self.apply_spatial_dabs();
                    if effective_tool != SculptTool::Grab {
                        self.apply_airbrush_dab();
                    }

                    let finished = self.stroke_sampler.as_ref().is_some_and(|sampler| {
                        sampler.is_released()
                            && !sampler.has_pending_path()
                            && !self.sculpt.has_pending_sample()
                    });
                    if finished {
                        self.finish_pointer_stroke();
                    }
                }
            });
        self.flush_frame_render_batch(context);
    }

    fn hit_at(&self, pointer: Pos2, camera: CameraFrame) -> Option<PointerHit> {
        let ray = camera.screen_ray(pointer)?;
        let hit = self
            .document
            .as_ref()?
            .mesh
            .as_ref()?
            .raycast(ray.origin, ray.direction)?;
        Some(PointerHit {
            hit,
            view_direction: ray.direction,
        })
    }

    fn sculpt_input(&self, camera: CameraFrame) -> SculptInput {
        SculptInput {
            camera,
            tool: self.tool,
            brush: self.brush,
            adaptive_topology: self.adaptive_topology,
            adaptive_detail: self.adaptive_detail,
            radius_points: self.brush_radius_points,
        }
    }

    fn brush_spacing(&self) -> f32 {
        (self.brush_radius_points * BRUSH_SPACING_RADIUS_FRACTION).max(1.0)
    }

    fn apply_spatial_dabs(&mut self) {
        let started = Instant::now();
        if self.sculpt.has_pending_sample() {
            self.continue_pending_sculpt_sample();
            if self.sculpt.has_pending_sample() || started.elapsed() >= SCULPT_FRAME_BUDGET {
                return;
            }
        }
        let mut processed = 0;
        while processed < MAX_DABS_PER_FRAME {
            let dab = self
                .stroke_sampler
                .as_mut()
                .and_then(StrokeSampler::next_spatial_dab);
            let Some(dab) = dab else {
                break;
            };
            self.apply_pointer_sample(dab.context, dab.position, Vec2::ZERO, dab.modifiers, None);
            processed += 1;
            if self.sculpt.has_pending_sample() || started.elapsed() >= SCULPT_FRAME_BUDGET {
                break;
            }
        }
        if processed != 0
            && let Some(sampler) = self.stroke_sampler.as_mut()
        {
            sampler.record_spatial_dab();
        }
    }

    fn apply_airbrush_dab(&mut self) {
        if !self.airbrush || self.tool == SculptTool::Grab || self.sculpt.has_pending_sample() {
            return;
        }
        let interval = self.airbrush_interval();
        let dab = self.stroke_sampler.as_ref().and_then(|sampler| {
            (!sampler.is_released() && sampler.airbrush_due(interval)).then_some(sampler.pointer())
        });
        let Some(dab) = dab else {
            return;
        };
        self.apply_pointer_sample(dab.context, dab.position, Vec2::ZERO, dab.modifiers, None);
        if let Some(sampler) = self.stroke_sampler.as_mut() {
            sampler.record_airbrush_dab();
        }
    }

    fn airbrush_interval(&self) -> Duration {
        Duration::from_secs_f32(
            self.airbrush_dabs_per_second
                .clamp(MIN_AIRBRUSH_DABS_PER_SECOND, MAX_AIRBRUSH_DABS_PER_SECOND)
                .recip(),
        )
    }

    fn finish_pointer_stroke(&mut self) {
        let outcome = self
            .document
            .as_ref()
            .and_then(|document| document.mesh.as_ref())
            .map(|mesh| self.sculpt.end_stroke(mesh))
            .unwrap_or_default();
        let history_entry = outcome
            .topology
            .map(|edit| HistoryEntry::Topology(Arc::new(edit)))
            .or_else(|| (!outcome.edit.is_empty()).then_some(HistoryEntry::Local(outcome.edit)));
        if let Some(history_entry) = history_entry {
            if let Some(document) = self.document.as_mut() {
                document.dirty = true;
            }
            let history_saved = self.history.record(history_entry);
            self.status = if history_saved {
                format!("{} stroke", self.tool.label())
            } else {
                format!("{} stroke (undo memory limit reached)", self.tool.label())
            };
        }
        if self
            .document
            .as_ref()
            .and_then(|document| document.bounds)
            .is_some_and(|bounds| !bounds.exact)
        {
            self.refresh_document_bounds();
        }
        self.stroke_sampler = None;
    }

    fn apply_pointer_sample(
        &mut self,
        input: SculptInput,
        pointer: Pos2,
        pointer_delta: Vec2,
        modifiers: Modifiers,
        pointer_hit: Option<PointerHit>,
    ) {
        let Some(pointer_hit) = pointer_hit.or_else(|| self.hit_at(pointer, input.camera)) else {
            return;
        };
        let hit = pointer_hit.hit;

        let depth_scale = (hit.distance / input.camera.distance().max(1.0e-6)).max(0.05);
        let units_per_point = input.camera.world_units_per_point() * depth_scale;
        let world_drag = input.camera.right() * pointer_delta.x * units_per_point
            - input.camera.up() * pointer_delta.y * units_per_point;
        let effective_tool = effective_tool(input.tool, modifiers);
        let mut settings = input.brush;
        settings.radius = input.radius_points * units_per_point;
        if effective_tool == SculptTool::Mask || !input.adaptive_topology {
            settings.remesh_target_edge_length = None;
        } else {
            settings.remesh_target_edge_length = Some(adaptive_target_edge_length(
                settings.radius,
                input.adaptive_detail,
                units_per_point,
            ));
        }
        let sample = BrushSample::from_hit(
            &hit,
            world_drag,
            pointer_hit.view_direction,
            1.0,
            modifiers.ctrl,
        );

        let changed = self.document.as_mut().is_some_and(|document| {
            document.mesh.as_mut().is_some_and(|mesh| {
                self.sculpt
                    .apply_sample(mesh, effective_tool, &settings, sample)
            })
        });
        self.consume_sculpt_step(changed, effective_tool != SculptTool::Mask);
    }

    fn continue_pending_sculpt_sample(&mut self) {
        let changed = self.document.as_mut().is_some_and(|document| {
            document
                .mesh
                .as_mut()
                .is_some_and(|mesh| self.sculpt.continue_pending_sample(mesh))
        });
        self.consume_sculpt_step(changed, true);
    }

    fn consume_sculpt_step(&mut self, changed: bool, geometry_changed: bool) {
        let committed = self.sculpt.take_sample_committed();
        let updated_vertices = self.sculpt.take_updated_vertices();
        let mesh_changes = self.sculpt.take_mesh_changes();
        if let Some(error) = self.sculpt.take_error() {
            self.error = Some(error);
        }
        if let Some(warning) = self.sculpt.take_warning() {
            self.status = warning;
        }
        if committed && let Some(document) = self.document.as_mut() {
            document.dirty = true;
        }
        if changed {
            if geometry_changed {
                self.update_document_bounds(&updated_vertices);
            }
            self.queue_frame_render_update(updated_vertices, mesh_changes);
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
                if ui.button("Export As…").clicked()
                    && self.export_as(Some(pending.clone()), context)
                {
                    self.pending_action = None;
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
        self.poll_background_task(&context);
        if context.input(|input| input.viewport().close_requested()) && !self.allow_close {
            if self
                .background_task
                .as_ref()
                .is_some_and(BackgroundTask::owns_mesh)
            {
                context.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.status = "Wait for the current mesh operation before closing".to_owned();
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
        if title != self.window_title {
            context.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }

        self.top_bar(root_ui, &context);
        self.status_bar(root_ui);
        self.tool_panel(root_ui);
        self.viewport(root_ui, &context);
        if self.error.is_some() {
            self.error_prompt(&context);
        } else {
            self.unsaved_prompt(&context);
        }

        if let Some(sampler) = &self.stroke_sampler {
            if sampler.has_pending_path() || self.sculpt.has_pending_sample() {
                context.request_repaint();
            } else if self.airbrush && self.tool != SculptTool::Grab && !sampler.is_released() {
                context.request_repaint_after(sampler.airbrush_wait(self.airbrush_interval()));
            }
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
    fn adaptive_target_never_creates_subpoint_edges() {
        let units_per_point = 0.25;
        let radius = MIN_BRUSH_RADIUS_POINTS * units_per_point;

        assert_eq!(
            adaptive_target_edge_length(radius, 0.03, units_per_point),
            units_per_point
        );
        assert_eq!(
            adaptive_target_edge_length(radius * 100.0, 0.12, units_per_point),
            radius * 12.0
        );
    }

    #[test]
    fn grouped_formats_face_counts() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn navigation_uses_right_drag_to_pan_and_middle_drag_to_orbit() {
        assert_eq!(
            navigation_action(PointerButton::Secondary),
            Some(NavigationAction::Pan)
        );
        assert_eq!(
            navigation_action(PointerButton::Middle),
            Some(NavigationAction::Orbit)
        );
        assert_eq!(navigation_action(PointerButton::Primary), None);
    }

    #[test]
    fn viewport_drag_sense_has_no_click_threshold_delay() {
        let sense = viewport_sense();

        assert!(sense.senses_drag());
        assert!(!sense.senses_click());
    }

    #[test]
    fn frame_render_batch_preserves_fixed_edits_around_topology_edits() {
        let mut batch = FrameRenderBatch::default();
        batch.queue(vec![1, 2], None);
        let mut topology = MeshChangeSet::default();
        topology.dirty_vertices = vec![3, 4];
        topology.dirty_faces = vec![7];
        topology.vertex_count = 12;
        topology.face_count = 9;
        batch.queue(vec![3], Some(topology));
        batch.queue(vec![5, 2], None);
        batch.changes.finalize(12, 9);

        assert!(batch.changed);
        assert!(batch.has_topology);
        assert_eq!(batch.changes.dirty_vertices, [1, 2, 3, 4, 5]);
        assert_eq!(batch.changes.dirty_faces, [7]);
        assert!(batch.vertices.is_empty());
    }

    #[test]
    fn queued_stroke_context_is_unchanged_by_later_camera_navigation() {
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut camera = Camera::default();
        let original = SculptInput {
            camera: camera.frame(viewport).unwrap(),
            tool: SculptTool::Draw,
            brush: BrushSettings::default(),
            adaptive_topology: true,
            adaptive_detail: DEFAULT_ADAPTIVE_DETAIL,
            radius_points: 55.0,
        };
        let mut sampler =
            StrokeSampler::begin(Pos2::ZERO, Modifiers::NONE, 10.0, original, Instant::now());
        sampler.enqueue_pointer(Pos2::new(20.0, 0.0), Modifiers::NONE, 10.0, original);

        camera.orbit(Vec2::new(80.0, 20.0));
        let navigated = camera.frame(viewport).unwrap();
        let queued = sampler.next_spatial_dab().unwrap();

        assert_eq!(queued.context, original);
        assert_ne!(queued.context.camera, navigated);
    }

    #[test]
    fn mesh_bounds_update_locally_and_rebuild_when_an_extreme_moves_inward() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::new(-2.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
            vec![[0, 2, 3], [1, 3, 2], [0, 3, 1], [0, 1, 2]],
        )
        .unwrap();
        let mut bounds = MeshBounds::from_mesh(&mesh).unwrap();

        mesh.positions[3].z = 3.0;
        bounds.update(&mesh, &[3]);
        assert!(bounds.exact);
        assert_eq!(bounds.maximum.z, 3.0);

        mesh.positions[1].x = 0.5;
        bounds.update(&mesh, &[1]);
        assert!(!bounds.exact);
        bounds = MeshBounds::from_mesh(&mesh).unwrap();
        assert_eq!(bounds.min_max(), mesh.bounds().unwrap());
    }
}
