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
    self, Align, Align2, Color32, Key, KeyboardShortcut, Layout, Modifiers, PointerButton, Pos2,
    Rect, RichText, Sense, Vec2,
};
use glam::Vec3;

use crate::{
    camera::{Camera, CameraFrame, CameraMode, FlyMovement, FlyMovementMode},
    history::{
        History, HistoryAction, HistoryDirection, HistoryEntry, LocalEdit, MaskChange,
        MeshSnapshot, PendingMeshReplacement,
    },
    mesh::{Mesh, RayHit},
    renderer::{BrushCursor, MeshGpuPreparer, PreparedMeshUpload, ViewportRenderer},
    sculpt::{
        Accumulation, BrushSample, BrushSettings, DabKind, SculptEngine, SculptTool, SymmetryAxis,
    },
    stl::{ImportReport, load_stl, save_stl_atomic},
    stroke::{MAX_DABS_PER_FRAME, StrokeSampler},
    voxel_remesh::{
        DEFAULT_REMESH_RESOLUTION, MAX_REMESH_RESOLUTION, MIN_REMESH_RESOLUTION, VoxelRemeshOutput,
        VoxelRemeshSettings, source_eligibility, voxel_remesh,
    },
};

#[derive(Clone, Copy)]
struct PointerHit {
    hit: RayHit,
    view_direction: Vec3,
}

#[derive(Clone, Copy)]
struct PointerDab {
    position: Pos2,
    delta: Vec2,
    pressure: f32,
    modifiers: Modifiers,
    hit: Option<PointerHit>,
    kind: DabKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SculptInput {
    camera: CameraFrame,
    tool: SculptTool,
    brush: BrushSettings,
    radius_points: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StrokeMethod {
    Spaced,
    Airbrush,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StrokeBehavior {
    method: StrokeMethod,
    accumulation: Accumulation,
}

impl StrokeBehavior {
    const CAPPED: Self = Self {
        method: StrokeMethod::Spaced,
        accumulation: Accumulation::Capped,
    };
    const ACCUMULATE: Self = Self {
        method: StrokeMethod::Spaced,
        accumulation: Accumulation::Accumulate,
    };
    const AIRBRUSH: Self = Self {
        method: StrokeMethod::Airbrush,
        accumulation: Accumulation::Accumulate,
    };
    const ALL: [Self; 3] = [Self::CAPPED, Self::ACCUMULATE, Self::AIRBRUSH];

    const fn label(self) -> &'static str {
        match self {
            Self::CAPPED => "Capped",
            Self::ACCUMULATE => "Accumulate",
            Self::AIRBRUSH => "Airbrush",
            _ => unreachable!(),
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::CAPPED => {
                "Limit each vertex to the strongest influence reached during one held stroke."
            }
            Self::ACCUMULATE => "Build up the effect with every distance-spaced dab while moving.",
            Self::AIRBRUSH => {
                "Build up while moving and emit timed dabs while the pointer is held still."
            }
            _ => unreachable!(),
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::CAPPED => Self::ACCUMULATE,
            Self::ACCUMULATE => Self::AIRBRUSH,
            Self::AIRBRUSH => Self::CAPPED,
            _ => Self::CAPPED,
        }
    }
}

impl Default for StrokeBehavior {
    fn default() -> Self {
        Self::CAPPED
    }
}

#[derive(Clone, Debug)]
struct ToolStrokeBehaviors {
    values: [StrokeBehavior; 9],
}

impl ToolStrokeBehaviors {
    fn get(&self, tool: SculptTool) -> StrokeBehavior {
        self.values[Self::index(tool)]
    }

    fn set(&mut self, tool: SculptTool, behavior: StrokeBehavior) {
        self.values[Self::index(tool)] = behavior;
    }

    const fn index(tool: SculptTool) -> usize {
        match tool {
            SculptTool::Grab => 0,
            SculptTool::Draw => 1,
            SculptTool::Clay => 2,
            SculptTool::Crease => 3,
            SculptTool::Inflate => 4,
            SculptTool::Smooth => 5,
            SculptTool::Pinch => 6,
            SculptTool::Flatten => 7,
            SculptTool::Mask => 8,
        }
    }
}

impl Default for ToolStrokeBehaviors {
    fn default() -> Self {
        Self {
            values: [StrokeBehavior::CAPPED; 9],
        }
    }
}

#[derive(Debug, Default)]
struct FrameRenderBatch {
    vertices: Vec<u32>,
    changed: bool,
}

impl FrameRenderBatch {
    fn clear(&mut self) {
        self.vertices.clear();
        self.changed = false;
    }

    fn queue(&mut self, updated_vertices: Vec<u32>) {
        self.changed = true;
        self.vertices.extend(updated_vertices);
    }
}

const INITIAL_BRUSH_RADIUS_POINTS: f32 = 55.0;
const MIN_BRUSH_RADIUS_POINTS: f32 = 4.0;
const MAX_BRUSH_RADIUS_POINTS: f32 = 300.0;
const DEFAULT_AIRBRUSH_DABS_PER_SECOND: f32 = 10.0;
const MIN_AIRBRUSH_DABS_PER_SECOND: f32 = 2.0;
const MAX_AIRBRUSH_DABS_PER_SECOND: f32 = 30.0;
const BRUSH_SPACING_RADIUS_FRACTION: f32 = 0.15;
const MOUSE_PRESSURE: f32 = 1.0;
const SCULPT_FRAME_BUDGET: Duration = Duration::from_millis(8);
const BRUSH_VALUE_STEP: f32 = 0.05;
const WHEEL_POINTS_PER_LINE: f32 = 40.0;
const MAX_FLY_WHEEL_POINTS: f32 = 240.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShortcutDirection {
    Decrease,
    Increase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShortcutAction {
    Open,
    Export,
    Undo,
    Redo,
    Frame,
    ToggleCameraMode,
    ToggleWireframe,
    SelectTool(SculptTool),
    AdjustRadius(ShortcutDirection),
    AdjustStrength(ShortcutDirection),
    AdjustHardness(ShortcutDirection),
    CycleStrokeApplication,
    ToggleBrushInvert,
    SetSymmetry(Option<SymmetryAxis>),
    ClearMask,
    InvertMask,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShortcutBinding {
    shortcut: KeyboardShortcut,
    action: ShortcutAction,
}

const fn shortcut(modifiers: Modifiers, key: Key, action: ShortcutAction) -> ShortcutBinding {
    ShortcutBinding {
        shortcut: KeyboardShortcut::new(modifiers, key),
        action,
    }
}

const SHORTCUT_BINDINGS: [ShortcutBinding; 31] = [
    // Put shortcuts with required Shift modifiers before their less-specific
    // variants because egui permits extra Shift for logical key matching.
    shortcut(
        Modifiers::CTRL.plus(Modifiers::SHIFT),
        Key::S,
        ShortcutAction::Export,
    ),
    shortcut(
        Modifiers::CTRL.plus(Modifiers::SHIFT),
        Key::Z,
        ShortcutAction::Redo,
    ),
    shortcut(Modifiers::CTRL, Key::O, ShortcutAction::Open),
    shortcut(Modifiers::CTRL, Key::Z, ShortcutAction::Undo),
    shortcut(Modifiers::CTRL, Key::Y, ShortcutAction::Redo),
    shortcut(Modifiers::CTRL, Key::Backspace, ShortcutAction::ClearMask),
    shortcut(Modifiers::CTRL, Key::I, ShortcutAction::InvertMask),
    shortcut(Modifiers::NONE, Key::F, ShortcutAction::Frame),
    shortcut(Modifiers::NONE, Key::V, ShortcutAction::ToggleCameraMode),
    shortcut(Modifiers::NONE, Key::W, ShortcutAction::ToggleWireframe),
    shortcut(
        Modifiers::NONE,
        Key::Num1,
        ShortcutAction::SelectTool(SculptTool::Draw),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num2,
        ShortcutAction::SelectTool(SculptTool::Clay),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num3,
        ShortcutAction::SelectTool(SculptTool::Crease),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num4,
        ShortcutAction::SelectTool(SculptTool::Inflate),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num5,
        ShortcutAction::SelectTool(SculptTool::Smooth),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num6,
        ShortcutAction::SelectTool(SculptTool::Pinch),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num7,
        ShortcutAction::SelectTool(SculptTool::Flatten),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num8,
        ShortcutAction::SelectTool(SculptTool::Grab),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Num9,
        ShortcutAction::SelectTool(SculptTool::Mask),
    ),
    shortcut(
        Modifiers::NONE,
        Key::OpenBracket,
        ShortcutAction::AdjustRadius(ShortcutDirection::Decrease),
    ),
    shortcut(
        Modifiers::NONE,
        Key::CloseBracket,
        ShortcutAction::AdjustRadius(ShortcutDirection::Increase),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Minus,
        ShortcutAction::AdjustStrength(ShortcutDirection::Decrease),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Equals,
        ShortcutAction::AdjustStrength(ShortcutDirection::Increase),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Comma,
        ShortcutAction::AdjustHardness(ShortcutDirection::Decrease),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Period,
        ShortcutAction::AdjustHardness(ShortcutDirection::Increase),
    ),
    shortcut(
        Modifiers::NONE,
        Key::A,
        ShortcutAction::CycleStrokeApplication,
    ),
    shortcut(Modifiers::NONE, Key::I, ShortcutAction::ToggleBrushInvert),
    shortcut(
        Modifiers::NONE,
        Key::Num0,
        ShortcutAction::SetSymmetry(None),
    ),
    shortcut(
        Modifiers::NONE,
        Key::X,
        ShortcutAction::SetSymmetry(Some(SymmetryAxis::X)),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Y,
        ShortcutAction::SetSymmetry(Some(SymmetryAxis::Y)),
    ),
    shortcut(
        Modifiers::NONE,
        Key::Z,
        ShortcutAction::SetSymmetry(Some(SymmetryAxis::Z)),
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShortcutAvailability {
    actions_ready: bool,
    mesh_ready: bool,
    can_undo: bool,
    can_redo: bool,
    tool: SculptTool,
}

fn shortcut_is_enabled(action: ShortcutAction, availability: ShortcutAvailability) -> bool {
    match action {
        ShortcutAction::ToggleWireframe => true,
        ShortcutAction::Open => availability.actions_ready,
        ShortcutAction::Undo => availability.mesh_ready && availability.can_undo,
        ShortcutAction::Redo => availability.mesh_ready && availability.can_redo,
        ShortcutAction::Export
        | ShortcutAction::Frame
        | ShortcutAction::ToggleCameraMode
        | ShortcutAction::SelectTool(_)
        | ShortcutAction::AdjustRadius(_)
        | ShortcutAction::AdjustStrength(_)
        | ShortcutAction::AdjustHardness(_)
        | ShortcutAction::ToggleBrushInvert
        | ShortcutAction::SetSymmetry(_)
        | ShortcutAction::ClearMask
        | ShortcutAction::InvertMask => availability.mesh_ready,
        ShortcutAction::CycleStrokeApplication => {
            availability.mesh_ready && availability.tool != SculptTool::Grab
        }
    }
}

fn adjusted_brush_value(
    value: f32,
    direction: ShortcutDirection,
    minimum: f32,
    maximum: f32,
) -> f32 {
    let delta = match direction {
        ShortcutDirection::Decrease => -BRUSH_VALUE_STEP,
        ShortcutDirection::Increase => BRUSH_VALUE_STEP,
    };
    (value + delta).clamp(minimum, maximum)
}

fn shortcuts_for(action: ShortcutAction) -> impl Iterator<Item = KeyboardShortcut> {
    SHORTCUT_BINDINGS
        .iter()
        .filter(move |binding| binding.action == action)
        .map(|binding| binding.shortcut)
}

fn shortcut_hint(context: &egui::Context, actions: &[ShortcutAction]) -> String {
    actions
        .iter()
        .flat_map(|&action| shortcuts_for(action))
        .map(|binding| context.format_shortcut(&binding))
        .collect::<Vec<_>>()
        .join(" / ")
}

fn shortcut_label(context: &egui::Context, label: &str, actions: &[ShortcutAction]) -> String {
    format!("{label} ({})", shortcut_hint(context, actions))
}

fn shortcut_tooltip(context: &egui::Context, actions: &[ShortcutAction]) -> String {
    format!("Shortcut: {}", shortcut_hint(context, actions))
}

struct MeshDocument {
    mesh: Option<Mesh>,
    bounds: Option<MeshBounds>,
    remesh_disabled_reason: Option<String>,
    remesh_bounds: Option<MeshBounds>,
    source_path: PathBuf,
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

    fn from_referenced_mesh(mesh: &Mesh) -> Option<Self> {
        let first_vertex = mesh.triangles.iter().flatten().copied().find(|&vertex| {
            mesh.positions
                .get(vertex as usize)
                .is_some_and(|position| position.is_finite())
        })?;
        let first = mesh.positions[first_vertex as usize];
        let mut bounds = Self {
            minimum: first,
            maximum: first,
            minimum_vertices: [first_vertex; 3],
            maximum_vertices: [first_vertex; 3],
            exact: true,
        };
        for &vertex in mesh.triangles.iter().flatten() {
            if let Some(&position) = mesh.positions.get(vertex as usize)
                && position.is_finite()
            {
                bounds.include(vertex, position);
            }
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
        self.update_where(mesh, changed_vertices, |_| true);
    }

    fn update_referenced(&mut self, mesh: &Mesh, changed_vertices: &[u32]) {
        self.update_where(mesh, changed_vertices, |vertex| {
            mesh.topology
                .vertex_triangles
                .get(vertex as usize)
                .is_some_and(|triangles| !triangles.is_empty())
        });
    }

    fn update_where(
        &mut self,
        mesh: &Mesh,
        changed_vertices: &[u32],
        include: impl Fn(u32) -> bool,
    ) {
        if self
            .minimum_vertices
            .iter()
            .chain(&self.maximum_vertices)
            .any(|&vertex| vertex as usize >= mesh.positions.len())
        {
            self.exact = false;
        }
        let invalidated = changed_vertices
            .iter()
            .copied()
            .filter(|&vertex| include(vertex))
            .any(|vertex| {
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
            if !include(vertex) {
                continue;
            }
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FlyKeyState {
    w: bool,
    s: bool,
    a: bool,
    d: bool,
    shift: bool,
    space: bool,
    ctrl: bool,
}

fn fly_movement(keys: FlyKeyState) -> FlyMovement {
    let axis = |positive: bool, negative: bool| {
        f32::from(u8::from(positive)) - f32::from(u8::from(negative))
    };
    FlyMovement {
        forward: axis(keys.w, keys.s),
        right: axis(keys.d, keys.a),
        up: axis(keys.space, keys.shift),
        boost: keys.ctrl,
    }
}

fn fly_look_delta(raw_motion: Option<Vec2>, pointer_delta: Vec2, pixels_per_point: f32) -> Vec2 {
    raw_motion.map_or(pointer_delta, |motion| {
        if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
            motion / pixels_per_point
        } else {
            motion
        }
    })
}

fn wheel_delta_points(unit: egui::MouseWheelUnit, delta_y: f32, viewport_height: f32) -> f32 {
    let scale = match unit {
        egui::MouseWheelUnit::Point => 1.0,
        egui::MouseWheelUnit::Line => WHEEL_POINTS_PER_LINE,
        egui::MouseWheelUnit::Page => viewport_height.max(0.0),
    };
    delta_y * scale
}

fn fly_wheel_points(events: &[egui::Event], viewport_height: f32) -> f32 {
    events
        .iter()
        .filter_map(|event| match event {
            egui::Event::MouseWheel { unit, delta, .. } => {
                Some(wheel_delta_points(*unit, delta.y, viewport_height))
            }
            _ => None,
        })
        .filter(|delta| delta.is_finite())
        .sum::<f32>()
        .clamp(-MAX_FLY_WHEEL_POINTS, MAX_FLY_WHEEL_POINTS)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FlyCaptureEligibility {
    mode: CameraMode,
    viewport_hovered: bool,
    focused: bool,
    has_mesh: bool,
    stroke_active: bool,
    mesh_job_active: bool,
    modal_active: bool,
}

fn fly_capture_eligible(state: FlyCaptureEligibility) -> bool {
    state.mode == CameraMode::Fly
        && state.viewport_hovered
        && state.focused
        && state.has_mesh
        && !state.stroke_active
        && !state.mesh_job_active
        && !state.modal_active
}

fn fly_shortcuts_suppressed(mode: CameraMode, captured: bool, secondary_button_down: bool) -> bool {
    captured || (mode == CameraMode::Fly && secondary_button_down)
}

fn fly_capture_should_release(
    captured: bool,
    secondary_button_down: bool,
    escape_pressed: bool,
    focused: bool,
    state_valid: bool,
) -> bool {
    captured && (!secondary_button_down || escape_pressed || !focused || !state_valid)
}

fn viewport_sense() -> Sense {
    Sense::drag()
}

fn fly_movement_mode_label(mode: FlyMovementMode) -> &'static str {
    match mode {
        FlyMovementMode::Level => "Level",
        FlyMovementMode::Free => "Free flight",
    }
}

fn quick_controls_copy(
    mode: CameraMode,
    fly_movement_mode: FlyMovementMode,
) -> (&'static str, &'static str) {
    match mode {
        CameraMode::Orbit => (
            "LMB drag · Shift smooth · Ctrl invert",
            "RMB drag pan · MMB drag orbit · Wheel zoom",
        ),
        CameraMode::Fly => match fly_movement_mode {
            FlyMovementMode::Level => (
                "LMB drag · Shift smooth · Ctrl invert",
                "Hold RMB · Mouse look · WASD horizontal · Shift/Space down/up · Ctrl boost · Wheel speed · Esc release",
            ),
            FlyMovementMode::Free => (
                "LMB drag · Shift smooth · Ctrl invert",
                "Hold RMB · Mouse look · W/S follow look · A/D strafe · Shift/Space down/up · Ctrl boost · Wheel speed · Esc release",
            ),
        },
    }
}

fn sculpting_allowed(fly_captured: bool) -> bool {
    !fly_captured
}

fn effective_tool(tool: SculptTool, modifiers: Modifiers) -> SculptTool {
    if modifiers.shift && tool != SculptTool::Mask {
        SculptTool::Smooth
    } else {
        tool
    }
}

fn stroke_behavior_for_press(
    behaviors: &ToolStrokeBehaviors,
    tool: SculptTool,
    modifiers: Modifiers,
) -> StrokeBehavior {
    let effective = effective_tool(tool, modifiers);
    if effective == SculptTool::Grab {
        StrokeBehavior::CAPPED
    } else {
        behaviors.get(effective)
    }
}

fn timed_dabs_enabled(
    behavior: StrokeBehavior,
    effective_tool: SculptTool,
    released: bool,
) -> bool {
    behavior.method == StrokeMethod::Airbrush && effective_tool != SculptTool::Grab && !released
}

fn active_tool_text(tool: SculptTool, brush_invert: bool, modifiers: Modifiers) -> String {
    let effective = effective_tool(tool, modifiers);
    let mut text = if effective != tool {
        format!("{} · temporary", effective.label())
    } else {
        effective.label().to_owned()
    };
    if brush_invert ^ modifiers.ctrl {
        text.push_str(" · inverted");
    }
    text
}

fn brush_cursor_color(tool: SculptTool, brush_invert: bool, modifiers: Modifiers) -> Color32 {
    if brush_invert ^ modifiers.ctrl {
        Color32::from_rgb(238, 128, 92)
    } else if effective_tool(tool, modifiers) != tool {
        Color32::from_rgb(166, 132, 245)
    } else {
        Color32::from_rgb(115, 205, 255)
    }
}

#[derive(Clone)]
enum PendingAction {
    OpenDialog,
    Import(PathBuf),
    Close,
}

fn remesh_disabled_reason(mesh: &Mesh) -> Option<String> {
    source_eligibility(mesh)
        .err()
        .map(|error| error.to_string())
}

fn remesh_bounds(mesh: &Mesh) -> Option<MeshBounds> {
    MeshBounds::from_referenced_mesh(mesh)
}

fn history_direction_label(direction: HistoryDirection) -> &'static str {
    match direction {
        HistoryDirection::Undo => "Undo",
        HistoryDirection::Redo => "Redo",
    }
}

enum WorkerJob {
    Import(PathBuf),
    Export {
        mesh: Box<Mesh>,
        path: PathBuf,
    },
    VoxelRemesh {
        mesh: Box<Mesh>,
        settings: VoxelRemeshSettings,
    },
    RestoreSnapshot {
        mesh: Box<Mesh>,
        target: Arc<MeshSnapshot>,
    },
    RecoverSnapshot {
        snapshot: Arc<MeshSnapshot>,
    },
}

type ImportResult = Result<
    (
        Box<Mesh>,
        ImportReport,
        Option<Box<PreparedMeshUpload>>,
        Option<String>,
        Option<MeshBounds>,
    ),
    String,
>;
type VoxelWorkerResult = Result<
    (
        VoxelRemeshOutput,
        Option<Box<PreparedMeshUpload>>,
        Option<String>,
        Option<MeshBounds>,
    ),
    String,
>;
type SnapshotWorkerResult = Result<
    (
        Box<Mesh>,
        Option<Box<PreparedMeshUpload>>,
        Option<String>,
        Option<MeshBounds>,
    ),
    String,
>;

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
    VoxelRemeshCheckpoint(Arc<MeshSnapshot>),
    VoxelRemesh {
        original: Box<Mesh>,
        source: Option<Arc<MeshSnapshot>>,
        result: Box<VoxelWorkerResult>,
    },
    RestoreSnapshotCheckpoint(Arc<MeshSnapshot>),
    RestoreSnapshot {
        original: Box<Mesh>,
        inverse: Option<Arc<MeshSnapshot>>,
        result: SnapshotWorkerResult,
    },
    RecoverSnapshot {
        result: SnapshotWorkerResult,
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
                                let remesh_disabled_reason = remesh_disabled_reason(&mesh);
                                let remesh_bounds = remesh_bounds(&mesh);
                                let upload = mesh_preparer
                                    .as_ref()
                                    .map(|preparer| Box::new(preparer.prepare_mesh(&mesh)));
                                Ok::<_, String>((
                                    Box::new(mesh),
                                    report,
                                    upload,
                                    remesh_disabled_reason,
                                    remesh_bounds,
                                ))
                            }))
                            .map_err(panic_message)
                            .and_then(|result| result);
                            WorkerResult::Import { path, result }
                        }
                        WorkerJob::Export { mesh, path } => {
                            match catch_unwind(AssertUnwindSafe(|| {
                                Arc::new(MeshSnapshot::capture(&mesh))
                            }))
                            .map_err(panic_message)
                            {
                                Ok(recovery) => {
                                    if result_sender
                                        .send(WorkerResult::ExportCheckpoint(recovery))
                                        .is_err()
                                    {
                                        break;
                                    }
                                    context.request_repaint();
                                    let result = catch_unwind(AssertUnwindSafe(|| {
                                        save_stl_atomic(&path, &mesh)
                                            .map_err(|error| error.to_string())
                                    }))
                                    .map_err(panic_message)
                                    .and_then(|result| result);
                                    WorkerResult::Export { path, mesh, result }
                                }
                                Err(error) => WorkerResult::Export {
                                    path,
                                    mesh,
                                    result: Err(format!(
                                        "Could not capture the export recovery snapshot: {error}"
                                    )),
                                },
                            }
                        }
                        WorkerJob::VoxelRemesh { mesh, settings } => {
                            match catch_unwind(AssertUnwindSafe(|| {
                                Arc::new(MeshSnapshot::capture(&mesh))
                            }))
                            .map_err(panic_message)
                            {
                                Ok(source) => {
                                    if result_sender
                                        .send(WorkerResult::VoxelRemeshCheckpoint(Arc::clone(
                                            &source,
                                        )))
                                        .is_err()
                                    {
                                        break;
                                    }
                                    context.request_repaint();
                                    let result = catch_unwind(AssertUnwindSafe(|| {
                                        let output = voxel_remesh(&mesh, settings)
                                            .map_err(|error| error.to_string())?;
                                        let remesh_disabled_reason =
                                            remesh_disabled_reason(&output.mesh);
                                        let remesh_bounds = remesh_bounds(&output.mesh);
                                        let upload = mesh_preparer.as_ref().map(|preparer| {
                                            Box::new(preparer.prepare_mesh(&output.mesh))
                                        });
                                        Ok::<_, String>((
                                            output,
                                            upload,
                                            remesh_disabled_reason,
                                            remesh_bounds,
                                        ))
                                    }))
                                    .map_err(panic_message)
                                    .and_then(|result| result);
                                    WorkerResult::VoxelRemesh {
                                        original: mesh,
                                        source: Some(source),
                                        result: Box::new(result),
                                    }
                                }
                                Err(error) => WorkerResult::VoxelRemesh {
                                    original: mesh,
                                    source: None,
                                    result: Box::new(Err(format!(
                                        "Could not capture the remesh history snapshot: {error}"
                                    ))),
                                },
                            }
                        }
                        WorkerJob::RestoreSnapshot { mesh, target } => {
                            match catch_unwind(AssertUnwindSafe(|| {
                                Arc::new(MeshSnapshot::capture(&mesh))
                            }))
                            .map_err(panic_message)
                            {
                                Ok(inverse) => {
                                    if result_sender
                                        .send(WorkerResult::RestoreSnapshotCheckpoint(Arc::clone(
                                            &inverse,
                                        )))
                                        .is_err()
                                    {
                                        break;
                                    }
                                    context.request_repaint();
                                    let result = catch_unwind(AssertUnwindSafe(|| {
                                        let restored = target
                                            .restore_mesh()
                                            .map_err(|error| error.to_string())?;
                                        let remesh_disabled_reason =
                                            remesh_disabled_reason(&restored);
                                        let remesh_bounds = remesh_bounds(&restored);
                                        let upload = mesh_preparer.as_ref().map(|preparer| {
                                            Box::new(preparer.prepare_mesh(&restored))
                                        });
                                        Ok::<_, String>((
                                            Box::new(restored),
                                            upload,
                                            remesh_disabled_reason,
                                            remesh_bounds,
                                        ))
                                    }))
                                    .map_err(panic_message)
                                    .and_then(|result| result);
                                    WorkerResult::RestoreSnapshot {
                                        original: mesh,
                                        inverse: Some(inverse),
                                        result,
                                    }
                                }
                                Err(error) => WorkerResult::RestoreSnapshot {
                                    original: mesh,
                                    inverse: None,
                                    result: Err(format!(
                                        "Could not capture the inverse history snapshot: {error}"
                                    )),
                                },
                            }
                        }
                        WorkerJob::RecoverSnapshot { snapshot } => {
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                let restored =
                                    snapshot.restore_mesh().map_err(|error| error.to_string())?;
                                let remesh_disabled_reason = remesh_disabled_reason(&restored);
                                let remesh_bounds = remesh_bounds(&restored);
                                let upload = mesh_preparer
                                    .as_ref()
                                    .map(|preparer| Box::new(preparer.prepare_mesh(&restored)));
                                Ok::<_, String>((
                                    Box::new(restored),
                                    upload,
                                    remesh_disabled_reason,
                                    remesh_bounds,
                                ))
                            }))
                            .map_err(panic_message)
                            .and_then(|result| result);
                            WorkerResult::RecoverSnapshot { result }
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
    VoxelRemesh {
        started: Instant,
        recovery: Option<Arc<MeshSnapshot>>,
        dirty_before: bool,
        settings: VoxelRemeshSettings,
    },
    RestoreSnapshot {
        started: Instant,
        recovery: Option<Arc<MeshSnapshot>>,
        dirty_before: bool,
        pending: PendingMeshReplacement,
    },
    RecoverSnapshot {
        started: Instant,
        recovery: Arc<MeshSnapshot>,
        dirty: bool,
        status: String,
        after: Option<PendingAction>,
    },
}

impl BackgroundTask {
    fn owns_mesh(&self) -> bool {
        matches!(
            self,
            Self::Export { .. }
                | Self::VoxelRemesh { .. }
                | Self::RestoreSnapshot { .. }
                | Self::RecoverSnapshot { .. }
        )
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
            Self::VoxelRemesh { started, .. } => ("Voxel remeshing".to_owned(), started),
            Self::RestoreSnapshot {
                started, pending, ..
            } => {
                let label = match pending.direction() {
                    HistoryDirection::Undo => "Preparing undo",
                    HistoryDirection::Redo => "Preparing redo",
                };
                (label.to_owned(), started)
            }
            Self::RecoverSnapshot { started, .. } => {
                ("Restoring the editable mesh".to_owned(), started)
            }
        };
        format!("{label}… {:.1}s", started.elapsed().as_secs_f32())
    }
}

pub struct SculptLiteApp {
    renderer: Option<ViewportRenderer>,
    renderer_error: Option<String>,
    camera: Camera,
    fly_captured: bool,
    document: Option<MeshDocument>,
    history: History,
    sculpt: SculptEngine,
    tool: SculptTool,
    brush: BrushSettings,
    stroke_behaviors: ToolStrokeBehaviors,
    active_stroke_behavior: Option<StrokeBehavior>,
    airbrush_dabs_per_second: f32,
    brush_radius_points: f32,
    voxel_remesh_settings: VoxelRemeshSettings,
    wireframe: bool,
    show_quick_controls: bool,
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
            fly_captured: false,
            document: None,
            history: History::default(),
            sculpt: SculptEngine::default(),
            tool: SculptTool::default(),
            brush: BrushSettings::default(),
            stroke_behaviors: ToolStrokeBehaviors::default(),
            active_stroke_behavior: None,
            airbrush_dabs_per_second: DEFAULT_AIRBRUSH_DABS_PER_SECOND,
            brush_radius_points: INITIAL_BRUSH_RADIUS_POINTS,
            voxel_remesh_settings: VoxelRemeshSettings::default(),
            wireframe: false,
            show_quick_controls: true,
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

    fn capture_fly(&mut self, context: &egui::Context) {
        if self.fly_captured {
            return;
        }
        self.fly_captured = true;
        context.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
            egui::viewport::CursorGrab::Locked,
        ));
        context.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
        context.request_repaint();
    }

    fn release_fly(&mut self, context: &egui::Context) {
        if !self.fly_captured {
            return;
        }
        self.fly_captured = false;
        context.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
            egui::viewport::CursorGrab::None,
        ));
        context.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
        context.request_repaint();
    }

    fn set_camera_mode(&mut self, mode: CameraMode, context: &egui::Context) {
        if self.camera.mode() == mode {
            return;
        }
        self.release_fly(context);
        self.camera.set_mode(mode);
    }

    fn enforce_fly_release(&mut self, context: &egui::Context) {
        if !self.fly_captured {
            return;
        }
        let has_mesh = self
            .document
            .as_ref()
            .is_some_and(|document| document.mesh.is_some());
        let state_valid = self.camera.mode() == CameraMode::Fly
            && has_mesh
            && !self.sculpt.is_stroking()
            && self.background_task.is_none()
            && self.pending_action.is_none()
            && self.error.is_none()
            && !context.input(|input| input.viewport().close_requested());
        let (secondary_button_down, escape_pressed, focused) = context.input(|input| {
            (
                input.pointer.button_down(PointerButton::Secondary),
                input.key_pressed(Key::Escape),
                input.focused,
            )
        });
        if fly_capture_should_release(
            self.fly_captured,
            secondary_button_down,
            escape_pressed,
            focused,
            state_valid,
        ) {
            self.release_fly(context);
        }
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
            self.release_fly(context);
            self.pending_action = Some(action);
        } else {
            self.perform_action(action, context);
        }
    }

    fn perform_action(&mut self, action: PendingAction, context: &egui::Context) {
        match action {
            PendingAction::OpenDialog => {
                self.release_fly(context);
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
                self.release_fly(context);
                self.allow_close = true;
                context.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn start_import(&mut self, path: PathBuf, context: &egui::Context) {
        self.release_fly(context);
        self.camera.set_mode(CameraMode::Orbit);
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
        remesh_disabled_reason: Option<String>,
        remesh_bounds: Option<MeshBounds>,
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
            format!("Loaded {} with topology warnings", path.display())
        } else {
            format!("Loaded {}", path.display())
        };
        self.document = Some(MeshDocument {
            mesh: Some(mesh),
            bounds,
            remesh_disabled_reason,
            remesh_bounds,
            source_path: path,
            dirty: false,
        });
        self.stroke_sampler = None;
        self.active_stroke_behavior = None;
        self.frame_render = FrameRenderBatch::default();
    }

    fn install_worker_mesh(
        &mut self,
        mesh: Mesh,
        upload: Option<PreparedMeshUpload>,
        remesh_disabled_reason: Option<String>,
        remesh_bounds: Option<MeshBounds>,
        dirty: bool,
    ) {
        let bounds = MeshBounds::from_mesh(&mesh);
        self.sculpt.reset_for_mesh(&mesh);
        if let (Some(renderer), Some(upload)) = (&self.renderer, upload) {
            renderer.install_prepared_mesh(upload);
        }
        if let Some(document) = self.document.as_mut() {
            document.mesh = Some(mesh);
            document.bounds = bounds;
            document.remesh_disabled_reason = remesh_disabled_reason;
            document.remesh_bounds = remesh_bounds;
            document.dirty = dirty;
        }
        self.stroke_sampler = None;
        self.active_stroke_behavior = None;
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
                    || Some("The mesh worker stopped".to_owned()),
                    |task| self.recover_background_task(task, context, true),
                );
                if let Some(message) = message {
                    self.error = Some(message);
                }
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
                    result: Ok((mesh, report, upload, remesh_disabled_reason, remesh_bounds)),
                },
            ) => self.install_import(
                path,
                *mesh,
                report,
                upload.map(|upload| *upload),
                remesh_disabled_reason,
                remesh_bounds,
            ),
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
            (
                BackgroundTask::VoxelRemesh {
                    started,
                    dirty_before,
                    settings,
                    ..
                },
                WorkerResult::VoxelRemeshCheckpoint(recovery),
            ) => {
                self.background_task = Some(BackgroundTask::VoxelRemesh {
                    started,
                    recovery: Some(recovery),
                    dirty_before,
                    settings,
                });
            }
            (
                BackgroundTask::VoxelRemesh { started, .. },
                WorkerResult::VoxelRemesh {
                    original,
                    source,
                    result,
                },
            ) => {
                let result = *result;
                let result = source
                    .ok_or_else(|| "The remesh history snapshot is unavailable".to_owned())
                    .and_then(|source| result.map(|prepared| (source, prepared)));
                match result {
                    Ok((source, (output, upload, remesh_disabled_reason, remesh_bounds))) => {
                        let VoxelRemeshOutput {
                            mesh,
                            voxel_size,
                            source_faces,
                            output_faces,
                        } = output;
                        self.install_worker_mesh(
                            mesh,
                            upload.map(|upload| *upload),
                            remesh_disabled_reason,
                            remesh_bounds,
                            true,
                        );
                        let undoable = self.history.record(HistoryEntry::MeshReplacement(source));
                        let undo_note = if undoable {
                            String::new()
                        } else {
                            " · not undoable (history memory limit)".to_owned()
                        };
                        self.status = format!(
                            "Voxel remeshed {} → {} faces in {:.1}s · voxel {:.6}{undo_note}",
                            grouped(source_faces),
                            grouped(output_faces),
                            started.elapsed().as_secs_f32(),
                            voxel_size,
                        );
                        drop(original);
                    }
                    Err(error) => {
                        if let Some(document) = self.document.as_mut() {
                            document.mesh = Some(*original);
                        }
                        self.status =
                            "Voxel remesh failed; the original mesh was left unchanged".to_owned();
                        self.error = Some(format!("Could not voxel remesh the mesh\n\n{error}"));
                    }
                }
            }
            (
                BackgroundTask::RestoreSnapshot {
                    started,
                    dirty_before,
                    pending,
                    ..
                },
                WorkerResult::RestoreSnapshotCheckpoint(recovery),
            ) => {
                self.background_task = Some(BackgroundTask::RestoreSnapshot {
                    started,
                    recovery: Some(recovery),
                    dirty_before,
                    pending,
                });
            }
            (
                BackgroundTask::RestoreSnapshot {
                    started,
                    dirty_before,
                    pending,
                    ..
                },
                WorkerResult::RestoreSnapshot {
                    original,
                    inverse,
                    result,
                },
            ) => {
                let direction = pending.direction();
                let result = inverse
                    .ok_or_else(|| "The inverse history snapshot is unavailable".to_owned())
                    .and_then(|inverse| result.map(|prepared| (inverse, prepared)));
                match result {
                    Ok((inverse, (mesh, upload, remesh_disabled_reason, remesh_bounds))) => {
                        match self.history.complete_mesh_replacement(pending, inverse) {
                            Ok(opposite_saved) => {
                                self.install_worker_mesh(
                                    *mesh,
                                    upload.map(|upload| *upload),
                                    remesh_disabled_reason,
                                    remesh_bounds,
                                    true,
                                );
                                let label = history_direction_label(direction);
                                let memory_note = if opposite_saved {
                                    ""
                                } else {
                                    " · opposite history unavailable (memory limit)"
                                };
                                self.status = format!(
                                    "{label} in {:.1}s{memory_note}",
                                    started.elapsed().as_secs_f32()
                                );
                                drop(original);
                            }
                            Err(error) => {
                                self.history.clear();
                                if let Some(document) = self.document.as_mut() {
                                    document.mesh = Some(*original);
                                    document.dirty = dirty_before;
                                }
                                self.error = Some(format!(
                                    "Could not complete {}\n\n{error}",
                                    history_direction_label(direction).to_lowercase()
                                ));
                            }
                        }
                    }
                    Err(error) => {
                        if let Err(history_error) = self.history.cancel_mesh_replacement(pending) {
                            self.history.clear();
                            self.worker_error = Some(history_error.to_string());
                        }
                        if let Some(document) = self.document.as_mut() {
                            document.mesh = Some(*original);
                            document.dirty = dirty_before;
                        }
                        self.status = format!(
                            "{} failed; the mesh was left unchanged",
                            history_direction_label(direction)
                        );
                        self.error = Some(format!(
                            "Could not prepare {}\n\n{error}",
                            history_direction_label(direction).to_lowercase()
                        ));
                    }
                }
            }
            (
                BackgroundTask::RecoverSnapshot {
                    dirty,
                    status,
                    after,
                    ..
                },
                WorkerResult::RecoverSnapshot { result },
            ) => match result {
                Ok((mesh, upload, remesh_disabled_reason, remesh_bounds)) => {
                    self.install_worker_mesh(
                        *mesh,
                        upload.map(|upload| *upload),
                        remesh_disabled_reason,
                        remesh_bounds,
                        dirty,
                    );
                    self.pending_action = after;
                    self.status = status;
                }
                Err(error) => {
                    self.pending_action = after;
                    self.error = Some(format!(
                        "Could not restore the editable mesh on the replacement worker\n\n{error}"
                    ));
                }
            },
            (task, _) => {
                let recovery = self.recover_background_task(task, context, false);
                self.error = Some(recovery.map_or_else(
                    || {
                        "The mesh worker returned an unexpected result; recovery is in progress"
                            .to_owned()
                    },
                    |recovery| {
                        format!("The mesh worker returned an unexpected result.\n\n{recovery}")
                    },
                ));
            }
        }
    }

    fn recover_background_task(
        &mut self,
        task: BackgroundTask,
        context: &egui::Context,
        restart_worker: bool,
    ) -> Option<String> {
        if matches!(&task, BackgroundTask::Import { .. }) {
            if restart_worker && let Err(error) = self.restart_worker(context) {
                return Some(format!(
                    "The mesh worker stopped before finishing the import, and its replacement could not start: {error}"
                ));
            }
            return Some("The mesh worker stopped before finishing the import".to_owned());
        }

        let (recovery, dirty, status, after) = match task {
            BackgroundTask::Import { .. } => unreachable!("imports returned above"),
            BackgroundTask::Export {
                recovery,
                dirty_before,
                after,
                ..
            } => (
                recovery,
                dirty_before,
                "The export stopped; the editable mesh was restored".to_owned(),
                after,
            ),
            BackgroundTask::VoxelRemesh {
                recovery,
                dirty_before,
                ..
            } => (
                recovery,
                dirty_before,
                "The voxel remesh stopped; the original mesh was restored".to_owned(),
                None,
            ),
            BackgroundTask::RestoreSnapshot {
                recovery,
                dirty_before,
                pending,
                ..
            } => {
                let direction = pending.direction();
                if let Err(error) = self.history.cancel_mesh_replacement(pending) {
                    self.history.clear();
                    self.worker_error = Some(error.to_string());
                }
                (
                    recovery,
                    dirty_before,
                    format!(
                        "The {} stopped; the editable mesh was restored",
                        history_direction_label(direction).to_lowercase()
                    ),
                    None,
                )
            }
            BackgroundTask::RecoverSnapshot {
                recovery,
                dirty,
                status,
                after,
                ..
            } => (Some(recovery), dirty, status, after),
        };

        let Some(recovery) = recovery else {
            self.pending_action = after;
            return Some(
                "The mesh worker stopped before supplying an editable-mesh recovery snapshot"
                    .to_owned(),
            );
        };
        if restart_worker && let Err(error) = self.restart_worker(context) {
            self.pending_action = after;
            return Some(format!(
                "The mesh worker stopped, and its replacement could not start: {error}"
            ));
        }
        let Some(worker) = &self.worker else {
            self.pending_action = after;
            return Some("The mesh worker is unavailable for recovery".to_owned());
        };
        match worker.sender.send(WorkerJob::RecoverSnapshot {
            snapshot: Arc::clone(&recovery),
        }) {
            Ok(()) => {
                self.background_task = Some(BackgroundTask::RecoverSnapshot {
                    started: Instant::now(),
                    recovery,
                    dirty,
                    status,
                    after,
                });
                self.status = "Restoring the editable mesh on the replacement worker".to_owned();
                context.request_repaint();
                None
            }
            Err(_) => {
                self.pending_action = after;
                Some("The replacement mesh worker stopped before recovery began".to_owned())
            }
        }
    }

    fn restart_worker(&mut self, context: &egui::Context) -> Result<(), String> {
        let mesh_preparer = self.renderer.as_ref().map(ViewportRenderer::mesh_preparer);
        let worker = BackgroundWorker::start(context.clone(), mesh_preparer)
            .map_err(|error| error.to_string())?;
        self.worker = Some(worker);
        self.worker_error = None;
        Ok(())
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
        self.release_fly(context);
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
        self.release_fly(context);
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

    fn start_voxel_remesh(&mut self, context: &egui::Context) {
        if self.background_task.is_some() || self.sculpt.is_stroking() {
            return;
        }
        if let Some(reason) = self
            .document
            .as_ref()
            .and_then(|document| document.remesh_disabled_reason.as_deref())
        {
            self.status = reason.to_owned();
            return;
        }
        let dirty_before = self
            .document
            .as_ref()
            .is_some_and(|document| document.dirty);
        let Some(mesh) = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.take())
        else {
            return;
        };
        self.release_fly(context);
        let Some(worker) = &self.worker else {
            if let Some(document) = self.document.as_mut() {
                document.mesh = Some(mesh);
            }
            self.error = Some("The mesh worker is unavailable".to_owned());
            return;
        };
        let settings = self.voxel_remesh_settings;
        match worker.sender.send(WorkerJob::VoxelRemesh {
            mesh: Box::new(mesh),
            settings,
        }) {
            Ok(()) => {
                self.background_task = Some(BackgroundTask::VoxelRemesh {
                    started: Instant::now(),
                    recovery: None,
                    dirty_before,
                    settings,
                });
                self.status = "Sampling the signed-distance field".to_owned();
                context.request_repaint();
            }
            Err(error) => {
                let WorkerJob::VoxelRemesh { mesh, .. } = error.0 else {
                    unreachable!("the submitted worker job remains a voxel remesh")
                };
                if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(*mesh);
                }
                self.worker_error = Some("The mesh worker stopped unexpectedly".to_owned());
            }
        }
    }

    fn start_snapshot_restore(&mut self, pending: PendingMeshReplacement, context: &egui::Context) {
        let direction = pending.direction();
        let target = Arc::clone(pending.target());
        let dirty_before = self
            .document
            .as_ref()
            .is_some_and(|document| document.dirty);
        let Some(mesh) = self
            .document
            .as_mut()
            .and_then(|document| document.mesh.take())
        else {
            if self.history.cancel_mesh_replacement(pending).is_err() {
                self.history.clear();
            }
            return;
        };
        self.release_fly(context);
        let Some(worker) = &self.worker else {
            if let Some(document) = self.document.as_mut() {
                document.mesh = Some(mesh);
            }
            if self.history.cancel_mesh_replacement(pending).is_err() {
                self.history.clear();
            }
            self.error = Some("The mesh worker is unavailable".to_owned());
            return;
        };
        match worker.sender.send(WorkerJob::RestoreSnapshot {
            mesh: Box::new(mesh),
            target,
        }) {
            Ok(()) => {
                self.background_task = Some(BackgroundTask::RestoreSnapshot {
                    started: Instant::now(),
                    recovery: None,
                    dirty_before,
                    pending,
                });
                self.status = format!("Preparing {}", history_direction_label(direction));
                context.request_repaint();
            }
            Err(error) => {
                let WorkerJob::RestoreSnapshot { mesh, .. } = error.0 else {
                    unreachable!("the submitted worker job remains a snapshot restoration")
                };
                if let Some(document) = self.document.as_mut() {
                    document.mesh = Some(*mesh);
                }
                if self.history.cancel_mesh_replacement(pending).is_err() {
                    self.history.clear();
                }
                self.worker_error = Some("The mesh worker stopped unexpectedly".to_owned());
            }
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

    fn begin_frame_render_batch(&mut self) {
        self.frame_render.clear();
    }

    fn queue_frame_render_update(&mut self, updated_vertices: Vec<u32>) {
        self.frame_render.queue(updated_vertices);
    }

    fn flush_frame_render_batch(&mut self, context: &egui::Context) {
        if !self.frame_render.changed {
            return;
        }
        self.frame_render.vertices.sort_unstable();
        self.frame_render.vertices.dedup();
        self.upload_vertices_partial(&self.frame_render.vertices);
        context.request_repaint();
    }

    fn frame_mesh(&mut self, context: &egui::Context) {
        self.release_fly(context);
        self.camera.set_mode(CameraMode::Orbit);
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
            document.remesh_bounds = document
                .mesh
                .as_ref()
                .and_then(MeshBounds::from_referenced_mesh);
        }
    }

    fn update_document_bounds(&mut self, changed_vertices: &[u32]) {
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let Some(mesh) = document.mesh.as_ref() else {
            document.bounds = None;
            document.remesh_bounds = None;
            return;
        };
        if let Some(bounds) = document.bounds.as_mut() {
            bounds.update(mesh, changed_vertices);
        } else {
            document.bounds = MeshBounds::from_mesh(mesh);
        }
        if let Some(bounds) = document.remesh_bounds.as_mut() {
            bounds.update_referenced(mesh, changed_vertices);
        } else {
            document.remesh_bounds = MeshBounds::from_referenced_mesh(mesh);
        }
    }

    fn undo(&mut self, context: &egui::Context) {
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
            HistoryAction::MeshReplacement(pending) => {
                self.start_snapshot_restore(pending, context);
            }
        }
    }

    fn redo(&mut self, context: &egui::Context) {
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
            HistoryAction::MeshReplacement(pending) => {
                self.start_snapshot_restore(pending, context);
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

    fn shortcut_availability(&self) -> ShortcutAvailability {
        let actions_ready = self.background_task.is_none() && !self.sculpt.is_stroking();
        let mesh_ready = actions_ready
            && self
                .document
                .as_ref()
                .is_some_and(|document| document.mesh.is_some());
        ShortcutAvailability {
            actions_ready,
            mesh_ready,
            can_undo: self.history.can_undo(),
            can_redo: self.history.can_redo(),
            tool: self.tool,
        }
    }

    fn apply_shortcut(&mut self, action: ShortcutAction, context: &egui::Context) {
        if !shortcut_is_enabled(action, self.shortcut_availability()) {
            return;
        }

        match action {
            ShortcutAction::Open => self.request_action(PendingAction::OpenDialog, context),
            ShortcutAction::Export => {
                self.export_as(None, context);
            }
            ShortcutAction::Undo => self.undo(context),
            ShortcutAction::Redo => self.redo(context),
            ShortcutAction::Frame => self.frame_mesh(context),
            ShortcutAction::ToggleCameraMode => {
                let mode = match self.camera.mode() {
                    CameraMode::Orbit => CameraMode::Fly,
                    CameraMode::Fly => CameraMode::Orbit,
                };
                self.set_camera_mode(mode, context);
            }
            ShortcutAction::ToggleWireframe => self.wireframe = !self.wireframe,
            ShortcutAction::SelectTool(tool) => self.tool = tool,
            ShortcutAction::AdjustRadius(direction) => match direction {
                ShortcutDirection::Decrease => {
                    self.brush_radius_points =
                        (self.brush_radius_points / 1.12).max(MIN_BRUSH_RADIUS_POINTS);
                }
                ShortcutDirection::Increase => {
                    self.brush_radius_points =
                        (self.brush_radius_points * 1.12).min(MAX_BRUSH_RADIUS_POINTS);
                }
            },
            ShortcutAction::AdjustStrength(direction) => {
                self.brush.strength =
                    adjusted_brush_value(self.brush.strength, direction, 0.01, 1.0);
            }
            ShortcutAction::AdjustHardness(direction) => {
                self.brush.falloff = adjusted_brush_value(self.brush.falloff, direction, 0.0, 0.95);
            }
            ShortcutAction::CycleStrokeApplication => {
                let behavior = self.stroke_behaviors.get(self.tool).next();
                self.stroke_behaviors.set(self.tool, behavior);
                self.status = format!("{} stroke: {}", self.tool.label(), behavior.label());
            }
            ShortcutAction::ToggleBrushInvert => self.brush.invert = !self.brush.invert,
            ShortcutAction::SetSymmetry(symmetry) => self.brush.symmetry = symmetry,
            ShortcutAction::ClearMask => self.edit_mask(false),
            ShortcutAction::InvertMask => self.edit_mask(true),
        }
    }

    fn handle_shortcuts(&mut self, context: &egui::Context) {
        let secondary_button_down =
            context.input(|input| input.pointer.button_down(PointerButton::Secondary));
        if fly_shortcuts_suppressed(self.camera.mode(), self.fly_captured, secondary_button_down)
            || context.egui_wants_keyboard_input()
            || self.pending_action.is_some()
            || self.error.is_some()
        {
            return;
        }

        let actions = context.input_mut(|input| {
            let mut actions = Vec::new();
            for binding in SHORTCUT_BINDINGS {
                if input.consume_shortcut(&binding.shortcut) && !actions.contains(&binding.action) {
                    actions.push(binding.action);
                }
            }
            actions
        });
        for action in actions {
            self.apply_shortcut(action, context);
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
                    .on_hover_text(shortcut_tooltip(context, &[ShortcutAction::Open]))
                    .clicked()
                {
                    self.request_action(PendingAction::OpenDialog, context);
                }
                if ui
                    .add_enabled(mesh_ready, egui::Button::new("Export As…"))
                    .on_hover_text(shortcut_tooltip(context, &[ShortcutAction::Export]))
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
                    .on_hover_text(shortcut_tooltip(context, &[ShortcutAction::Undo]))
                    .clicked()
                {
                    self.undo(context);
                }
                if ui
                    .add_enabled(
                        mesh_ready && self.history.can_redo(),
                        egui::Button::new("Redo"),
                    )
                    .on_hover_text(shortcut_tooltip(context, &[ShortcutAction::Redo]))
                    .clicked()
                {
                    self.redo(context);
                }
                ui.separator();
                if ui
                    .add_enabled(mesh_ready, egui::Button::new("Frame"))
                    .on_hover_text(shortcut_tooltip(context, &[ShortcutAction::Frame]))
                    .clicked()
                {
                    self.frame_mesh(context);
                }
                let camera_shortcut =
                    shortcut_tooltip(context, &[ShortcutAction::ToggleCameraMode]);
                if ui
                    .add_enabled(
                        mesh_ready,
                        egui::Button::selectable(self.camera.mode() == CameraMode::Orbit, "Orbit"),
                    )
                    .on_hover_text(&camera_shortcut)
                    .clicked()
                {
                    self.set_camera_mode(CameraMode::Orbit, context);
                }
                if ui
                    .add_enabled(
                        mesh_ready,
                        egui::Button::selectable(self.camera.mode() == CameraMode::Fly, "Fly"),
                    )
                    .on_hover_text(&camera_shortcut)
                    .clicked()
                {
                    self.set_camera_mode(CameraMode::Fly, context);
                }
                if self.camera.mode() == CameraMode::Fly {
                    let mut movement_mode = self.camera.fly_movement_mode();
                    ui.add_enabled_ui(mesh_ready && !self.fly_captured, |ui| {
                        egui::ComboBox::from_id_salt("fly_movement_mode")
                            .selected_text(fly_movement_mode_label(movement_mode))
                            .show_ui(ui, |ui| {
                                for mode in [FlyMovementMode::Level, FlyMovementMode::Free] {
                                    ui.selectable_value(
                                        &mut movement_mode,
                                        mode,
                                        fly_movement_mode_label(mode),
                                    );
                                }
                            })
                            .response
                            .on_hover_text("Choose whether W/S stays level or follows look pitch");
                    });
                    self.camera.set_fly_movement_mode(movement_mode);
                }
                let wireframe_label =
                    shortcut_label(context, "Wireframe", &[ShortcutAction::ToggleWireframe]);
                ui.checkbox(&mut self.wireframe, wireframe_label);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new("SculptLite").strong());
                });
            });
        });
    }

    fn tool_panel(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::left("tools")
            .resizable(false)
            .default_size(240.0)
            .show(root_ui, |ui| {
                ui.heading("Sculpt");
                let mesh_ready = self.background_task.is_none()
                    && !self.sculpt.is_stroking()
                    && self
                        .document
                        .as_ref()
                        .is_some_and(|document| document.mesh.is_some());
                ui.separator();

                let footer_height = 50.0;
                egui::ScrollArea::vertical()
                    .max_height((ui.available_height() - footer_height).max(0.0))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Tools").strong());
                        ui.add_enabled_ui(mesh_ready, |ui| {
                            let spacing = 6.0;
                            let button_width =
                                two_column_item_width(ui.available_width(), spacing);
                            egui::Grid::new("tool_grid")
                                .num_columns(2)
                                .spacing([spacing, 6.0])
                                .show(ui, |ui| {
                                    for (index, tool) in SculptTool::ALL.into_iter().enumerate() {
                                        let action = ShortcutAction::SelectTool(tool);
                                        let key = shortcut_hint(ui.ctx(), &[action]);
                                        let selected = self.tool == tool;
                                        let mut text = RichText::new(shortcut_label(
                                            ui.ctx(),
                                            tool.label(),
                                            &[action],
                                        ));
                                        if selected {
                                            text = text.strong();
                                        }
                                        let mut button = egui::Button::selectable(selected, text);
                                        if selected {
                                            button = button
                                                .fill(ui.visuals().selection.bg_fill)
                                                .stroke(ui.visuals().selection.stroke);
                                        }
                                        if ui
                                            .add_sized([button_width, 30.0], button)
                                            .on_hover_text(format!(
                                                "{}\nShortcut: {key}",
                                                tool.description()
                                            ))
                                            .clicked()
                                        {
                                            self.tool = tool;
                                        }
                                        if index % 2 == 1 {
                                            ui.end_row();
                                        }
                                    }
                                });
                        });

                        ui.separator();
                        egui::CollapsingHeader::new(RichText::new("Brush").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_enabled_ui(mesh_ready, |ui| {
                                    ui.label(shortcut_label(
                                        ui.ctx(),
                                        "Radius",
                                        &[
                                            ShortcutAction::AdjustRadius(
                                                ShortcutDirection::Decrease,
                                            ),
                                            ShortcutAction::AdjustRadius(
                                                ShortcutDirection::Increase,
                                            ),
                                        ],
                                    ));
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.brush_radius_points,
                                            MIN_BRUSH_RADIUS_POINTS..=MAX_BRUSH_RADIUS_POINTS,
                                        )
                                        .suffix(" px")
                                        .logarithmic(true),
                                    );
                                    ui.label(shortcut_label(
                                        ui.ctx(),
                                        "Strength",
                                        &[
                                            ShortcutAction::AdjustStrength(
                                                ShortcutDirection::Decrease,
                                            ),
                                            ShortcutAction::AdjustStrength(
                                                ShortcutDirection::Increase,
                                            ),
                                        ],
                                    ));
                                    ui.add(egui::Slider::new(
                                        &mut self.brush.strength,
                                        0.01..=1.0,
                                    ));
                                    ui.label(shortcut_label(
                                        ui.ctx(),
                                        "Hardness",
                                        &[
                                            ShortcutAction::AdjustHardness(
                                                ShortcutDirection::Decrease,
                                            ),
                                            ShortcutAction::AdjustHardness(
                                                ShortcutDirection::Increase,
                                            ),
                                        ],
                                    ));
                                    ui.add(egui::Slider::new(
                                        &mut self.brush.falloff,
                                        0.0..=0.95,
                                    ));
                                    let invert_label = shortcut_label(
                                        ui.ctx(),
                                        "Invert brush",
                                        &[ShortcutAction::ToggleBrushInvert],
                                    );
                                    ui.checkbox(&mut self.brush.invert, invert_label);
                                });
                            });

                        ui.separator();
                        egui::CollapsingHeader::new(RichText::new("Stroke").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_enabled_ui(mesh_ready, |ui| {
                                    let supports_application = self.tool != SculptTool::Grab;
                                    let current = self.stroke_behaviors.get(self.tool);
                                    let mut selected = current;
                                    ui.label(shortcut_label(
                                        ui.ctx(),
                                        "Application",
                                        &[ShortcutAction::CycleStrokeApplication],
                                    ));
                                    ui.add_enabled_ui(supports_application, |ui| {
                                        egui::ComboBox::from_id_salt("stroke_application")
                                            .width(ui.available_width())
                                            .selected_text(selected.label())
                                            .show_ui(ui, |ui| {
                                                for behavior in StrokeBehavior::ALL {
                                                    ui.selectable_value(
                                                        &mut selected,
                                                        behavior,
                                                        behavior.label(),
                                                    )
                                                    .on_hover_text(behavior.description());
                                                }
                                            });
                                    });
                                    if supports_application && selected != current {
                                        self.stroke_behaviors.set(self.tool, selected);
                                        self.status = format!(
                                            "{} stroke: {}",
                                            self.tool.label(),
                                            selected.label()
                                        );
                                    }
                                    if !supports_application {
                                        ui.small("Grab is movement-driven.");
                                    } else {
                                        ui.small(selected.description());
                                    }
                                    ui.label("Rate");
                                    ui.add_enabled(
                                        supports_application
                                            && selected.method == StrokeMethod::Airbrush,
                                        egui::Slider::new(
                                            &mut self.airbrush_dabs_per_second,
                                            MIN_AIRBRUSH_DABS_PER_SECOND
                                                ..=MAX_AIRBRUSH_DABS_PER_SECOND,
                                        )
                                        .suffix(" dabs/s"),
                                    );
                                });
                            });

                        ui.separator();
                        egui::CollapsingHeader::new(RichText::new("Symmetry").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_enabled_ui(mesh_ready, |ui| {
                                    egui::ComboBox::from_id_salt("symmetry")
                                        .width(ui.available_width())
                                        .selected_text(match self.brush.symmetry {
                                            None => shortcut_label(
                                                ui.ctx(),
                                                "Off",
                                                &[ShortcutAction::SetSymmetry(None)],
                                            ),
                                            Some(SymmetryAxis::X) => shortcut_label(
                                                ui.ctx(),
                                                "X axis",
                                                &[ShortcutAction::SetSymmetry(Some(
                                                    SymmetryAxis::X,
                                                ))],
                                            ),
                                            Some(SymmetryAxis::Y) => shortcut_label(
                                                ui.ctx(),
                                                "Y axis",
                                                &[ShortcutAction::SetSymmetry(Some(
                                                    SymmetryAxis::Y,
                                                ))],
                                            ),
                                            Some(SymmetryAxis::Z) => shortcut_label(
                                                ui.ctx(),
                                                "Z axis",
                                                &[ShortcutAction::SetSymmetry(Some(
                                                    SymmetryAxis::Z,
                                                ))],
                                            ),
                                        })
                                        .show_ui(ui, |ui| {
                                            for (symmetry, label) in [
                                                (None, "Off"),
                                                (Some(SymmetryAxis::X), "X axis"),
                                                (Some(SymmetryAxis::Y), "Y axis"),
                                                (Some(SymmetryAxis::Z), "Z axis"),
                                            ] {
                                                let action =
                                                    ShortcutAction::SetSymmetry(symmetry);
                                                let label = shortcut_label(
                                                    ui.ctx(),
                                                    label,
                                                    &[action],
                                                );
                                                ui.selectable_value(
                                                    &mut self.brush.symmetry,
                                                    symmetry,
                                                    label,
                                                );
                                            }
                                        });
                                });
                            });

                        ui.separator();
                        egui::CollapsingHeader::new(RichText::new("Mask").strong())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_enabled_ui(mesh_ready, |ui| {
                                    if self.tool == SculptTool::Mask {
                                        ui.colored_label(
                                            ui.visuals().selection.bg_fill,
                                            "Mask painting active",
                                        );
                                        ui.small("Paint to protect the surface; hold Ctrl to erase.");
                                    } else {
                                        ui.small("Masks protect the surface from sculpting tools.");
                                    }
                                    ui.horizontal(|ui| {
                                        let width = two_column_item_width(
                                            ui.available_width(),
                                            ui.spacing().item_spacing.x,
                                        );
                                        if ui
                                            .add_sized([width, 26.0], egui::Button::new("Clear"))
                                            .on_hover_text(shortcut_tooltip(
                                                ui.ctx(),
                                                &[ShortcutAction::ClearMask],
                                            ))
                                            .clicked()
                                        {
                                            self.edit_mask(false);
                                        }
                                        if ui
                                            .add_sized([width, 26.0], egui::Button::new("Invert"))
                                            .on_hover_text(shortcut_tooltip(
                                                ui.ctx(),
                                                &[ShortcutAction::InvertMask],
                                            ))
                                            .clicked()
                                        {
                                            self.edit_mask(true);
                                        }
                                    });
                                });
                            });

                        ui.separator();
                        let mesh_heading = self.document.as_ref().map_or_else(
                            || "Mesh · none".to_owned(),
                            |document| {
                                document.mesh.as_ref().map_or_else(
                                    || "Mesh · working…".to_owned(),
                                    |mesh| {
                                        format!("Mesh · {} faces", grouped(mesh.triangles.len()))
                                    },
                                )
                            },
                        );
                        let mut remesh_requested = false;
                        egui::CollapsingHeader::new(RichText::new(mesh_heading).strong())
                            .id_salt("mesh_section")
                            .default_open(false)
                            .show(ui, |ui| {
                                if let Some(document) = &self.document {
                                    if let Some(mesh) = &document.mesh {
                                        ui.label(format!(
                                            "{} vertices · {} faces",
                                            grouped(mesh.positions.len()),
                                            grouped(mesh.triangles.len())
                                        ));
                                        if let Some((minimum, maximum)) =
                                            document.bounds.map(MeshBounds::min_max)
                                        {
                                            let size = maximum - minimum;
                                            ui.label(format!(
                                                "{:.1} × {:.1} × {:.1}",
                                                size.x, size.y, size.z
                                            ));
                                        }
                                    } else {
                                        ui.label("Mesh operation in progress");
                                    }
                                } else {
                                    ui.label("Import an STL mesh to begin sculpting.");
                                }

                                ui.separator();
                                ui.label(format!(
                                    "Voxel resolution (default {DEFAULT_REMESH_RESOLUTION})"
                                ));
                                ui.add_enabled(
                                    mesh_ready,
                                    egui::Slider::new(
                                        &mut self.voxel_remesh_settings.resolution,
                                        MIN_REMESH_RESOLUTION..=MAX_REMESH_RESOLUTION,
                                    )
                                    .suffix(" cells"),
                                );
                                if let Some((minimum, maximum)) = self
                                    .document
                                    .as_ref()
                                    .and_then(|document| document.remesh_bounds)
                                    .map(MeshBounds::min_max)
                                {
                                    let longest_axis = (maximum - minimum).max_element();
                                    let size = longest_axis
                                        / self.voxel_remesh_settings.resolution as f32;
                                    ui.small(format!("Voxel size: {size:.6}"));
                                }
                                ui.small(
                                    RichText::new(
                                        "Sub-voxel features can disappear or merge. This replaces the whole mesh.",
                                    )
                                    .color(Color32::YELLOW),
                                );

                                let disabled_reason = self
                                    .document
                                    .as_ref()
                                    .and_then(|document| document.remesh_disabled_reason.as_deref());
                                if let Some(reason) = disabled_reason {
                                    ui.small(RichText::new(reason).color(Color32::LIGHT_RED));
                                }
                                let button = ui.add_enabled(
                                    mesh_ready && disabled_reason.is_none(),
                                    egui::Button::new("Voxel remesh"),
                                );
                                if button.clicked() {
                                    remesh_requested = true;
                                }
                            });
                        if remesh_requested {
                            self.start_voxel_remesh(ui.ctx());
                        }
                    });

                ui.separator();
                ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                    let mib = self.history.bytes_used() as f64 / (1024.0 * 1024.0);
                    let budget_mib = self.history.byte_budget() as f64 / (1024.0 * 1024.0);
                    ui.small(format!("Undo: {mib:.1} / {budget_mib:.0} MiB"));
                    ui.small("Shift: smooth · Ctrl: invert");
                    match self.camera.mode() {
                        CameraMode::Orbit => {
                            ui.small("RMB: pan · MMB: orbit · Wheel: zoom");
                        }
                        CameraMode::Fly => {
                            let movement = match self.camera.fly_movement_mode() {
                                FlyMovementMode::Level => "WASD: horizontal",
                                FlyMovementMode::Free => "W/S: follow look · A/D: strafe",
                            };
                            ui.small(format!(
                                "Hold RMB: look/fly · {movement} · Shift/Space: height · Ctrl: boost · Wheel: speed"
                            ));
                        }
                    }
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
                let has_mesh = self
                    .document
                    .as_ref()
                    .is_some_and(|document| document.mesh.is_some());
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
                match self.camera.mode() {
                    CameraMode::Orbit => {
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
                    }
                    CameraMode::Fly => {
                        let (focused, secondary_pressed) = context.input(|input| {
                            (
                                input.focused,
                                input.pointer.button_pressed(PointerButton::Secondary),
                            )
                        });
                        if secondary_pressed
                            && fly_capture_eligible(FlyCaptureEligibility {
                                mode: self.camera.mode(),
                                viewport_hovered: response.hovered(),
                                focused,
                                has_mesh,
                                stroke_active: self.sculpt.is_stroking(),
                                mesh_job_active: self.background_task.is_some(),
                                modal_active: self.pending_action.is_some() || self.error.is_some(),
                            })
                        {
                            self.capture_fly(context);
                        }
                        if self.fly_captured {
                            let (look_delta, movement, wheel_points, delta_seconds) = context
                                .input(|input| {
                                    let keys = FlyKeyState {
                                        w: input.key_down(Key::W),
                                        s: input.key_down(Key::S),
                                        a: input.key_down(Key::A),
                                        d: input.key_down(Key::D),
                                        shift: input.modifiers.shift,
                                        space: input.key_down(Key::Space),
                                        ctrl: input.modifiers.ctrl,
                                    };
                                    (
                                        fly_look_delta(
                                            input.pointer.motion(),
                                            input.pointer.delta(),
                                            input.pixels_per_point(),
                                        ),
                                        fly_movement(keys),
                                        fly_wheel_points(&input.events, rect.height()),
                                        input.stable_dt,
                                    )
                                });
                            self.camera.fly_look(look_delta);
                            self.camera.fly_move(movement, delta_seconds);
                            if wheel_points != 0.0 {
                                self.camera.adjust_fly_speed(wheel_points);
                            }
                        }
                    }
                }

                let Some(camera_frame) = self.camera.frame(rect) else {
                    return;
                };
                let modifiers = context.input(|input| input.modifiers);
                let cursor = (!self.fly_captured)
                    .then_some(pointer)
                    .flatten()
                    .filter(|position| rect.contains(*position))
                    .map(|position| {
                        let mut cursor = BrushCursor::new(position, self.brush_radius_points);
                        cursor.active = self.sculpt.is_stroking();
                        cursor.color = brush_cursor_color(self.tool, self.brush.invert, modifiers);
                        cursor
                    });
                if let Some(renderer) = &self.renderer {
                    renderer.update_camera(camera_frame);
                    renderer.paint(ui, rect, cursor, self.wireframe);
                } else {
                    ui.painter()
                        .rect_filled(rect, 0.0, Color32::from_rgb(20, 22, 25));
                }

                if self.show_quick_controls && has_mesh {
                    let mode = self.camera.mode();
                    let (sculpt_controls, view_controls) =
                        quick_controls_copy(mode, self.camera.fly_movement_mode());
                    let view_label = match mode {
                        CameraMode::Orbit => "ORBIT",
                        CameraMode::Fly => "FLY",
                    };
                    let overlay_width = (rect.width() - 24.0).clamp(240.0, 720.0);
                    let mut dismiss = false;
                    egui::Area::new(egui::Id::new("quick_controls_overlay"))
                        .order(egui::Order::Foreground)
                        .fixed_pos(rect.min + Vec2::splat(12.0))
                        .constrain_to(rect)
                        .default_width(overlay_width)
                        .show(context, |ui| {
                            egui::Frame::NONE
                                .fill(Color32::from_black_alpha(210))
                                .corner_radius(6)
                                .inner_margin(10)
                                .show(ui, |ui| {
                                    ui.set_width(overlay_width);
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new("CONTROLS").strong());
                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                dismiss = ui
                                                    .small_button("×")
                                                    .on_hover_text("Hide controls")
                                                    .clicked();
                                            },
                                        );
                                    });
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            RichText::new("SCULPT")
                                                .small()
                                                .strong()
                                                .color(Color32::from_rgb(120, 200, 235)),
                                        );
                                        ui.small(sculpt_controls);
                                    });
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            RichText::new(view_label)
                                                .small()
                                                .strong()
                                                .color(Color32::from_rgb(255, 198, 92)),
                                        );
                                        ui.small(view_controls);
                                    });
                                });
                        });
                    if dismiss {
                        self.show_quick_controls = false;
                    }
                }

                let (badge_color, badge_text) = match self.camera.mode() {
                    CameraMode::Orbit => (
                        brush_cursor_color(self.tool, self.brush.invert, modifiers),
                        active_tool_text(self.tool, self.brush.invert, modifiers),
                    ),
                    CameraMode::Fly if self.fly_captured => (
                        Color32::from_rgb(255, 198, 92),
                        format!(
                            "Fly · active · {:.3} radii/s · Esc to release",
                            self.camera.fly_speed()
                        ),
                    ),
                    CameraMode::Fly => (
                        brush_cursor_color(self.tool, self.brush.invert, modifiers),
                        format!(
                            "Fly · hold RMB · {:.3} radii/s · {}",
                            self.camera.fly_speed(),
                            active_tool_text(self.tool, self.brush.invert, modifiers)
                        ),
                    ),
                };
                let badge_galley = ui.painter().layout_no_wrap(
                    badge_text,
                    egui::FontId::proportional(14.0),
                    badge_color,
                );
                let badge_padding = Vec2::new(10.0, 6.0);
                let badge_size = badge_galley.size() + badge_padding * 2.0;
                let badge_rect = Rect::from_min_size(
                    Pos2::new(rect.right() - badge_size.x - 12.0, rect.top() + 12.0),
                    badge_size,
                );
                ui.painter()
                    .rect_filled(badge_rect, 5.0, Color32::from_black_alpha(190));
                ui.painter()
                    .galley(badge_rect.min + badge_padding, badge_galley, badge_color);

                let (primary_press, primary_down, primary_released) = context.input(|input| {
                    let primary_press = input.events.iter().find_map(|event| match event {
                        egui::Event::PointerButton {
                            pos,
                            button: PointerButton::Primary,
                            pressed: true,
                            modifiers,
                        } => Some((*pos, *modifiers)),
                        _ => None,
                    });
                    (
                        primary_press,
                        input.pointer.button_down(PointerButton::Primary),
                        input.pointer.button_released(PointerButton::Primary),
                    )
                });
                let viewport_primary_press = primary_press.filter(|(position, _modifiers)| {
                    response.interact_rect.contains(*position)
                        && context.layer_id_at(*position) == Some(response.layer_id)
                });
                if sculpting_allowed(self.fly_captured)
                    && self.background_task.is_none()
                    && !self.sculpt.is_stroking()
                    && let Some((position, modifiers)) = viewport_primary_press
                    && let Some(pointer_hit) = self.hit_at(position, camera_frame)
                    && let Some(document) = self.document.as_ref()
                    && let Some(mesh) = document.mesh.as_ref()
                {
                    self.sculpt.begin_stroke(mesh);
                    let effective_tool = effective_tool(self.tool, modifiers);
                    self.active_stroke_behavior = Some(stroke_behavior_for_press(
                        &self.stroke_behaviors,
                        self.tool,
                        modifiers,
                    ));
                    let sculpt_input = self.sculpt_input(camera_frame);
                    self.stroke_sampler = Some(StrokeSampler::begin(
                        position,
                        MOUSE_PRESSURE,
                        modifiers,
                        self.brush_spacing(),
                        sculpt_input,
                        Instant::now(),
                    ));
                    if effective_tool != SculptTool::Grab {
                        let initial_dab = self
                            .stroke_sampler
                            .as_mut()
                            .and_then(StrokeSampler::take_initial_dab);
                        if let Some(initial_dab) = initial_dab {
                            self.apply_pointer_sample(
                                initial_dab.context,
                                PointerDab {
                                    position: initial_dab.position,
                                    delta: Vec2::ZERO,
                                    pressure: initial_dab.pressure,
                                    modifiers: initial_dab.modifiers,
                                    hit: Some(pointer_hit),
                                    kind: DabKind::Spatial,
                                },
                            );
                        }
                    }
                }

                if sculpting_allowed(self.fly_captured) && self.sculpt.is_stroking() {
                    let modifiers = context.input(|input| input.modifiers);
                    let sculpt_input = self.sculpt_input(camera_frame);
                    let brush_spacing = self.brush_spacing();
                    let effective_tool = effective_tool(sculpt_input.tool, modifiers);
                    if (primary_down || primary_released)
                        && let Some(position) = pointer
                    {
                        if effective_tool == SculptTool::Grab {
                            let delta = self
                                .stroke_sampler
                                .as_mut()
                                .and_then(|sampler| {
                                    sampler.consume_grab_delta(position, MOUSE_PRESSURE)
                                })
                                .unwrap_or(Vec2::ZERO);
                            if delta.length_sq() > f32::EPSILON {
                                self.apply_pointer_sample(
                                    sculpt_input,
                                    PointerDab {
                                        position,
                                        delta,
                                        pressure: MOUSE_PRESSURE,
                                        modifiers,
                                        hit: None,
                                        kind: DabKind::Spatial,
                                    },
                                );
                                if let Some(sampler) = self.stroke_sampler.as_mut() {
                                    sampler.record_spatial_dab();
                                }
                            }
                        } else if let Some(sampler) = self.stroke_sampler.as_mut() {
                            sampler.enqueue_pointer(
                                position,
                                MOUSE_PRESSURE,
                                modifiers,
                                brush_spacing,
                                sculpt_input,
                            );
                        }
                    }
                    if primary_released && let Some(sampler) = self.stroke_sampler.as_mut() {
                        sampler.release();
                    }

                    self.apply_spatial_dabs();
                    if effective_tool != SculptTool::Grab {
                        self.apply_airbrush_dab();
                    }

                    let finished = self.stroke_sampler.as_ref().is_some_and(|sampler| {
                        sampler.is_released() && !sampler.has_pending_path()
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
        let mut brush = self.brush;
        brush.accumulation = self
            .active_stroke_behavior
            .unwrap_or_else(|| self.stroke_behaviors.get(self.tool))
            .accumulation;
        SculptInput {
            camera,
            tool: self.tool,
            brush,
            radius_points: self.brush_radius_points,
        }
    }

    fn brush_spacing(&self) -> f32 {
        (self.brush_radius_points * BRUSH_SPACING_RADIUS_FRACTION).max(1.0)
    }

    fn apply_spatial_dabs(&mut self) {
        let started = Instant::now();
        let mut processed = 0;
        while processed < MAX_DABS_PER_FRAME {
            let dab = self
                .stroke_sampler
                .as_mut()
                .and_then(StrokeSampler::next_spatial_dab);
            let Some(dab) = dab else {
                break;
            };
            self.apply_pointer_sample(
                dab.context,
                PointerDab {
                    position: dab.position,
                    delta: Vec2::ZERO,
                    pressure: dab.pressure,
                    modifiers: dab.modifiers,
                    hit: None,
                    kind: DabKind::Spatial,
                },
            );
            processed += 1;
            if started.elapsed() >= SCULPT_FRAME_BUDGET {
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
        let behavior = self.active_stroke_behavior.unwrap_or_default();
        let interval = self.airbrush_interval();
        let dab = self.stroke_sampler.as_ref().and_then(|sampler| {
            let pointer = sampler.pointer();
            (timed_dabs_enabled(
                behavior,
                effective_tool(pointer.context.tool, pointer.modifiers),
                sampler.is_released(),
            ) && sampler.airbrush_due(interval))
            .then_some(pointer)
        });
        let Some(dab) = dab else {
            return;
        };
        self.apply_pointer_sample(
            dab.context,
            PointerDab {
                position: dab.position,
                delta: Vec2::ZERO,
                pressure: dab.pressure,
                modifiers: dab.modifiers,
                hit: None,
                kind: DabKind::Timed,
            },
        );
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
        let history_entry = (!outcome.edit.is_empty()).then_some(HistoryEntry::Local(outcome.edit));
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
        if self.document.as_ref().is_some_and(|document| {
            document.bounds.is_some_and(|bounds| !bounds.exact)
                || document.remesh_bounds.is_some_and(|bounds| !bounds.exact)
        }) {
            self.refresh_document_bounds();
        }
        self.stroke_sampler = None;
        self.active_stroke_behavior = None;
    }

    fn apply_pointer_sample(&mut self, input: SculptInput, dab: PointerDab) {
        let Some(pointer_hit) = dab.hit.or_else(|| self.hit_at(dab.position, input.camera)) else {
            return;
        };
        let hit = pointer_hit.hit;

        let depth_scale = (hit.distance / input.camera.distance().max(1.0e-6)).max(0.05);
        let units_per_point = input.camera.world_units_per_point() * depth_scale;
        let world_drag = input.camera.right() * dab.delta.x * units_per_point
            - input.camera.up() * dab.delta.y * units_per_point;
        let effective_tool = effective_tool(input.tool, dab.modifiers);
        let mut settings = input.brush;
        settings.radius = input.radius_points * units_per_point;
        let sample = BrushSample::from_hit(
            &hit,
            world_drag,
            pointer_hit.view_direction,
            dab.pressure,
            dab.modifiers.ctrl,
        );

        let changed = self.document.as_mut().is_some_and(|document| {
            document.mesh.as_mut().is_some_and(|mesh| {
                self.sculpt
                    .apply_sample(mesh, effective_tool, &settings, sample, dab.kind)
            })
        });
        self.consume_sculpt_step(changed, effective_tool != SculptTool::Mask);
    }

    fn consume_sculpt_step(&mut self, changed: bool, geometry_changed: bool) {
        let committed = self.sculpt.take_sample_committed();
        let updated_vertices = self.sculpt.take_updated_vertices();
        if let Some(error) = self.sculpt.take_error() {
            self.error = Some(error);
        }
        if committed && let Some(document) = self.document.as_mut() {
            document.dirty = true;
        }
        if changed {
            if geometry_changed {
                self.update_document_bounds(&updated_vertices);
            }
            self.queue_frame_render_update(updated_vertices);
        }
    }

    fn unsaved_prompt(&mut self, context: &egui::Context) {
        let Some(pending) = self.pending_action.clone() else {
            return;
        };
        let response = egui::Modal::new(egui::Id::new("unsaved_sculpt")).show(context, |ui| {
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
        if response.should_close() {
            self.pending_action = None;
        }
    }

    fn error_prompt(&mut self, context: &egui::Context) {
        let Some(message) = self.error.clone() else {
            return;
        };
        let response = egui::Modal::new(egui::Id::new("sculpt_lite_error")).show(context, |ui| {
            ui.heading("SculptLite error");
            ui.label(message);
            if ui.button("Close").clicked() {
                self.error = None;
            }
        });
        if response.should_close() {
            self.error = None;
        }
    }
}

impl eframe::App for SculptLiteApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let context = root_ui.ctx().clone();
        self.poll_background_task(&context);
        self.enforce_fly_release(&context);
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
            if sampler.has_pending_path() {
                context.request_repaint();
            } else if timed_dabs_enabled(
                self.active_stroke_behavior.unwrap_or_default(),
                effective_tool(sampler.pointer().context.tool, sampler.pointer().modifiers),
                sampler.is_released(),
            ) {
                context.request_repaint_after(sampler.airbrush_wait(self.airbrush_interval()));
            }
        }
        if self.background_task.is_some() {
            context.request_repaint_after(Duration::from_millis(100));
        }
        if self.fly_captured {
            context.request_repaint();
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

fn two_column_item_width(available_width: f32, item_spacing: f32) -> f32 {
    ((available_width - item_spacing) / 2.0).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_tetrahedron(scale: f32) -> Mesh {
        Mesh::new(
            vec![
                Vec3::new(1.0, 1.0, 1.0) * scale,
                Vec3::new(-1.0, -1.0, 1.0) * scale,
                Vec3::new(-1.0, 1.0, -1.0) * scale,
                Vec3::new(1.0, -1.0, -1.0) * scale,
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        .unwrap()
    }

    #[test]
    fn voxel_worker_failure_returns_the_exact_original() {
        let worker = BackgroundWorker::start(egui::Context::default(), None).unwrap();
        let source = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).unwrap();
        let expected = MeshSnapshot::capture(&source);
        worker
            .sender
            .send(WorkerJob::VoxelRemesh {
                mesh: Box::new(source),
                settings: VoxelRemeshSettings { resolution: 32 },
            })
            .unwrap();

        let checkpoint = worker
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let WorkerResult::VoxelRemeshCheckpoint(checkpoint) = checkpoint else {
            panic!("remesh checkpoint expected");
        };
        assert_eq!(checkpoint.as_ref(), &expected);

        let result = worker
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let WorkerResult::VoxelRemesh {
            original,
            source,
            result,
        } = result
        else {
            panic!("remesh result expected");
        };
        assert!(result.is_err());
        assert_eq!(MeshSnapshot::capture(&original), expected);
        assert_eq!(source.unwrap().as_ref(), &expected);
    }

    #[test]
    fn snapshot_restore_worker_prepares_an_exact_transaction() {
        let worker = BackgroundWorker::start(egui::Context::default(), None).unwrap();
        let mut current = closed_tetrahedron(1.0);
        current.mask = vec![0.0, 0.25, 0.5, 1.0];
        let current_snapshot = MeshSnapshot::capture(&current);
        let mut target = closed_tetrahedron(2.0);
        target.mask = vec![1.0, 0.5, 0.25, 0.0];
        let target_snapshot = Arc::new(MeshSnapshot::capture(&target));
        worker
            .sender
            .send(WorkerJob::RestoreSnapshot {
                mesh: Box::new(current),
                target: Arc::clone(&target_snapshot),
            })
            .unwrap();

        let checkpoint = worker
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let WorkerResult::RestoreSnapshotCheckpoint(checkpoint) = checkpoint else {
            panic!("restore checkpoint expected");
        };
        assert_eq!(checkpoint.as_ref(), &current_snapshot);

        let result = worker
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let WorkerResult::RestoreSnapshot {
            original,
            inverse,
            result,
        } = result
        else {
            panic!("restore result expected");
        };
        assert_eq!(MeshSnapshot::capture(&original), current_snapshot);
        assert_eq!(inverse.unwrap().as_ref(), &current_snapshot);
        let (restored, upload, disabled_reason, remesh_bounds) = result.unwrap();
        assert!(upload.is_none());
        assert!(disabled_reason.is_none());
        assert!(remesh_bounds.is_some_and(|bounds| {
            let (minimum, maximum) = bounds.min_max();
            (maximum - minimum).max_element() > 0.0
        }));
        assert_eq!(MeshSnapshot::capture(&restored), *target_snapshot);
    }

    #[test]
    fn recovery_job_rebuilds_and_prepares_a_snapshot_off_thread() {
        let worker = BackgroundWorker::start(egui::Context::default(), None).unwrap();
        let mut source = closed_tetrahedron(1.0);
        source.mask = vec![0.0, 0.2, 0.6, 1.0];
        let snapshot = Arc::new(MeshSnapshot::capture(&source));
        worker
            .sender
            .send(WorkerJob::RecoverSnapshot {
                snapshot: Arc::clone(&snapshot),
            })
            .unwrap();

        let result = worker
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let WorkerResult::RecoverSnapshot { result } = result else {
            panic!("recovery result expected");
        };
        let (restored, upload, disabled_reason, remesh_bounds) = result.unwrap();
        assert!(upload.is_none());
        assert!(disabled_reason.is_none());
        assert!(remesh_bounds.is_some_and(|bounds| {
            let (minimum, maximum) = bounds.min_max();
            (maximum - minimum).max_element() > 0.0
        }));
        assert_eq!(MeshSnapshot::capture(&restored), *snapshot);
    }

    #[test]
    fn grouped_formats_face_counts() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn two_column_row_does_not_grow_its_panel_across_frames() {
        let context = egui::Context::default();

        for frame in 0..16 {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1_280.0, 800.0))),
                time: Some(f64::from(frame) / 60.0),
                ..Default::default()
            };
            let mut panel_width = 0.0;
            let _ = context.run_ui(input, |root_ui| {
                let panel = egui::Panel::left("tools")
                    .resizable(false)
                    .default_size(240.0)
                    .show(root_ui, |ui| {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                egui::CollapsingHeader::new("Mask").default_open(true).show(
                                    ui,
                                    |ui| {
                                        ui.horizontal(|ui| {
                                            let width = two_column_item_width(
                                                ui.available_width(),
                                                ui.spacing().item_spacing.x,
                                            );
                                            ui.add_sized([width, 26.0], egui::Button::new("Clear"));
                                            ui.add_sized(
                                                [width, 26.0],
                                                egui::Button::new("Invert"),
                                            );
                                        });
                                    },
                                );
                            });
                    });
                panel_width = panel.response.rect.width();
                egui::CentralPanel::default().show(root_ui, |_| {});
            });

            assert_eq!(panel_width, 240.0, "panel grew on frame {frame}");
        }
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
    fn quick_controls_copy_matches_the_active_camera_mode() {
        let (orbit_sculpt, orbit_view) =
            quick_controls_copy(CameraMode::Orbit, FlyMovementMode::Level);
        assert!(orbit_sculpt.contains("LMB drag"));
        assert!(orbit_view.contains("RMB drag pan"));
        assert!(orbit_view.contains("MMB drag orbit"));

        let (level_sculpt, level_view) =
            quick_controls_copy(CameraMode::Fly, FlyMovementMode::Level);
        assert_eq!(level_sculpt, orbit_sculpt);
        assert!(level_view.contains("Hold RMB"));
        assert!(level_view.contains("WASD horizontal"));
        assert!(level_view.contains("Shift/Space down/up"));
        assert!(level_view.contains("Ctrl boost"));

        let (free_sculpt, free_view) = quick_controls_copy(CameraMode::Fly, FlyMovementMode::Free);
        assert_eq!(free_sculpt, orbit_sculpt);
        assert!(free_view.contains("W/S follow look"));
        assert!(free_view.contains("A/D strafe"));
    }

    #[test]
    fn fly_movement_mode_labels_are_concise() {
        assert_eq!(fly_movement_mode_label(FlyMovementMode::Level), "Level");
        assert_eq!(
            fly_movement_mode_label(FlyMovementMode::Free),
            "Free flight"
        );
    }

    #[test]
    fn fly_key_mapping_uses_expected_signed_axes() {
        assert_eq!(
            fly_movement(FlyKeyState {
                w: true,
                a: true,
                space: true,
                ..FlyKeyState::default()
            }),
            FlyMovement {
                forward: 1.0,
                right: -1.0,
                up: 1.0,
                ..FlyMovement::default()
            }
        );
        assert_eq!(
            fly_movement(FlyKeyState {
                shift: true,
                ..FlyKeyState::default()
            }),
            FlyMovement {
                up: -1.0,
                ..FlyMovement::default()
            }
        );
        assert_eq!(
            fly_movement(FlyKeyState {
                w: true,
                s: true,
                a: true,
                d: true,
                shift: true,
                space: true,
                ctrl: false,
            }),
            FlyMovement::default()
        );
        assert!(
            fly_movement(FlyKeyState {
                ctrl: true,
                ..FlyKeyState::default()
            })
            .boost
        );
    }

    #[test]
    fn fly_wheel_units_convert_to_points_and_clamp_spikes() {
        assert_eq!(
            wheel_delta_points(egui::MouseWheelUnit::Point, 3.5, 600.0),
            3.5
        );
        assert_eq!(
            wheel_delta_points(egui::MouseWheelUnit::Line, 2.0, 600.0),
            80.0
        );
        assert_eq!(
            wheel_delta_points(egui::MouseWheelUnit::Page, -0.5, 600.0),
            -300.0
        );
        let events = [egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Page,
            delta: Vec2::new(0.0, 20.0),
            phase: egui::TouchPhase::Move,
            modifiers: Modifiers::NONE,
        }];
        assert_eq!(fly_wheel_points(&events, 600.0), MAX_FLY_WHEEL_POINTS);
    }

    #[test]
    fn fly_look_prefers_raw_motion_with_pointer_fallback() {
        assert_eq!(
            fly_look_delta(Some(Vec2::new(8.0, 4.0)), Vec2::new(1.0, 2.0), 2.0),
            Vec2::new(4.0, 2.0)
        );
        assert_eq!(
            fly_look_delta(None, Vec2::new(1.0, 2.0), 2.0),
            Vec2::new(1.0, 2.0)
        );
    }

    #[test]
    fn fly_capture_eligibility_rejects_strokes_mesh_jobs_and_modals() {
        let ready = FlyCaptureEligibility {
            mode: CameraMode::Fly,
            viewport_hovered: true,
            focused: true,
            has_mesh: true,
            stroke_active: false,
            mesh_job_active: false,
            modal_active: false,
        };
        assert!(fly_capture_eligible(ready));
        assert!(!fly_capture_eligible(FlyCaptureEligibility {
            stroke_active: true,
            ..ready
        }));
        assert!(!fly_capture_eligible(FlyCaptureEligibility {
            mesh_job_active: true,
            ..ready
        }));
        assert!(!fly_capture_eligible(FlyCaptureEligibility {
            modal_active: true,
            ..ready
        }));
        assert!(!fly_capture_eligible(FlyCaptureEligibility {
            viewport_hovered: false,
            ..ready
        }));
        assert!(!fly_capture_eligible(FlyCaptureEligibility {
            mode: CameraMode::Orbit,
            ..ready
        }));
    }

    #[test]
    fn active_flight_suppresses_shortcuts_and_releases_on_escape_or_focus_loss() {
        assert!(fly_shortcuts_suppressed(CameraMode::Fly, true, true));
        assert!(fly_shortcuts_suppressed(CameraMode::Fly, false, true));
        assert!(!fly_shortcuts_suppressed(CameraMode::Orbit, false, true));
        assert!(fly_capture_should_release(true, true, true, true, true));
        assert!(fly_capture_should_release(true, true, false, false, true));
        assert!(fly_capture_should_release(true, false, false, true, true));
        assert!(fly_capture_should_release(true, true, false, true, false));
        assert!(!fly_capture_should_release(true, true, false, true, true));
    }

    #[test]
    fn releasing_flight_restores_existing_sculpt_interactions() {
        assert!(!sculpting_allowed(true));
        assert!(sculpting_allowed(false));
        assert_eq!(
            effective_tool(SculptTool::Draw, Modifiers::NONE),
            SculptTool::Draw
        );
        assert_eq!(
            effective_tool(SculptTool::Grab, Modifiers::NONE),
            SculptTool::Grab
        );
        assert_eq!(
            effective_tool(SculptTool::Draw, Modifiers::SHIFT),
            SculptTool::Smooth
        );
        assert_eq!(
            active_tool_text(SculptTool::Draw, false, Modifiers::CTRL),
            "Draw · inverted"
        );
    }

    #[test]
    fn active_tool_feedback_explains_temporary_modifiers() {
        assert_eq!(
            active_tool_text(SculptTool::Clay, false, Modifiers::NONE),
            "Clay"
        );
        assert_eq!(
            active_tool_text(SculptTool::Clay, false, Modifiers::SHIFT),
            "Smooth · temporary"
        );
        assert_eq!(
            active_tool_text(SculptTool::Clay, false, Modifiers::CTRL),
            "Clay · inverted"
        );
        assert_eq!(
            active_tool_text(
                SculptTool::Clay,
                false,
                Modifiers::SHIFT.plus(Modifiers::CTRL)
            ),
            "Smooth · temporary · inverted"
        );
        assert_eq!(
            active_tool_text(SculptTool::Mask, false, Modifiers::SHIFT),
            "Mask"
        );
        assert_eq!(
            active_tool_text(SculptTool::Clay, true, Modifiers::NONE),
            "Clay · inverted"
        );
        assert_eq!(
            active_tool_text(SculptTool::Clay, true, Modifiers::CTRL),
            "Clay"
        );
    }

    #[test]
    fn stroke_application_cycles_and_is_remembered_per_effective_tool() {
        assert_eq!(StrokeBehavior::CAPPED.next(), StrokeBehavior::ACCUMULATE);
        assert_eq!(StrokeBehavior::ACCUMULATE.next(), StrokeBehavior::AIRBRUSH);
        assert_eq!(StrokeBehavior::AIRBRUSH.next(), StrokeBehavior::CAPPED);

        let mut behaviors = ToolStrokeBehaviors::default();
        behaviors.set(SculptTool::Draw, StrokeBehavior::ACCUMULATE);
        behaviors.set(SculptTool::Smooth, StrokeBehavior::AIRBRUSH);

        let captured = stroke_behavior_for_press(&behaviors, SculptTool::Draw, Modifiers::NONE);
        behaviors.set(SculptTool::Draw, StrokeBehavior::AIRBRUSH);
        assert_eq!(captured, StrokeBehavior::ACCUMULATE);
        behaviors.set(SculptTool::Draw, StrokeBehavior::ACCUMULATE);

        assert_eq!(
            stroke_behavior_for_press(&behaviors, SculptTool::Draw, Modifiers::NONE),
            StrokeBehavior::ACCUMULATE
        );
        assert_eq!(
            stroke_behavior_for_press(&behaviors, SculptTool::Draw, Modifiers::SHIFT),
            StrokeBehavior::AIRBRUSH
        );
        assert_eq!(
            stroke_behavior_for_press(&behaviors, SculptTool::Clay, Modifiers::NONE),
            StrokeBehavior::CAPPED
        );
        assert_eq!(
            stroke_behavior_for_press(&behaviors, SculptTool::Grab, Modifiers::NONE),
            StrokeBehavior::CAPPED
        );
    }

    #[test]
    fn only_active_airbrush_strokes_schedule_timed_dabs() {
        assert!(!timed_dabs_enabled(
            StrokeBehavior::CAPPED,
            SculptTool::Draw,
            false
        ));
        assert!(!timed_dabs_enabled(
            StrokeBehavior::ACCUMULATE,
            SculptTool::Draw,
            false
        ));
        assert!(timed_dabs_enabled(
            StrokeBehavior::AIRBRUSH,
            SculptTool::Draw,
            false
        ));
        assert!(!timed_dabs_enabled(
            StrokeBehavior::AIRBRUSH,
            SculptTool::Grab,
            false
        ));
        assert!(!timed_dabs_enabled(
            StrokeBehavior::AIRBRUSH,
            SculptTool::Draw,
            true
        ));
    }

    #[test]
    fn cursor_color_tracks_temporary_brush_behavior() {
        let normal = brush_cursor_color(SculptTool::Clay, false, Modifiers::NONE);
        let smooth = brush_cursor_color(SculptTool::Clay, false, Modifiers::SHIFT);
        let inverted = brush_cursor_color(SculptTool::Clay, false, Modifiers::CTRL);

        assert_ne!(normal, smooth);
        assert_ne!(normal, inverted);
        assert_eq!(
            brush_cursor_color(
                SculptTool::Clay,
                false,
                Modifiers::SHIFT.plus(Modifiers::CTRL)
            ),
            inverted
        );
        assert_eq!(
            brush_cursor_color(SculptTool::Mask, false, Modifiers::SHIFT),
            normal
        );
        assert_eq!(
            brush_cursor_color(SculptTool::Clay, true, Modifiers::NONE),
            inverted
        );
        assert_eq!(
            brush_cursor_color(SculptTool::Clay, true, Modifiers::CTRL),
            normal
        );
    }

    #[test]
    fn every_sculpt_tool_has_help_text() {
        for tool in SculptTool::ALL {
            assert!(!tool.description().is_empty(), "missing help for {tool}");
        }
    }

    #[test]
    fn viewport_drag_sense_has_no_click_threshold_delay() {
        let sense = viewport_sense();

        assert!(sense.senses_drag());
        assert!(!sense.senses_click());
    }

    #[test]
    fn mouse_samples_use_full_pressure() {
        assert_eq!(MOUSE_PRESSURE, 1.0);
    }

    #[test]
    fn tool_shortcuts_follow_workflow_order() {
        let assignments = [
            (SculptTool::Draw, Key::Num1),
            (SculptTool::Clay, Key::Num2),
            (SculptTool::Crease, Key::Num3),
            (SculptTool::Inflate, Key::Num4),
            (SculptTool::Smooth, Key::Num5),
            (SculptTool::Pinch, Key::Num6),
            (SculptTool::Flatten, Key::Num7),
            (SculptTool::Grab, Key::Num8),
            (SculptTool::Mask, Key::Num9),
        ];

        assert_eq!(SculptTool::ALL, assignments.map(|(tool, _shortcut)| tool));
        for (tool, key) in assignments {
            assert_eq!(
                shortcuts_for(ShortcutAction::SelectTool(tool)).collect::<Vec<_>>(),
                [KeyboardShortcut::new(Modifiers::NONE, key)]
            );
        }
    }

    #[test]
    fn inline_shortcut_labels_put_the_key_after_the_label_in_parentheses() {
        let context = egui::Context::default();
        assert_eq!(
            shortcut_label(
                &context,
                "Draw",
                &[ShortcutAction::SelectTool(SculptTool::Draw)]
            ),
            "Draw (1)"
        );
        assert_eq!(
            shortcut_label(&context, "Off", &[ShortcutAction::SetSymmetry(None)]),
            "Off (0)"
        );
    }

    #[test]
    fn shortcut_bindings_are_unique_and_prioritize_specific_redo() {
        for (index, binding) in SHORTCUT_BINDINGS.iter().enumerate() {
            assert!(
                SHORTCUT_BINDINGS[..index]
                    .iter()
                    .all(|earlier| earlier.shortcut != binding.shortcut),
                "duplicate shortcut: {:?}",
                binding.shortcut
            );
        }

        let shifted_redo = KeyboardShortcut::new(Modifiers::CTRL.plus(Modifiers::SHIFT), Key::Z);
        let undo = KeyboardShortcut::new(Modifiers::CTRL, Key::Z);
        let shifted_redo_index = SHORTCUT_BINDINGS
            .iter()
            .position(|binding| binding.shortcut == shifted_redo)
            .unwrap();
        let undo_index = SHORTCUT_BINDINGS
            .iter()
            .position(|binding| binding.shortcut == undo)
            .unwrap();
        assert!(shifted_redo_index < undo_index);

        assert_eq!(
            shortcuts_for(ShortcutAction::ToggleBrushInvert).collect::<Vec<_>>(),
            [KeyboardShortcut::new(Modifiers::NONE, Key::I)]
        );
        assert_eq!(
            shortcuts_for(ShortcutAction::InvertMask).collect::<Vec<_>>(),
            [KeyboardShortcut::new(Modifiers::CTRL, Key::I)]
        );
        assert_eq!(
            shortcuts_for(ShortcutAction::ToggleCameraMode).collect::<Vec<_>>(),
            [KeyboardShortcut::new(Modifiers::NONE, Key::V)]
        );
        assert_eq!(
            shortcuts_for(ShortcutAction::CycleStrokeApplication).collect::<Vec<_>>(),
            [KeyboardShortcut::new(Modifiers::NONE, Key::A)]
        );
    }

    #[test]
    fn shortcut_availability_matches_control_state() {
        let ready = ShortcutAvailability {
            actions_ready: true,
            mesh_ready: true,
            can_undo: true,
            can_redo: false,
            tool: SculptTool::Draw,
        };
        assert!(shortcut_is_enabled(ShortcutAction::Open, ready));
        assert!(shortcut_is_enabled(ShortcutAction::Undo, ready));
        assert!(!shortcut_is_enabled(ShortcutAction::Redo, ready));
        assert!(shortcut_is_enabled(
            ShortcutAction::CycleStrokeApplication,
            ready
        ));

        assert!(!shortcut_is_enabled(
            ShortcutAction::CycleStrokeApplication,
            ShortcutAvailability {
                tool: SculptTool::Grab,
                ..ready
            }
        ));

        let unavailable = ShortcutAvailability {
            actions_ready: false,
            mesh_ready: false,
            can_undo: true,
            can_redo: true,
            tool: SculptTool::Draw,
        };
        assert!(!shortcut_is_enabled(ShortcutAction::Open, unavailable));
        assert!(!shortcut_is_enabled(
            ShortcutAction::SelectTool(SculptTool::Clay),
            unavailable
        ));
        assert!(shortcut_is_enabled(
            ShortcutAction::ToggleWireframe,
            unavailable
        ));
    }

    #[test]
    fn brush_shortcut_steps_clamp_to_slider_bounds() {
        assert_eq!(
            adjusted_brush_value(0.5, ShortcutDirection::Decrease, 0.01, 1.0),
            0.45
        );
        assert_eq!(
            adjusted_brush_value(0.5, ShortcutDirection::Increase, 0.01, 1.0),
            0.55
        );
        assert_eq!(
            adjusted_brush_value(0.01, ShortcutDirection::Decrease, 0.01, 1.0),
            0.01
        );
        assert_eq!(
            adjusted_brush_value(0.95, ShortcutDirection::Increase, 0.0, 0.95),
            0.95
        );
    }

    #[test]
    fn frame_render_batch_merges_fixed_topology_edits() {
        let mut batch = FrameRenderBatch::default();
        batch.queue(vec![1, 2]);
        batch.queue(vec![3, 4]);
        batch.queue(vec![5, 2]);
        batch.vertices.sort_unstable();
        batch.vertices.dedup();

        assert!(batch.changed);
        assert_eq!(batch.vertices, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn queued_stroke_context_is_unchanged_by_later_camera_navigation() {
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut camera = Camera::default();
        let original = SculptInput {
            camera: camera.frame(viewport).unwrap(),
            tool: SculptTool::Draw,
            brush: BrushSettings::default(),
            radius_points: 55.0,
        };
        let mut sampler = StrokeSampler::begin(
            Pos2::ZERO,
            1.0,
            Modifiers::NONE,
            10.0,
            original,
            Instant::now(),
        );
        sampler.enqueue_pointer(Pos2::new(20.0, 0.0), 1.0, Modifiers::NONE, 10.0, original);

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

    #[test]
    fn remesh_bounds_track_only_referenced_vertices() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::new(10_000.0, 10_000.0, 10_000.0),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let mut bounds = MeshBounds::from_referenced_mesh(&mesh).unwrap();
        assert_eq!(bounds.min_max(), (Vec3::ZERO, Vec3::ONE - Vec3::Z));

        mesh.positions[3] = Vec3::splat(20_000.0);
        bounds.update_referenced(&mesh, &[3]);
        assert_eq!(bounds.min_max(), (Vec3::ZERO, Vec3::ONE - Vec3::Z));

        mesh.positions[1].x = 2.0;
        bounds.update_referenced(&mesh, &[1]);
        assert_eq!(bounds.maximum.x, 2.0);
    }
}
