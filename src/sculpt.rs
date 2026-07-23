use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
};

use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

use crate::{
    history::{LocalEdit, MaskChange, PositionChange},
    mesh::{
        Mesh, MeshChangeSet, MeshEditDelta, MeshEditRecorder, RayHit, RemeshSettings,
        VertexRemeshEvent, VertexTraversalScratch,
    },
};

// Keep each adaptive-topology step bounded. A brush sample may use multiple
// steps, but deformation is deferred until the local mesh has enough support.
const MAX_ADAPTIVE_TOPOLOGY_EDITS_PER_STEP: usize = 64;
const MAX_ADAPTIVE_TOPOLOGY_STEPS: usize = 48;
const ADAPTIVE_SPLIT_THRESHOLD: f32 = 2.0;
const ADAPTIVE_COLLAPSE_THRESHOLD: f32 = 0.5;
const MIN_ADAPTIVE_SUPPORT_INFLUENCE: f32 = 0.05;
const MAX_ADAPTIVE_SUPPORT_INFLUENCE_STEP: f32 = 0.35;
const SAFE_DEFORMATION_SEARCH_STEPS: usize = 6;
const MIN_SAFE_DEFORMATION_FACTOR: f32 = 1.0 / 64.0;
const MAX_TOPOLOGY_INTERSECTION_FACES_PER_STEP: usize = 256;
const MAX_DEFORMATION_INTERSECTION_FACES_PER_STEP: usize = 512;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SculptTool {
    Grab,
    #[default]
    Draw,
    Clay,
    Crease,
    Inflate,
    Smooth,
    Pinch,
    Flatten,
    Mask,
}

impl SculptTool {
    pub const ALL: [Self; 9] = [
        Self::Draw,
        Self::Clay,
        Self::Crease,
        Self::Inflate,
        Self::Smooth,
        Self::Pinch,
        Self::Flatten,
        Self::Grab,
        Self::Mask,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Grab => "Grab",
            Self::Draw => "Draw",
            Self::Clay => "Clay",
            Self::Crease => "Crease",
            Self::Inflate => "Inflate",
            Self::Smooth => "Smooth",
            Self::Pinch => "Pinch",
            Self::Flatten => "Flatten",
            Self::Mask => "Mask",
        }
    }

    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Grab => "Pull a region of the surface with the pointer.",
            Self::Draw => "Raise the surface, or lower it while inverted.",
            Self::Clay => "Build or trim broad forms against a local surface plane.",
            Self::Crease => "Cut or raise a sharp line while pinching its sides together.",
            Self::Inflate => "Expand the surface along its normals, or deflate it while inverted.",
            Self::Smooth => "Relax surface variation without changing the overall form quickly.",
            Self::Pinch => "Pull nearby surface points toward the brush center.",
            Self::Flatten => "Move the surface toward the brush's local plane.",
            Self::Mask => "Paint protection that limits the effect of sculpting tools.",
        }
    }
}

impl fmt::Display for SculptTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymmetryAxis {
    X,
    Y,
    Z,
}

impl SymmetryAxis {
    #[must_use]
    pub fn reflect_point(self, point: Vec3, plane_center: Vec3) -> Vec3 {
        let mut reflected = point;
        match self {
            Self::X => reflected.x = 2.0 * plane_center.x - reflected.x,
            Self::Y => reflected.y = 2.0 * plane_center.y - reflected.y,
            Self::Z => reflected.z = 2.0 * plane_center.z - reflected.z,
        }
        reflected
    }

    #[must_use]
    pub fn reflect_vector(self, vector: Vec3) -> Vec3 {
        match self {
            Self::X => Vec3::new(-vector.x, vector.y, vector.z),
            Self::Y => Vec3::new(vector.x, -vector.y, vector.z),
            Self::Z => Vec3::new(vector.x, vector.y, -vector.z),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Accumulation {
    #[default]
    Capped,
    Accumulate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushSettings {
    /// Brush radius in mesh/world coordinates.
    pub radius: f32,
    /// Normalized effect strength. Values above one are accepted for expert use.
    pub strength: f32,
    /// Hard core of the brush from 0 (fully soft) to 1 (hard edged).
    pub falloff: f32,
    /// World-space remeshing target. `None` keeps the existing topology.
    pub remesh_target_edge_length: Option<f32>,
    pub invert: bool,
    pub symmetry: Option<SymmetryAxis>,
    pub accumulation: Accumulation,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 1.0,
            strength: 0.35,
            falloff: 0.15,
            remesh_target_edge_length: None,
            invert: false,
            symmetry: None,
            accumulation: Accumulation::Capped,
        }
    }
}

/// One world-space sample of an active pointer stroke.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushSample {
    pub center: Vec3,
    pub normal: Vec3,
    /// World-space pointer movement since the previous accepted sample.
    pub drag_delta: Vec3,
    /// Direction from the camera towards the mesh.
    pub view_direction: Vec3,
    pub seed_triangle: u32,
    pub pressure: f32,
    /// Temporary inversion, normally supplied by the Ctrl modifier.
    pub invert_modifier: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DabKind {
    Spatial,
    Timed,
}

impl BrushSample {
    #[must_use]
    pub fn from_hit(
        hit: &RayHit,
        drag_delta: Vec3,
        view_direction: Vec3,
        pressure: f32,
        invert_modifier: bool,
    ) -> Self {
        Self {
            center: hit.position,
            normal: hit.normal,
            drag_delta,
            view_direction,
            seed_triangle: hit.triangle,
            pressure,
            invert_modifier,
        }
    }

    fn reflected(self, axis: SymmetryAxis, plane_center: Vec3, seed_triangle: u32) -> Self {
        Self {
            center: axis.reflect_point(self.center, plane_center),
            normal: axis.reflect_vector(self.normal),
            drag_delta: axis.reflect_vector(self.drag_delta),
            view_direction: axis.reflect_vector(self.view_direction),
            seed_triangle,
            ..self
        }
    }
}

#[derive(Clone, Debug)]
pub struct StrokeState {
    symmetry_center: Vec3,
    mirrored_seed: Option<u32>,
    original: StrokeOriginals,
    recorder: MeshEditRecorder,
    coverage: CoverageMap,
    planes: HashMap<PlaneChannel, BrushFrame>,
    density_checked: bool,
    warning: Option<String>,
    grab: Option<GrabState>,
}

#[derive(Clone, Copy, Debug)]
struct GrabVertex {
    base: Vec3,
    primary_influence: f32,
    mirrored_influence: f32,
}

#[derive(Clone, Debug, Default)]
struct GrabState {
    vertices: HashMap<u32, GrabVertex>,
    desired_delta: Vec3,
    symmetry: Option<SymmetryAxis>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PassSide {
    Primary,
    Mirrored,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlaneChannel {
    tool: SculptTool,
    side: PassSide,
}

#[derive(Clone, Copy, Debug)]
struct BrushFrame {
    origin: Vec3,
    normal: Vec3,
}

#[derive(Clone, Debug, Default)]
struct StrokeOriginals {
    positions: HashMap<u32, Vec3>,
    masks: HashMap<u32, f32>,
}

#[derive(Clone, Debug)]
struct PendingAdaptiveSample {
    tool: SculptTool,
    settings: BrushSettings,
    samples: SmallVec<[BrushSample; 2]>,
    coverage_channel: Option<CoverageChannel>,
    remapped_coverage: CoverageMap,
    staged_coverage: CoverageMap,
    staged_planes: HashMap<PlaneChannel, BrushFrame>,
    recorder: MeshEditRecorder,
    steps: usize,
    topology_edit_limit: usize,
    topology_changed: bool,
    next_topology_pass: usize,
    stalled_topology_passes: usize,
    topology_stage: Option<PendingTopologyValidation>,
    deformation_stage: Option<PendingDeformationStage>,
}

#[derive(Clone, Debug)]
struct PendingTopologyValidation {
    recorder: MeshEditRecorder,
    changes: MeshChangeSet,
    next_face: usize,
    next_topology_pass: usize,
    stalled_topology_passes: usize,
    support_patch_failed: bool,
}

#[derive(Clone, Copy, Debug)]
struct SafeDeformationSearch {
    low: f32,
    high: f32,
    completed_steps: usize,
}

#[derive(Clone, Debug)]
enum PendingDeformationStage {
    Search(SafeDeformationSearch),
    Validate(DeformationValidation),
}

#[derive(Clone, Debug)]
struct DeformationValidation {
    purpose: DeformationValidationPurpose,
    affected: Vec<u32>,
    faces: Vec<u32>,
    next_face: usize,
}

#[derive(Clone, Copy, Debug)]
enum DeformationValidationPurpose {
    FullStrength,
    SearchTrial {
        search: SafeDeformationSearch,
        factor: f32,
    },
    SearchFinal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CoveragePolarity {
    Add,
    Subtract,
    Neutral,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CoverageChannel {
    tool: SculptTool,
    polarity: CoveragePolarity,
}

type CoverageKey = (CoverageChannel, u32);
type CoverageMap = HashMap<CoverageKey, f32>;

impl CoverageChannel {
    fn for_sample(tool: SculptTool, inverted: bool) -> Option<Self> {
        let polarity = match tool {
            SculptTool::Grab => return None,
            SculptTool::Smooth => CoveragePolarity::Neutral,
            SculptTool::Draw
            | SculptTool::Clay
            | SculptTool::Crease
            | SculptTool::Inflate
            | SculptTool::Pinch
            | SculptTool::Flatten
            | SculptTool::Mask => {
                if inverted {
                    CoveragePolarity::Subtract
                } else {
                    CoveragePolarity::Add
                }
            }
        };
        Some(Self { tool, polarity })
    }
}

fn remap_coverage(coverage: &mut CoverageMap, events: &[VertexRemeshEvent]) {
    for &event in events {
        match event {
            VertexRemeshEvent::Created { vertex } => {
                coverage.retain(|&(_, candidate), _| candidate != vertex);
            }
            VertexRemeshEvent::Split {
                vertex,
                first,
                second,
            } => {
                let channels = coverage
                    .keys()
                    .filter_map(|&(channel, candidate)| {
                        (candidate == first || candidate == second).then_some(channel)
                    })
                    .collect::<HashSet<_>>();
                for channel in channels {
                    let first = coverage.get(&(channel, first)).copied().unwrap_or(0.0);
                    let second = coverage.get(&(channel, second)).copied().unwrap_or(0.0);
                    coverage.insert((channel, vertex), (first + second) * 0.5);
                }
            }
            VertexRemeshEvent::Collapsed { keep, remove } => {
                let channels = coverage
                    .keys()
                    .filter_map(|&(channel, candidate)| {
                        (candidate == keep || candidate == remove).then_some(channel)
                    })
                    .collect::<HashSet<_>>();
                for channel in channels {
                    let kept = coverage.get(&(channel, keep)).copied().unwrap_or(0.0);
                    let removed = coverage.remove(&(channel, remove)).unwrap_or(0.0);
                    coverage.insert((channel, keep), (kept + removed) * 0.5);
                }
            }
            VertexRemeshEvent::Remapped { from, to } => {
                let moved = coverage
                    .iter()
                    .filter_map(|(&(channel, candidate), &influence)| {
                        (candidate == from).then_some((channel, influence))
                    })
                    .collect::<Vec<_>>();
                coverage.retain(|&(_, candidate), _| candidate != from && candidate != to);
                for (channel, influence) in moved {
                    coverage.insert((channel, to), influence);
                }
            }
        }
    }
}

/// The editable result of a completed stroke. Adaptive topology is already
/// applied while sampling; `topology` stores its exact whole-stroke history.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrokeOutcome {
    pub edit: LocalEdit,
    pub topology: Option<MeshEditDelta>,
    pub warning: Option<String>,
}

#[derive(Default, Debug)]
pub struct SculptEngine {
    active: Option<StrokeState>,
    pending_sample: Option<PendingAdaptiveSample>,
    symmetry_center: Option<Vec3>,
    traversal: VertexTraversalScratch,
    source_positions: HashMap<u32, Vec3>,
    updated_vertices: Vec<u32>,
    mesh_changes: Option<MeshChangeSet>,
    warning: Option<String>,
    error: Option<String>,
    sample_committed: bool,
}

impl SculptEngine {
    #[must_use]
    pub fn is_stroking(&self) -> bool {
        self.active.is_some()
    }

    #[must_use]
    pub fn has_pending_sample(&self) -> bool {
        self.pending_sample.is_some()
    }

    pub fn cancel_pending_sample(&mut self, mesh: &mut Mesh, message: &str) -> bool {
        self.clear_step_outputs();
        let Some(pending) = self.pending_sample.take() else {
            return false;
        };
        self.rollback_pending_sample(mesh, pending, message, true)
    }

    pub fn continue_pending_sample_safely(&mut self, mesh: &mut Mesh) -> Result<bool, String> {
        let checkpoint = self.pending_sample.clone();
        match catch_unwind(AssertUnwindSafe(|| self.continue_pending_sample(mesh))) {
            Ok(changed) => Ok(changed),
            Err(payload) => {
                self.clear_step_outputs();
                self.pending_sample = None;
                if let Some(pending) = checkpoint {
                    self.rollback_pending_sample_without_message(mesh, pending);
                }
                let detail = payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_owned());
                Err(format!(
                    "Adaptive topology panicked and the brush sample was rolled back: {detail}"
                ))
            }
        }
    }

    /// Resets per-document state after importing or replacing a mesh.
    pub fn reset_for_mesh(&mut self, mesh: &Mesh) {
        self.active = None;
        self.pending_sample = None;
        self.symmetry_center = Some(mesh.center().unwrap_or(Vec3::ZERO));
        self.traversal = VertexTraversalScratch::default();
        self.source_positions = HashMap::new();
        self.updated_vertices = Vec::new();
        self.mesh_changes = None;
        self.warning = None;
        self.error = None;
        self.sample_committed = false;
    }

    pub fn begin_stroke(&mut self, mesh: &Mesh) {
        let symmetry_center = *self
            .symmetry_center
            .get_or_insert_with(|| mesh.center().unwrap_or(Vec3::ZERO));
        self.active = Some(StrokeState {
            symmetry_center,
            mirrored_seed: None,
            original: StrokeOriginals::default(),
            recorder: MeshEditRecorder::new(mesh),
            coverage: CoverageMap::new(),
            planes: HashMap::new(),
            density_checked: false,
            warning: None,
            grab: None,
        });
        self.pending_sample = None;
        self.updated_vertices.clear();
        self.mesh_changes = None;
        self.warning = None;
        self.error = None;
        self.sample_committed = false;
    }

    /// Captures the region affected by Grab without deforming it.
    pub fn anchor_grab(
        &mut self,
        mesh: &Mesh,
        settings: &BrushSettings,
        sample: BrushSample,
    ) -> bool {
        let Some(stroke) = self.active.as_ref() else {
            return false;
        };
        if settings.radius <= f32::EPSILON
            || !settings.radius.is_finite()
            || !sample.center.is_finite()
            || !sample.normal.is_finite()
            || !sample.view_direction.is_finite()
        {
            return false;
        }
        let symmetry_center = stroke.symmetry_center;
        let mut samples = SmallVec::<[(BrushSample, PassSide); 2]>::new();
        samples.push((sample, PassSide::Primary));
        if let Some(axis) = settings.symmetry {
            let center = axis.reflect_point(sample.center, symmetry_center);
            if let Some(seed) = mesh.nearest_triangle(center) {
                samples.push((
                    sample.reflected(axis, symmetry_center, seed),
                    PassSide::Mirrored,
                ));
            }
        }

        let mut passes = SmallVec::<[PreparedPass; 2]>::new();
        for (sample, side) in samples {
            passes.push(PreparedPass::new(
                mesh,
                sample,
                settings.radius,
                side,
                &mut self.traversal,
            ));
        }
        let strength = settings.strength.abs() * sample.pressure.clamp(0.0, 1.0);
        let mut vertices = HashMap::<u32, GrabVertex>::new();
        for pass in &passes {
            for &vertex in &pass.vertices {
                let Some(&position) = mesh.positions.get(vertex as usize) else {
                    continue;
                };
                let falloff = brush_falloff(
                    position.distance(pass.sample.center) / settings.radius,
                    settings.falloff,
                );
                let unmasked = 1.0
                    - mesh
                        .mask
                        .get(vertex as usize)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0);
                let influence = (strength * falloff * unmasked).min(1.0);
                if influence <= f32::EPSILON {
                    continue;
                }
                let entry = vertices.entry(vertex).or_insert(GrabVertex {
                    base: position,
                    primary_influence: 0.0,
                    mirrored_influence: 0.0,
                });
                match pass.side {
                    PassSide::Primary => entry.primary_influence = influence,
                    PassSide::Mirrored => entry.mirrored_influence = influence,
                }
            }
        }
        if vertices.is_empty() {
            return false;
        }
        let warning = density_warning(mesh, &passes, settings.radius);
        let stroke = self.active.as_mut().expect("active stroke checked above");
        stroke.density_checked = true;
        stroke.warning = warning;
        stroke.grab = Some(GrabState {
            vertices,
            desired_delta: Vec3::ZERO,
            symmetry: settings.symmetry,
        });
        true
    }

    /// Moves an anchored Grab selection without requiring another surface hit.
    pub fn apply_grab_delta(&mut self, mesh: &mut Mesh, drag_delta: Vec3, pressure: f32) -> bool {
        self.clear_step_outputs();
        if !drag_delta.is_finite() || !pressure.is_finite() {
            return false;
        }
        let Some(stroke) = self.active.as_mut() else {
            return false;
        };
        let Some(grab) = stroke.grab.as_mut() else {
            return false;
        };
        grab.desired_delta += drag_delta * pressure.clamp(0.0, 1.0);
        let desired = grab.desired_delta;
        let mut full = HashMap::with_capacity(grab.vertices.len());
        for (&vertex, grabbed) in &grab.vertices {
            let primary = grabbed.primary_influence;
            let mirrored = grabbed.mirrored_influence;
            let offset = if primary > 0.0 && mirrored > 0.0 {
                let reflected = grab
                    .symmetry
                    .map_or(desired, |axis| axis.reflect_vector(desired));
                (desired * primary + reflected * mirrored) / (primary + mirrored)
                    * primary.max(mirrored)
            } else if mirrored > 0.0 {
                grab.symmetry
                    .map_or(desired, |axis| axis.reflect_vector(desired))
                    * mirrored
            } else {
                desired * primary
            };
            let next = grabbed.base + offset;
            if mesh
                .positions
                .get(vertex as usize)
                .is_some_and(|current| *current != next)
            {
                full.insert(vertex, next);
            }
        }
        if full.is_empty() {
            return false;
        }

        let mut accepted = None;
        for scale in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
            let candidates = full
                .iter()
                .filter_map(|(&vertex, &target)| {
                    let grabbed = grab.vertices.get(&vertex)?;
                    Some((vertex, grabbed.base + (target - grabbed.base) * scale))
                })
                .collect::<HashMap<_, _>>();
            if let Some(faces) = mesh.validated_deformation_candidates(&candidates) {
                accepted = Some((candidates, faces));
                break;
            }
        }
        let Some((candidates, faces)) = accepted else {
            self.error = Some(
                "Grab could not find a safe deformation step; move the pointer back toward the anchored region"
                    .to_owned(),
            );
            return false;
        };
        let mut affected = Vec::with_capacity(candidates.len());
        for (vertex, next) in candidates {
            let index = vertex as usize;
            let Some(position) = mesh.positions.get_mut(index) else {
                continue;
            };
            stroke.original.positions.entry(vertex).or_insert(*position);
            *position = next;
            affected.push(vertex);
        }
        self.updated_vertices = mesh.update_deformed_faces(&faces);
        self.updated_vertices.extend(affected);
        self.updated_vertices.sort_unstable();
        self.updated_vertices.dedup();
        self.sample_committed = true;
        true
    }

    /// Takes the deduplicated vertex IDs changed by the latest brush sample.
    ///
    /// This includes mask-only edits and is cleared before every sample, so an
    /// empty result always means the latest sample made no editable change.
    #[must_use]
    pub fn take_updated_vertices(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.updated_vertices)
    }

    #[must_use]
    pub fn take_mesh_changes(&mut self) -> Option<MeshChangeSet> {
        self.mesh_changes.take()
    }

    #[must_use]
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    #[must_use]
    pub fn take_warning(&mut self) -> Option<String> {
        self.warning.take()
    }

    /// Returns whether the latest completed sample committed an editable change.
    /// Intermediate topology preparation and a subsequent rollback both return false.
    #[must_use]
    pub fn take_sample_committed(&mut self) -> bool {
        std::mem::take(&mut self.sample_committed)
    }

    /// Applies one brush sample and returns whether it changed the mesh.
    pub fn apply_sample(
        &mut self,
        mesh: &mut Mesh,
        tool: SculptTool,
        settings: &BrushSettings,
        sample: BrushSample,
        dab_kind: DabKind,
    ) -> bool {
        self.clear_step_outputs();
        if self.pending_sample.is_some() {
            return false;
        }
        let Some(stroke) = self.active.as_ref() else {
            return false;
        };
        if mesh.positions.is_empty()
            || !settings.radius.is_finite()
            || settings.radius <= f32::EPSILON
            || !settings.strength.is_finite()
            || settings.strength.abs() <= f32::EPSILON
            || !sample.center.is_finite()
            || !sample.normal.is_finite()
            || !sample.view_direction.is_finite()
            || !sample.pressure.is_finite()
            || sample.pressure.clamp(0.0, 1.0) <= 0.0
        {
            return false;
        }

        let symmetry_center = stroke.symmetry_center;
        let cached_mirrored_seed = stroke.mirrored_seed;
        let mut samples = SmallVec::<[BrushSample; 2]>::from_slice(&[sample]);

        if let Some(axis) = settings.symmetry {
            let mirrored_center = axis.reflect_point(sample.center, symmetry_center);
            let seed_triangle = cached_mirrored_seed
                .filter(|&seed| triangle_is_near(mesh, seed, mirrored_center, settings.radius))
                .or_else(|| mesh.nearest_triangle(mirrored_center));
            if let Some(seed_triangle) = seed_triangle {
                let mirrored = sample.reflected(axis, symmetry_center, seed_triangle);
                samples.push(mirrored);
                if let Some(stroke) = self.active.as_mut() {
                    stroke.mirrored_seed = Some(seed_triangle);
                }
            }
        }

        let target_edge_length = settings
            .remesh_target_edge_length
            .filter(|target| target.is_finite() && *target > 0.0);
        let adaptive =
            !matches!(tool, SculptTool::Grab | SculptTool::Mask) && target_edge_length.is_some();
        if adaptive {
            let remapped_coverage = self
                .active
                .as_ref()
                .expect("active stroke checked above")
                .coverage
                .clone();
            let coverage_channel = (dab_kind == DabKind::Spatial
                && settings.accumulation == Accumulation::Capped)
                .then(|| {
                    CoverageChannel::for_sample(tool, settings.invert ^ sample.invert_modifier)
                })
                .flatten();
            self.pending_sample = Some(PendingAdaptiveSample {
                tool,
                settings: *settings,
                samples,
                coverage_channel,
                remapped_coverage,
                staged_coverage: CoverageMap::new(),
                staged_planes: HashMap::new(),
                recorder: MeshEditRecorder::new(mesh),
                steps: 0,
                topology_edit_limit: MAX_ADAPTIVE_TOPOLOGY_EDITS_PER_STEP,
                topology_changed: false,
                next_topology_pass: 0,
                stalled_topology_passes: 0,
                topology_stage: None,
                deformation_stage: None,
            });
            return false;
        }
        for brush_sample in &mut samples {
            if let Some(seed) = mesh.nearest_triangle(brush_sample.center) {
                brush_sample.seed_triangle = seed;
            }
        }
        let mut passes = SmallVec::<[PreparedPass; 2]>::new();
        for (index, brush_sample) in samples.into_iter().enumerate() {
            passes.push(PreparedPass::new(
                mesh,
                brush_sample,
                settings.radius,
                if index == 0 {
                    PassSide::Primary
                } else {
                    PassSide::Mirrored
                },
                &mut self.traversal,
            ));
        }

        capture_source_positions(mesh, tool, &passes, &mut self.source_positions);

        if let Some(stroke) = self.active.as_mut()
            && !stroke.density_checked
        {
            stroke.density_checked = true;
            stroke.warning = density_warning(mesh, &passes, settings.radius);
        }

        let (frames, staged_planes) = {
            let stroke = self.active.as_ref().expect("active stroke checked above");
            prepare_brush_frames(
                mesh,
                tool,
                settings,
                &passes,
                &self.source_positions,
                &stroke.planes,
            )
        };

        let channel = (dab_kind == DabKind::Spatial
            && settings.accumulation == Accumulation::Capped)
            .then(|| CoverageChannel::for_sample(tool, settings.invert ^ sample.invert_modifier))
            .flatten();
        let mut staged_coverage = CoverageMap::new();
        let mut proposals = ProposedEdits::default();
        {
            let stroke = self.active.as_mut().expect("active stroke checked above");
            let mut coverage = SampleCoverage {
                channel,
                committed: &stroke.coverage,
                staged: &mut staged_coverage,
            };
            let context = ProposalContext {
                mesh,
                tool,
                settings,
                source_positions: &self.source_positions,
            };
            propose_passes(&context, &passes, &frames, &mut coverage, &mut proposals);
        }
        if tool == SculptTool::Mask {
            if proposals.masks.is_empty() {
                return false;
            }
            let stroke = self.active.as_mut().expect("active stroke checked above");
            let mut affected = Vec::with_capacity(proposals.masks.len());
            for (vertex, next) in proposals.masks {
                let index = vertex as usize;
                let Some(value) = mesh.mask.get_mut(index) else {
                    continue;
                };
                if *value == next {
                    continue;
                }
                stroke.original.masks.entry(vertex).or_insert(*value);
                *value = next;
                affected.push(vertex);
            }
            if affected.is_empty() {
                return false;
            }
            self.commit_coverage(staged_coverage);
            self.commit_planes(staged_planes);
            self.updated_vertices = affected;
            self.sample_committed = true;
            return true;
        }

        let candidates = candidate_positions(mesh, &proposals.positions, 1.0);
        if candidates.is_empty() {
            return false;
        }
        let mut accepted = None;
        for scale in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
            let scaled = if scale == 1.0 {
                candidates.clone()
            } else {
                candidate_positions(mesh, &proposals.positions, scale)
            };
            if let Some(faces) = mesh.validated_deformation_candidates(&scaled) {
                accepted = Some((scale, scaled, faces));
                break;
            }
        }
        let Some((accepted_scale, candidates, faces)) = accepted else {
            self.error = Some(
                "Sculpt sample rejected an invalid local mesh update; no safe partial step was available"
                    .to_owned(),
            );
            return false;
        };
        let stroke = self.active.as_mut().expect("active stroke checked above");
        let mut affected = Vec::with_capacity(candidates.len());
        for (vertex, next) in candidates {
            let index = vertex as usize;
            let Some(position) = mesh.positions.get_mut(index) else {
                continue;
            };
            if *position == next {
                continue;
            }
            if tool != SculptTool::Grab
                && let Some(grabbed) = stroke
                    .grab
                    .as_mut()
                    .and_then(|grab| grab.vertices.get_mut(&vertex))
            {
                grabbed.base += next - *position;
            }
            stroke.original.positions.entry(vertex).or_insert(*position);
            *position = next;
            affected.push(vertex);
        }
        self.updated_vertices = mesh.update_deformed_faces(&faces);
        self.updated_vertices.extend(affected);
        self.updated_vertices.sort_unstable();
        self.updated_vertices.dedup();
        self.commit_coverage_scaled(staged_coverage, accepted_scale);
        self.commit_planes(staged_planes);
        self.sample_committed = true;
        true
    }

    /// Continues a quality-gated adaptive sample for one bounded topology step.
    /// Returns whether this step changed the visible mesh.
    pub fn continue_pending_sample(&mut self, mesh: &mut Mesh) -> bool {
        self.clear_step_outputs();
        let Some(mut pending) = self.pending_sample.take() else {
            return false;
        };
        if self.active.is_none() {
            return self.rollback_pending_sample(
                mesh,
                pending,
                "Adaptive topology lost its active stroke; the brush sample was rolled back",
                true,
            );
        }
        if let Some(validation) = pending.topology_stage.take() {
            return self.continue_topology_validation(mesh, pending, validation);
        }
        if let Some(stage) = pending.deformation_stage.take() {
            return match stage {
                PendingDeformationStage::Search(search) => {
                    self.start_safe_deformation_trial(mesh, pending, search)
                }
                PendingDeformationStage::Validate(validation) => {
                    self.continue_deformation_validation(mesh, pending, validation)
                }
            };
        }

        pending.steps += 1;
        let target = pending
            .settings
            .remesh_target_edge_length
            .expect("pending adaptive samples have a remesh target");
        let maximum_edge_length = target * ADAPTIVE_SPLIT_THRESHOLD;
        let mut topology_changes = MeshChangeSet::default();
        let mut topology_edits = 0;
        let mut step_recorder = MeshEditRecorder::new(mesh);
        let mut next_topology_pass = pending.next_topology_pass;
        let mut stalled_topology_passes = pending.stalled_topology_passes;
        let mut regular_pass_processed = false;
        let mut support_patch_failed = false;

        // Round-robin one regular remesh pass per step. Each side gets the full
        // bounded edit slice without making a symmetric dab pay for two growing
        // region scans in one UI frame.
        for offset in 0..pending.samples.len() {
            let pass_index = (pending.next_topology_pass + offset) % pending.samples.len();
            let brush_sample = pending.samples[pass_index];
            let pass = PreparedPass::new(
                mesh,
                brush_sample,
                pending.settings.radius,
                if pass_index == 0 {
                    PassSide::Primary
                } else {
                    PassSide::Mirrored
                },
                &mut self.traversal,
            );
            let seed = mesh
                .nearest_triangle(brush_sample.center)
                .unwrap_or(brush_sample.seed_triangle);
            if pass.vertices.is_empty() {
                // The support patch is a small, bounded atomic topology operation.
                // Splitting it across frames would expose a hole in the surface.
                if let Some((added_vertices, changes)) = mesh.insert_brush_support_patch(
                    seed,
                    brush_sample.center,
                    pending.settings.radius,
                    target,
                    &mut step_recorder,
                ) {
                    topology_edits += added_vertices;
                    topology_changes.merge(changes);
                    stalled_topology_passes = 0;
                } else {
                    support_patch_failed = true;
                }
                continue;
            }
            if pass.has_remesh_support(
                mesh,
                pending.settings.radius,
                pending.settings.strength,
                pending.settings.falloff,
                maximum_edge_length,
            ) {
                continue;
            }
            if regular_pass_processed {
                continue;
            }
            regular_pass_processed = true;
            next_topology_pass = (pass_index + 1) % pending.samples.len();

            let active = mesh.brush_remesh_vertices(
                seed,
                brush_sample.center,
                pending.settings.radius,
                target,
                brush_sample.view_direction,
            );
            if active.is_empty() {
                stalled_topology_passes += 1;
                continue;
            }
            let mut remesh = RemeshSettings::new(target);
            remesh.iterations = 4;
            remesh.max_topology_edits = pending.topology_edit_limit;
            remesh.split_threshold = ADAPTIVE_SPLIT_THRESHOLD;
            remesh.collapse_threshold = ADAPTIVE_COLLAPSE_THRESHOLD;
            remesh.relaxation = 0.0;
            let outcome = mesh.remesh_brush_region(
                &active,
                brush_sample.center,
                pending.settings.radius,
                remesh,
                &mut step_recorder,
            );
            let pass_edits = outcome.stats.splits + outcome.stats.collapses + outcome.stats.flips;
            topology_edits += pass_edits;
            topology_changes.merge(outcome.changes);
            if pass_edits == 0 {
                stalled_topology_passes += 1;
            } else {
                stalled_topology_passes = 0;
            }
        }

        if topology_edits != 0 {
            if !mesh.local_changes_are_valid(&topology_changes) {
                step_recorder.finish(mesh).apply_before(mesh);
                return self.retry_invalid_topology_step(mesh, pending);
            }
            pending.topology_stage = Some(PendingTopologyValidation {
                recorder: step_recorder,
                changes: topology_changes,
                next_face: 0,
                next_topology_pass,
                stalled_topology_passes,
                support_patch_failed,
            });
            self.warning = Some("Validating intersection-free mesh detail".to_owned());
            self.pending_sample = Some(pending);
            return false;
        }
        pending.next_topology_pass = next_topology_pass;
        pending.stalled_topology_passes = stalled_topology_passes;
        self.finish_topology_step(mesh, pending, topology_changes, false, support_patch_failed)
    }

    fn continue_topology_validation(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
        mut validation: PendingTopologyValidation,
    ) -> bool {
        let end = (validation.next_face + MAX_TOPOLOGY_INTERSECTION_FACES_PER_STEP)
            .min(validation.changes.dirty_faces.len());
        if mesh.faces_have_self_intersections(
            &validation.changes.dirty_faces[validation.next_face..end],
        ) {
            validation.recorder.finish(mesh).apply_before(mesh);
            return self.retry_invalid_topology_step(mesh, pending);
        }
        validation.next_face = end;
        if end < validation.changes.dirty_faces.len() {
            pending.topology_stage = Some(validation);
            self.warning = Some("Validating intersection-free mesh detail".to_owned());
            self.pending_sample = Some(pending);
            return false;
        }

        pending.recorder.absorb_recorder(validation.recorder, mesh);
        pending.next_topology_pass = validation.next_topology_pass;
        pending.stalled_topology_passes = validation.stalled_topology_passes;
        pending.topology_changed = true;
        self.finish_topology_step(
            mesh,
            pending,
            validation.changes,
            true,
            validation.support_patch_failed,
        )
    }

    fn retry_invalid_topology_step(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
    ) -> bool {
        if pending.topology_edit_limit == 1 || pending.steps >= MAX_ADAPTIVE_TOPOLOGY_STEPS {
            return self.rollback_pending_sample(
                mesh,
                pending,
                "Adaptive topology could not produce an intersection-free support mesh; the brush sample was rolled back",
                true,
            );
        }
        pending.topology_edit_limit = (pending.topology_edit_limit / 2).max(1);
        self.warning = Some("Preparing intersection-free mesh detail".to_owned());
        self.pending_sample = Some(pending);
        false
    }

    fn finish_topology_step(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
        topology_changes: MeshChangeSet,
        topology_changed_this_step: bool,
        support_patch_failed: bool,
    ) -> bool {
        if topology_changed_this_step {
            remap_coverage(
                &mut pending.remapped_coverage,
                &topology_changes.vertex_remesh_events,
            );
        }
        let maximum_edge_length = pending
            .settings
            .remesh_target_edge_length
            .expect("pending adaptive samples have a remesh target")
            * ADAPTIVE_SPLIT_THRESHOLD;
        for brush_sample in &mut pending.samples {
            if let Some(seed) = mesh.nearest_triangle(brush_sample.center) {
                brush_sample.seed_triangle = seed;
            }
        }
        let mut passes = SmallVec::<[PreparedPass; 2]>::new();
        for (index, &brush_sample) in pending.samples.iter().enumerate() {
            passes.push(PreparedPass::new(
                mesh,
                brush_sample,
                pending.settings.radius,
                if index == 0 {
                    PassSide::Primary
                } else {
                    PassSide::Mirrored
                },
                &mut self.traversal,
            ));
        }

        let support_ready = !passes.iter().any(|pass| pass.vertices.is_empty())
            && passes.iter().all(|pass| {
                pass.has_remesh_support(
                    mesh,
                    pending.settings.radius,
                    pending.settings.strength,
                    pending.settings.falloff,
                    maximum_edge_length,
                )
            });
        if !support_ready {
            if support_patch_failed
                || pending.stalled_topology_passes >= pending.samples.len()
                || pending.steps >= MAX_ADAPTIVE_TOPOLOGY_STEPS
            {
                return self.rollback_pending_sample(
                    mesh,
                    pending,
                    "Mesh resolution could not safely support this brush; the brush sample was rolled back",
                    false,
                );
            }
            self.publish_adaptive_changes(mesh, topology_changes, &[]);
            self.warning = Some("Preparing mesh detail for this brush".to_owned());
            self.pending_sample = Some(pending);
            return true;
        }
        if topology_changed_this_step {
            self.publish_adaptive_changes(mesh, topology_changes, &[]);
            self.warning = Some("Preparing safe brush deformation".to_owned());
            self.pending_sample = Some(pending);
            return true;
        }

        self.start_deformation_validation(
            mesh,
            pending,
            DeformationValidationPurpose::FullStrength,
            1.0,
        )
    }

    fn start_safe_deformation_trial(
        &mut self,
        mesh: &mut Mesh,
        pending: PendingAdaptiveSample,
        search: SafeDeformationSearch,
    ) -> bool {
        if search.completed_steps >= SAFE_DEFORMATION_SEARCH_STEPS {
            if search.low < MIN_SAFE_DEFORMATION_FACTOR {
                return self.rollback_pending_sample(
                    mesh,
                    pending,
                    "Brush movement rejected to prevent self-intersection",
                    false,
                );
            }
            return self.start_deformation_validation(
                mesh,
                pending,
                DeformationValidationPurpose::SearchFinal,
                search.low,
            );
        }

        let factor = (search.low + search.high) * 0.5;
        self.start_deformation_validation(
            mesh,
            pending,
            DeformationValidationPurpose::SearchTrial { search, factor },
            factor,
        )
    }

    fn start_deformation_validation(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
        purpose: DeformationValidationPurpose,
        strength_factor: f32,
    ) -> bool {
        let Some(passes) = self.prepare_pending_deformation_passes(mesh, &mut pending) else {
            return self.rollback_pending_sample(
                mesh,
                pending,
                "Adaptive topology lost brush support during deformation validation; the brush sample was rolled back",
                true,
            );
        };
        capture_source_positions(mesh, pending.tool, &passes, &mut self.source_positions);
        let mut scaled = pending.settings;
        scaled.strength *= strength_factor;
        let (frames, staged_planes) = {
            let stroke = self.active.as_ref().expect("active stroke checked above");
            prepare_brush_frames(
                mesh,
                pending.tool,
                &scaled,
                &passes,
                &self.source_positions,
                &stroke.planes,
            )
        };
        let mut staged_coverage = CoverageMap::new();
        let mut proposals = ProposedEdits::default();
        {
            let mut coverage = SampleCoverage {
                channel: pending.coverage_channel,
                committed: &pending.remapped_coverage,
                staged: &mut staged_coverage,
            };
            let context = ProposalContext {
                mesh,
                tool: pending.tool,
                settings: &scaled,
                source_positions: &self.source_positions,
            };
            propose_passes(&context, &passes, &frames, &mut coverage, &mut proposals);
        }
        let candidates = candidate_positions(mesh, &proposals.positions, 1.0);
        if candidates.is_empty() {
            return match purpose {
                DeformationValidationPurpose::FullStrength => {
                    self.rollback_pending_sample_without_message(mesh, pending)
                }
                DeformationValidationPurpose::SearchFinal => self.rollback_pending_sample(
                    mesh,
                    pending,
                    "Brush movement rejected to prevent self-intersection",
                    false,
                ),
                DeformationValidationPurpose::SearchTrial { .. } => {
                    self.complete_deformation_validation(mesh, pending, purpose, false, &[], &[])
                }
            };
        }

        let Some(faces) = mesh.validated_deformation_candidates(&candidates) else {
            return self.complete_deformation_validation(mesh, pending, purpose, false, &[], &[]);
        };

        let record = !matches!(purpose, DeformationValidationPurpose::SearchTrial { .. });
        let mut affected = Vec::with_capacity(candidates.len());
        {
            let stroke = self.active.as_mut().expect("active stroke checked above");
            for (vertex, next) in candidates {
                let index = vertex as usize;
                if record {
                    pending.recorder.record_vertex(mesh, vertex);
                }
                let Some(position) = mesh.positions.get_mut(index) else {
                    continue;
                };
                if record {
                    stroke.original.positions.entry(vertex).or_insert(*position);
                }
                *position = next;
                affected.push(vertex);
            }
        }
        affected.sort_unstable();
        if record {
            pending.staged_coverage = staged_coverage;
            pending.staged_planes = staged_planes;
        }
        mesh.update_deformed_faces(&faces);
        let validation = DeformationValidation {
            purpose,
            affected,
            faces,
            next_face: 0,
        };
        if validation.faces.len() <= MAX_DEFORMATION_INTERSECTION_FACES_PER_STEP {
            return self.continue_deformation_validation(mesh, pending, validation);
        }
        pending.deformation_stage = Some(PendingDeformationStage::Validate(validation));
        self.warning = Some("Validating safe brush deformation".to_owned());
        self.pending_sample = Some(pending);
        false
    }

    fn prepare_pending_deformation_passes(
        &mut self,
        mesh: &Mesh,
        pending: &mut PendingAdaptiveSample,
    ) -> Option<SmallVec<[PreparedPass; 2]>> {
        for brush_sample in &mut pending.samples {
            if let Some(seed) = mesh.nearest_triangle(brush_sample.center) {
                brush_sample.seed_triangle = seed;
            }
        }
        let mut passes = SmallVec::<[PreparedPass; 2]>::new();
        for (index, &brush_sample) in pending.samples.iter().enumerate() {
            passes.push(PreparedPass::new(
                mesh,
                brush_sample,
                pending.settings.radius,
                if index == 0 {
                    PassSide::Primary
                } else {
                    PassSide::Mirrored
                },
                &mut self.traversal,
            ));
        }
        let maximum_edge_length = pending
            .settings
            .remesh_target_edge_length
            .expect("pending adaptive samples have a remesh target")
            * ADAPTIVE_SPLIT_THRESHOLD;
        passes
            .iter()
            .all(|pass| {
                pass.has_remesh_support(
                    mesh,
                    pending.settings.radius,
                    pending.settings.strength,
                    pending.settings.falloff,
                    maximum_edge_length,
                )
            })
            .then_some(passes)
    }

    fn continue_deformation_validation(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
        mut validation: DeformationValidation,
    ) -> bool {
        let end = (validation.next_face + MAX_DEFORMATION_INTERSECTION_FACES_PER_STEP)
            .min(validation.faces.len());
        let intersects =
            mesh.faces_have_self_intersections(&validation.faces[validation.next_face..end]);
        if intersects {
            restore_deformation_faces(
                mesh,
                &validation.affected,
                &self.source_positions,
                &validation.faces,
            );
            return self.complete_deformation_validation(
                mesh,
                pending,
                validation.purpose,
                false,
                &validation.affected,
                &validation.faces,
            );
        }
        validation.next_face = end;
        if end < validation.faces.len() {
            pending.deformation_stage = Some(PendingDeformationStage::Validate(validation));
            self.warning = Some("Validating safe brush deformation".to_owned());
            self.pending_sample = Some(pending);
            return false;
        }
        self.complete_deformation_validation(
            mesh,
            pending,
            validation.purpose,
            true,
            &validation.affected,
            &validation.faces,
        )
    }

    fn complete_deformation_validation(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
        purpose: DeformationValidationPurpose,
        safe: bool,
        affected: &[u32],
        faces: &[u32],
    ) -> bool {
        match purpose {
            DeformationValidationPurpose::FullStrength if safe => {
                self.finish_pending_adaptive_sample(mesh, pending, true, affected, faces)
            }
            DeformationValidationPurpose::FullStrength => {
                pending.staged_coverage.clear();
                pending.staged_planes.clear();
                pending.deformation_stage =
                    Some(PendingDeformationStage::Search(SafeDeformationSearch {
                        low: 0.0,
                        high: 1.0,
                        completed_steps: 0,
                    }));
                self.warning = Some("Finding a safe brush deformation".to_owned());
                self.pending_sample = Some(pending);
                false
            }
            DeformationValidationPurpose::SearchTrial { mut search, factor } => {
                if safe {
                    restore_deformation_faces(mesh, affected, &self.source_positions, faces);
                    search.low = factor;
                } else {
                    search.high = factor;
                }
                search.completed_steps += 1;
                pending.deformation_stage = Some(PendingDeformationStage::Search(search));
                self.warning = Some("Finding a safe brush deformation".to_owned());
                self.pending_sample = Some(pending);
                false
            }
            DeformationValidationPurpose::SearchFinal if safe => {
                self.warning =
                    Some("Brush movement limited to prevent self-intersection".to_owned());
                self.finish_pending_adaptive_sample(mesh, pending, true, affected, faces)
            }
            DeformationValidationPurpose::SearchFinal => self.rollback_pending_sample(
                mesh,
                pending,
                "Brush movement rejected to prevent self-intersection",
                false,
            ),
        }
    }

    fn finish_pending_adaptive_sample(
        &mut self,
        mesh: &mut Mesh,
        mut pending: PendingAdaptiveSample,
        deformation_changed: bool,
        affected: &[u32],
        faces: &[u32],
    ) -> bool {
        if deformation_changed {
            self.updated_vertices = mesh.update_deformed_faces(faces);
            self.updated_vertices.extend(affected.iter().copied());
            self.updated_vertices.sort_unstable();
            self.updated_vertices.dedup();
        }
        let sample_changed = pending.topology_changed || deformation_changed;
        if sample_changed {
            for (key, influence) in std::mem::take(&mut pending.staged_coverage) {
                let maximum = pending.remapped_coverage.entry(key).or_default();
                *maximum = maximum.max(influence);
            }
            self.active
                .as_mut()
                .expect("active stroke checked above")
                .coverage = std::mem::take(&mut pending.remapped_coverage);
            self.commit_planes(std::mem::take(&mut pending.staged_planes));
            self.active
                .as_mut()
                .expect("active stroke checked above")
                .recorder
                .absorb_recorder(pending.recorder, mesh);
            self.sample_committed = true;
        }
        if deformation_changed {
            let updated_vertices = std::mem::take(&mut self.updated_vertices);
            self.publish_adaptive_changes(mesh, MeshChangeSet::default(), &updated_vertices);
        }
        deformation_changed
    }

    fn commit_coverage(&mut self, staged: CoverageMap) {
        let Some(stroke) = self.active.as_mut() else {
            return;
        };
        for (key, influence) in staged {
            let maximum = stroke.coverage.entry(key).or_default();
            *maximum = maximum.max(influence);
        }
    }
    fn commit_coverage_scaled(&mut self, staged: CoverageMap, scale: f32) {
        let Some(stroke) = self.active.as_mut() else {
            return;
        };
        for (key, requested) in staged {
            let previous = stroke.coverage.entry(key).or_default();
            *previous += (requested - *previous).max(0.0) * scale;
        }
    }

    fn commit_planes(&mut self, staged: HashMap<PlaneChannel, BrushFrame>) {
        if let Some(stroke) = self.active.as_mut() {
            stroke.planes.extend(staged);
        }
    }
    fn clear_step_outputs(&mut self) {
        self.updated_vertices.clear();
        self.mesh_changes = None;
        self.warning = None;
        self.error = None;
        self.sample_committed = false;
    }

    fn publish_adaptive_changes(
        &mut self,
        mesh: &Mesh,
        mut changes: MeshChangeSet,
        updated_vertices: &[u32],
    ) {
        changes.include_vertices(updated_vertices.iter().copied());
        changes.finalize(mesh.positions.len(), mesh.triangles.len());
        self.updated_vertices
            .extend(changes.dirty_vertices.iter().copied());
        self.updated_vertices.sort_unstable();
        self.updated_vertices.dedup();
        self.mesh_changes = Some(changes);
    }

    fn rollback_pending_sample(
        &mut self,
        mesh: &mut Mesh,
        pending: PendingAdaptiveSample,
        message: &str,
        is_error: bool,
    ) -> bool {
        let delta = pending.recorder.finish(mesh);
        let changed = !delta.is_empty();
        if changed {
            let changes = delta.apply_before(mesh);
            self.updated_vertices = changes.dirty_vertices.clone();
            self.mesh_changes = Some(changes);
        }
        if is_error {
            self.error = Some(message.to_owned());
        } else {
            self.warning = Some(message.to_owned());
        }
        changed
    }

    fn rollback_pending_sample_without_message(
        &mut self,
        mesh: &mut Mesh,
        pending: PendingAdaptiveSample,
    ) -> bool {
        let delta = pending.recorder.finish(mesh);
        let changed = !delta.is_empty();
        if changed {
            let changes = delta.apply_before(mesh);
            self.updated_vertices = changes.dirty_vertices.clone();
            self.mesh_changes = Some(changes);
        }
        changed
    }

    /// Ends the active stroke and returns any deferred topology work.
    #[must_use]
    pub fn end_stroke(&mut self, mesh: &Mesh) -> StrokeOutcome {
        let Some(mut stroke) = self.active.take() else {
            return StrokeOutcome::default();
        };
        let positions = stroke
            .original
            .positions
            .drain()
            .filter_map(|(vertex, before)| {
                let after = mesh.positions.get(vertex as usize).copied()?;
                Some(PositionChange {
                    vertex,
                    before,
                    after,
                })
            })
            .collect();
        let masks = stroke
            .original
            .masks
            .drain()
            .filter_map(|(vertex, before)| {
                let after = mesh.mask.get(vertex as usize).copied()?;
                Some(MaskChange {
                    vertex,
                    before,
                    after,
                })
            })
            .collect();
        let edit = LocalEdit::new(positions, masks);
        let topology = stroke.recorder.finish(mesh);
        let topology = (!topology.is_empty() && topology.topology_changed()).then_some(topology);
        let warning = stroke.warning.take();
        if edit.is_empty() && topology.is_none() && warning.is_none() {
            return StrokeOutcome::default();
        }
        StrokeOutcome {
            edit,
            topology,
            warning,
        }
    }
}

#[derive(Debug)]
struct PreparedPass {
    sample: BrushSample,
    vertices: Vec<u32>,
    side: PassSide,
}

struct SampleCoverage<'a> {
    channel: Option<CoverageChannel>,
    committed: &'a CoverageMap,
    staged: &'a mut CoverageMap,
}

impl SampleCoverage<'_> {
    fn limited_influence(&mut self, vertex: u32, current: f32) -> f32 {
        let Some(channel) = self.channel else {
            return current;
        };
        let key = (channel, vertex);
        let previous = self
            .committed
            .get(&key)
            .copied()
            .unwrap_or(0.0)
            .max(self.staged.get(&key).copied().unwrap_or(0.0));
        if current <= previous {
            return 0.0;
        }
        self.staged.insert(key, current);
        current - previous
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProposalAccumulator {
    weighted_unit: Vec3,
    influence_sum: f32,
    maximum_influence: f32,
}

type ProposalMap = HashMap<u32, Vec3>;
type MaskProposalMap = HashMap<u32, f32>;

struct ProposalContext<'a> {
    mesh: &'a Mesh,
    tool: SculptTool,
    settings: &'a BrushSettings,
    source_positions: &'a HashMap<u32, Vec3>,
}

#[derive(Default)]
struct ProposedEdits {
    positions: ProposalMap,
    masks: MaskProposalMap,
}

fn propose_passes(
    context: &ProposalContext<'_>,
    passes: &[PreparedPass],
    frames: &[BrushFrame],
    coverage: &mut SampleCoverage<'_>,
    output: &mut ProposedEdits,
) {
    let mut accumulated = HashMap::<u32, ProposalAccumulator>::new();
    let mut mask_influence = HashMap::<u32, f32>::new();
    for (pass, frame) in passes.iter().zip(frames) {
        propose_pass(context, pass, *frame, &mut accumulated, &mut mask_influence);
    }

    let inverted = context.settings.invert
        ^ passes
            .first()
            .is_some_and(|pass| pass.sample.invert_modifier);
    let direction = if inverted { -1.0 } else { 1.0 };
    if context.tool == SculptTool::Mask {
        for (vertex, maximum) in mask_influence {
            let influence = coverage.limited_influence(vertex, maximum);
            let Some(&before) = context.mesh.mask.get(vertex as usize) else {
                continue;
            };
            let next = (before + direction * influence).clamp(0.0, 1.0);
            if next != before {
                output.masks.insert(vertex, next);
            }
        }
        return;
    }

    for (vertex, accumulated) in accumulated {
        if accumulated.influence_sum <= f32::EPSILON {
            continue;
        }
        let influence = coverage.limited_influence(vertex, accumulated.maximum_influence);
        if influence <= f32::EPSILON {
            continue;
        }
        let unit = accumulated.weighted_unit / accumulated.influence_sum;
        let displacement = unit * influence.min(1.0);
        if displacement.is_finite() && displacement.length_squared() > f32::EPSILON.powi(2) {
            output.positions.insert(vertex, displacement);
        }
    }
}

fn restore_deformation_faces(
    mesh: &mut Mesh,
    affected: &[u32],
    source_positions: &HashMap<u32, Vec3>,
    faces: &[u32],
) {
    for &vertex in affected {
        let Some(&source) = source_positions.get(&vertex) else {
            continue;
        };
        if let Some(position) = mesh.positions.get_mut(vertex as usize) {
            *position = source;
        }
    }
    mesh.update_deformed_faces(faces);
}

impl PreparedPass {
    fn new(
        mesh: &Mesh,
        sample: BrushSample,
        radius: f32,
        side: PassSide,
        traversal: &mut VertexTraversalScratch,
    ) -> Self {
        let mut vertices = Vec::new();
        mesh.connected_front_facing_vertices(
            sample.seed_triangle,
            sample.center,
            radius,
            sample.view_direction,
            traversal,
            &mut vertices,
        );
        Self {
            sample,
            vertices,
            side,
        }
    }

    fn has_remesh_support(
        &self,
        mesh: &Mesh,
        radius: f32,
        strength: f32,
        falloff: f32,
        maximum_edge_length: f32,
    ) -> bool {
        let maximum_edge_squared = maximum_edge_length * maximum_edge_length * (1.0 + 1.0e-5);
        !self.vertices.is_empty()
            && self.vertices.iter().all(|&vertex| {
                let index = vertex as usize;
                let Some(&position) = mesh.positions.get(index) else {
                    return false;
                };
                let weight = brush_falloff(position.distance(self.sample.center) / radius, falloff);
                let influence = weight * strength.abs() * self.sample.pressure.clamp(0.0, 1.0);
                if influence < MIN_ADAPTIVE_SUPPORT_INFLUENCE {
                    return true;
                }
                mesh.topology
                    .vertex_neighbors
                    .get(index)
                    .is_some_and(|neighbors| {
                        !neighbors.is_empty()
                            && neighbors.iter().all(|&neighbor| {
                                let neighbor_position = mesh.positions[neighbor as usize];
                                let neighbor_weight = brush_falloff(
                                    neighbor_position.distance(self.sample.center) / radius,
                                    falloff,
                                );
                                let neighbor_influence = neighbor_weight
                                    * strength.abs()
                                    * self.sample.pressure.clamp(0.0, 1.0);
                                let edge_is_short = position.distance_squared(neighbor_position)
                                    <= maximum_edge_squared;
                                let midpoint_weight = brush_falloff(
                                    position
                                        .midpoint(neighbor_position)
                                        .distance(self.sample.center)
                                        / radius,
                                    falloff,
                                );
                                let midpoint_influence = midpoint_weight
                                    * strength.abs()
                                    * self.sample.pressure.clamp(0.0, 1.0);
                                edge_is_short
                                    || ((influence - neighbor_influence).abs()
                                        <= MAX_ADAPTIVE_SUPPORT_INFLUENCE_STEP
                                        && (midpoint_influence
                                            - (influence + neighbor_influence) * 0.5)
                                            .abs()
                                            <= MAX_ADAPTIVE_SUPPORT_INFLUENCE_STEP)
                            })
                    })
            })
    }
}

fn propose_pass(
    context: &ProposalContext<'_>,
    pass: &PreparedPass,
    frame: BrushFrame,
    proposals: &mut HashMap<u32, ProposalAccumulator>,
    mask_influence: &mut HashMap<u32, f32>,
) {
    let mesh = context.mesh;
    let tool = context.tool;
    let settings = context.settings;
    let source_positions = context.source_positions;
    let radius = settings.radius;
    let pressure = pass.sample.pressure.clamp(0.0, 1.0);
    let strength = settings.strength.abs() * pressure;
    if strength <= f32::EPSILON {
        return;
    }

    let inverted = settings.invert ^ pass.sample.invert_modifier;
    let direction = if inverted { -1.0 } else { 1.0 };
    let brush_normal = frame.normal;
    let smooth_units =
        (tool == SculptTool::Smooth).then(|| taubin_smooth_units(mesh, pass, source_positions));
    for &vertex in &pass.vertices {
        let index = vertex as usize;
        let Some(&position) = source_positions.get(&vertex) else {
            continue;
        };
        let distance = position.distance(pass.sample.center);
        let weight = brush_falloff(distance / radius, settings.falloff);
        if weight <= f32::EPSILON {
            continue;
        }

        if tool == SculptTool::Mask {
            let maximum = mask_influence.entry(vertex).or_default();
            *maximum = maximum.max(strength * weight);
            continue;
        }

        let unmasked = 1.0 - mesh.mask.get(index).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let influence = strength * weight * unmasked;
        if influence <= f32::EPSILON {
            continue;
        }

        let unit_displacement = match tool {
            SculptTool::Grab => pass.sample.drag_delta,
            SculptTool::Draw => brush_normal * (direction * radius * 0.15),
            SculptTool::Clay => {
                let plane_normal = frame.normal;
                let plane_point =
                    frame.origin + plane_normal * (direction * settings.radius * 0.15);
                let signed_plane_distance = (plane_point - position).dot(plane_normal);
                plane_normal * signed_plane_distance
            }
            SculptTool::Crease => {
                let to_center = pass.sample.center - position;
                let tangent = to_center - brush_normal * to_center.dot(brush_normal);
                brush_normal * (direction * radius * 0.15) + tangent * (2.0 / 3.0)
            }
            SculptTool::Inflate => {
                let mut normal = mesh
                    .normals
                    .get(index)
                    .copied()
                    .unwrap_or(brush_normal)
                    .normalize_or_zero();
                if normal.dot(brush_normal) < 0.0 {
                    normal = -normal;
                }
                normal * (direction * radius * 0.15)
            }
            SculptTool::Smooth => {
                let Some(unit) = smooth_units.as_ref().and_then(|units| units.get(&vertex)) else {
                    continue;
                };
                *unit
            }
            SculptTool::Pinch => {
                let to_center = pass.sample.center - position;
                let tangent = to_center - brush_normal * to_center.dot(brush_normal);
                tangent * direction
            }
            SculptTool::Flatten => {
                let signed_plane_distance = (frame.origin - position).dot(brush_normal);
                brush_normal * (direction * signed_plane_distance)
            }
            SculptTool::Mask => unreachable!("mask is handled before geometry brushes"),
        };

        if !unit_displacement.is_finite()
            || unit_displacement.length_squared() <= f32::EPSILON.powi(2)
        {
            continue;
        }
        let proposal = proposals.entry(vertex).or_default();
        proposal.weighted_unit += unit_displacement * influence;
        proposal.influence_sum += influence;
        proposal.maximum_influence = proposal.maximum_influence.max(influence);
    }
}

fn candidate_positions(mesh: &Mesh, proposals: &ProposalMap, scale: f32) -> HashMap<u32, Vec3> {
    proposals
        .iter()
        .filter_map(|(&vertex, &displacement)| {
            let before = mesh.positions.get(vertex as usize).copied()?;
            let next = before + displacement * scale;
            (next != before).then_some((vertex, next))
        })
        .collect()
}

fn prepare_brush_frames(
    mesh: &Mesh,
    tool: SculptTool,
    settings: &BrushSettings,
    passes: &[PreparedPass],
    source_positions: &HashMap<u32, Vec3>,
    committed: &HashMap<PlaneChannel, BrushFrame>,
) -> (Vec<BrushFrame>, HashMap<PlaneChannel, BrushFrame>) {
    let mut frames = Vec::with_capacity(passes.len());
    let mut staged = HashMap::new();
    for pass in passes {
        let raw = area_weighted_brush_frame(mesh, settings, pass, source_positions);
        if matches!(tool, SculptTool::Clay | SculptTool::Flatten) {
            let channel = PlaneChannel {
                tool,
                side: pass.side,
            };
            let stable = committed.get(&channel).map_or(raw, |previous| {
                let mut normal = raw.normal;
                if normal.dot(previous.normal) < 0.0 {
                    normal = -normal;
                }
                BrushFrame {
                    origin: previous.origin.lerp(raw.origin, 0.35),
                    normal: (previous.normal * 0.65 + normal * 0.35).normalize_or_zero(),
                }
            });
            staged.insert(channel, stable);
            frames.push(stable);
        } else {
            frames.push(raw);
        }
    }
    (frames, staged)
}

fn area_weighted_brush_frame(
    mesh: &Mesh,
    settings: &BrushSettings,
    pass: &PreparedPass,
    source_positions: &HashMap<u32, Vec3>,
) -> BrushFrame {
    let fallback_normal = pass.sample.normal.normalize_or_zero();
    let mut weighted_position = Vec3::ZERO;
    let mut weighted_normal = Vec3::ZERO;
    let mut weight_sum = 0.0;
    for &vertex in &pass.vertices {
        let index = vertex as usize;
        let Some(&position) = source_positions.get(&vertex) else {
            continue;
        };
        let falloff = brush_falloff(
            position.distance(pass.sample.center) / settings.radius,
            settings.falloff,
        );
        if falloff <= f32::EPSILON {
            continue;
        }
        let area = vertex_surface_area(mesh, vertex, source_positions);
        let weight = falloff * area.max(f32::EPSILON);
        let mut normal = mesh
            .normals
            .get(index)
            .copied()
            .unwrap_or(fallback_normal)
            .normalize_or_zero();
        if normal.dot(fallback_normal) < 0.0 {
            normal = -normal;
        }
        weighted_position += position * weight;
        weighted_normal += normal * weight;
        weight_sum += weight;
    }
    if weight_sum <= f32::EPSILON {
        return BrushFrame {
            origin: pass.sample.center,
            normal: fallback_normal,
        };
    }
    let plane_normal = weighted_normal.normalize_or_zero();
    BrushFrame {
        origin: weighted_position / weight_sum,
        normal: if plane_normal == Vec3::ZERO {
            fallback_normal
        } else {
            plane_normal
        },
    }
}

fn source_position(
    mesh: &Mesh,
    source_positions: &HashMap<u32, Vec3>,
    vertex: u32,
) -> Option<Vec3> {
    source_positions
        .get(&vertex)
        .copied()
        .or_else(|| mesh.positions.get(vertex as usize).copied())
}

fn vertex_surface_area(mesh: &Mesh, vertex: u32, source_positions: &HashMap<u32, Vec3>) -> f32 {
    mesh.topology
        .vertex_triangles
        .get(vertex as usize)
        .into_iter()
        .flatten()
        .filter_map(|&face| mesh.triangles.get(face as usize))
        .filter_map(|triangle| {
            let [a, b, c] = triangle.map(|index| source_position(mesh, source_positions, index));
            Some((b? - a?).cross(c? - a?).length() / 6.0)
        })
        .sum()
}

fn taubin_smooth_units(
    mesh: &Mesh,
    pass: &PreparedPass,
    source_positions: &HashMap<u32, Vec3>,
) -> HashMap<u32, Vec3> {
    const LAMBDA: f32 = 0.5;
    const MU: f32 = -0.53;

    let mut first_pass = HashMap::with_capacity(pass.vertices.len());
    for &vertex in &pass.vertices {
        if mesh
            .topology
            .non_manifold_vertices
            .get(vertex as usize)
            .copied()
            .unwrap_or(true)
        {
            continue;
        }
        let Some(position) = source_position(mesh, source_positions, vertex) else {
            continue;
        };
        let neighbors = smoothing_neighbors(mesh, vertex);
        if neighbors.is_empty() {
            continue;
        }
        let average = neighbors
            .iter()
            .filter_map(|&neighbor| source_position(mesh, source_positions, neighbor))
            .sum::<Vec3>()
            / neighbors.len() as f32;
        first_pass.insert(vertex, position + (average - position) * LAMBDA);
    }

    let mut units = HashMap::with_capacity(first_pass.len());
    for (&vertex, &first_position) in &first_pass {
        let neighbors = smoothing_neighbors(mesh, vertex);
        if neighbors.is_empty() {
            continue;
        }
        let average = neighbors
            .iter()
            .filter_map(|&neighbor| {
                first_pass
                    .get(&neighbor)
                    .copied()
                    .or_else(|| source_position(mesh, source_positions, neighbor))
            })
            .sum::<Vec3>()
            / neighbors.len() as f32;
        let Some(original) = source_position(mesh, source_positions, vertex) else {
            continue;
        };
        units.insert(
            vertex,
            (first_position - original) + (average - first_position) * MU,
        );
    }
    units
}

fn smoothing_neighbors(mesh: &Mesh, vertex: u32) -> Vec<u32> {
    let Some(neighbors) = mesh.topology.vertex_neighbors.get(vertex as usize) else {
        return Vec::new();
    };
    if !mesh
        .topology
        .boundary_vertices
        .get(vertex as usize)
        .copied()
        .unwrap_or(false)
    {
        return neighbors.to_vec();
    }
    neighbors
        .iter()
        .copied()
        .filter(|&neighbor| {
            let edge = if vertex < neighbor {
                (vertex, neighbor)
            } else {
                (neighbor, vertex)
            };
            mesh.topology
                .edge_faces
                .get(&edge)
                .is_some_and(|faces| faces.len() == 1)
        })
        .collect()
}

fn density_warning(mesh: &Mesh, passes: &[PreparedPass], radius: f32) -> Option<String> {
    let mut edges = HashSet::new();
    let mut lengths = Vec::new();
    for pass in passes {
        for &vertex in &pass.vertices {
            let Some(neighbors) = mesh.topology.vertex_neighbors.get(vertex as usize) else {
                continue;
            };
            for &neighbor in neighbors {
                let edge = if vertex < neighbor {
                    (vertex, neighbor)
                } else {
                    (neighbor, vertex)
                };
                if edges.insert(edge)
                    && let (Some(a), Some(b)) = (
                        mesh.positions.get(edge.0 as usize),
                        mesh.positions.get(edge.1 as usize),
                    )
                {
                    let length = a.distance(*b);
                    if length.is_finite() && length > f32::EPSILON {
                        lengths.push(length);
                    }
                }
            }
        }
    }
    if lengths.is_empty() {
        return None;
    }
    lengths.sort_by(f32::total_cmp);
    let median = lengths[lengths.len() / 2];
    (radius < median * 2.0).then(|| {
        "Brush is undersampled by this mesh; increase its radius or voxel-remesh a closed mesh at finer resolution"
            .to_owned()
    })
}

/// Smooth radial brush weight with an optional hard inner core.
#[must_use]
pub fn brush_falloff(normalized_distance: f32, hardness: f32) -> f32 {
    if !normalized_distance.is_finite() || normalized_distance >= 1.0 {
        return 0.0;
    }
    if normalized_distance <= 0.0 {
        return 1.0;
    }

    let hardness = hardness.clamp(0.0, 0.999_9);
    if normalized_distance <= hardness {
        return 1.0;
    }
    let t = ((normalized_distance - hardness) / (1.0 - hardness)).clamp(0.0, 1.0);
    let smoothstep = t * t * (3.0 - 2.0 * t);
    1.0 - smoothstep
}

fn capture_source_positions(
    mesh: &Mesh,
    tool: SculptTool,
    passes: &[PreparedPass],
    source: &mut HashMap<u32, Vec3>,
) {
    let selected_count = passes.iter().map(|pass| pass.vertices.len()).sum();
    source.clear();
    source.reserve(selected_count);
    for pass in passes {
        for &vertex in &pass.vertices {
            if let Some(&position) = mesh.positions.get(vertex as usize) {
                source.entry(vertex).or_insert(position);
            }
            if tool == SculptTool::Smooth
                && let Some(neighbors) = mesh.topology.vertex_neighbors.get(vertex as usize)
            {
                for &neighbor in neighbors {
                    if let Some(&position) = mesh.positions.get(neighbor as usize) {
                        source.entry(neighbor).or_insert(position);
                    }
                }
            }
        }
    }
}

fn triangle_is_near(mesh: &Mesh, triangle: u32, point: Vec3, radius: f32) -> bool {
    let Some(vertices) = mesh.triangles.get(triangle as usize) else {
        return false;
    };
    vertices.iter().any(|&vertex| {
        mesh.positions
            .get(vertex as usize)
            .is_some_and(|position| position.distance_squared(point) <= (radius * 2.0).powi(2))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{History, HistoryAction, HistoryEntry};
    use std::{sync::Arc, time::Instant};

    fn grid() -> Mesh {
        let positions = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let triangles = vec![
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ];
        Mesh::new(positions, triangles).expect("valid grid")
    }

    fn octahedron() -> Mesh {
        Mesh::new(
            vec![
                Vec3::X,
                Vec3::Y,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ],
            vec![
                [4, 0, 1],
                [4, 1, 2],
                [4, 2, 3],
                [4, 3, 0],
                [5, 1, 0],
                [5, 2, 1],
                [5, 3, 2],
                [5, 0, 3],
            ],
        )
        .expect("valid octahedron")
    }

    fn sample(center: Vec3, seed_triangle: u32) -> BrushSample {
        BrushSample {
            center,
            normal: Vec3::Z,
            drag_delta: Vec3::ZERO,
            view_direction: Vec3::NEG_Z,
            seed_triangle,
            pressure: 1.0,
            invert_modifier: false,
        }
    }

    fn test_settings() -> BrushSettings {
        BrushSettings {
            radius: 2.1,
            strength: 1.0,
            falloff: 0.0,
            remesh_target_edge_length: None,
            invert: false,
            symmetry: None,
            accumulation: Accumulation::Capped,
        }
    }

    fn drain_pending_sample(engine: &mut SculptEngine, mesh: &mut Mesh) -> usize {
        let mut steps = 0;
        while engine.has_pending_sample() {
            assert!(steps < 1_024, "adaptive sample did not terminate");
            engine.continue_pending_sample(mesh);
            let error = engine.take_error();
            assert!(error.is_none(), "adaptive continuation failed: {error:?}");
            steps += 1;
        }
        steps
    }

    fn coverage_mesh() -> Mesh {
        let mut mesh = grid();
        mesh.positions[4].z = 0.35;
        let _ = mesh.rebuild();
        mesh
    }

    fn coverage_settings() -> BrushSettings {
        BrushSettings {
            radius: 2.1,
            strength: 0.1,
            falloff: 0.95,
            remesh_target_edge_length: None,
            invert: false,
            symmetry: None,
            accumulation: Accumulation::Capped,
        }
    }

    fn editable_state(mesh: &Mesh) -> (Vec<Vec3>, Vec<f32>) {
        (mesh.positions.clone(), mesh.mask.clone())
    }

    fn assert_editable_mesh_eq(actual: &Mesh, expected: &Mesh) {
        assert_eq!(actual.positions, expected.positions);
        assert_eq!(actual.triangles, expected.triangles);
        assert_eq!(actual.mask, expected.mask);
    }

    #[test]
    fn falloff_has_stable_center_and_edge_values() {
        assert_eq!(brush_falloff(0.0, 0.0), 1.0);
        assert_eq!(brush_falloff(1.0, 0.0), 0.0);
        assert_eq!(brush_falloff(1.5, 0.5), 0.0);
        assert_eq!(brush_falloff(0.25, 0.5), 1.0);
        assert!((brush_falloff(0.5, 0.0) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn adaptive_vertex_events_remap_capped_coverage() {
        let channel = CoverageChannel {
            tool: SculptTool::Draw,
            polarity: CoveragePolarity::Add,
        };
        let mut coverage = CoverageMap::from([((channel, 0), 0.2), ((channel, 1), 0.6)]);
        remap_coverage(
            &mut coverage,
            &[
                VertexRemeshEvent::Created { vertex: 2 },
                VertexRemeshEvent::Split {
                    vertex: 2,
                    first: 0,
                    second: 1,
                },
            ],
        );
        assert!((coverage[&(channel, 2)] - 0.4).abs() < 1.0e-6);

        remap_coverage(
            &mut coverage,
            &[VertexRemeshEvent::Collapsed { keep: 0, remove: 1 }],
        );
        assert!((coverage[&(channel, 0)] - 0.4).abs() < 1.0e-6);
        assert!(!coverage.contains_key(&(channel, 1)));

        remap_coverage(
            &mut coverage,
            &[VertexRemeshEvent::Remapped { from: 2, to: 1 }],
        );
        assert!((coverage[&(channel, 1)] - 0.4).abs() < 1.0e-6);
        assert!(!coverage.contains_key(&(channel, 2)));
    }

    #[test]
    fn spatial_dab_tools_do_not_accumulate_until_the_next_stroke() {
        let tools = [
            SculptTool::Draw,
            SculptTool::Clay,
            SculptTool::Crease,
            SculptTool::Inflate,
            SculptTool::Smooth,
            SculptTool::Pinch,
            SculptTool::Flatten,
            SculptTool::Mask,
        ];
        for tool in tools {
            let mut mesh = coverage_mesh();
            let settings = coverage_settings();
            let brush_sample = sample(mesh.positions[4], 0);
            let mut engine = SculptEngine::default();
            engine.begin_stroke(&mesh);

            assert!(
                engine.apply_sample(&mut mesh, tool, &settings, brush_sample, DabKind::Spatial,),
                "{tool} first spatial dab"
            );
            let after_first = editable_state(&mesh);
            assert!(
                !engine.apply_sample(&mut mesh, tool, &settings, brush_sample, DabKind::Spatial,),
                "{tool} repeated spatial dab"
            );
            assert_eq!(editable_state(&mesh), after_first, "{tool} retrace");

            assert!(!engine.end_stroke(&mesh).edit.is_empty());
            engine.begin_stroke(&mesh);
            assert!(
                engine.apply_sample(&mut mesh, tool, &settings, brush_sample, DabKind::Spatial,),
                "{tool} next stroke"
            );
        }
    }

    #[test]
    fn accumulating_spatial_and_timed_dabs_keep_building_up() {
        for tool in [
            SculptTool::Draw,
            SculptTool::Clay,
            SculptTool::Crease,
            SculptTool::Inflate,
            SculptTool::Smooth,
            SculptTool::Pinch,
            SculptTool::Flatten,
            SculptTool::Mask,
        ] {
            let mut mesh = coverage_mesh();
            let mut settings = coverage_settings();
            settings.accumulation = Accumulation::Accumulate;
            let brush_sample = sample(mesh.positions[4], 0);
            let mut engine = SculptEngine::default();
            engine.begin_stroke(&mesh);

            assert!(engine.apply_sample(
                &mut mesh,
                tool,
                &settings,
                brush_sample,
                DabKind::Spatial,
            ));
            let after_first = editable_state(&mesh);
            assert!(
                engine.apply_sample(&mut mesh, tool, &settings, brush_sample, DabKind::Spatial,),
                "{tool} second accumulating spatial dab"
            );
            assert_ne!(editable_state(&mesh), after_first, "{tool} accumulation");

            let after_spatial = editable_state(&mesh);
            assert!(
                engine.apply_sample(&mut mesh, tool, &settings, brush_sample, DabKind::Timed,),
                "{tool} timed dab"
            );
            assert_ne!(editable_state(&mesh), after_spatial, "{tool} timed buildup");
        }
    }

    #[test]
    fn accumulating_spatial_and_timed_samples_have_identical_effects() {
        for tool in [
            SculptTool::Draw,
            SculptTool::Clay,
            SculptTool::Crease,
            SculptTool::Inflate,
            SculptTool::Smooth,
            SculptTool::Pinch,
            SculptTool::Flatten,
            SculptTool::Mask,
        ] {
            let mut spatial = coverage_mesh();
            let mut timed = spatial.clone();
            let mut settings = coverage_settings();
            settings.accumulation = Accumulation::Accumulate;
            let brush_sample = sample(spatial.positions[4], 0);
            let mut spatial_engine = SculptEngine::default();
            let mut timed_engine = SculptEngine::default();
            spatial_engine.begin_stroke(&spatial);
            timed_engine.begin_stroke(&timed);

            assert!(spatial_engine.apply_sample(
                &mut spatial,
                tool,
                &settings,
                brush_sample,
                DabKind::Spatial,
            ));
            assert!(timed_engine.apply_sample(
                &mut timed,
                tool,
                &settings,
                brush_sample,
                DabKind::Timed,
            ));
            assert_eq!(editable_state(&spatial), editable_state(&timed), "{tool}");
        }
    }

    #[test]
    fn spatial_coverage_tracks_tool_polarity_pressure_and_masking() {
        let settings = coverage_settings();
        let mut mesh = coverage_mesh();
        let brush_sample = sample(mesh.positions[4], 0);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Smooth,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        let after_smooth = editable_state(&mesh);
        let mut inverted = brush_sample;
        inverted.invert_modifier = true;
        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Smooth,
            &settings,
            inverted,
            DabKind::Spatial,
        ));
        assert_eq!(editable_state(&mesh), after_smooth);
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            inverted,
            DabKind::Spatial,
        ));

        let mut pressure_mesh = grid();
        let mut pressure_sample = sample(Vec3::ZERO, 0);
        pressure_sample.pressure = 0.25;
        let mut pressure_engine = SculptEngine::default();
        pressure_engine.begin_stroke(&pressure_mesh);
        assert!(pressure_engine.apply_sample(
            &mut pressure_mesh,
            SculptTool::Draw,
            &settings,
            pressure_sample,
            DabKind::Spatial,
        ));
        let first_height = pressure_mesh.positions[4].z;
        pressure_sample.pressure = 0.5;
        assert!(pressure_engine.apply_sample(
            &mut pressure_mesh,
            SculptTool::Draw,
            &settings,
            pressure_sample,
            DabKind::Spatial,
        ));
        assert!((pressure_mesh.positions[4].z - first_height * 2.0).abs() < 1.0e-6);
        assert!(!pressure_engine.apply_sample(
            &mut pressure_mesh,
            SculptTool::Draw,
            &settings,
            pressure_sample,
            DabKind::Spatial,
        ));

        let mut masked = grid();
        masked.mask[4] = 0.5;
        let mut masked_engine = SculptEngine::default();
        masked_engine.begin_stroke(&masked);
        assert!(masked_engine.apply_sample(
            &mut masked,
            SculptTool::Draw,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        ));
        assert!((masked.positions[4].z - first_height * 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn mirrored_spatial_passes_share_coverage() {
        let mut regular = grid();
        let mut mirrored = grid();
        let settings = coverage_settings();
        let mut symmetry_settings = settings;
        symmetry_settings.symmetry = Some(SymmetryAxis::X);
        let brush_sample = sample(Vec3::new(0.1, 0.0, 0.0), 0);
        let mut regular_engine = SculptEngine::default();
        let mut mirrored_engine = SculptEngine::default();
        regular_engine.begin_stroke(&regular);
        mirrored_engine.begin_stroke(&mirrored);

        assert!(regular_engine.apply_sample(
            &mut regular,
            SculptTool::Mask,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        assert!(mirrored_engine.apply_sample(
            &mut mirrored,
            SculptTool::Mask,
            &symmetry_settings,
            brush_sample,
            DabKind::Spatial,
        ));
        assert_eq!(mirrored.mask, regular.mask);
    }

    #[test]
    fn rejected_spatial_sample_does_not_consume_coverage() {
        let mut mesh = grid();
        let mut settings = test_settings();
        settings.falloff = 1.0;
        settings.strength = 0.1;
        let mut brush_sample = sample(Vec3::ZERO, 0);
        brush_sample.normal = Vec3::NAN;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        brush_sample.normal = Vec3::Z;
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
    }

    #[test]
    fn one_spatial_click_records_one_history_entry_in_each_accumulation_mode() {
        for accumulation in [Accumulation::Capped, Accumulation::Accumulate] {
            let mut mesh = grid();
            let before = mesh.clone();
            let mut engine = SculptEngine::default();
            let mut settings = coverage_settings();
            settings.accumulation = accumulation;
            engine.begin_stroke(&mesh);
            assert!(engine.apply_sample(
                &mut mesh,
                SculptTool::Draw,
                &settings,
                sample(Vec3::ZERO, 0),
                DabKind::Spatial,
            ));
            let outcome = engine.end_stroke(&mesh);
            let mut history = History::default();
            assert!(history.record(HistoryEntry::Local(outcome.edit)));

            assert!(matches!(
                history.undo(&mut mesh),
                HistoryAction::Local { .. }
            ));
            assert_editable_mesh_eq(&mesh, &before);
            assert!(matches!(history.undo(&mut mesh), HistoryAction::Empty));
        }
    }

    #[test]
    fn a_full_mask_prevents_deformation() {
        let mut mesh = grid();
        mesh.mask[4] = 1.0;
        let before = mesh.positions[4];
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &test_settings(),
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        ));
        assert_eq!(mesh.positions[4], before);
        assert!(mesh.positions.iter().any(|position| position.z > 0.0));
    }

    #[test]
    fn symmetry_mirrors_brush_deformation() {
        let mut mesh = grid();
        let mut settings = test_settings();
        settings.radius = 0.8;
        settings.falloff = 0.4;
        settings.symmetry = Some(SymmetryAxis::X);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::new(1.0, 0.0, 0.0), 2),
            DabKind::Spatial,
        ));
        assert!(mesh.positions[5].z > 0.0);
        assert_eq!(mesh.positions[3].z, mesh.positions[5].z);
    }

    #[test]
    fn empty_prepared_pass_never_has_remesh_support() {
        let mesh = grid();
        let pass = PreparedPass {
            sample: sample(Vec3::ZERO, 0),
            vertices: Vec::new(),
            side: PassSide::Primary,
        };

        assert!(!pass.has_remesh_support(&mesh, 0.1, 1.0, 0.0, 0.02));
    }

    #[test]
    fn inversion_reverses_an_inflate_brush() {
        let mut outward = grid();
        let mut inward = grid();
        let settings = test_settings();
        let mut normal_engine = SculptEngine::default();
        let mut inverse_engine = SculptEngine::default();
        normal_engine.begin_stroke(&outward);
        inverse_engine.begin_stroke(&inward);
        let normal_sample = sample(Vec3::ZERO, 0);
        let mut inverse_sample = normal_sample;
        inverse_sample.invert_modifier = true;

        normal_engine.apply_sample(
            &mut outward,
            SculptTool::Inflate,
            &settings,
            normal_sample,
            DabKind::Spatial,
        );
        inverse_engine.apply_sample(
            &mut inward,
            SculptTool::Inflate,
            &settings,
            inverse_sample,
            DabKind::Spatial,
        );

        assert!(outward.positions[4].z > 0.0);
        assert_eq!(outward.positions[4].z, -inward.positions[4].z);
    }

    #[test]
    fn clay_adds_and_subtracts_while_flattening_to_its_local_plane() {
        let mut added = grid();
        let mut subtracted = grid();
        let mut settings = test_settings();
        settings.radius = 0.75;
        settings.strength = 0.5;
        let mut add_engine = SculptEngine::default();
        let mut subtract_engine = SculptEngine::default();
        add_engine.begin_stroke(&added);
        subtract_engine.begin_stroke(&subtracted);
        let add_sample = sample(Vec3::ZERO, 0);
        let mut subtract_sample = add_sample;
        subtract_sample.invert_modifier = true;

        assert!(add_engine.apply_sample(
            &mut added,
            SculptTool::Clay,
            &settings,
            add_sample,
            DabKind::Spatial,
        ));
        assert!(subtract_engine.apply_sample(
            &mut subtracted,
            SculptTool::Clay,
            &settings,
            subtract_sample,
            DabKind::Spatial,
        ));
        assert!(added.positions[4].z > 0.0);
        assert_eq!(added.positions[4].z, -subtracted.positions[4].z);

        let mut uneven = grid();
        uneven.positions[4].z = 0.4;
        let _ = uneven.rebuild();
        let mut planar_settings = test_settings();
        planar_settings.falloff = 0.95;
        let center = uneven.positions[4];
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&uneven);
        assert!(engine.apply_sample(
            &mut uneven,
            SculptTool::Clay,
            &planar_settings,
            sample(center, 0),
            DabKind::Spatial,
        ));
        let minimum = uneven
            .positions
            .iter()
            .map(|position| position.z)
            .fold(f32::INFINITY, f32::min);
        let maximum = uneven
            .positions
            .iter()
            .map(|position| position.z)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(maximum - minimum < 1.0e-5);
    }

    #[test]
    fn crease_builds_ridges_and_grooves_while_always_pinching_inward() {
        let mut ridge = grid();
        let mut groove = grid();
        let mut settings = test_settings();
        settings.radius = 1.1;
        settings.strength = 0.2;
        settings.falloff = 0.95;
        let mut ridge_engine = SculptEngine::default();
        let mut groove_engine = SculptEngine::default();
        ridge_engine.begin_stroke(&ridge);
        groove_engine.begin_stroke(&groove);
        let ridge_sample = sample(Vec3::ZERO, 0);
        let mut groove_sample = ridge_sample;
        groove_sample.invert_modifier = true;

        assert!(ridge_engine.apply_sample(
            &mut ridge,
            SculptTool::Crease,
            &settings,
            ridge_sample,
            DabKind::Spatial,
        ));
        assert!(groove_engine.apply_sample(
            &mut groove,
            SculptTool::Crease,
            &settings,
            groove_sample,
            DabKind::Spatial,
        ));
        assert!(ridge.positions[4].z > 0.0);
        assert_eq!(ridge.positions[4].z, -groove.positions[4].z);
        assert!(ridge.positions[5].x < 1.0);
        assert!(groove.positions[5].x < 1.0);
        assert_eq!(ridge.positions[5].x, groove.positions[5].x);
    }

    #[test]
    fn clay_and_crease_scale_linearly_with_pressure() {
        for tool in [SculptTool::Clay, SculptTool::Crease] {
            let mut low = grid();
            let mut high = grid();
            let mut settings = test_settings();
            settings.radius = 0.75;
            settings.strength = 1.0;
            let mut low_engine = SculptEngine::default();
            let mut high_engine = SculptEngine::default();
            low_engine.begin_stroke(&low);
            high_engine.begin_stroke(&high);
            let mut low_sample = sample(Vec3::ZERO, 0);
            low_sample.pressure = 0.25;
            let mut high_sample = low_sample;
            high_sample.pressure = 0.5;

            assert!(low_engine.apply_sample(
                &mut low,
                tool,
                &settings,
                low_sample,
                DabKind::Spatial,
            ));
            assert!(high_engine.apply_sample(
                &mut high,
                tool,
                &settings,
                high_sample,
                DabKind::Spatial,
            ));
            assert!((high.positions[4].z - low.positions[4].z * 2.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn ineffective_pressure_cannot_deform_or_start_adaptive_topology() {
        for tool in [SculptTool::Clay, SculptTool::Crease] {
            for pressure in [0.0, -1.0, f32::NAN, f32::INFINITY] {
                let mut mesh = octahedron();
                let before = mesh.clone();
                let mut settings = test_settings();
                settings.radius = 1.2;
                settings.remesh_target_edge_length = Some(0.24);
                let mut brush_sample = sample(Vec3::splat(1.0 / 3.0), 0);
                brush_sample.pressure = pressure;
                let mut engine = SculptEngine::default();
                engine.begin_stroke(&mesh);

                assert!(!engine.apply_sample(
                    &mut mesh,
                    tool,
                    &settings,
                    brush_sample,
                    DabKind::Spatial,
                ));
                assert!(!engine.has_pending_sample());
                assert!(!engine.take_sample_committed());
                assert!(engine.take_mesh_changes().is_none());
                assert_editable_mesh_eq(&mesh, &before);
                assert_eq!(engine.end_stroke(&mesh), StrokeOutcome::default());
            }
        }
    }

    #[test]
    fn clay_and_crease_respect_full_partial_masks_and_symmetry() {
        for tool in [SculptTool::Clay, SculptTool::Crease] {
            let mut fully_masked = grid();
            fully_masked.mask[4] = 1.0;
            let before = fully_masked.positions.clone();
            let mut settings = test_settings();
            settings.radius = 0.75;
            settings.strength = 0.5;
            let mut engine = SculptEngine::default();
            engine.begin_stroke(&fully_masked);
            assert!(!engine.apply_sample(
                &mut fully_masked,
                tool,
                &settings,
                sample(Vec3::ZERO, 0),
                DabKind::Spatial,
            ));
            assert_eq!(fully_masked.positions, before);

            let mut unmasked = grid();
            let mut partially_masked = grid();
            partially_masked.mask[4] = 0.5;
            let mut unmasked_engine = SculptEngine::default();
            let mut partial_engine = SculptEngine::default();
            unmasked_engine.begin_stroke(&unmasked);
            partial_engine.begin_stroke(&partially_masked);
            assert!(unmasked_engine.apply_sample(
                &mut unmasked,
                tool,
                &settings,
                sample(Vec3::ZERO, 0),
                DabKind::Spatial,
            ));
            assert!(partial_engine.apply_sample(
                &mut partially_masked,
                tool,
                &settings,
                sample(Vec3::ZERO, 0),
                DabKind::Spatial,
            ));
            assert!(
                (partially_masked.positions[4].z - unmasked.positions[4].z * 0.5).abs() < 1.0e-6
            );

            let mut symmetric = grid();
            let mut symmetry_settings = settings;
            symmetry_settings.radius = 0.8;
            symmetry_settings.symmetry = Some(SymmetryAxis::X);
            let mut symmetry_engine = SculptEngine::default();
            symmetry_engine.begin_stroke(&symmetric);
            assert!(symmetry_engine.apply_sample(
                &mut symmetric,
                tool,
                &symmetry_settings,
                sample(Vec3::X * 0.8, 2),
                DabKind::Spatial,
            ));
            assert!(symmetric.positions[5].z > 0.0);
            assert_eq!(symmetric.positions[3].z, symmetric.positions[5].z);
            assert_eq!(symmetric.positions[3].x, -symmetric.positions[5].x);
        }
    }

    #[test]
    fn clay_and_crease_fixed_strokes_have_exact_undo_and_redo() {
        for tool in [SculptTool::Clay, SculptTool::Crease] {
            let mut mesh = grid();
            let before = mesh.clone();
            let mut settings = test_settings();
            settings.radius = 0.75;
            settings.strength = 0.5;
            let mut engine = SculptEngine::default();
            engine.begin_stroke(&mesh);
            assert!(engine.apply_sample(
                &mut mesh,
                tool,
                &settings,
                sample(Vec3::ZERO, 0),
                DabKind::Spatial,
            ));
            let outcome = engine.end_stroke(&mesh);
            assert!(outcome.topology.is_none());
            assert!(!outcome.edit.is_empty());
            let after = mesh.clone();
            let mut history = History::default();
            assert!(history.record(HistoryEntry::Local(outcome.edit)));

            assert!(matches!(
                history.undo(&mut mesh),
                HistoryAction::Local { .. }
            ));
            assert_editable_mesh_eq(&mesh, &before);
            assert!(matches!(
                history.redo(&mut mesh),
                HistoryAction::Local { .. }
            ));
            assert_editable_mesh_eq(&mesh, &after);
        }
    }

    #[test]
    fn adaptive_clay_and_crease_finish_valid_and_have_exact_undo_and_redo() {
        for tool in [SculptTool::Clay, SculptTool::Crease] {
            let mut mesh = octahedron();
            let before = mesh.clone();
            let mut settings = test_settings();
            settings.radius = 1.2;
            settings.strength = 1.0;
            settings.remesh_target_edge_length = Some(0.24);
            let center = Vec3::splat(1.0 / 3.0);
            let mut engine = SculptEngine::default();
            engine.begin_stroke(&mesh);

            engine.apply_sample(
                &mut mesh,
                tool,
                &settings,
                sample(center, 0),
                DabKind::Spatial,
            );
            drain_pending_sample(&mut engine, &mut mesh);
            assert!(engine.take_sample_committed());
            mesh.validate().unwrap();
            let faces = (0..mesh.triangles.len() as u32).collect::<Vec<_>>();
            assert!(!mesh.faces_have_self_intersections(&faces));
            let outcome = engine.end_stroke(&mesh);
            let topology = outcome
                .topology
                .expect("adaptive brush records topology history");
            let after = mesh.clone();
            let mut history = History::default();
            assert!(history.record(HistoryEntry::Topology(Arc::new(topology))));

            let HistoryAction::Topology(pending) = history.undo(&mut mesh) else {
                panic!("topology undo expected");
            };
            pending.edit().apply_before(&mut mesh);
            history.complete_topology(pending).unwrap();
            assert_editable_mesh_eq(&mesh, &before);
            let HistoryAction::Topology(pending) = history.redo(&mut mesh) else {
                panic!("topology redo expected");
            };
            pending.edit().apply_after(&mut mesh);
            history.complete_topology(pending).unwrap();
            assert_editable_mesh_eq(&mesh, &after);
        }
    }

    #[test]
    fn smoothing_reduces_a_local_peak() {
        let mut mesh = grid();
        mesh.positions[4].z = 1.0;
        let _ = mesh.rebuild();
        let before = mesh.positions[4].z;
        let center = mesh.positions[4];
        let mut settings = test_settings();
        settings.radius = 0.75;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Smooth,
            &settings,
            sample(center, 0),
            DabKind::Spatial,
        ));
        assert!(mesh.positions[4].z < before);
    }

    #[test]
    fn mask_brush_adds_and_inverted_mask_brush_erases() {
        let mut mesh = grid();
        let settings = test_settings();
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        let mut brush_sample = sample(Vec3::ZERO, 0);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Mask,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        assert!(mesh.mask[4] > 0.0);

        brush_sample.invert_modifier = true;
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Mask,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        assert_eq!(mesh.mask[4], 0.0);
        let outcome = engine.end_stroke(&mesh);
        assert!(outcome.edit.is_empty());
        assert!(outcome.topology.is_none());
    }

    #[test]
    fn adaptive_topology_is_applied_during_the_stroke() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::X,
                Vec3::Y,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ],
            vec![
                [4, 0, 1],
                [4, 1, 2],
                [4, 2, 3],
                [4, 3, 0],
                [5, 1, 0],
                [5, 2, 1],
                [5, 3, 2],
                [5, 0, 3],
            ],
        )
        .unwrap();
        let before_positions = mesh.positions.clone();
        let before_triangles = mesh.triangles.clone();
        let before_mask = mesh.mask.clone();
        let original_vertex_count = mesh.positions.len();
        let original_triangles = mesh.triangles.clone();
        let mut settings = test_settings();
        settings.radius = 1.2;
        settings.remesh_target_edge_length = Some(settings.radius * 0.2);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::splat(1.0 / 3.0), 0),
            DabKind::Spatial,
        ));
        assert!(engine.has_pending_sample());
        assert!(engine.take_mesh_changes().is_none());
        assert_eq!(mesh.triangles, original_triangles);
        let mut updated = Vec::new();
        let changes = loop {
            engine.continue_pending_sample(&mut mesh);
            assert!(engine.take_error().is_none());
            updated.extend(engine.take_updated_vertices());
            if let Some(changes) = engine.take_mesh_changes() {
                break changes;
            }
        };
        assert!(!changes.dirty_faces.is_empty());
        while engine.has_pending_sample() {
            engine.continue_pending_sample(&mut mesh);
            assert!(engine.take_error().is_none());
            let _ = engine.take_updated_vertices();
            let _ = engine.take_mesh_changes();
        }

        let center = Vec3::splat(1.0 / 3.0);
        let seed_triangle = mesh.nearest_triangle(center).unwrap();
        engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(center, seed_triangle),
            DabKind::Spatial,
        );
        let mut second_sample_vertices = engine.take_updated_vertices();
        while engine.has_pending_sample() {
            engine.continue_pending_sample(&mut mesh);
            assert!(engine.take_error().is_none());
            second_sample_vertices.extend(engine.take_updated_vertices());
            let _ = engine.take_mesh_changes();
        }
        assert!(
            second_sample_vertices
                .iter()
                .any(|&vertex| vertex as usize >= original_vertex_count),
            "later brush samples must select topology created earlier in the stroke"
        );
        let outcome = engine.end_stroke(&mesh);

        assert!(!updated.is_empty());
        let topology = outcome
            .topology
            .expect("adaptive stroke records exact topology history");
        topology.apply_before(&mut mesh);
        assert_eq!(mesh.positions, before_positions);
        assert_eq!(mesh.triangles, before_triangles);
        assert_eq!(mesh.mask, before_mask);
    }

    #[test]
    fn small_adaptive_brush_builds_bounded_support_without_spikes_or_stalls() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::X,
                Vec3::Y,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ],
            vec![
                [4, 0, 1],
                [4, 1, 2],
                [4, 2, 3],
                [4, 3, 0],
                [5, 1, 0],
                [5, 2, 1],
                [5, 3, 2],
                [5, 0, 3],
            ],
        )
        .unwrap();
        let before = mesh.clone();
        let center = Vec3::splat(1.0 / 3.0);
        let mut settings = test_settings();
        settings.radius = 0.1;
        settings.strength = 0.2;
        settings.remesh_target_edge_length = Some(settings.radius * 0.2);
        let original_vertex_count = mesh.positions.len();
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        let mut longest_dab = std::time::Duration::ZERO;

        for dab in 0..5 {
            let seed = mesh.nearest_triangle(center).unwrap();
            let dab_started = Instant::now();
            let changed = engine.apply_sample(
                &mut mesh,
                SculptTool::Draw,
                &settings,
                sample(center, seed),
                DabKind::Spatial,
            );
            longest_dab = longest_dab.max(dab_started.elapsed());
            assert!(engine.take_error().is_none());
            let _ = engine.take_warning();
            if dab == 0 {
                assert!(!changed);
                assert!(engine.has_pending_sample());
                assert!(mesh.positions.len() - original_vertex_count <= 24);
            }
            while engine.has_pending_sample() {
                let step_started = Instant::now();
                engine.continue_pending_sample(&mut mesh);
                longest_dab = longest_dab.max(step_started.elapsed());
                assert!(engine.take_error().is_none());
                let _ = engine.take_warning();
            }
            if dab == 0 {
                assert!(engine.take_sample_committed());
            } else {
                let _ = engine.take_sample_committed();
            }
        }

        assert!(mesh.positions.len() <= 256);
        mesh.validate().unwrap();
        let faces = (0..mesh.triangles.len() as u32).collect::<Vec<_>>();
        assert!(!mesh.faces_have_self_intersections(&faces));
        if !cfg!(debug_assertions) {
            assert!(
                longest_dab < std::time::Duration::from_millis(8),
                "small adaptive dab exceeded one frame: {longest_dab:?}"
            );
        }
        let seed = mesh.nearest_triangle(center).unwrap();
        let pass = PreparedPass::new(
            &mesh,
            sample(center, seed),
            settings.radius,
            PassSide::Primary,
            &mut VertexTraversalScratch::default(),
        );
        assert!(pass.has_remesh_support(
            &mesh,
            settings.radius,
            settings.strength,
            settings.falloff,
            settings.remesh_target_edge_length.unwrap() * ADAPTIVE_SPLIT_THRESHOLD,
        ));
        let topology = engine
            .end_stroke(&mesh)
            .topology
            .expect("small adaptive stroke records its support patch");
        topology.apply_before(&mut mesh);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
    }

    #[test]
    fn adaptive_deformation_waits_for_quality_support_across_steps() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::X,
                Vec3::Y,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ],
            vec![
                [4, 0, 1],
                [4, 1, 2],
                [4, 2, 3],
                [4, 3, 0],
                [5, 1, 0],
                [5, 2, 1],
                [5, 3, 2],
                [5, 0, 3],
            ],
        )
        .unwrap();
        let before = mesh.clone();
        let original_vertex_count = mesh.positions.len();
        let center = Vec3::splat(1.0 / 3.0);
        let mut settings = test_settings();
        settings.radius = 0.1;
        settings.remesh_target_edge_length = Some(settings.radius * 0.2);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(center, 0),
            DabKind::Spatial,
        ));
        assert!(engine.has_pending_sample());
        assert!(!engine.take_sample_committed());
        assert_eq!(&mesh.positions[..original_vertex_count], &before.positions);
        assert!(
            mesh.positions[original_vertex_count..]
                .iter()
                .all(|position| (position.element_sum() - 1.0).abs() < 1.0e-5)
        );

        let continuation_steps = drain_pending_sample(&mut engine, &mut mesh);
        assert!(continuation_steps < MAX_ADAPTIVE_TOPOLOGY_STEPS);
        assert!(engine.take_sample_committed());
        assert!(
            mesh.positions[original_vertex_count..]
                .iter()
                .any(|position| position.element_sum() > 1.0 + 1.0e-5)
        );
        mesh.validate().unwrap();
        let faces = (0..mesh.triangles.len() as u32).collect::<Vec<_>>();
        assert!(!mesh.faces_have_self_intersections(&faces));

        let topology = engine
            .end_stroke(&mesh)
            .topology
            .expect("completed adaptive sample records exact rollback data");
        topology.apply_before(&mut mesh);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
    }

    #[test]
    fn finest_adaptive_symmetry_builds_and_deforms_both_support_patches() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::X,
                Vec3::Y,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ],
            vec![
                [4, 0, 1],
                [4, 1, 2],
                [4, 2, 3],
                [4, 3, 0],
                [5, 1, 0],
                [5, 2, 1],
                [5, 3, 2],
                [5, 0, 3],
            ],
        )
        .unwrap();
        let before = mesh.clone();
        let original_vertex_count = mesh.positions.len();
        let center = Vec3::splat(1.0 / 3.0);
        let mut settings = test_settings();
        settings.radius = 0.1;
        settings.strength = 0.2;
        settings.remesh_target_edge_length = Some(settings.radius * 0.03);
        settings.symmetry = Some(SymmetryAxis::X);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        let mut longest_step = std::time::Duration::ZERO;

        let started = Instant::now();
        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(center, 0),
            DabKind::Spatial,
        ));
        longest_step = longest_step.max(started.elapsed());
        while engine.has_pending_sample() {
            let started = Instant::now();
            engine.continue_pending_sample(&mut mesh);
            longest_step = longest_step.max(started.elapsed());
            assert!(engine.take_error().is_none());
        }

        assert!(engine.take_sample_committed());
        let added = &mesh.positions[original_vertex_count..];
        assert_eq!(added.len(), 206);
        assert_eq!(
            added.iter().filter(|position| position.x > 0.0).count(),
            added.iter().filter(|position| position.x < 0.0).count()
        );
        assert!(
            added
                .iter()
                .any(|position| position.x > 0.0 && position.element_sum() > 1.0 + 1.0e-5)
        );
        assert!(added.iter().any(|position| {
            position.x < 0.0 && -position.x + position.y + position.z > 1.0 + 1.0e-5
        }));
        assert!(mesh.positions.iter().all(|position| {
            let mirrored = Vec3::new(-position.x, position.y, position.z);
            mesh.positions
                .iter()
                .any(|candidate| candidate.distance_squared(mirrored) < 1.0e-10)
        }));
        mesh.validate().unwrap();
        let faces = (0..mesh.triangles.len() as u32).collect::<Vec<_>>();
        assert!(!mesh.faces_have_self_intersections(&faces));
        if !cfg!(debug_assertions) {
            assert!(
                longest_step < std::time::Duration::from_millis(8),
                "symmetric adaptive step exceeded one frame: {longest_step:?}"
            );
        }

        let topology = engine
            .end_stroke(&mesh)
            .topology
            .expect("symmetric support patches produce topology history");
        topology.apply_before(&mut mesh);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
    }

    #[test]
    fn adaptive_symmetry_splits_regular_remesh_budget_between_both_sides() {
        let component_positions = [
            Vec3::X,
            Vec3::Y,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let component_triangles = [
            [4, 0, 1],
            [4, 1, 2],
            [4, 2, 3],
            [4, 3, 0],
            [5, 1, 0],
            [5, 2, 1],
            [5, 3, 2],
            [5, 0, 3],
        ];
        let mut positions = component_positions
            .map(|position| position + Vec3::X * 2.0)
            .to_vec();
        positions.extend(
            positions
                .clone()
                .into_iter()
                .map(|position| Vec3::new(-position.x, position.y, position.z)),
        );
        let mut triangles = component_triangles.to_vec();
        triangles.extend(component_triangles.map(|triangle| triangle.map(|vertex| vertex + 6)));
        let mut mesh = Mesh::new(positions, triangles).unwrap();
        let before = mesh.clone();
        let center = Vec3::new(2.0, 0.0, 0.0) + Vec3::splat(1.0 / 3.0);
        let seed = mesh.nearest_triangle(center).unwrap();
        let mut settings = test_settings();
        settings.radius = 1.2;
        settings.strength = 0.8;
        settings.remesh_target_edge_length = Some(settings.radius * 0.09);
        settings.symmetry = Some(SymmetryAxis::X);
        let mut engine = SculptEngine {
            symmetry_center: Some(Vec3::ZERO),
            ..SculptEngine::default()
        };
        engine.begin_stroke(&mesh);
        let mut longest_step = std::time::Duration::ZERO;
        let mut brush_sample = sample(center, seed);
        brush_sample.normal = Vec3::ONE.normalize();

        let started = Instant::now();
        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        let elapsed = started.elapsed();
        longest_step = longest_step.max(elapsed);
        assert!(engine.has_pending_sample());
        let started = Instant::now();
        engine.continue_pending_sample(&mut mesh);
        longest_step = longest_step.max(started.elapsed());
        assert!(engine.take_error().is_none());
        let positive_vertices = mesh
            .positions
            .iter()
            .filter(|position| position.x > 0.0)
            .count();
        let negative_vertices = mesh
            .positions
            .iter()
            .filter(|position| position.x < 0.0)
            .count();
        assert!((positive_vertices > 6) ^ (negative_vertices > 6));

        let started = Instant::now();
        assert!(engine.continue_pending_sample(&mut mesh));
        longest_step = longest_step.max(started.elapsed());
        assert!(engine.take_error().is_none());
        let started = Instant::now();
        assert!(!engine.continue_pending_sample(&mut mesh));
        longest_step = longest_step.max(started.elapsed());
        assert!(engine.take_error().is_none());
        assert!(
            mesh.positions
                .iter()
                .filter(|position| position.x > 0.0)
                .count()
                > 6
        );
        assert!(
            mesh.positions
                .iter()
                .filter(|position| position.x < 0.0)
                .count()
                > 6
        );

        let mut continuation_steps = 2;
        while engine.has_pending_sample() {
            assert!(continuation_steps < 1_024);
            let started = Instant::now();
            engine.continue_pending_sample(&mut mesh);
            let elapsed = started.elapsed();
            longest_step = longest_step.max(elapsed);
            assert!(engine.take_error().is_none());
            continuation_steps += 1;
        }

        assert!(engine.take_sample_committed());
        assert!(mesh.positions.iter().all(|position| {
            let mirrored = Vec3::new(-position.x, position.y, position.z);
            mesh.positions
                .iter()
                .any(|candidate| candidate.distance_squared(mirrored) < 1.0e-8)
        }));
        mesh.validate().unwrap();
        let faces = (0..mesh.triangles.len() as u32).collect::<Vec<_>>();
        assert!(!mesh.faces_have_self_intersections(&faces));
        if !cfg!(debug_assertions) {
            assert!(
                longest_step < std::time::Duration::from_millis(8),
                "symmetric adaptive remesh step exceeded one frame: {longest_step:?}"
            );
        }

        let topology = engine
            .end_stroke(&mesh)
            .topology
            .expect("symmetric remeshing produces topology history");
        topology.apply_before(&mut mesh);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
    }

    #[test]
    fn adaptive_symmetry_rejects_supported_primary_when_mirrored_pass_is_empty() {
        let positions = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.01, 0.0, 0.0),
            Vec3::new(1.0, 0.01, 0.0),
            Vec3::new(0.99, 0.0, 0.0),
            Vec3::new(1.0, -0.01, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let triangles = vec![[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 6, 7]];
        let mut mesh = Mesh::new(positions, triangles).unwrap();
        let before = mesh.clone();
        let mut settings = test_settings();
        settings.radius = 0.05;
        settings.strength = 0.2;
        settings.remesh_target_edge_length = Some(0.02);
        settings.symmetry = Some(SymmetryAxis::X);
        let primary_sample = sample(Vec3::X, 0);
        let mirrored_center = Vec3::NEG_X;
        let mirrored_seed = mesh.nearest_triangle(mirrored_center).unwrap();
        let mirrored_sample = primary_sample.reflected(SymmetryAxis::X, Vec3::ZERO, mirrored_seed);
        let primary_pass = PreparedPass::new(
            &mesh,
            primary_sample,
            settings.radius,
            PassSide::Primary,
            &mut VertexTraversalScratch::default(),
        );
        let mirrored_pass = PreparedPass::new(
            &mesh,
            mirrored_sample,
            settings.radius,
            PassSide::Mirrored,
            &mut VertexTraversalScratch::default(),
        );
        assert!(primary_pass.has_remesh_support(
            &mesh,
            settings.radius,
            settings.strength,
            settings.falloff,
            settings.remesh_target_edge_length.unwrap() * ADAPTIVE_SPLIT_THRESHOLD,
        ));
        assert!(mirrored_pass.vertices.is_empty());
        assert!(!mirrored_pass.has_remesh_support(
            &mesh,
            settings.radius,
            settings.strength,
            settings.falloff,
            settings.remesh_target_edge_length.unwrap() * ADAPTIVE_SPLIT_THRESHOLD,
        ));

        let mut engine = SculptEngine {
            symmetry_center: Some(Vec3::ZERO),
            ..SculptEngine::default()
        };
        engine.begin_stroke(&mesh);
        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            primary_sample,
            DabKind::Spatial,
        ));
        drain_pending_sample(&mut engine, &mut mesh);
        assert!(!engine.has_pending_sample());
        assert!(!engine.take_sample_committed());
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
    }

    #[test]
    fn adaptive_sample_does_not_deform_when_boundary_topology_cannot_add_support() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let before = mesh.clone();
        let mut settings = test_settings();
        settings.radius = 0.2;
        settings.remesh_target_edge_length = Some(0.01);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(before.positions[0], 0),
            DabKind::Spatial,
        ));
        drain_pending_sample(&mut engine, &mut mesh);
        assert!(!engine.has_pending_sample());
        assert!(!engine.take_sample_committed());
        assert_eq!(
            engine.take_warning().as_deref(),
            Some(
                "Mesh resolution could not safely support this brush; the brush sample was rolled back"
            )
        );
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert!(engine.end_stroke(&mesh).edit.is_empty());
    }

    #[test]
    fn adaptive_topology_keeps_edges_that_are_already_usable() {
        let mut mesh = Mesh::new(
            vec![
                Vec3::X,
                Vec3::Y,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ],
            vec![
                [4, 0, 1],
                [4, 1, 2],
                [4, 2, 3],
                [4, 3, 0],
                [5, 1, 0],
                [5, 2, 1],
                [5, 3, 2],
                [5, 0, 3],
            ],
        )
        .unwrap();
        let mut settings = test_settings();
        settings.radius = 2.0;
        settings.remesh_target_edge_length = Some(0.8);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::splat(1.0 / 3.0), 0),
            DabKind::Spatial,
        ));
        drain_pending_sample(&mut engine, &mut mesh);
        assert!(engine.take_sample_committed());
        assert_eq!(mesh.positions.len(), 6);
        assert_eq!(mesh.triangles.len(), 8);
    }

    #[test]
    fn stroke_records_each_original_vertex_once_across_samples() {
        let mut mesh = grid();
        let before = mesh.positions.clone();
        let settings = test_settings();
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        );
        engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        );
        let outcome = engine.end_stroke(&mesh);

        assert!(!outcome.edit.positions.is_empty());
        assert!(
            outcome
                .edit
                .positions
                .windows(2)
                .all(|changes| changes[0].vertex < changes[1].vertex)
        );
        for change in &outcome.edit.positions {
            assert_eq!(change.before, before[change.vertex as usize]);
            assert_eq!(change.after, mesh.positions[change.vertex as usize]);
        }
    }

    #[test]
    fn fixed_topology_step_limits_a_degenerate_position_edit_before_refreshing_derived_data() {
        let mut mesh = grid();
        let before_positions = mesh.positions.clone();
        let before_normals = mesh.normals.clone();
        let mut settings = test_settings();
        settings.falloff = 1.0;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Pinch,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        ));

        assert_ne!(mesh.positions, before_positions);
        assert_eq!(mesh.normals, before_normals);
        assert!(!engine.take_updated_vertices().is_empty());
        assert!(engine.take_error().is_none());
        assert!(!engine.end_stroke(&mesh).edit.is_empty());
    }

    #[test]
    fn anchored_grab_uses_press_selection_and_total_pointer_displacement() {
        let mut mesh = grid();
        let before = mesh.positions.clone();
        let settings = test_settings();
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        assert!(engine.anchor_grab(&mesh, &settings, sample(Vec3::ZERO, 0)));

        assert!(engine.apply_grab_delta(&mut mesh, Vec3::new(0.25, 0.0, 0.0), 1.0));
        assert!(mesh.positions[4].x > before[4].x);
        assert!(engine.apply_grab_delta(&mut mesh, Vec3::new(-0.25, 0.0, 0.0), 1.0));
        assert_eq!(mesh.positions, before);
        assert!(engine.end_stroke(&mesh).edit.is_empty());
    }

    #[test]
    fn adaptive_topology_clamps_a_foldover_instead_of_accepting_it() {
        let mut mesh = grid();
        let before = mesh.positions.clone();
        let mut settings = test_settings();
        settings.falloff = 0.95;
        settings.remesh_target_edge_length = Some(1.0);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        engine.apply_sample(
            &mut mesh,
            SculptTool::Pinch,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        );
        assert!(engine.has_pending_sample());
        let continuation_steps = drain_pending_sample(&mut engine, &mut mesh);
        assert!(continuation_steps >= SAFE_DEFORMATION_SEARCH_STEPS);
        assert_eq!(
            engine.take_warning().as_deref(),
            Some("Brush movement limited to prevent self-intersection")
        );
        assert_ne!(mesh.positions, before);
        assert!(mesh.validate().is_ok());
        assert!(!mesh.faces_have_self_intersections(&(0..8).collect::<Vec<_>>()));
    }

    #[test]
    fn grab_symmetry_cancels_axis_motion_on_the_plane() {
        let mut mesh = grid();
        let mut settings = test_settings();
        settings.symmetry = Some(SymmetryAxis::X);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        assert!(engine.anchor_grab(&mesh, &settings, sample(Vec3::ZERO, 0)));
        assert!(engine.apply_grab_delta(&mut mesh, Vec3::new(0.25, 0.2, 0.0), 1.0));

        assert!(mesh.positions[4].x.abs() < 1.0e-6);
        assert!(mesh.positions[4].y > 0.0);
    }

    #[test]
    fn temporary_smoothing_rebases_an_active_grab_anchor() {
        let mut mesh = grid();
        let mut settings = test_settings();
        settings.accumulation = Accumulation::Accumulate;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        assert!(engine.anchor_grab(&mesh, &settings, sample(Vec3::ZERO, 0)));
        assert!(engine.apply_grab_delta(&mut mesh, Vec3::new(0.25, 0.0, 0.0), 1.0));

        let positions_before_smooth = mesh.positions.clone();
        let bases_before = engine
            .active
            .as_ref()
            .and_then(|stroke| stroke.grab.as_ref())
            .unwrap()
            .vertices
            .iter()
            .map(|(&vertex, grabbed)| (vertex, grabbed.base))
            .collect::<HashMap<_, _>>();
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Smooth,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        ));

        let grab = engine
            .active
            .as_ref()
            .and_then(|stroke| stroke.grab.as_ref())
            .unwrap();
        for (&vertex, grabbed) in &grab.vertices {
            let smooth_delta =
                mesh.positions[vertex as usize] - positions_before_smooth[vertex as usize];
            assert!(
                grabbed
                    .base
                    .abs_diff_eq(bases_before[&vertex] + smooth_delta, 1.0e-6)
            );
        }
    }

    #[test]
    fn taubin_smoothing_keeps_more_volume_than_a_full_laplacian_step() {
        fn volume(mesh: &Mesh) -> f32 {
            mesh.triangles
                .iter()
                .map(|triangle| {
                    let [a, b, c] = triangle.map(|vertex| mesh.positions[vertex as usize]);
                    a.dot(b.cross(c)) / 6.0
                })
                .sum::<f32>()
                .abs()
        }

        let mut mesh = octahedron();
        let before = volume(&mesh);
        let mut settings = test_settings();
        settings.radius = 3.0;
        settings.falloff = 1.0;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Smooth,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        ));

        assert!(volume(&mesh) > before * 0.4);
    }

    #[test]
    fn undersampled_brush_reports_one_non_blocking_warning() {
        let mut mesh = grid();
        let mut settings = test_settings();
        settings.radius = 0.25;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::ZERO, 0),
            DabKind::Spatial,
        ));
        let outcome = engine.end_stroke(&mesh);
        assert!(outcome.warning.is_some());
        assert!(!outcome.edit.is_empty());
    }

    #[test]
    #[ignore = "release-mode performance envelope"]
    fn million_face_fixed_and_adaptive_sculpt_samples() {
        const CELLS: usize = 708;
        let row = CELLS + 1;
        let mut positions = Vec::with_capacity(row * row);
        for y in 0..=CELLS {
            for x in 0..=CELLS {
                positions.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let mut triangles = Vec::with_capacity(CELLS * CELLS * 2);
        for y in 0..CELLS {
            for x in 0..CELLS {
                let a = (y * row + x) as u32;
                let b = a + 1;
                let c = a + row as u32;
                let d = c + 1;
                triangles.push([a, b, d]);
                triangles.push([a, d, c]);
            }
        }
        let mut mesh = Mesh::new(positions, triangles).unwrap();
        let center = Vec3::new(CELLS as f32 * 0.5, CELLS as f32 * 0.5, 0.0);
        let seed = mesh.nearest_triangle(center).unwrap();
        let mut settings = BrushSettings {
            radius: 10.0,
            strength: 0.1,
            falloff: 0.15,
            remesh_target_edge_length: None,
            invert: false,
            symmetry: None,
            accumulation: Accumulation::Capped,
        };
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        let fixed_started = Instant::now();
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(center, seed),
            DabKind::Spatial,
        ));
        let fixed_elapsed = fixed_started.elapsed();
        let _ = engine.take_updated_vertices();
        let _ = engine.end_stroke(&mesh);

        settings.remesh_target_edge_length = Some(settings.radius * 0.09);
        let seed = mesh.nearest_triangle(center).unwrap();
        engine.begin_stroke(&mesh);
        let step_started = Instant::now();
        engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(center, seed),
            DabKind::Spatial,
        );
        let mut adaptive_elapsed = step_started.elapsed();
        let mut published_changes = engine.take_mesh_changes().is_some();
        while engine.has_pending_sample() {
            let step_started = Instant::now();
            engine.continue_pending_sample(&mut mesh);
            let step_elapsed = step_started.elapsed();
            adaptive_elapsed = adaptive_elapsed.max(step_elapsed);
            published_changes |= engine.take_mesh_changes().is_some();
            assert!(
                step_elapsed < std::time::Duration::from_millis(8),
                "million-face adaptive continuation exceeded one frame: {step_elapsed:?}"
            );
            assert!(engine.take_error().is_none());
        }
        assert!(
            adaptive_elapsed < std::time::Duration::from_millis(8),
            "million-face adaptive sample exceeded one frame: {adaptive_elapsed:?}"
        );
        assert!(published_changes);
        let _ = engine.end_stroke(&mesh);

        eprintln!(
            "million-face sculpt sample: fixed={fixed_elapsed:?}, adaptive={adaptive_elapsed:?}"
        );
    }
}
