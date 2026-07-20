use std::fmt;

use glam::Vec3;
use hashbrown::HashMap;
use smallvec::SmallVec;

use crate::{
    history::{LocalEdit, MaskChange, PositionChange},
    mesh::{
        Mesh, MeshChangeSet, MeshEditDelta, MeshEditRecorder, RayHit, RemeshSettings,
        VertexTraversalScratch,
    },
};

const MAX_ADAPTIVE_TOPOLOGY_EDITS_PER_DAB: usize = 512;
const MAX_ADAPTIVE_REMESH_ITERATIONS: u32 = 8;
const SAFE_DEFORMATION_SEARCH_STEPS: usize = 6;
const MIN_SAFE_DEFORMATION_FACTOR: f32 = 1.0 / 64.0;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SculptTool {
    Grab,
    #[default]
    Draw,
    Inflate,
    Smooth,
    Pinch,
    Flatten,
    Mask,
}

impl SculptTool {
    pub const ALL: [Self; 7] = [
        Self::Grab,
        Self::Draw,
        Self::Inflate,
        Self::Smooth,
        Self::Pinch,
        Self::Flatten,
        Self::Mask,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Grab => "Grab",
            Self::Draw => "Draw",
            Self::Inflate => "Inflate / Deflate",
            Self::Smooth => "Smooth",
            Self::Pinch => "Pinch",
            Self::Flatten => "Flatten",
            Self::Mask => "Mask",
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

    fn distance_to_plane(self, point: Vec3, plane_center: Vec3) -> f32 {
        match self {
            Self::X => (point.x - plane_center.x).abs(),
            Self::Y => (point.y - plane_center.y).abs(),
            Self::Z => (point.z - plane_center.z).abs(),
        }
    }
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
}

#[derive(Clone, Debug, Default)]
struct StrokeOriginals {
    positions: HashMap<u32, Vec3>,
    masks: HashMap<u32, f32>,
}

/// The editable result of a completed stroke. Adaptive topology is already
/// applied while sampling; `topology` stores its exact whole-stroke history.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrokeOutcome {
    pub edit: LocalEdit,
    pub topology: Option<MeshEditDelta>,
}

#[derive(Default, Debug)]
pub struct SculptEngine {
    active: Option<StrokeState>,
    symmetry_center: Option<Vec3>,
    traversal: VertexTraversalScratch,
    source_positions: HashMap<u32, Vec3>,
    updated_vertices: Vec<u32>,
    mesh_changes: Option<MeshChangeSet>,
    warning: Option<String>,
    error: Option<String>,
}

impl SculptEngine {
    #[must_use]
    pub fn is_stroking(&self) -> bool {
        self.active.is_some()
    }

    /// Resets per-document state after importing or replacing a mesh.
    pub fn reset_for_mesh(&mut self, mesh: &Mesh) {
        self.active = None;
        self.symmetry_center = Some(mesh.center().unwrap_or(Vec3::ZERO));
        self.traversal = VertexTraversalScratch::default();
        self.source_positions = HashMap::new();
        self.updated_vertices = Vec::new();
        self.mesh_changes = None;
        self.warning = None;
        self.error = None;
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
        });
        self.updated_vertices.clear();
        self.mesh_changes = None;
        self.warning = None;
        self.error = None;
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

    /// Applies one brush sample and returns whether it changed the mesh.
    pub fn apply_sample(
        &mut self,
        mesh: &mut Mesh,
        tool: SculptTool,
        settings: &BrushSettings,
        sample: BrushSample,
    ) -> bool {
        self.updated_vertices.clear();
        self.mesh_changes = None;
        self.warning = None;
        self.error = None;
        let Some(stroke) = self.active.as_ref() else {
            return false;
        };
        if mesh.positions.is_empty()
            || !settings.radius.is_finite()
            || settings.radius <= f32::EPSILON
            || !settings.strength.is_finite()
            || settings.strength.abs() <= f32::EPSILON
            || !sample.center.is_finite()
        {
            return false;
        }

        let symmetry_center = stroke.symmetry_center;
        let cached_mirrored_seed = stroke.mirrored_seed;
        let mut samples = SmallVec::<[BrushSample; 2]>::from_slice(&[sample]);

        if let Some(axis) = settings.symmetry
            && axis.distance_to_plane(sample.center, symmetry_center) > settings.radius * 1.0e-4
        {
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
        let adaptive = tool != SculptTool::Mask && target_edge_length.is_some();
        let mut topology_recorder = adaptive.then(|| MeshEditRecorder::new(mesh));
        let mut topology_changes = MeshChangeSet::default();
        let mut topology_edits = 0;

        if let (Some(target), Some(recorder)) = (target_edge_length, topology_recorder.as_mut()) {
            for brush_sample in &samples {
                let seed = mesh
                    .nearest_triangle(brush_sample.center)
                    .unwrap_or(brush_sample.seed_triangle);
                let active = mesh.brush_remesh_vertices(
                    seed,
                    brush_sample.center,
                    settings.radius,
                    target,
                    brush_sample.view_direction,
                );
                let remaining = MAX_ADAPTIVE_TOPOLOGY_EDITS_PER_DAB.saturating_sub(topology_edits);
                if active.is_empty() || remaining == 0 {
                    continue;
                }
                let mut remesh = RemeshSettings::new(target);
                remesh.iterations = MAX_ADAPTIVE_REMESH_ITERATIONS;
                remesh.max_topology_edits = remaining;
                remesh.relaxation = 0.0;
                let outcome = mesh.remesh_region(&active, remesh, recorder);
                topology_edits +=
                    outcome.stats.splits + outcome.stats.collapses + outcome.stats.flips;
                topology_changes.merge(outcome.changes);
            }

            let topology_valid = mesh.local_changes_are_valid(&topology_changes);
            let topology_intersects =
                mesh.faces_have_self_intersections(&topology_changes.dirty_faces);
            if !topology_valid || topology_intersects {
                topology_recorder
                    .take()
                    .expect("adaptive samples have a topology recorder")
                    .finish(mesh)
                    .apply_before(mesh);
                self.error = Some(
                    "Adaptive topology rejected an invalid local mesh update; the latest brush sample was rolled back"
                        .to_owned(),
                );
                return false;
            }
        }

        for brush_sample in &mut samples {
            if let Some(seed) = mesh.nearest_triangle(brush_sample.center) {
                brush_sample.seed_triangle = seed;
            }
        }
        let mut passes = SmallVec::<[PreparedPass; 2]>::new();
        for brush_sample in samples {
            passes.push(PreparedPass::new(
                mesh,
                brush_sample,
                settings.radius,
                &mut self.traversal,
            ));
        }

        if adaptive && passes.iter().all(|pass| pass.vertices.is_empty()) {
            topology_recorder
                .take()
                .expect("adaptive samples have a topology recorder")
                .finish(mesh)
                .apply_before(mesh);
            self.warning = Some("Mesh resolution too low for this brush".to_owned());
            return false;
        }

        capture_source_positions(mesh, tool, &passes, &mut self.source_positions);
        let baseline_faces = if adaptive {
            capture_face_crosses(mesh, &passes)
        } else {
            HashMap::new()
        };

        if adaptive {
            let mut deformation_recorder = Some(MeshEditRecorder::new(mesh));
            let (mut changed, mut affected) = apply_passes(
                mesh,
                tool,
                settings,
                &passes,
                &self.source_positions,
                &mut self
                    .active
                    .as_mut()
                    .expect("active stroke checked above")
                    .original,
                deformation_recorder.as_mut(),
            );
            let mut safe = false;
            if changed {
                safe = refresh_and_validate_deformation(
                    mesh,
                    &mut affected,
                    &baseline_faces,
                    &mut self.updated_vertices,
                );
            }

            if changed && !safe {
                deformation_recorder
                    .take()
                    .expect("deformation attempt has a recorder")
                    .finish(mesh)
                    .apply_before(mesh);
                let mut low = 0.0_f32;
                let mut high = 1.0_f32;
                for _ in 0..SAFE_DEFORMATION_SEARCH_STEPS {
                    let factor = (low + high) * 0.5;
                    let mut scaled = *settings;
                    scaled.strength *= factor;
                    let mut trial_recorder = MeshEditRecorder::new(mesh);
                    let (trial_changed, mut trial_affected) = apply_passes(
                        mesh,
                        tool,
                        &scaled,
                        &passes,
                        &self.source_positions,
                        &mut self
                            .active
                            .as_mut()
                            .expect("active stroke checked above")
                            .original,
                        Some(&mut trial_recorder),
                    );
                    let trial_safe = trial_changed
                        && refresh_and_validate_deformation(
                            mesh,
                            &mut trial_affected,
                            &baseline_faces,
                            &mut Vec::new(),
                        );
                    trial_recorder.finish(mesh).apply_before(mesh);
                    if trial_safe {
                        low = factor;
                    } else {
                        high = factor;
                    }
                }

                changed = false;
                affected.clear();
                self.updated_vertices.clear();
                if low >= MIN_SAFE_DEFORMATION_FACTOR {
                    let mut scaled = *settings;
                    scaled.strength *= low;
                    deformation_recorder = Some(MeshEditRecorder::new(mesh));
                    (changed, affected) = apply_passes(
                        mesh,
                        tool,
                        &scaled,
                        &passes,
                        &self.source_positions,
                        &mut self
                            .active
                            .as_mut()
                            .expect("active stroke checked above")
                            .original,
                        deformation_recorder.as_mut(),
                    );
                    safe = changed
                        && refresh_and_validate_deformation(
                            mesh,
                            &mut affected,
                            &baseline_faces,
                            &mut self.updated_vertices,
                        );
                    if !safe {
                        deformation_recorder
                            .take()
                            .expect("deformation attempt has a recorder")
                            .finish(mesh)
                            .apply_before(mesh);
                        changed = false;
                        self.updated_vertices.clear();
                    }
                }
                self.warning =
                    Some("Brush movement limited to prevent self-intersection".to_owned());
            }

            if changed {
                topology_recorder
                    .as_mut()
                    .expect("adaptive samples have a topology recorder")
                    .absorb_recorder(
                        deformation_recorder
                            .take()
                            .expect("successful deformation has a recorder"),
                        mesh,
                    );
            }
            topology_changes
                .dirty_vertices
                .extend(self.updated_vertices.iter().copied());
            topology_changes.finalize(mesh.positions.len(), mesh.triangles.len());
            self.updated_vertices
                .extend(topology_changes.dirty_vertices.iter().copied());
            self.updated_vertices.sort_unstable();
            self.updated_vertices.dedup();

            let sample_changed = changed || topology_edits != 0;
            if sample_changed {
                self.mesh_changes = Some(topology_changes);
                if let Some(stroke) = self.active.as_mut() {
                    stroke.recorder.absorb_recorder(
                        topology_recorder
                            .take()
                            .expect("adaptive samples have a topology recorder"),
                        mesh,
                    );
                }
            }
            return sample_changed;
        }

        let (changed, mut affected) = apply_passes(
            mesh,
            tool,
            settings,
            &passes,
            &self.source_positions,
            &mut self
                .active
                .as_mut()
                .expect("active stroke checked above")
                .original,
            None,
        );
        if !changed {
            return false;
        }
        affected.sort_unstable();
        affected.dedup();
        if tool == SculptTool::Mask {
            self.updated_vertices = affected;
            return true;
        }
        let Some(faces) = mesh.validated_deformation_faces(&affected) else {
            for &vertex in &affected {
                if let Some(&position) = self.source_positions.get(&vertex)
                    && let Some(target) = mesh.positions.get_mut(vertex as usize)
                {
                    *target = position;
                }
            }
            self.error = Some(
                "Sculpt sample rejected an invalid local mesh update; the latest brush sample was rolled back"
                    .to_owned(),
            );
            return false;
        };
        self.updated_vertices = mesh.update_deformed_faces(&faces);
        self.updated_vertices.extend(affected);
        self.updated_vertices.sort_unstable();
        self.updated_vertices.dedup();
        true
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
        if edit.is_empty() && topology.is_none() {
            return StrokeOutcome::default();
        }
        StrokeOutcome { edit, topology }
    }
}

#[derive(Debug)]
struct PreparedPass {
    sample: BrushSample,
    vertices: Vec<u32>,
}

struct SampleEdits<'a> {
    affected: &'a mut Vec<u32>,
    original: &'a mut StrokeOriginals,
    recorder: Option<&'a mut MeshEditRecorder>,
}

fn apply_passes(
    mesh: &mut Mesh,
    tool: SculptTool,
    settings: &BrushSettings,
    passes: &[PreparedPass],
    source_positions: &HashMap<u32, Vec3>,
    original: &mut StrokeOriginals,
    recorder: Option<&mut MeshEditRecorder>,
) -> (bool, Vec<u32>) {
    let mut affected = Vec::new();
    let mut edits = SampleEdits {
        affected: &mut affected,
        original,
        recorder,
    };
    let mut changed = false;
    for pass in passes {
        changed |= apply_pass(mesh, tool, settings, pass, source_positions, &mut edits);
    }
    affected.sort_unstable();
    affected.dedup();
    (changed, affected)
}

fn capture_face_crosses(mesh: &Mesh, passes: &[PreparedPass]) -> HashMap<u32, Vec3> {
    let mut faces = HashMap::new();
    for vertex in passes.iter().flat_map(|pass| pass.vertices.iter().copied()) {
        let Some(incident) = mesh.topology.vertex_triangles.get(vertex as usize) else {
            continue;
        };
        for &face in incident {
            let Some(&triangle) = mesh.triangles.get(face as usize) else {
                continue;
            };
            let [a, b, c] = triangle.map(|index| mesh.positions[index as usize]);
            faces.entry(face).or_insert_with(|| (b - a).cross(c - a));
        }
    }
    faces
}

fn refresh_and_validate_deformation(
    mesh: &mut Mesh,
    affected: &mut Vec<u32>,
    baseline_faces: &HashMap<u32, Vec3>,
    updated_vertices: &mut Vec<u32>,
) -> bool {
    affected.sort_unstable();
    affected.dedup();
    let Some(faces) = mesh.validated_deformation_faces(affected) else {
        return false;
    };
    if faces.iter().any(|face| {
        let Some(baseline) = baseline_faces.get(face) else {
            return true;
        };
        let triangle = mesh.triangles[*face as usize];
        let [a, b, c] = triangle.map(|index| mesh.positions[index as usize]);
        baseline.dot((b - a).cross(c - a)) <= 0.0
    }) {
        return false;
    }
    updated_vertices.clear();
    updated_vertices.extend(mesh.update_deformed_faces(&faces));
    updated_vertices.extend(affected.iter().copied());
    updated_vertices.sort_unstable();
    updated_vertices.dedup();
    !mesh.faces_have_self_intersections(&faces)
}

impl PreparedPass {
    fn new(
        mesh: &Mesh,
        sample: BrushSample,
        radius: f32,
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
        Self { sample, vertices }
    }
}

fn apply_pass(
    mesh: &mut Mesh,
    tool: SculptTool,
    settings: &BrushSettings,
    pass: &PreparedPass,
    source_positions: &HashMap<u32, Vec3>,
    edits: &mut SampleEdits<'_>,
) -> bool {
    let radius = settings.radius;
    let pressure = pass.sample.pressure.clamp(0.0, 1.0);
    let strength = settings.strength.abs() * pressure;
    if strength <= f32::EPSILON {
        return false;
    }

    let inverted = settings.invert ^ pass.sample.invert_modifier;
    let direction = if inverted { -1.0 } else { 1.0 };
    let brush_normal = pass.sample.normal.normalize_or_zero();
    let mut changed = false;

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
            let Some(before) = mesh.mask.get(index).copied() else {
                continue;
            };
            let next = (before + direction * strength * weight).clamp(0.0, 1.0);
            if next != before {
                if let Some(recorder) = edits.recorder.as_deref_mut() {
                    recorder.record_vertex(mesh, vertex);
                }
                edits.original.masks.entry(vertex).or_insert(before);
                mesh.mask[index] = next;
                edits.affected.push(vertex);
                changed = true;
            }
            continue;
        }

        let unmasked = 1.0 - mesh.mask.get(index).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let influence = strength * weight * unmasked;
        if influence <= f32::EPSILON {
            continue;
        }

        let displacement = match tool {
            SculptTool::Grab => pass.sample.drag_delta * influence,
            SculptTool::Draw => brush_normal * (direction * radius * 0.15 * influence),
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
                normal * (direction * radius * 0.15 * influence)
            }
            SculptTool::Smooth => {
                let Some(neighbors) = mesh.topology.vertex_neighbors.get(index) else {
                    continue;
                };
                if neighbors.is_empty() {
                    continue;
                }
                let average = neighbors
                    .iter()
                    .filter_map(|neighbor| source_positions.get(neighbor))
                    .copied()
                    .sum::<Vec3>()
                    / neighbors.len() as f32;
                (average - position) * influence.min(1.0)
            }
            SculptTool::Pinch => {
                let to_center = pass.sample.center - position;
                let tangent = to_center - brush_normal * to_center.dot(brush_normal);
                tangent * (direction * influence.min(1.0))
            }
            SculptTool::Flatten => {
                let signed_plane_distance = (pass.sample.center - position).dot(brush_normal);
                brush_normal * (direction * signed_plane_distance * influence.min(1.0))
            }
            SculptTool::Mask => unreachable!("mask is handled before geometry brushes"),
        };

        if !displacement.is_finite() || displacement.length_squared() <= f32::EPSILON.powi(2) {
            continue;
        }
        let next = position + displacement;
        if next != mesh.positions[index] {
            if let Some(recorder) = edits.recorder.as_deref_mut() {
                recorder.record_vertex(mesh, vertex);
            }
            edits
                .original
                .positions
                .entry(vertex)
                .or_insert(mesh.positions[index]);
            mesh.positions[index] = next;
            edits.affected.push(vertex);
            changed = true;
        }
    }

    changed
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
    use std::time::Instant;

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
        }
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
        ));
        assert!(mesh.positions[5].z > 0.0);
        assert_eq!(mesh.positions[3].z, mesh.positions[5].z);
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

        normal_engine.apply_sample(&mut outward, SculptTool::Inflate, &settings, normal_sample);
        inverse_engine.apply_sample(&mut inward, SculptTool::Inflate, &settings, inverse_sample);

        assert!(outward.positions[4].z > 0.0);
        assert_eq!(outward.positions[4].z, -inward.positions[4].z);
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

        assert!(engine.apply_sample(&mut mesh, SculptTool::Smooth, &settings, sample(center, 0),));
        assert!(mesh.positions[4].z < before);
    }

    #[test]
    fn mask_brush_adds_and_inverted_mask_brush_erases() {
        let mut mesh = grid();
        let settings = test_settings();
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);
        let mut brush_sample = sample(Vec3::ZERO, 0);

        assert!(engine.apply_sample(&mut mesh, SculptTool::Mask, &settings, brush_sample,));
        assert!(mesh.mask[4] > 0.0);

        brush_sample.invert_modifier = true;
        assert!(engine.apply_sample(&mut mesh, SculptTool::Mask, &settings, brush_sample,));
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

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::splat(1.0 / 3.0), 0),
        ));
        let updated = engine.take_updated_vertices();
        let changes = engine
            .take_mesh_changes()
            .expect("adaptive sample produces renderer changes");
        assert!(!changes.dirty_faces.is_empty());
        assert_ne!(mesh.triangles, original_triangles);

        let center = Vec3::splat(1.0 / 3.0);
        let seed_triangle = mesh.nearest_triangle(center).unwrap();
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(center, seed_triangle),
        ));
        let second_sample_vertices = engine.take_updated_vertices();
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
    fn adaptive_topology_supports_a_brush_smaller_than_its_seed_face() {
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
        let center = Vec3::splat(1.0 / 3.0);
        let seed = mesh.nearest_triangle(center).unwrap();
        let mut settings = test_settings();
        settings.radius = 0.1;
        settings.remesh_target_edge_length = Some(settings.radius * 0.2);
        let before = mesh.positions.clone();
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(&mut mesh, SculptTool::Draw, &settings, sample(center, seed),));
        assert_ne!(mesh.positions, before);
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
        );
        engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::ZERO, 0),
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
    fn fixed_topology_rejects_a_degenerate_position_edit_before_refreshing_derived_data() {
        let mut mesh = grid();
        let before_positions = mesh.positions.clone();
        let before_normals = mesh.normals.clone();
        let mut settings = test_settings();
        settings.falloff = 1.0;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Pinch,
            &settings,
            sample(Vec3::ZERO, 0),
        ));

        assert_eq!(mesh.positions, before_positions);
        assert_eq!(mesh.normals, before_normals);
        assert!(engine.take_updated_vertices().is_empty());
        assert!(engine.take_mesh_changes().is_none());
        assert!(engine.take_error().is_some());
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

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Pinch,
            &settings,
            sample(Vec3::ZERO, 0),
        ));
        assert_eq!(
            engine.take_warning().as_deref(),
            Some("Brush movement limited to prevent self-intersection")
        );
        assert_ne!(mesh.positions, before);
        assert!(
            mesh.validated_deformation_faces(&(0..9).collect::<Vec<_>>())
                .is_some()
        );
        assert!(!mesh.faces_have_self_intersections(&(0..8).collect::<Vec<_>>()));
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
        };
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        let fixed_started = Instant::now();
        assert!(engine.apply_sample(&mut mesh, SculptTool::Draw, &settings, sample(center, seed),));
        let fixed_elapsed = fixed_started.elapsed();
        let _ = engine.take_updated_vertices();
        let _ = engine.end_stroke(&mesh);

        settings.remesh_target_edge_length = Some(settings.radius * 0.09);
        let seed = mesh.nearest_triangle(center).unwrap();
        engine.begin_stroke(&mesh);
        let adaptive_started = Instant::now();
        assert!(engine.apply_sample(&mut mesh, SculptTool::Draw, &settings, sample(center, seed),));
        let adaptive_elapsed = adaptive_started.elapsed();
        assert!(engine.take_mesh_changes().is_some());
        let _ = engine.end_stroke(&mesh);

        eprintln!(
            "million-face sculpt sample: fixed={fixed_elapsed:?}, adaptive={adaptive_elapsed:?}"
        );
    }
}
