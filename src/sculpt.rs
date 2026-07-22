use std::fmt;

use glam::Vec3;
use hashbrown::HashMap;
use smallvec::SmallVec;

use crate::{
    history::{LocalEdit, MaskChange, PositionChange},
    mesh::{Mesh, RayHit, VertexTraversalScratch},
};

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

    fn distance_to_plane(self, point: Vec3, plane_center: Vec3) -> f32 {
        match self {
            Self::X => (point.x - plane_center.x).abs(),
            Self::Y => (point.y - plane_center.y).abs(),
            Self::Z => (point.z - plane_center.z).abs(),
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
    coverage: CoverageMap,
}

#[derive(Clone, Debug, Default)]
struct StrokeOriginals {
    positions: HashMap<u32, Vec3>,
    masks: HashMap<u32, f32>,
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrokeOutcome {
    pub edit: LocalEdit,
}

#[derive(Default, Debug)]
pub struct SculptEngine {
    active: Option<StrokeState>,
    symmetry_center: Option<Vec3>,
    traversal: VertexTraversalScratch,
    source_positions: HashMap<u32, Vec3>,
    updated_vertices: Vec<u32>,
    error: Option<String>,
    sample_committed: bool,
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
            coverage: CoverageMap::new(),
        });
        self.updated_vertices.clear();
        self.error = None;
        self.sample_committed = false;
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
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    /// Returns whether the latest completed sample committed an editable change.
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
        let Some(stroke) = self.active.as_ref() else {
            return false;
        };
        if mesh.positions.is_empty()
            || !settings.radius.is_finite()
            || settings.radius <= f32::EPSILON
            || !settings.strength.is_finite()
            || settings.strength.abs() <= f32::EPSILON
            || !sample.center.is_finite()
            || !sample.pressure.is_finite()
            || sample.pressure.clamp(0.0, 1.0) <= 0.0
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

        capture_source_positions(mesh, tool, &passes, &mut self.source_positions);

        let channel = (dab_kind == DabKind::Spatial
            && settings.accumulation == Accumulation::Capped)
            .then(|| CoverageChannel::for_sample(tool, settings.invert ^ sample.invert_modifier))
            .flatten();
        let mut staged_coverage = CoverageMap::new();
        let (changed, mut affected) = {
            let stroke = self.active.as_mut().expect("active stroke checked above");
            let mut coverage = SampleCoverage {
                channel,
                committed: &stroke.coverage,
                staged: &mut staged_coverage,
            };
            apply_passes(
                mesh,
                tool,
                settings,
                &passes,
                &self.source_positions,
                &mut stroke.original,
                &mut coverage,
            )
        };
        if !changed {
            return false;
        }
        affected.sort_unstable();
        affected.dedup();
        if tool == SculptTool::Mask {
            self.commit_coverage(staged_coverage);
            self.updated_vertices = affected;
            self.sample_committed = true;
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
        self.commit_coverage(staged_coverage);
        self.sample_committed = true;
        true
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

    fn clear_step_outputs(&mut self) {
        self.updated_vertices.clear();
        self.error = None;
        self.sample_committed = false;
    }

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
        if edit.is_empty() {
            return StrokeOutcome::default();
        }
        StrokeOutcome { edit }
    }
}

#[derive(Debug)]
struct PreparedPass {
    sample: BrushSample,
    vertices: Vec<u32>,
}

struct SampleEdits<'a, 'coverage> {
    affected: &'a mut Vec<u32>,
    original: &'a mut StrokeOriginals,
    coverage: &'a mut SampleCoverage<'coverage>,
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

    fn is_limited(&self) -> bool {
        self.channel.is_some()
    }
}

fn apply_passes(
    mesh: &mut Mesh,
    tool: SculptTool,
    settings: &BrushSettings,
    passes: &[PreparedPass],
    source_positions: &HashMap<u32, Vec3>,
    original: &mut StrokeOriginals,
    coverage: &mut SampleCoverage<'_>,
) -> (bool, Vec<u32>) {
    let mut affected = Vec::new();
    let mut edits = SampleEdits {
        affected: &mut affected,
        original,
        coverage,
    };
    let mut changed = false;
    for pass in passes {
        changed |= apply_pass(mesh, tool, settings, pass, source_positions, &mut edits);
    }
    affected.sort_unstable();
    affected.dedup();
    (changed, affected)
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
    edits: &mut SampleEdits<'_, '_>,
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
    let clay_plane = if tool == SculptTool::Clay {
        clay_surface_plane(
            mesh,
            settings,
            pass,
            source_positions,
            brush_normal,
            direction,
        )
    } else {
        None
    };
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
            let influence = edits.coverage.limited_influence(vertex, strength * weight);
            if influence <= f32::EPSILON {
                continue;
            }
            let next = (before + direction * influence).clamp(0.0, 1.0);
            if next != before {
                edits.original.masks.entry(vertex).or_insert(before);
                mesh.mask[index] = next;
                edits.affected.push(vertex);
                changed = true;
            }
            continue;
        }

        let unmasked = 1.0 - mesh.mask.get(index).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let influence = edits
            .coverage
            .limited_influence(vertex, strength * weight * unmasked);
        if influence <= f32::EPSILON {
            continue;
        }

        let displacement = match tool {
            SculptTool::Grab => pass.sample.drag_delta * influence,
            SculptTool::Draw => brush_normal * (direction * radius * 0.15 * influence),
            SculptTool::Clay => {
                let Some((plane_point, plane_normal)) = clay_plane else {
                    continue;
                };
                let signed_plane_distance = (plane_point - position).dot(plane_normal);
                plane_normal * signed_plane_distance * influence.min(1.0)
            }
            SculptTool::Crease => {
                let to_center = pass.sample.center - position;
                let tangent = to_center - brush_normal * to_center.dot(brush_normal);
                brush_normal * (direction * radius * 0.15 * influence)
                    + tangent * ((2.0 / 3.0) * influence.min(1.0))
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
        let base = if edits.coverage.is_limited() {
            mesh.positions[index]
        } else {
            position
        };
        let next = base + displacement;
        if next != mesh.positions[index] {
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

fn clay_surface_plane(
    mesh: &Mesh,
    settings: &BrushSettings,
    pass: &PreparedPass,
    source_positions: &HashMap<u32, Vec3>,
    brush_normal: Vec3,
    direction: f32,
) -> Option<(Vec3, Vec3)> {
    let mut weighted_position = Vec3::ZERO;
    let mut weighted_normal = Vec3::ZERO;
    let mut weight_sum = 0.0;
    for &vertex in &pass.vertices {
        let index = vertex as usize;
        let Some(&position) = source_positions.get(&vertex) else {
            continue;
        };
        let weight = brush_falloff(
            position.distance(pass.sample.center) / settings.radius,
            settings.falloff,
        );
        if weight <= f32::EPSILON {
            continue;
        }
        let mut normal = mesh
            .normals
            .get(index)
            .copied()
            .unwrap_or(brush_normal)
            .normalize_or_zero();
        if normal.dot(brush_normal) < 0.0 {
            normal = -normal;
        }
        weighted_position += position * weight;
        weighted_normal += normal * weight;
        weight_sum += weight;
    }
    if weight_sum <= f32::EPSILON {
        return None;
    }
    let plane_normal = weighted_normal.normalize_or_zero();
    if plane_normal == Vec3::ZERO {
        return None;
    }
    let plane_point =
        weighted_position / weight_sum + plane_normal * (direction * settings.radius * 0.15);
    Some((plane_point, plane_normal))
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
            invert: false,
            symmetry: None,
            accumulation: Accumulation::Capped,
        }
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
        let brush_sample = sample(Vec3::ZERO, 0);
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(!engine.apply_sample(
            &mut mesh,
            SculptTool::Pinch,
            &settings,
            brush_sample,
            DabKind::Spatial,
        ));
        settings.strength = 0.1;
        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Pinch,
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
    fn ineffective_pressure_cannot_deform() {
        for tool in [SculptTool::Clay, SculptTool::Crease] {
            for pressure in [0.0, -1.0, f32::NAN, f32::INFINITY] {
                let mut mesh = octahedron();
                let before = mesh.clone();
                let mut settings = test_settings();
                settings.radius = 1.2;
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
                assert!(!engine.take_sample_committed());
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
            DabKind::Spatial,
        ));

        assert_eq!(mesh.positions, before_positions);
        assert_eq!(mesh.normals, before_normals);
        assert!(engine.take_updated_vertices().is_empty());
        assert!(engine.take_error().is_some());
        assert!(engine.end_stroke(&mesh).edit.is_empty());
    }

    #[test]
    #[ignore = "release-mode performance envelope"]
    fn million_face_fixed_sculpt_sample() {
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
        let settings = BrushSettings {
            radius: 10.0,
            strength: 0.1,
            falloff: 0.15,
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

        assert!(
            fixed_elapsed < std::time::Duration::from_millis(8),
            "million-face fixed sample exceeded one frame: {fixed_elapsed:?}"
        );
        eprintln!("million-face fixed sculpt sample: {fixed_elapsed:?}");
    }
}
