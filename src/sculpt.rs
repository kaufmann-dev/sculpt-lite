use std::{collections::HashMap, fmt};

use glam::Vec3;

use crate::mesh::{Mesh, RayHit, RemeshSettings};

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
    /// Remeshing target edge length as a fraction of brush radius. Set to zero
    /// to disable remeshing for a stroke.
    pub detail: f32,
    pub invert: bool,
    pub symmetry: Option<SymmetryAxis>,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 1.0,
            strength: 0.35,
            falloff: 0.15,
            detail: 0.12,
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
    affected_vertices: Vec<u32>,
    remesh_target: Option<f32>,
    changed: bool,
}

/// Topology work collected during a stroke and safe to execute after the
/// interactive pointer gesture has finished.
#[derive(Clone, Debug, PartialEq)]
pub struct RemeshRequest {
    pub affected_vertices: Vec<u32>,
    pub settings: RemeshSettings,
}

/// The editable result of a completed stroke.
///
/// Remeshing is deliberately returned as work instead of being applied here so
/// callers can move a large mesh to a worker thread without cloning it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrokeOutcome {
    pub changed: bool,
    pub remesh: Option<RemeshRequest>,
}

#[derive(Default, Debug)]
pub struct SculptEngine {
    active: Option<StrokeState>,
    symmetry_center: Option<Vec3>,
    updated_vertices: Vec<u32>,
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
        self.updated_vertices.clear();
    }

    pub fn begin_stroke(&mut self, mesh: &Mesh) {
        let symmetry_center = *self
            .symmetry_center
            .get_or_insert_with(|| mesh.center().unwrap_or(Vec3::ZERO));
        self.active = Some(StrokeState {
            symmetry_center,
            mirrored_seed: None,
            affected_vertices: Vec::new(),
            remesh_target: None,
            changed: false,
        });
        self.updated_vertices.clear();
    }

    /// Takes the deduplicated vertex IDs changed by the latest brush sample.
    ///
    /// This includes mask-only edits and is cleared before every sample, so an
    /// empty result always means the latest sample made no editable change.
    #[must_use]
    pub fn take_updated_vertices(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.updated_vertices)
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
        let mut passes = Vec::with_capacity(2);
        passes.push(PreparedPass::new(mesh, sample, settings.radius));

        if let Some(axis) = settings.symmetry
            && axis.distance_to_plane(sample.center, symmetry_center) > settings.radius * 1.0e-4
        {
            let mirrored_center = axis.reflect_point(sample.center, symmetry_center);
            let seed_triangle = cached_mirrored_seed
                .filter(|&seed| triangle_is_near(mesh, seed, mirrored_center, settings.radius))
                .or_else(|| mesh.nearest_triangle(mirrored_center));
            if let Some(seed_triangle) = seed_triangle {
                let mirrored = sample.reflected(axis, symmetry_center, seed_triangle);
                passes.push(PreparedPass::new(mesh, mirrored, settings.radius));
                if let Some(stroke) = self.active.as_mut() {
                    stroke.mirrored_seed = Some(seed_triangle);
                }
            }
        }

        let source_positions = capture_source_positions(mesh, tool, &passes);
        let mut changed = false;
        let mut affected = Vec::new();
        for pass in &passes {
            changed |= apply_pass(mesh, tool, settings, pass, &source_positions, &mut affected);
        }

        if changed {
            affected.sort_unstable();
            affected.dedup();

            if tool != SculptTool::Mask {
                self.updated_vertices = mesh.update_deformed_vertices(&affected);
                self.updated_vertices.extend(affected.iter().copied());
                self.updated_vertices.sort_unstable();
                self.updated_vertices.dedup();

                if settings.detail.is_finite()
                    && settings.detail > 0.0
                    && let Some(stroke) = self.active.as_mut()
                {
                    let target_edge_length = settings.radius * settings.detail.clamp(0.01, 1.0);
                    stroke.affected_vertices.extend(affected);
                    stroke.remesh_target =
                        Some(stroke.remesh_target.map_or(target_edge_length, |current| {
                            current.min(target_edge_length)
                        }));
                }
            } else {
                self.updated_vertices.clone_from(&affected);
            }
        }

        if changed && let Some(stroke) = self.active.as_mut() {
            stroke.changed = true;
        }
        changed
    }

    /// Ends the active stroke and returns any deferred topology work.
    #[must_use]
    pub fn end_stroke(&mut self) -> StrokeOutcome {
        let Some(mut stroke) = self.active.take() else {
            return StrokeOutcome::default();
        };
        if !stroke.changed {
            return StrokeOutcome::default();
        }

        let remesh = stroke.remesh_target.map(|target_edge_length| {
            stroke.affected_vertices.sort_unstable();
            stroke.affected_vertices.dedup();
            let mut remesh = RemeshSettings::new(target_edge_length);
            remesh.iterations = 1;
            remesh.relaxation = 0.0;
            RemeshRequest {
                affected_vertices: stroke.affected_vertices,
                settings: remesh,
            }
        });
        StrokeOutcome {
            changed: true,
            remesh,
        }
    }
}

#[derive(Debug)]
struct PreparedPass {
    sample: BrushSample,
    vertices: Vec<u32>,
}

impl PreparedPass {
    fn new(mesh: &Mesh, sample: BrushSample, radius: f32) -> Self {
        Self {
            vertices: mesh.connected_front_facing_vertices(
                sample.seed_triangle,
                sample.center,
                radius,
                sample.view_direction,
            ),
            sample,
        }
    }
}

fn apply_pass(
    mesh: &mut Mesh,
    tool: SculptTool,
    settings: &BrushSettings,
    pass: &PreparedPass,
    source_positions: &HashMap<u32, Vec3>,
    affected: &mut Vec<u32>,
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
            let Some(mask) = mesh.mask.get_mut(index) else {
                continue;
            };
            let next = (*mask + direction * strength * weight).clamp(0.0, 1.0);
            if next != *mask {
                *mask = next;
                affected.push(vertex);
                changed = true;
            }
            continue;
        }

        let unmasked = 1.0 - mesh.mask.get(index).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let influence = strength * weight * unmasked;
        if influence <= f32::EPSILON {
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
        let displacement = match tool {
            SculptTool::Grab => pass.sample.drag_delta * influence,
            SculptTool::Draw => brush_normal * (direction * radius * 0.15 * influence),
            SculptTool::Inflate => normal * (direction * radius * 0.15 * influence),
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
        if let Some(output) = mesh.positions.get_mut(index) {
            let next = position + displacement;
            if next != *output {
                *output = next;
                affected.push(vertex);
                changed = true;
            }
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
) -> HashMap<u32, Vec3> {
    let selected_count = passes.iter().map(|pass| pass.vertices.len()).sum();
    let mut source = HashMap::with_capacity(selected_count);
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
    source
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
            detail: 0.0,
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
        let outcome = engine.end_stroke();
        assert!(outcome.changed);
        assert!(outcome.remesh.is_none());
    }

    #[test]
    fn end_stroke_returns_remeshing_without_applying_it() {
        let mut mesh = grid();
        let original_triangles = mesh.triangles.clone();
        let mut settings = test_settings();
        settings.radius = 0.8;
        settings.detail = 0.2;
        let mut engine = SculptEngine::default();
        engine.begin_stroke(&mesh);

        assert!(engine.apply_sample(
            &mut mesh,
            SculptTool::Draw,
            &settings,
            sample(Vec3::ZERO, 0),
        ));
        let updated = engine.take_updated_vertices();
        let outcome = engine.end_stroke();

        assert!(outcome.changed);
        let remesh = outcome.remesh.expect("geometry stroke requests remeshing");
        assert!(!updated.is_empty());
        assert!(
            remesh
                .affected_vertices
                .iter()
                .all(|vertex| updated.contains(vertex))
        );
        assert!((remesh.settings.target_edge_length - 0.16).abs() < 1.0e-6);
        assert_eq!(mesh.triangles, original_triangles);
    }
}
