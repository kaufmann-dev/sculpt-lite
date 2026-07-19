use std::collections::VecDeque;

use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use thiserror::Error;

pub type EdgeKey = (u32, u32);

const TRIANGLE_RELATIVE_EPSILON: f32 = 1.0e-12;
const BVH_LEAF_SIZE: usize = 8;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum MeshError {
    #[error("vertex {index} has non-finite coordinates")]
    NonFiniteVertex { index: usize },
    #[error(
        "triangle {triangle} references vertex {index}, but the mesh has only {vertex_count} vertices"
    )]
    IndexOutOfBounds {
        triangle: usize,
        index: u32,
        vertex_count: usize,
    },
    #[error("the mesh has too many vertices for 32-bit indices")]
    TooManyVertices,
    #[error("vertex normal count {actual} does not match vertex count {expected}")]
    NormalCountMismatch { actual: usize, expected: usize },
    #[error("mask count {actual} does not match vertex count {expected}")]
    MaskCountMismatch { actual: usize, expected: usize },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CleanupReport {
    pub input_vertices: usize,
    pub input_triangles: usize,
    pub welded_vertices: usize,
    pub removed_invalid_faces: usize,
    pub removed_degenerate_faces: usize,
    pub removed_duplicate_faces: usize,
    pub flipped_faces: usize,
    pub output_vertices: usize,
    pub output_triangles: usize,
    pub boundary_edges: usize,
    pub boundary_vertices: usize,
    pub non_manifold_edges: usize,
    pub non_manifold_vertices: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub position: Vec3,
    pub normal: Vec3,
    pub triangle: u32,
    pub distance: f32,
    pub barycentric: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RemeshSettings {
    pub target_edge_length: f32,
    pub iterations: u32,
    pub split_threshold: f32,
    pub collapse_threshold: f32,
    pub enable_flips: bool,
    pub relaxation: f32,
}

impl RemeshSettings {
    pub fn new(target_edge_length: f32) -> Self {
        Self {
            target_edge_length,
            ..Self::default()
        }
    }
}

impl Default for RemeshSettings {
    fn default() -> Self {
        Self {
            target_edge_length: 1.0,
            iterations: 1,
            split_threshold: 4.0 / 3.0,
            collapse_threshold: 0.7,
            enable_flips: true,
            relaxation: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RemeshStats {
    pub splits: usize,
    pub collapses: usize,
    pub flips: usize,
    pub relaxed_vertices: usize,
    pub iterations: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MeshBvh {
    nodes: Vec<BvhNode>,
    triangle_indices: Vec<u32>,
    triangle_leaves: Vec<u32>,
    parents: Vec<u32>,
    dirty_marks: Vec<u32>,
    dirty_generation: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

impl BvhNode {
    fn is_leaf(self) -> bool {
        self.count != 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshTopology {
    pub vertex_neighbors: Vec<SmallVec<[u32; 8]>>,
    pub vertex_triangles: Vec<SmallVec<[u32; 8]>>,
    pub edge_faces: HashMap<EdgeKey, SmallVec<[u32; 2]>>,
    pub boundary_vertices: Vec<bool>,
    pub non_manifold_vertices: Vec<bool>,
    pub bvh: MeshBvh,
}

impl MeshTopology {
    pub fn boundary_edge_count(&self) -> usize {
        self.edge_faces
            .values()
            .filter(|faces| faces.len() == 1)
            .count()
    }

    pub fn non_manifold_edge_count(&self) -> usize {
        self.edge_faces
            .values()
            .filter(|faces| faces.len() > 2)
            .count()
    }

    pub fn boundary_vertex_count(&self) -> usize {
        self.boundary_vertices
            .iter()
            .filter(|&&value| value)
            .count()
    }

    pub fn non_manifold_vertex_count(&self) -> usize {
        self.non_manifold_vertices
            .iter()
            .filter(|&&value| value)
            .count()
    }

    fn protected_neighborhood(&self) -> Vec<bool> {
        let mut protected = self
            .boundary_vertices
            .iter()
            .zip(&self.non_manifold_vertices)
            .map(|(&boundary, &non_manifold)| boundary || non_manifold)
            .collect::<Vec<_>>();
        let seeds = protected.clone();
        for (vertex, &is_seed) in seeds.iter().enumerate() {
            if !is_seed {
                continue;
            }
            for &neighbor in &self.vertex_neighbors[vertex] {
                protected[neighbor as usize] = true;
            }
        }
        protected
    }

    fn vertex_in_protected_neighborhood(&self, vertex: u32) -> bool {
        let index = vertex as usize;
        self.boundary_vertices.get(index).copied().unwrap_or(true)
            || self
                .non_manifold_vertices
                .get(index)
                .copied()
                .unwrap_or(true)
            || self
                .vertex_neighbors
                .get(index)
                .into_iter()
                .flatten()
                .any(|&neighbor| {
                    self.boundary_vertices[neighbor as usize]
                        || self.non_manifold_vertices[neighbor as usize]
                })
    }

    fn active_edges(&self, active: &[bool]) -> Vec<EdgeKey> {
        let mut edges = Vec::new();
        for (vertex, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            let Ok(a) = u32::try_from(vertex) else {
                continue;
            };
            for &b in &self.vertex_neighbors[vertex] {
                if a < b && active.get(b as usize).copied().unwrap_or(false) {
                    edges.push((a, b));
                }
            }
        }
        edges
    }

    fn build(positions: &[Vec3], triangles: &[[u32; 3]]) -> Self {
        let mut topology = Self {
            vertex_neighbors: vec![SmallVec::new(); positions.len()],
            vertex_triangles: vec![SmallVec::new(); positions.len()],
            edge_faces: HashMap::with_capacity(triangles.len().saturating_mul(3) / 2),
            boundary_vertices: vec![false; positions.len()],
            non_manifold_vertices: vec![false; positions.len()],
            bvh: MeshBvh::default(),
        };

        for (face_index, triangle) in triangles.iter().enumerate() {
            for &vertex in triangle {
                topology.vertex_triangles[vertex as usize].push(face_index as u32);
            }
            for edge_index in 0..3 {
                let a = triangle[edge_index];
                let b = triangle[(edge_index + 1) % 3];
                topology.vertex_neighbors[a as usize].push(b);
                topology.vertex_neighbors[b as usize].push(a);
                topology
                    .edge_faces
                    .entry(edge_key(a, b))
                    .or_default()
                    .push(face_index as u32);
            }
        }

        for neighbors in &mut topology.vertex_neighbors {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        for (&(a, b), faces) in &topology.edge_faces {
            match faces.len() {
                1 => {
                    topology.boundary_vertices[a as usize] = true;
                    topology.boundary_vertices[b as usize] = true;
                }
                2 => {}
                _ => {
                    topology.non_manifold_vertices[a as usize] = true;
                    topology.non_manifold_vertices[b as usize] = true;
                }
            }
        }

        // Detect bow-tie vertices and invalid boundary fans, not just edges with >2 faces. Reusing
        // one face-mark array avoids allocating hash maps for every vertex on large meshes.
        let mut face_marks = vec![0_u32; triangles.len()];
        let mut mark = 0_u32;
        let mut face_queue = Vec::new();
        for vertex in 0..positions.len() {
            let incident = &topology.vertex_triangles[vertex];
            if incident.is_empty() {
                continue;
            }
            let mut boundary_degree = 0;
            for &neighbor in &topology.vertex_neighbors[vertex] {
                let key = edge_key(vertex as u32, neighbor);
                let faces = &topology.edge_faces[&key];
                if faces.len() == 1 {
                    boundary_degree += 1;
                }
            }
            if boundary_degree != 0 && boundary_degree != 2 {
                topology.non_manifold_vertices[vertex] = true;
            }

            mark = mark.wrapping_add(1);
            if mark == 0 {
                face_marks.fill(0);
                mark = 1;
            }
            face_queue.clear();
            face_queue.push(incident[0]);
            face_marks[incident[0] as usize] = mark;
            let mut visited_count = 0;
            while let Some(face) = face_queue.pop() {
                visited_count += 1;
                let triangle = triangles[face as usize];
                for edge_index in 0..3 {
                    let a = triangle[edge_index];
                    let b = triangle[(edge_index + 1) % 3];
                    if a != vertex as u32 && b != vertex as u32 {
                        continue;
                    }
                    let adjacent_faces = &topology.edge_faces[&edge_key(a, b)];
                    if adjacent_faces.len() == 2 {
                        for &neighbor_face in adjacent_faces {
                            if face_marks[neighbor_face as usize] != mark {
                                face_marks[neighbor_face as usize] = mark;
                                face_queue.push(neighbor_face);
                            }
                        }
                    }
                }
            }
            if visited_count != incident.len() {
                topology.non_manifold_vertices[vertex] = true;
            }
        }

        topology.bvh = MeshBvh::build(positions, triangles);
        topology
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub normals: Vec<Vec3>,
    pub mask: Vec<f32>,
    pub topology: MeshTopology,
}

#[derive(Default)]
pub(crate) struct TriangleSoupBuilder {
    positions: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    vertices: HashMap<[u32; 3], u32>,
    input_vertices: usize,
    input_triangles: usize,
}

impl TriangleSoupBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_triangle_capacity(triangle_count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(triangle_count.saturating_mul(3)),
            triangles: Vec::with_capacity(triangle_count),
            vertices: HashMap::new(),
            input_vertices: 0,
            input_triangles: 0,
        }
    }

    pub(crate) fn push_triangle(&mut self, triangle: [Vec3; 3]) -> Result<(), MeshError> {
        if self.input_vertices > u32::MAX as usize - 3 {
            return Err(MeshError::TooManyVertices);
        }

        let mut indices = [0_u32; 3];
        for (corner, position) in triangle.into_iter().enumerate() {
            if !position.is_finite() {
                return Err(MeshError::NonFiniteVertex {
                    index: self.input_vertices + corner,
                });
            }
            let bits = [
                position.x.to_bits(),
                position.y.to_bits(),
                position.z.to_bits(),
            ];
            indices[corner] = if let Some(&index) = self.vertices.get(&bits) {
                index
            } else {
                let index =
                    u32::try_from(self.positions.len()).map_err(|_| MeshError::TooManyVertices)?;
                self.vertices.insert(bits, index);
                self.positions.push(position);
                index
            };
        }
        self.triangles.push(indices);
        self.input_triangles += 1;
        self.input_vertices += 3;
        Ok(())
    }

    pub(crate) fn finish(self) -> (Mesh, CleanupReport) {
        let unique_vertices = self.positions.len();
        let mut mesh = Mesh {
            positions: self.positions,
            triangles: self.triangles,
            normals: vec![Vec3::ZERO; unique_vertices],
            mask: vec![0.0; unique_vertices],
            topology: MeshTopology::default(),
        };
        let mut report = mesh.rebuild();
        report.input_vertices = self.input_vertices;
        report.input_triangles = self.input_triangles;
        report.welded_vertices = report.input_vertices.saturating_sub(unique_vertices);
        (mesh, report)
    }
}

impl Mesh {
    #[cfg(test)]
    pub fn new(positions: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Result<Self, MeshError> {
        validate_positions_and_indices(&positions, &triangles)?;
        let vertex_count = positions.len();
        let mut mesh = Self {
            positions,
            triangles,
            normals: vec![Vec3::ZERO; vertex_count],
            mask: vec![0.0; vertex_count],
            topology: MeshTopology::default(),
        };
        mesh.rebuild();
        Ok(mesh)
    }

    #[cfg(test)]
    pub fn from_triangle_soup(soup: &[[Vec3; 3]]) -> Result<(Self, CleanupReport), MeshError> {
        if soup.len().saturating_mul(3) > u32::MAX as usize {
            return Err(MeshError::TooManyVertices);
        }
        let mut builder = TriangleSoupBuilder::with_triangle_capacity(soup.len());
        for &triangle in soup {
            builder.push_triangle(triangle)?;
        }
        Ok(builder.finish())
    }

    pub fn validate(&self) -> Result<(), MeshError> {
        validate_positions_and_indices(&self.positions, &self.triangles)?;
        if self.normals.len() != self.positions.len() {
            return Err(MeshError::NormalCountMismatch {
                actual: self.normals.len(),
                expected: self.positions.len(),
            });
        }
        if self.mask.len() != self.positions.len() {
            return Err(MeshError::MaskCountMismatch {
                actual: self.mask.len(),
                expected: self.positions.len(),
            });
        }
        Ok(())
    }

    /// Restores all derived data and removes unusable faces.
    ///
    /// Faces with invalid indices, non-finite coordinates, repeated vertices, negligible area,
    /// or duplicate vertex sets are removed. Public editing code may therefore mutate the geometry
    /// arrays directly and call `rebuild` before rendering or querying the mesh.
    pub fn rebuild(&mut self) -> CleanupReport {
        let mut report = CleanupReport {
            input_vertices: self.positions.len(),
            input_triangles: self.triangles.len(),
            ..CleanupReport::default()
        };
        let vertex_count = self.positions.len();
        let mut unique_faces = HashSet::with_capacity(self.triangles.len());
        self.triangles.retain(|triangle| {
            if triangle.iter().any(|&index| index as usize >= vertex_count)
                || triangle
                    .iter()
                    .any(|&index| !self.positions[index as usize].is_finite())
            {
                report.removed_invalid_faces += 1;
                return false;
            }
            if !triangle_is_valid(&self.positions, *triangle) {
                report.removed_degenerate_faces += 1;
                return false;
            }
            let key = sorted_triangle(*triangle);
            if !unique_faces.insert(key) {
                report.removed_duplicate_faces += 1;
                return false;
            }
            true
        });

        self.mask.resize(self.positions.len(), 0.0);
        for weight in &mut self.mask {
            *weight = if weight.is_finite() {
                weight.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        self.topology = MeshTopology::build(&self.positions, &self.triangles);
        report.flipped_faces =
            orient_manifold_faces(&mut self.triangles, &self.topology.edge_faces);
        self.recompute_normals();

        report.output_vertices = self.positions.len();
        report.output_triangles = self.triangles.len();
        report.boundary_edges = self.topology.boundary_edge_count();
        report.boundary_vertices = self.topology.boundary_vertex_count();
        report.non_manifold_edges = self.topology.non_manifold_edge_count();
        report.non_manifold_vertices = self.topology.non_manifold_vertex_count();
        report
    }

    pub fn recompute_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), Vec3::ZERO);
        for &triangle in &self.triangles {
            let [a, b, c] = triangle.map(|index| self.positions[index as usize]);
            let weighted_normal = (b - a).cross(c - a);
            if !weighted_normal.is_finite() {
                continue;
            }
            for vertex in triangle {
                self.normals[vertex as usize] += weighted_normal;
            }
        }
        for normal in &mut self.normals {
            *normal = normal.try_normalize().unwrap_or(Vec3::ZERO);
        }
        self.topology.bvh.refit(&self.positions, &self.triangles);
    }

    /// Recomputes only normals and BVH branches touched by moved vertices.
    ///
    /// Returns the complete set of vertices whose render normals changed, so
    /// the renderer can issue compact partial vertex-buffer writes.
    pub fn update_deformed_vertices(&mut self, moved_vertices: &[u32]) -> Vec<u32> {
        let mut affected_faces = moved_vertices
            .iter()
            .filter_map(|&vertex| self.topology.vertex_triangles.get(vertex as usize))
            .flat_map(|faces| faces.iter().copied())
            .collect::<Vec<_>>();
        affected_faces.sort_unstable();
        affected_faces.dedup();
        if affected_faces.is_empty() {
            return Vec::new();
        }

        let mut normal_vertices = affected_faces
            .iter()
            .filter_map(|&face| self.triangles.get(face as usize))
            .flat_map(|triangle| triangle.iter().copied())
            .collect::<Vec<_>>();
        normal_vertices.sort_unstable();
        normal_vertices.dedup();

        for &vertex in &normal_vertices {
            let Some(normal) = self.normals.get_mut(vertex as usize) else {
                continue;
            };
            *normal = self.topology.vertex_triangles[vertex as usize]
                .iter()
                .filter_map(|&face| self.triangles.get(face as usize))
                .map(|&triangle| triangle_cross(&self.positions, triangle))
                .sum::<Vec3>()
                .try_normalize()
                .unwrap_or(Vec3::ZERO);
        }
        self.topology
            .bvh
            .refit_triangles(&self.positions, &self.triangles, &affected_faces);
        normal_vertices
    }

    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut positions = self
            .positions
            .iter()
            .copied()
            .filter(|position| position.is_finite());
        let first = positions.next()?;
        let mut min = first;
        let mut max = first;
        for position in positions {
            min = min.min(position);
            max = max.max(position);
        }
        Some((min, max))
    }

    pub fn center(&self) -> Option<Vec3> {
        self.bounds().map(|(min, max)| min.midpoint(max))
    }

    #[cfg(test)]
    pub fn diagonal(&self) -> f32 {
        self.bounds()
            .map(|(min, max)| min.distance(max))
            .unwrap_or(0.0)
    }

    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite() {
            return None;
        }
        self.topology.bvh.raycast(
            &self.positions,
            &self.triangles,
            origin,
            direction,
            &self.normals,
        )
    }

    pub fn nearest_triangle(&self, point: Vec3) -> Option<u32> {
        if !point.is_finite() {
            return None;
        }
        self.topology
            .bvh
            .nearest_triangle(&self.positions, &self.triangles, point)
    }

    pub fn connected_front_facing_vertices(
        &self,
        seed_triangle: u32,
        center: Vec3,
        radius: f32,
        view_direction: Vec3,
    ) -> Vec<u32> {
        let Some(seed) = self.triangles.get(seed_triangle as usize) else {
            return Vec::new();
        };
        if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return Vec::new();
        }
        let radius_squared = radius * radius;
        let view_direction = view_direction.try_normalize();
        let seed_faces_toward_view = view_direction.is_some_and(|view| {
            let seed_normal = seed
                .iter()
                .filter_map(|&vertex| self.normals.get(vertex as usize))
                .copied()
                .sum::<Vec3>();
            seed_normal.dot(view) > 0.0
        });
        let mut visited = vec![false; self.positions.len()];
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        for &vertex in seed {
            if !visited[vertex as usize] {
                queue.push_back(vertex);
            }
        }
        while let Some(vertex) = queue.pop_front() {
            let index = vertex as usize;
            if visited[index] {
                continue;
            }
            visited[index] = true;
            let position = self.positions[index];
            if position.distance_squared(center) > radius_squared {
                continue;
            }
            let front_facing = view_direction.is_none_or(|view| {
                let normal = self.normals[index];
                normal == Vec3::ZERO
                    || if seed_faces_toward_view {
                        normal.dot(view) >= 0.0
                    } else {
                        normal.dot(view) <= 0.0
                    }
            });
            if !front_facing {
                continue;
            }
            result.push(vertex);
            for &neighbor in &self.topology.vertex_neighbors[index] {
                if !visited[neighbor as usize] {
                    queue.push_back(neighbor);
                }
            }
        }
        result
    }

    pub fn remesh_region(&mut self, vertices: &[u32], settings: RemeshSettings) -> RemeshStats {
        if !settings.target_edge_length.is_finite()
            || settings.target_edge_length <= 0.0
            || settings.iterations == 0
        {
            return RemeshStats::default();
        }
        if self.topology.vertex_neighbors.len() != self.positions.len() {
            self.rebuild();
        }
        let mut active = vec![false; self.positions.len()];
        for &vertex in vertices {
            if let Some(value) = active.get_mut(vertex as usize) {
                *value = true;
            }
        }
        if !active.iter().any(|&value| value) {
            return RemeshStats::default();
        }

        let split_length = settings.target_edge_length * settings.split_threshold.max(1.01);
        let collapse_length = settings.target_edge_length
            * settings
                .collapse_threshold
                .clamp(0.01, settings.split_threshold.min(0.99));
        let mut stats = RemeshStats::default();

        for _ in 0..settings.iterations {
            stats.iterations += 1;
            stats.splits += self.split_active_edges_batch(&mut active, split_length);
            stats.collapses += self.collapse_active_edges_batch(&mut active, collapse_length);

            if settings.enable_flips {
                stats.flips += self.flip_active_edges_batch(&active);
            }

            let relaxation = settings.relaxation.clamp(0.0, 1.0);
            if relaxation > 0.0 {
                stats.relaxed_vertices += self.relax_active_vertices(&active, relaxation);
                self.recompute_normals();
            }
        }

        stats
    }

    /// Splits a maximal set of long edges whose incident faces do not overlap,
    /// then rebuilds derived topology once for the whole set.
    fn split_active_edges_batch(&mut self, active: &mut Vec<bool>, threshold: f32) -> usize {
        let mut candidates = self
            .topology
            .active_edges(active)
            .into_iter()
            .filter_map(|edge @ (a, b)| {
                let faces = self.topology.edge_faces.get(&edge)?;
                let ai = a as usize;
                let bi = b as usize;
                if faces.len() != 2
                    || self.topology.vertex_in_protected_neighborhood(a)
                    || self.topology.vertex_in_protected_neighborhood(b)
                {
                    return None;
                }
                let length = self.positions[ai].distance(self.positions[bi]);
                (length > threshold).then_some((edge, [faces[0], faces[1]], length))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            right
                .2
                .total_cmp(&left.2)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut used_faces = HashSet::new();
        let mut selected = Vec::new();
        for (edge, faces, _) in candidates {
            if faces.iter().any(|face| used_faces.contains(face)) {
                continue;
            }
            for &face in &faces {
                used_faces.insert(face);
            }
            selected.push((edge, faces));
        }

        let mut split_count = 0;
        for ((a, b), faces) in selected {
            let Ok(new_vertex) = u32::try_from(self.positions.len()) else {
                break;
            };
            let first_triangle = self.triangles[faces[0] as usize];
            let second_triangle = self.triangles[faces[1] as usize];
            let Some((first_a, first_b)) = split_triangle(first_triangle, a, b, new_vertex) else {
                continue;
            };
            let Some((second_a, second_b)) = split_triangle(second_triangle, a, b, new_vertex)
            else {
                continue;
            };

            self.positions
                .push(self.positions[a as usize].midpoint(self.positions[b as usize]));
            self.mask
                .push((self.mask[a as usize] + self.mask[b as usize]) * 0.5);
            self.normals.push(
                (self.normals[a as usize] + self.normals[b as usize])
                    .try_normalize()
                    .unwrap_or(Vec3::ZERO),
            );
            active.push(true);
            self.triangles[faces[0] as usize] = first_a;
            self.triangles[faces[1] as usize] = second_a;
            self.triangles.push(first_b);
            self.triangles.push(second_b);
            split_count += 1;
        }
        if split_count != 0 {
            self.rebuild();
        }
        split_count
    }

    /// Collapses a maximal set of one-ring-disjoint short edges in one compacting pass.
    fn collapse_active_edges_batch(&mut self, active: &mut Vec<bool>, threshold: f32) -> usize {
        let mut candidates = self
            .topology
            .active_edges(active)
            .into_iter()
            .filter_map(|(a, b)| {
                let faces = self.topology.edge_faces.get(&(a, b))?;
                let ai = a as usize;
                let bi = b as usize;
                if faces.len() != 2
                    || self.topology.vertex_in_protected_neighborhood(a)
                    || self.topology.vertex_in_protected_neighborhood(b)
                {
                    return None;
                }
                let length = self.positions[ai].distance(self.positions[bi]);
                (length < threshold).then_some(((a, b), length))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut blocked = vec![false; self.positions.len()];
        let mut selected = Vec::new();
        for ((keep, remove), _) in candidates {
            if blocked[keep as usize]
                || blocked[remove as usize]
                || !self.collapse_candidate_is_safe(keep, remove)
            {
                continue;
            }
            selected.push((keep, remove));
            blocked[keep as usize] = true;
            blocked[remove as usize] = true;
            for &neighbor in self.topology.vertex_neighbors[keep as usize]
                .iter()
                .chain(&self.topology.vertex_neighbors[remove as usize])
            {
                blocked[neighbor as usize] = true;
            }
        }
        if selected.is_empty() {
            return 0;
        }

        let mut removed_to_keep = vec![None; self.positions.len()];
        let mut merged_positions = HashMap::with_capacity(selected.len());
        let mut merged_masks = HashMap::with_capacity(selected.len());
        let mut merged_active = HashMap::with_capacity(selected.len());
        for &(keep, remove) in &selected {
            removed_to_keep[remove as usize] = Some(keep);
            merged_positions.insert(
                keep,
                self.positions[keep as usize].midpoint(self.positions[remove as usize]),
            );
            merged_masks.insert(
                keep,
                (self.mask[keep as usize] + self.mask[remove as usize]) * 0.5,
            );
            merged_active.insert(keep, active[keep as usize] || active[remove as usize]);
        }

        let mut old_to_new = vec![u32::MAX; self.positions.len()];
        let mut new_positions = Vec::with_capacity(self.positions.len() - selected.len());
        let mut new_masks = Vec::with_capacity(self.mask.len() - selected.len());
        let mut new_active = Vec::with_capacity(active.len() - selected.len());
        for vertex in 0..self.positions.len() {
            if removed_to_keep[vertex].is_some() {
                continue;
            }
            let old = vertex as u32;
            old_to_new[vertex] = new_positions.len() as u32;
            new_positions.push(
                merged_positions
                    .get(&old)
                    .copied()
                    .unwrap_or(self.positions[vertex]),
            );
            new_masks.push(merged_masks.get(&old).copied().unwrap_or(self.mask[vertex]));
            new_active.push(merged_active.get(&old).copied().unwrap_or(active[vertex]));
        }
        for (removed, keep) in removed_to_keep.iter().enumerate() {
            if let Some(keep) = keep {
                old_to_new[removed] = old_to_new[*keep as usize];
            }
        }

        let mut new_triangles = Vec::with_capacity(self.triangles.len());
        let mut unique = HashSet::with_capacity(self.triangles.len());
        for triangle in &self.triangles {
            let replacement = triangle.map(|vertex| old_to_new[vertex as usize]);
            if replacement[0] == replacement[1]
                || replacement[1] == replacement[2]
                || replacement[2] == replacement[0]
                || !triangle_is_valid(&new_positions, replacement)
                || !unique.insert(sorted_triangle(replacement))
            {
                continue;
            }
            new_triangles.push(replacement);
        }

        self.positions = new_positions;
        self.mask = new_masks;
        self.normals = vec![Vec3::ZERO; self.positions.len()];
        self.triangles = new_triangles;
        *active = new_active;
        self.rebuild();
        selected.len()
    }

    fn collapse_candidate_is_safe(&self, keep: u32, remove: u32) -> bool {
        let edge = edge_key(keep, remove);
        let Some(faces) = self.topology.edge_faces.get(&edge) else {
            return false;
        };
        if faces.len() != 2 {
            return false;
        }
        let keep_neighbors = self.topology.vertex_neighbors[keep as usize]
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let remove_neighbors = self.topology.vertex_neighbors[remove as usize]
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let common = keep_neighbors
            .intersection(&remove_neighbors)
            .copied()
            .collect::<HashSet<_>>();
        let opposites = faces
            .iter()
            .filter_map(|&face| opposite_vertex(self.triangles[face as usize], keep, remove))
            .collect::<HashSet<_>>();
        if common != opposites || opposites.len() != 2 {
            return false;
        }

        let merged_position =
            self.positions[keep as usize].midpoint(self.positions[remove as usize]);
        let mut incident = self.topology.vertex_triangles[keep as usize].clone();
        incident.extend(
            self.topology.vertex_triangles[remove as usize]
                .iter()
                .copied(),
        );
        incident.sort_unstable();
        incident.dedup();
        for face in incident {
            let triangle = self.triangles[face as usize];
            let old_positions = triangle.map(|index| self.positions[index as usize]);
            let mut replacement = triangle;
            for vertex in &mut replacement {
                if *vertex == remove {
                    *vertex = keep;
                }
            }
            if replacement[0] == replacement[1]
                || replacement[1] == replacement[2]
                || replacement[2] == replacement[0]
            {
                continue;
            }
            let new_positions = replacement.map(|index| {
                if index == keep {
                    merged_position
                } else {
                    self.positions[index as usize]
                }
            });
            if !positions_form_valid_triangle(new_positions) {
                return false;
            }
            let old_normal =
                (old_positions[1] - old_positions[0]).cross(old_positions[2] - old_positions[0]);
            let new_normal =
                (new_positions[1] - new_positions[0]).cross(new_positions[2] - new_positions[0]);
            if old_normal.dot(new_normal) <= 0.0 {
                return false;
            }
        }
        true
    }

    /// Flips a maximal set of face-disjoint edges, followed by one topology rebuild.
    fn flip_active_edges_batch(&mut self, active: &[bool]) -> usize {
        let mut candidates = self
            .topology
            .active_edges(active)
            .into_iter()
            .filter_map(|(a, b)| {
                let faces = self.topology.edge_faces.get(&(a, b))?;
                if faces.len() != 2
                    || self.topology.vertex_in_protected_neighborhood(a)
                    || self.topology.vertex_in_protected_neighborhood(b)
                {
                    return None;
                }
                let first = self.triangles[faces[0] as usize];
                let second = self.triangles[faces[1] as usize];
                let c = opposite_vertex(first, a, b)?;
                let d = opposite_vertex(second, a, b)?;
                if c == d
                    || !active.get(c as usize).copied().unwrap_or(false)
                    || !active.get(d as usize).copied().unwrap_or(false)
                    || self.topology.vertex_in_protected_neighborhood(c)
                    || self.topology.vertex_in_protected_neighborhood(d)
                    || self.topology.edge_faces.contains_key(&edge_key(c, d))
                {
                    return None;
                }
                let old_quality = triangle_quality(&self.positions, first)
                    .min(triangle_quality(&self.positions, second));
                let (new_first, new_second) = flipped_triangles(first, second, a, b, c, d)?;
                let new_quality = triangle_quality(&self.positions, new_first)
                    .min(triangle_quality(&self.positions, new_second));
                let improvement = new_quality - old_quality;
                if improvement <= 1.0e-4
                    || !triangle_is_valid(&self.positions, new_first)
                    || !triangle_is_valid(&self.positions, new_second)
                {
                    return None;
                }
                let old_normal = triangle_cross(&self.positions, first)
                    + triangle_cross(&self.positions, second);
                let new_normal = triangle_cross(&self.positions, new_first)
                    + triangle_cross(&self.positions, new_second);
                (old_normal.dot(new_normal) > 0.0).then_some((
                    [faces[0], faces[1]],
                    edge_key(c, d),
                    new_first,
                    new_second,
                    improvement,
                ))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| right.4.total_cmp(&left.4));

        let mut used_faces = HashSet::new();
        let mut new_edges = HashSet::new();
        let mut flip_count = 0;
        for (faces, new_edge, first, second, _) in candidates {
            if faces.iter().any(|face| used_faces.contains(face)) || !new_edges.insert(new_edge) {
                continue;
            }
            used_faces.insert(faces[0]);
            used_faces.insert(faces[1]);
            self.triangles[faces[0] as usize] = first;
            self.triangles[faces[1] as usize] = second;
            flip_count += 1;
        }
        if flip_count != 0 {
            self.rebuild();
        }
        flip_count
    }

    fn relax_active_vertices(&mut self, active: &[bool], amount: f32) -> usize {
        let protected = self.topology.protected_neighborhood();
        let mut replacements = Vec::new();
        for (vertex, &is_active) in active.iter().enumerate() {
            if !is_active || protected[vertex] || self.topology.vertex_neighbors[vertex].is_empty()
            {
                continue;
            }
            let neighbors = &self.topology.vertex_neighbors[vertex];
            let average = neighbors
                .iter()
                .map(|&neighbor| self.positions[neighbor as usize])
                .sum::<Vec3>()
                / neighbors.len() as f32;
            let delta = average - self.positions[vertex];
            let normal = self.normals[vertex];
            let tangent_delta = if normal == Vec3::ZERO {
                delta
            } else {
                delta - normal * delta.dot(normal)
            };
            let replacement = self.positions[vertex] + tangent_delta * amount;
            if replacement.is_finite() {
                replacements.push((vertex, replacement));
            }
        }
        for (vertex, replacement) in &replacements {
            self.positions[*vertex] = *replacement;
        }
        replacements.len()
    }
}

impl MeshBvh {
    fn build(positions: &[Vec3], triangles: &[[u32; 3]]) -> Self {
        if triangles.is_empty() {
            return Self::default();
        }
        let estimated_nodes = triangles
            .len()
            .div_ceil(BVH_LEAF_SIZE / 2)
            .saturating_mul(2);
        let mut bvh = Self {
            nodes: Vec::with_capacity(estimated_nodes),
            triangle_indices: (0..triangles.len() as u32).collect(),
            triangle_leaves: vec![u32::MAX; triangles.len()],
            parents: Vec::with_capacity(estimated_nodes),
            dirty_marks: Vec::with_capacity(estimated_nodes),
            dirty_generation: 0,
        };
        let centroids = triangles
            .iter()
            .map(|&triangle| triangle_centroid(positions, triangle))
            .collect::<Vec<_>>();
        bvh.build_node(
            positions,
            triangles,
            &centroids,
            0,
            bvh.triangle_indices.len(),
            u32::MAX,
        );
        bvh
    }

    fn build_node(
        &mut self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        centroids: &[Vec3],
        start: usize,
        end: usize,
        parent: u32,
    ) -> u32 {
        let node_index = self.nodes.len() as u32;
        self.nodes.push(BvhNode::default());
        self.parents.push(parent);
        self.dirty_marks.push(0);
        let (centroid_min, centroid_max) =
            centroid_range_bounds(centroids, &self.triangle_indices[start..end]);
        let count = end - start;
        if count <= BVH_LEAF_SIZE {
            let (min, max, _, _) =
                triangle_range_bounds(positions, triangles, &self.triangle_indices[start..end]);
            self.nodes[node_index as usize] = BvhNode {
                min,
                max,
                start: start as u32,
                count: count as u32,
                ..BvhNode::default()
            };
            for &triangle in &self.triangle_indices[start..end] {
                self.triangle_leaves[triangle as usize] = node_index;
            }
            return node_index;
        }

        let extent = centroid_max - centroid_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let middle = start + count / 2;
        self.triangle_indices[start..end].select_nth_unstable_by(
            middle - start,
            |&left, &right| {
                centroids[left as usize][axis].total_cmp(&centroids[right as usize][axis])
            },
        );
        let left = self.build_node(positions, triangles, centroids, start, middle, node_index);
        let right = self.build_node(positions, triangles, centroids, middle, end, node_index);
        let left_node = self.nodes[left as usize];
        let right_node = self.nodes[right as usize];
        self.nodes[node_index as usize] = BvhNode {
            min: left_node.min.min(right_node.min),
            max: left_node.max.max(right_node.max),
            left,
            right,
            start: 0,
            count: 0,
        };
        node_index
    }

    fn refit(&mut self, positions: &[Vec3], triangles: &[[u32; 3]]) {
        if self.nodes.is_empty()
            || self.triangle_indices.len() != triangles.len()
            || self
                .triangle_indices
                .iter()
                .any(|&triangle| triangle as usize >= triangles.len())
        {
            return;
        }
        for node_index in (0..self.nodes.len()).rev() {
            let node = self.nodes[node_index];
            let (min, max) = if node.is_leaf() {
                let range = node.start as usize..(node.start + node.count) as usize;
                let (min, max, _, _) =
                    triangle_range_bounds(positions, triangles, &self.triangle_indices[range]);
                (min, max)
            } else {
                let left = self.nodes[node.left as usize];
                let right = self.nodes[node.right as usize];
                (left.min.min(right.min), left.max.max(right.max))
            };
            self.nodes[node_index].min = min;
            self.nodes[node_index].max = max;
        }
    }

    fn refit_triangles(
        &mut self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        affected_faces: &[u32],
    ) {
        if affected_faces.is_empty() {
            return;
        }
        if self.nodes.is_empty()
            || self.triangle_indices.len() != triangles.len()
            || self.triangle_leaves.len() != triangles.len()
            || self.parents.len() != self.nodes.len()
            || self.dirty_marks.len() != self.nodes.len()
        {
            *self = Self::build(positions, triangles);
            return;
        }
        if affected_faces.len() >= triangles.len() / 4 {
            self.refit(positions, triangles);
            return;
        }

        self.dirty_generation = self.dirty_generation.wrapping_add(1);
        if self.dirty_generation == 0 {
            self.dirty_marks.fill(0);
            self.dirty_generation = 1;
        }
        let generation = self.dirty_generation;
        let mut dirty_nodes = Vec::with_capacity(affected_faces.len().saturating_mul(4));
        for &face in affected_faces {
            let Some(&leaf) = self.triangle_leaves.get(face as usize) else {
                continue;
            };
            if leaf == u32::MAX {
                continue;
            }
            let mut node = leaf;
            while node != u32::MAX {
                let mark = &mut self.dirty_marks[node as usize];
                if *mark == generation {
                    break;
                }
                *mark = generation;
                dirty_nodes.push(node);
                node = self.parents[node as usize];
            }
        }

        // Nodes are allocated before their children, so descending indices always
        // update children before the parent that encloses them.
        dirty_nodes.sort_unstable_by(|left, right| right.cmp(left));
        for node_index in dirty_nodes {
            let node = self.nodes[node_index as usize];
            let (min, max) = if node.is_leaf() {
                let range = node.start as usize..(node.start + node.count) as usize;
                let (min, max, _, _) =
                    triangle_range_bounds(positions, triangles, &self.triangle_indices[range]);
                (min, max)
            } else {
                let left = self.nodes[node.left as usize];
                let right = self.nodes[node.right as usize];
                (left.min.min(right.min), left.max.max(right.max))
            };
            self.nodes[node_index as usize].min = min;
            self.nodes[node_index as usize].max = max;
        }
    }

    fn raycast(
        &self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        origin: Vec3,
        direction: Vec3,
        normals: &[Vec3],
    ) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut stack = Vec::from([0_u32]);
        let mut best: Option<RayHit> = None;
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index as usize];
            let max_distance = best.map_or(f32::INFINITY, |hit| hit.distance);
            let Some(near) = ray_aabb(origin, direction, node.min, node.max) else {
                continue;
            };
            if near > max_distance {
                continue;
            }
            if node.is_leaf() {
                let range = node.start as usize..(node.start + node.count) as usize;
                for &triangle_index in &self.triangle_indices[range] {
                    let triangle = triangles[triangle_index as usize];
                    let vertices = triangle.map(|index| positions[index as usize]);
                    let Some((distance, barycentric)) = ray_triangle(origin, direction, vertices)
                    else {
                        continue;
                    };
                    if distance >= best.map_or(f32::INFINITY, |hit| hit.distance) {
                        continue;
                    }
                    let position = origin + direction * distance;
                    let interpolated = normals[triangle[0] as usize] * barycentric.x
                        + normals[triangle[1] as usize] * barycentric.y
                        + normals[triangle[2] as usize] * barycentric.z;
                    let mut normal = interpolated.try_normalize().unwrap_or_else(|| {
                        (vertices[1] - vertices[0])
                            .cross(vertices[2] - vertices[0])
                            .normalize_or_zero()
                    });
                    if normal.dot(direction) > 0.0 {
                        normal = -normal;
                    }
                    best = Some(RayHit {
                        position,
                        normal,
                        triangle: triangle_index,
                        distance,
                        barycentric,
                    });
                }
            } else {
                let left = self.nodes[node.left as usize];
                let right = self.nodes[node.right as usize];
                let left_near = ray_aabb(origin, direction, left.min, left.max);
                let right_near = ray_aabb(origin, direction, right.min, right.max);
                match (left_near, right_near) {
                    (Some(left_distance), Some(right_distance)) => {
                        if left_distance < right_distance {
                            stack.push(node.right);
                            stack.push(node.left);
                        } else {
                            stack.push(node.left);
                            stack.push(node.right);
                        }
                    }
                    (Some(_), None) => stack.push(node.left),
                    (None, Some(_)) => stack.push(node.right),
                    (None, None) => {}
                }
            }
        }
        best
    }

    fn nearest_triangle(
        &self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        point: Vec3,
    ) -> Option<u32> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut stack = Vec::from([0_u32]);
        let mut best_triangle = None;
        let mut best_distance = f32::INFINITY;
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index as usize];
            if point_aabb_distance_squared(point, node.min, node.max) > best_distance {
                continue;
            }
            if node.is_leaf() {
                let range = node.start as usize..(node.start + node.count) as usize;
                for &triangle_index in &self.triangle_indices[range] {
                    let triangle = triangles[triangle_index as usize];
                    let vertices = triangle.map(|index| positions[index as usize]);
                    let distance = point_triangle_distance_squared(point, vertices);
                    if distance < best_distance {
                        best_distance = distance;
                        best_triangle = Some(triangle_index);
                    }
                }
            } else {
                let left = self.nodes[node.left as usize];
                let right = self.nodes[node.right as usize];
                let left_distance = point_aabb_distance_squared(point, left.min, left.max);
                let right_distance = point_aabb_distance_squared(point, right.min, right.max);
                if left_distance < right_distance {
                    if right_distance <= best_distance {
                        stack.push(node.right);
                    }
                    if left_distance <= best_distance {
                        stack.push(node.left);
                    }
                } else {
                    if left_distance <= best_distance {
                        stack.push(node.left);
                    }
                    if right_distance <= best_distance {
                        stack.push(node.right);
                    }
                }
            }
        }
        best_triangle
    }
}

fn validate_positions_and_indices(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
) -> Result<(), MeshError> {
    if positions.len() > u32::MAX as usize {
        return Err(MeshError::TooManyVertices);
    }
    for (index, position) in positions.iter().enumerate() {
        if !position.is_finite() {
            return Err(MeshError::NonFiniteVertex { index });
        }
    }
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        for &index in triangle {
            if index as usize >= positions.len() {
                return Err(MeshError::IndexOutOfBounds {
                    triangle: triangle_index,
                    index,
                    vertex_count: positions.len(),
                });
            }
        }
    }
    Ok(())
}

fn edge_key(a: u32, b: u32) -> EdgeKey {
    if a < b { (a, b) } else { (b, a) }
}

fn sorted_triangle(mut triangle: [u32; 3]) -> [u32; 3] {
    triangle.sort_unstable();
    triangle
}

fn triangle_cross(positions: &[Vec3], triangle: [u32; 3]) -> Vec3 {
    let [a, b, c] = triangle.map(|index| positions[index as usize]);
    (b - a).cross(c - a)
}

fn triangle_is_valid(positions: &[Vec3], triangle: [u32; 3]) -> bool {
    if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
        return false;
    }
    positions_form_valid_triangle(triangle.map(|index| positions[index as usize]))
}

fn positions_form_valid_triangle([a, b, c]: [Vec3; 3]) -> bool {
    let ab = b - a;
    let ac = c - a;
    let bc = c - b;
    let max_edge_squared = ab
        .length_squared()
        .max(ac.length_squared())
        .max(bc.length_squared());
    let cross_squared = ab.cross(ac).length_squared();
    max_edge_squared.is_finite()
        && cross_squared.is_finite()
        && max_edge_squared > 0.0
        && cross_squared > max_edge_squared * max_edge_squared * TRIANGLE_RELATIVE_EPSILON
}

fn orient_manifold_faces(
    triangles: &mut [[u32; 3]],
    edge_faces: &HashMap<EdgeKey, SmallVec<[u32; 2]>>,
) -> usize {
    let mut adjacency = vec![SmallVec::<[(u32, bool); 3]>::new(); triangles.len()];
    for (&(a, b), occurrences) in edge_faces.iter().filter(|(_, faces)| faces.len() == 2) {
        let first_face = occurrences[0];
        let second_face = occurrences[1];
        let first_direction = triangle_edge_direction(triangles[first_face as usize], a, b);
        let second_direction = triangle_edge_direction(triangles[second_face as usize], a, b);
        let parity_changes = first_direction == second_direction;
        adjacency[first_face as usize].push((second_face, parity_changes));
        adjacency[second_face as usize].push((first_face, parity_changes));
    }

    let mut flips = vec![None; triangles.len()];
    for seed in 0..triangles.len() {
        if flips[seed].is_some() {
            continue;
        }
        flips[seed] = Some(false);
        let mut queue = VecDeque::from([seed as u32]);
        while let Some(face) = queue.pop_front() {
            let face_flip = flips[face as usize].unwrap_or(false);
            for &(neighbor, parity_changes) in &adjacency[face as usize] {
                let expected = face_flip ^ parity_changes;
                if flips[neighbor as usize].is_none() {
                    flips[neighbor as usize] = Some(expected);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    let mut flipped_count = 0;
    for (triangle, flip) in triangles.iter_mut().zip(flips) {
        if flip.unwrap_or(false) {
            triangle.swap(1, 2);
            flipped_count += 1;
        }
    }
    flipped_count
}

fn triangle_edge_direction(triangle: [u32; 3], a: u32, b: u32) -> bool {
    (0..3).any(|index| triangle[index] == a && triangle[(index + 1) % 3] == b)
}

fn split_triangle(
    triangle: [u32; 3],
    a: u32,
    b: u32,
    midpoint: u32,
) -> Option<([u32; 3], [u32; 3])> {
    for index in 0..3 {
        let u = triangle[index];
        let v = triangle[(index + 1) % 3];
        if (u == a && v == b) || (u == b && v == a) {
            let opposite = triangle[(index + 2) % 3];
            return Some(([u, midpoint, opposite], [midpoint, v, opposite]));
        }
    }
    None
}

fn opposite_vertex(triangle: [u32; 3], a: u32, b: u32) -> Option<u32> {
    triangle
        .into_iter()
        .find(|&vertex| vertex != a && vertex != b)
}

fn flipped_triangles(
    first: [u32; 3],
    second: [u32; 3],
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> Option<([u32; 3], [u32; 3])> {
    for index in 0..3 {
        let u = first[index];
        let v = first[(index + 1) % 3];
        if !((u == a && v == b) || (u == b && v == a)) {
            continue;
        }
        let second_has_reverse = (0..3)
            .any(|second_index| second[second_index] == v && second[(second_index + 1) % 3] == u);
        if second_has_reverse {
            return Some(([c, d, v], [d, c, u]));
        }
    }
    None
}

fn triangle_quality(positions: &[Vec3], triangle: [u32; 3]) -> f32 {
    let [a, b, c] = triangle.map(|index| positions[index as usize]);
    let twice_area = (b - a).cross(c - a).length();
    let denominator = a.distance_squared(b) + b.distance_squared(c) + c.distance_squared(a);
    if denominator <= 0.0 {
        0.0
    } else {
        2.0 * 3.0_f32.sqrt() * twice_area / denominator
    }
}

fn triangle_centroid(positions: &[Vec3], triangle: [u32; 3]) -> Vec3 {
    let [a, b, c] = triangle.map(|index| positions[index as usize]);
    (a + b + c) / 3.0
}

fn triangle_range_bounds(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    indices: &[u32],
) -> (Vec3, Vec3, Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut centroid_min = Vec3::splat(f32::INFINITY);
    let mut centroid_max = Vec3::splat(f32::NEG_INFINITY);
    for &triangle_index in indices {
        let triangle = triangles[triangle_index as usize];
        for vertex in triangle {
            let position = positions[vertex as usize];
            min = min.min(position);
            max = max.max(position);
        }
        let centroid = triangle_centroid(positions, triangle);
        centroid_min = centroid_min.min(centroid);
        centroid_max = centroid_max.max(centroid);
    }
    (min, max, centroid_min, centroid_max)
}

fn centroid_range_bounds(centroids: &[Vec3], indices: &[u32]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &triangle_index in indices {
        let centroid = centroids[triangle_index as usize];
        min = min.min(centroid);
        max = max.max(centroid);
    }
    (min, max)
}

fn ray_aabb(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let mut near: f32 = 0.0;
    let mut far = f32::INFINITY;
    for axis in 0..3 {
        let direction_axis = direction[axis];
        if direction_axis.abs() < 1.0e-20 {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return None;
            }
            continue;
        }
        let inverse = direction_axis.recip();
        let mut first = (min[axis] - origin[axis]) * inverse;
        let mut second = (max[axis] - origin[axis]) * inverse;
        if first > second {
            std::mem::swap(&mut first, &mut second);
        }
        near = near.max(first);
        far = far.min(second);
        if near > far {
            return None;
        }
    }
    (far >= 0.0).then_some(near)
}

fn ray_triangle(origin: Vec3, direction: Vec3, [a, b, c]: [Vec3; 3]) -> Option<(f32, Vec3)> {
    let edge_ab = b - a;
    let edge_ac = c - a;
    let perpendicular = direction.cross(edge_ac);
    let determinant = edge_ab.dot(perpendicular);
    let scale = edge_ab.length() * edge_ac.length();
    if determinant.abs() <= scale * 1.0e-7 {
        return None;
    }
    let inverse = determinant.recip();
    let origin_offset = origin - a;
    let v_weight = origin_offset.dot(perpendicular) * inverse;
    if !(-1.0e-6..=1.0 + 1.0e-6).contains(&v_weight) {
        return None;
    }
    let cross = origin_offset.cross(edge_ab);
    let w_weight = direction.dot(cross) * inverse;
    if w_weight < -1.0e-6 || v_weight + w_weight > 1.0 + 1.0e-6 {
        return None;
    }
    let distance = edge_ac.dot(cross) * inverse;
    if distance < 0.0 || !distance.is_finite() {
        return None;
    }
    let u_weight = 1.0 - v_weight - w_weight;
    Some((distance, Vec3::new(u_weight, v_weight, w_weight)))
}

fn point_aabb_distance_squared(point: Vec3, min: Vec3, max: Vec3) -> f32 {
    let below = (min - point).max(Vec3::ZERO);
    let above = (point - max).max(Vec3::ZERO);
    (below + above).length_squared()
}

// Closest-point regions from Real-Time Collision Detection, Christer Ericson.
fn point_triangle_distance_squared(point: Vec3, [a, b, c]: [Vec3; 3]) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }

    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return point.distance_squared(a + ab * v);
    }

    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return point.distance_squared(a + ac * w);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let edge = c - b;
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return point.distance_squared(b + edge * w);
    }

    let denominator = (va + vb + vc).recip();
    let v = vb * denominator;
    let w = vc * denominator;
    point.distance_squared(a + ab * v + ac * w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn square() -> Mesh {
        Mesh::new(
            vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .unwrap()
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
        .unwrap()
    }

    #[test]
    fn rejects_non_finite_vertices_and_bad_indices() {
        assert!(matches!(
            Mesh::new(vec![Vec3::new(f32::NAN, 0.0, 0.0)], vec![]),
            Err(MeshError::NonFiniteVertex { .. })
        ));
        assert!(matches!(
            Mesh::new(vec![Vec3::ZERO; 3], vec![[0, 1, 3]]),
            Err(MeshError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn rebuild_removes_degenerate_and_duplicate_faces() {
        let mut mesh = Mesh {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(2.0, 0.0, 0.0)],
            triangles: vec![[0, 1, 2], [2, 1, 0], [0, 1, 3], [0, 0, 2], [0, 1, 99]],
            normals: Vec::new(),
            mask: Vec::new(),
            topology: MeshTopology::default(),
        };
        let report = mesh.rebuild();
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
        assert_eq!(report.removed_duplicate_faces, 1);
        assert_eq!(report.removed_degenerate_faces, 2);
        assert_eq!(report.removed_invalid_faces, 1);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert_eq!(mesh.mask.len(), mesh.positions.len());
    }

    #[test]
    fn triangle_soup_welds_only_exact_bit_patterns() {
        let soup = [
            [Vec3::ZERO, Vec3::X, Vec3::Y],
            [Vec3::ZERO, Vec3::Y, Vec3::Z],
        ];
        let (mesh, report) = Mesh::from_triangle_soup(&soup).unwrap();
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(report.welded_vertices, 2);
    }

    #[test]
    fn manifold_neighbors_are_oriented_oppositely() {
        let mesh = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::X + Vec3::Y],
            // Both input faces traverse 1 -> 2; the second must be flipped.
            vec![[0, 1, 2], [1, 2, 3]],
        )
        .unwrap();
        let directions = mesh
            .triangles
            .iter()
            .map(|triangle| {
                (0..3).any(|index| triangle[index] == 1 && triangle[(index + 1) % 3] == 2)
            })
            .collect::<Vec<_>>();
        assert_ne!(directions[0], directions[1]);
    }

    #[test]
    fn topology_reports_boundaries_and_non_manifold_edges() {
        let boundary = square();
        assert_eq!(boundary.topology.boundary_edge_count(), 4);
        assert_eq!(boundary.topology.boundary_vertex_count(), 4);

        let non_manifold = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::NEG_Y, Vec3::Z],
            vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
        )
        .unwrap();
        assert_eq!(non_manifold.topology.non_manifold_edge_count(), 1);
        assert!(non_manifold.topology.non_manifold_vertices[0]);
        assert!(non_manifold.topology.non_manifold_vertices[1]);
    }

    #[test]
    fn active_edge_collection_matches_topology_filter() {
        let mesh = octahedron();
        let mut active = vec![false; mesh.positions.len()];
        for vertex in [0_usize, 1, 4, 5] {
            active[vertex] = true;
        }
        let mut expected = mesh
            .topology
            .edge_faces
            .keys()
            .copied()
            .filter(|&(a, b)| active[a as usize] && active[b as usize])
            .collect::<Vec<_>>();
        expected.sort_unstable();

        assert_eq!(mesh.topology.active_edges(&active), expected);
    }

    #[test]
    fn bounds_center_and_diagonal_preserve_coordinates() {
        let mesh = Mesh::new(
            vec![Vec3::new(-2.0, 3.0, 4.0), Vec3::new(6.0, 7.0, 8.0)],
            vec![],
        )
        .unwrap();
        assert_eq!(
            mesh.bounds(),
            Some((Vec3::new(-2.0, 3.0, 4.0), Vec3::new(6.0, 7.0, 8.0)))
        );
        assert_eq!(mesh.center(), Some(Vec3::new(2.0, 5.0, 6.0)));
        assert!((mesh.diagonal() - 9.797_959).abs() < 1.0e-5);
    }

    #[test]
    fn bvh_raycast_returns_nearest_hit_and_barycentrics() {
        let mesh = square();
        let hit = mesh
            .raycast(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z)
            .unwrap();
        assert!((hit.distance - 2.0).abs() < 1.0e-6);
        assert!(hit.position.abs_diff_eq(Vec3::new(0.25, 0.25, 0.0), 1.0e-6));
        assert!((hit.barycentric.element_sum() - 1.0).abs() < 1.0e-6);
        assert!(hit.normal.dot(Vec3::Z).abs() > 0.99);
        assert!(
            mesh.raycast(Vec3::new(3.0, 3.0, 2.0), Vec3::NEG_Z)
                .is_none()
        );
    }

    #[test]
    fn bvh_nearest_triangle_finds_the_closest_component() {
        let mesh = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(11.0, 0.0, 0.0),
                Vec3::new(10.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .unwrap();
        assert_eq!(mesh.nearest_triangle(Vec3::new(10.2, 0.2, 3.0)), Some(1));
        assert_eq!(mesh.nearest_triangle(Vec3::new(0.2, 0.2, 3.0)), Some(0));
    }

    #[test]
    fn local_deformation_refresh_matches_full_normals_and_refits_bvh() {
        let mut local = octahedron();
        local.positions[0] = Vec3::new(2.5, 0.2, 0.1);
        let updated = local.update_deformed_vertices(&[0]);

        let mut full = octahedron();
        full.positions[0] = local.positions[0];
        full.recompute_normals();

        assert!(updated.contains(&0));
        for (local_normal, full_normal) in local.normals.iter().zip(&full.normals) {
            assert!(local_normal.abs_diff_eq(*full_normal, 1.0e-6));
        }
        let local_hit = local
            .raycast(Vec3::new(1.8, 0.1, 2.0), Vec3::NEG_Z)
            .expect("the expanded mesh bounds must be pickable after a local refit");
        let full_hit = full.raycast(Vec3::new(1.8, 0.1, 2.0), Vec3::NEG_Z).unwrap();
        assert!((local_hit.distance - full_hit.distance).abs() < 1.0e-5);
        assert_eq!(local_hit.triangle, full_hit.triangle);
    }

    #[test]
    fn connected_front_facing_collection_stays_on_seed_component() {
        let mesh = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::new(0.1, 0.1, 0.05),
                Vec3::new(1.1, 0.1, 0.05),
                Vec3::new(0.1, 1.1, 0.05),
            ],
            vec![[0, 1, 2], [3, 5, 4]],
        )
        .unwrap();
        let selected =
            mesh.connected_front_facing_vertices(0, Vec3::new(0.3, 0.3, 0.0), 2.0, Vec3::NEG_Z);
        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&vertex| vertex < 3));
    }

    #[test]
    fn connected_selection_accepts_consistently_reversed_winding() {
        let mesh = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 2, 1]]).unwrap();
        let selected = mesh.connected_front_facing_vertices(0, Vec3::splat(0.2), 2.0, Vec3::NEG_Z);
        assert_eq!(selected.len(), 3);
        let hit = mesh.raycast(Vec3::new(0.2, 0.2, 1.0), Vec3::NEG_Z).unwrap();
        assert!(hit.normal.dot(Vec3::NEG_Z) <= 0.0);
    }

    #[test]
    fn dynamic_remesh_splits_long_interior_edges_and_preserves_validity() {
        let mut mesh = octahedron();
        let original_vertices = mesh.positions.len();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let stats = mesh.remesh_region(
            &active,
            RemeshSettings {
                target_edge_length: 0.9,
                iterations: 1,
                enable_flips: false,
                relaxation: 0.0,
                ..RemeshSettings::default()
            },
        );
        assert!(stats.splits > 0);
        assert!(mesh.positions.len() > original_vertices);
        mesh.validate().unwrap();
        assert!(
            mesh.triangles
                .iter()
                .all(|&triangle| triangle_is_valid(&mesh.positions, triangle))
        );
    }

    #[test]
    fn dynamic_remesh_batches_safe_short_edge_collapses() {
        let mut mesh = octahedron();
        let original_vertices = mesh.positions.len();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let stats = mesh.remesh_region(
            &active,
            RemeshSettings {
                target_edge_length: 2.5,
                iterations: 1,
                enable_flips: false,
                relaxation: 0.0,
                ..RemeshSettings::default()
            },
        );
        assert!(stats.collapses > 0);
        assert!(mesh.positions.len() < original_vertices);
        mesh.validate().unwrap();
        assert!(
            mesh.triangles
                .iter()
                .all(|&triangle| triangle_is_valid(&mesh.positions, triangle))
        );
    }

    #[test]
    fn dynamic_remesh_batches_quality_improving_edge_flips() {
        let mut mesh = octahedron();
        mesh.positions[0] *= 3.0;
        mesh.positions[1] *= 3.0;
        mesh.rebuild();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let stats = mesh.remesh_region(
            &active,
            RemeshSettings {
                target_edge_length: 1.0,
                iterations: 1,
                split_threshold: 10.0,
                collapse_threshold: 0.01,
                enable_flips: true,
                relaxation: 0.0,
            },
        );
        assert!(stats.flips > 0);
        mesh.validate().unwrap();
    }

    #[test]
    fn remeshing_protects_open_boundaries() {
        let mut mesh = square();
        let before = mesh.clone();
        let active = [0, 1, 2, 3];
        let stats = mesh.remesh_region(
            &active,
            RemeshSettings {
                target_edge_length: 0.1,
                relaxation: 1.0,
                ..RemeshSettings::default()
            },
        );
        assert_eq!(stats.splits, 0);
        assert_eq!(stats.collapses, 0);
        assert_eq!(stats.relaxed_vertices, 0);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
    }

    #[test]
    fn no_op_local_remesh_preserves_mesh_exactly() {
        let mut mesh = octahedron();
        let before = mesh.clone();
        let stats = mesh.remesh_region(
            &[0, 1, 4],
            RemeshSettings {
                target_edge_length: 1.0,
                iterations: 1,
                split_threshold: 10.0,
                collapse_threshold: 0.01,
                enable_flips: false,
                relaxation: 0.0,
            },
        );

        assert_eq!(stats.splits, 0);
        assert_eq!(stats.collapses, 0);
        assert_eq!(stats.flips, 0);
        assert_eq!(stats.relaxed_vertices, 0);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
    }

    #[test]
    fn closest_point_regions_cover_faces_edges_and_vertices() {
        let triangle = [Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(
            (point_triangle_distance_squared(Vec3::new(0.25, 0.25, 2.0), triangle) - 4.0).abs()
                < 1.0e-6
        );
        assert!(
            (point_triangle_distance_squared(Vec3::new(-1.0, 0.0, 0.0), triangle) - 1.0).abs()
                < 1.0e-6
        );
        assert!(
            (point_triangle_distance_squared(Vec3::new(0.75, 0.75, 0.0), triangle) - 0.125).abs()
                < 1.0e-6
        );
    }

    #[test]
    #[ignore = "release-mode performance envelope"]
    fn million_face_mesh_build_pick_and_local_remesh() {
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

        let build_started = Instant::now();
        let mut mesh = Mesh::new(positions, triangles).unwrap();
        let build_elapsed = build_started.elapsed();
        assert!(mesh.triangles.len() >= 1_000_000);

        let center = Vec3::new(CELLS as f32 * 0.5, CELLS as f32 * 0.5, 0.0);
        let pick_started = Instant::now();
        let hit = mesh.raycast(center + Vec3::Z * 10.0, Vec3::NEG_Z);
        let pick_elapsed = pick_started.elapsed();
        assert!(hit.is_some());

        let half_patch = 10;
        let middle = CELLS / 2;
        let mut active = Vec::new();
        for y in middle - half_patch..=middle + half_patch {
            for x in middle - half_patch..=middle + half_patch {
                active.push((y * row + x) as u32);
            }
        }
        for &vertex in &active {
            mesh.positions[vertex as usize].z = 0.01;
        }
        let deform_refresh_started = Instant::now();
        let updated_vertices = mesh.update_deformed_vertices(&active);
        let deform_refresh_elapsed = deform_refresh_started.elapsed();
        assert!(updated_vertices.len() >= active.len());
        let remesh_started = Instant::now();
        let stats = mesh.remesh_region(
            &active,
            RemeshSettings {
                target_edge_length: 0.9,
                iterations: 1,
                enable_flips: true,
                relaxation: 0.0,
                ..RemeshSettings::default()
            },
        );
        let remesh_elapsed = remesh_started.elapsed();
        assert!(stats.splits > 0);
        mesh.validate().unwrap();

        eprintln!(
            "million-face benchmark: build={build_elapsed:?}, pick={pick_elapsed:?}, deform_refresh={deform_refresh_elapsed:?}, local_remesh={remesh_elapsed:?}"
        );
    }
}
