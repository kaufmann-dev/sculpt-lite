use std::{collections::VecDeque, mem::size_of};

use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use thiserror::Error;

pub type EdgeKey = (u32, u32);

const TRIANGLE_RELATIVE_EPSILON: f32 = 1.0e-12;
const BVH_LEAF_SIZE: usize = 8;
const EDIT_GROWTH_DIVISOR: usize = 8;
const MIN_EDIT_GROWTH: usize = 64;
const MAX_HASH_EDIT_GROWTH: usize = 65_536;
const INLINE_VERTEX_CLASSIFICATION_FACES: usize = 16;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVertexState {
    pub position: Vec3,
    pub mask: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotChange<T> {
    pub index: u32,
    pub before: Option<T>,
    pub after: Option<T>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshEditDelta {
    pub before_vertex_count: usize,
    pub after_vertex_count: usize,
    pub before_face_count: usize,
    pub after_face_count: usize,
    pub vertices: Vec<SlotChange<MeshVertexState>>,
    pub faces: Vec<SlotChange<[u32; 3]>>,
}

impl MeshEditDelta {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.faces.is_empty()
    }

    #[must_use]
    pub fn topology_changed(&self) -> bool {
        self.before_vertex_count != self.after_vertex_count
            || self.before_face_count != self.after_face_count
            || !self.faces.is_empty()
    }

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.vertices.len() * size_of::<SlotChange<MeshVertexState>>()
            + self.faces.len() * size_of::<SlotChange<[u32; 3]>>()
    }

    pub fn apply_before(&self, mesh: &mut Mesh) -> MeshChangeSet {
        mesh.apply_edit_delta(self, false)
    }

    pub fn apply_after(&self, mesh: &mut Mesh) -> MeshChangeSet {
        mesh.apply_edit_delta(self, true)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshChangeSet {
    pub dirty_vertices: Vec<u32>,
    pub dirty_faces: Vec<u32>,
    pub added_edges: Vec<EdgeKey>,
    pub removed_edges: Vec<EdgeKey>,
    pub vertex_count: usize,
    pub face_count: usize,
    edge_deltas: HashMap<EdgeKey, EdgeDelta>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeDelta {
    Added,
    Removed,
}

impl MeshChangeSet {
    fn normalize(&mut self) {
        self.dirty_vertices.sort_unstable();
        self.dirty_vertices.dedup();
        self.dirty_faces.sort_unstable();
        self.dirty_faces.dedup();

        self.added_edges.clear();
        self.removed_edges.clear();
        for (&edge, &delta) in &self.edge_deltas {
            match delta {
                EdgeDelta::Added => self.added_edges.push(edge),
                EdgeDelta::Removed => self.removed_edges.push(edge),
            }
        }
        self.added_edges.sort_unstable();
        self.removed_edges.sort_unstable();
    }

    fn record_edge_change(&mut self, edge: EdgeKey, next: EdgeDelta) {
        use hashbrown::hash_map::Entry;

        match self.edge_deltas.entry(edge) {
            Entry::Vacant(entry) => {
                entry.insert(next);
            }
            Entry::Occupied(entry) if *entry.get() == next => {}
            Entry::Occupied(entry) => {
                entry.remove();
            }
        }
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.dirty_vertices.extend(other.dirty_vertices);
        self.dirty_faces.extend(other.dirty_faces);
        for (edge, delta) in other.edge_deltas {
            self.record_edge_change(edge, delta);
        }
        self.vertex_count = other.vertex_count;
        self.face_count = other.face_count;
    }

    pub(crate) fn include_vertices(&mut self, vertices: impl IntoIterator<Item = u32>) {
        self.dirty_vertices.extend(vertices);
    }

    pub(crate) fn finalize(&mut self, vertex_count: usize, face_count: usize) {
        self.vertex_count = vertex_count;
        self.face_count = face_count;
        self.normalize();
    }

    pub(crate) fn clear(&mut self) {
        self.dirty_vertices.clear();
        self.dirty_faces.clear();
        self.added_edges.clear();
        self.removed_edges.clear();
        self.edge_deltas.clear();
        self.vertex_count = 0;
        self.face_count = 0;
    }
}

#[derive(Clone, Debug)]
pub struct MeshEditRecorder {
    before_vertex_count: usize,
    before_face_count: usize,
    vertices: HashMap<u32, Option<MeshVertexState>>,
    faces: HashMap<u32, Option<[u32; 3]>>,
}

impl MeshEditRecorder {
    #[must_use]
    pub fn new(mesh: &Mesh) -> Self {
        Self {
            before_vertex_count: mesh.positions.len(),
            before_face_count: mesh.triangles.len(),
            vertices: HashMap::new(),
            faces: HashMap::new(),
        }
    }

    pub fn record_vertex(&mut self, mesh: &Mesh, vertex: u32) {
        self.vertices.entry(vertex).or_insert_with(|| {
            let index = vertex as usize;
            mesh.positions
                .get(index)
                .copied()
                .map(|position| MeshVertexState {
                    position,
                    mask: mesh.mask.get(index).copied().unwrap_or(0.0),
                })
        });
    }

    pub fn record_face(&mut self, mesh: &Mesh, face: u32) {
        self.faces
            .entry(face)
            .or_insert_with(|| mesh.triangles.get(face as usize).copied());
    }

    pub fn absorb_recorder(&mut self, recorder: Self, mesh: &Mesh) {
        let Self {
            vertices, faces, ..
        } = recorder;
        let mut additional_vertices = 0;
        for (&index, &before) in &vertices {
            if mesh_vertex_state(mesh, index) != before && !self.vertices.contains_key(&index) {
                additional_vertices += 1;
            }
        }
        let mut additional_faces = 0;
        for (&index, &before) in &faces {
            if mesh.triangles.get(index as usize).copied() != before
                && !self.faces.contains_key(&index)
            {
                additional_faces += 1;
            }
        }
        self.vertices.reserve(additional_vertices);
        self.faces.reserve(additional_faces);

        for (index, before) in vertices {
            if mesh_vertex_state(mesh, index) != before {
                self.vertices.entry(index).or_insert(before);
            }
        }
        for (index, before) in faces {
            if mesh.triangles.get(index as usize).copied() != before {
                self.faces.entry(index).or_insert(before);
            }
        }
    }

    #[must_use]
    pub fn finish(self, mesh: &Mesh) -> MeshEditDelta {
        let mut vertices = self
            .vertices
            .into_iter()
            .filter_map(|(index, before)| {
                let after = mesh_vertex_state(mesh, index);
                (before != after).then_some(SlotChange {
                    index,
                    before,
                    after,
                })
            })
            .collect::<Vec<_>>();
        let mut faces = self
            .faces
            .into_iter()
            .filter_map(|(index, before)| {
                let after = mesh.triangles.get(index as usize).copied();
                (before != after).then_some(SlotChange {
                    index,
                    before,
                    after,
                })
            })
            .collect::<Vec<_>>();
        vertices.sort_unstable_by_key(|change| change.index);
        faces.sort_unstable_by_key(|change| change.index);
        MeshEditDelta {
            before_vertex_count: self.before_vertex_count,
            after_vertex_count: mesh.positions.len(),
            before_face_count: self.before_face_count,
            after_face_count: mesh.triangles.len(),
            vertices,
            faces,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RemeshOutcome {
    pub stats: RemeshStats,
    pub changes: MeshChangeSet,
}

impl std::ops::Deref for RemeshOutcome {
    type Target = RemeshStats;

    fn deref(&self) -> &Self::Target {
        &self.stats
    }
}

#[derive(Clone, Debug)]
pub struct MeshBvh {
    root: u32,
    nodes: Vec<BvhNode>,
    leaf_faces: Vec<Vec<u32>>,
    triangle_leaves: Vec<u32>,
    free_nodes: Vec<u32>,
    free_leaves: Vec<u32>,
}

/// Reusable storage for connected brush-region traversals.
///
/// Clearing only the vertices touched by the previous traversal keeps brush
/// selection proportional to the local region instead of the complete mesh.
#[derive(Debug, Default)]
pub(crate) struct VertexTraversalScratch {
    visited: Vec<u8>,
    touched: Vec<u32>,
    queue: VecDeque<u32>,
}

impl VertexTraversalScratch {
    fn begin(&mut self, vertex_count: usize) {
        for vertex in self.touched.drain(..) {
            self.visited[vertex as usize] = 0;
        }
        if self.visited.len() < vertex_count {
            self.visited.resize(vertex_count, 0);
        }
        self.queue.clear();
    }

    fn enqueue_once(&mut self, vertex: u32) {
        let index = vertex as usize;
        if index >= self.visited.len() || self.visited[index] != 0 {
            return;
        }
        self.visited[index] = 1;
        self.touched.push(vertex);
        self.queue.push_back(vertex);
    }
}

impl Default for MeshBvh {
    fn default() -> Self {
        Self {
            root: u32::MAX,
            nodes: Vec::new(),
            leaf_faces: Vec::new(),
            triangle_leaves: Vec::new(),
            free_nodes: Vec::new(),
            free_leaves: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    parent: u32,
    left: u32,
    right: u32,
    leaf: u32,
    height: i32,
}

impl Default for BvhNode {
    fn default() -> Self {
        Self {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            parent: u32::MAX,
            left: u32::MAX,
            right: u32::MAX,
            leaf: u32::MAX,
            height: -1,
        }
    }
}

impl BvhNode {
    fn is_leaf(self) -> bool {
        self.height == 0 && self.leaf != u32::MAX
    }

    fn is_live(self) -> bool {
        self.height >= 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshTopology {
    pub vertex_neighbors: Vec<SmallVec<[u32; 8]>>,
    pub vertex_triangles: Vec<SmallVec<[u32; 8]>>,
    pub edge_faces: HashMap<EdgeKey, SmallVec<[u32; 2]>>,
    face_lookup: HashMap<[u32; 3], u32>,
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

    fn active_edges(&self, active: &HashSet<u32>) -> Vec<EdgeKey> {
        let mut edges = Vec::new();
        for &a in active {
            let Some(neighbors) = self.vertex_neighbors.get(a as usize) else {
                continue;
            };
            for &b in neighbors {
                if a < b && active.contains(&b) {
                    edges.push((a, b));
                }
            }
        }
        edges
    }

    fn build(positions: &[Vec3], triangles: &[[u32; 3]]) -> Self {
        let mut topology = Self {
            vertex_neighbors: filled_vec_with_headroom(positions.len(), SmallVec::new()),
            vertex_triangles: filled_vec_with_headroom(positions.len(), SmallVec::new()),
            edge_faces: HashMap::with_capacity(bounded_hash_editing_capacity(
                triangles.len().saturating_mul(3) / 2,
            )),
            face_lookup: HashMap::with_capacity(bounded_hash_editing_capacity(triangles.len())),
            boundary_vertices: filled_vec_with_headroom(positions.len(), false),
            non_manifold_vertices: filled_vec_with_headroom(positions.len(), false),
            bvh: MeshBvh::default(),
        };

        for (face_index, triangle) in triangles.iter().enumerate() {
            topology
                .face_lookup
                .insert(sorted_triangle(*triangle), face_index as u32);
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

        // The edge estimate is exact for closed manifolds but can be low for
        // boundary-heavy or disconnected inputs. Guarantee bounded spare room
        // from the actual map sizes while this build still runs off the UI path.
        topology
            .edge_faces
            .reserve(bounded_hash_edit_growth(topology.edge_faces.len()));
        topology
            .face_lookup
            .reserve(bounded_hash_edit_growth(topology.face_lookup.len()));

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
        let Self {
            mut positions,
            mut triangles,
            vertices,
            input_vertices,
            input_triangles,
        } = self;
        drop(vertices);
        let unique_vertices = positions.len();
        reserve_edit_headroom(&mut positions);
        reserve_edit_headroom(&mut triangles);
        let mut mesh = Mesh {
            positions,
            triangles,
            normals: filled_vec_with_headroom(unique_vertices, Vec3::ZERO),
            mask: filled_vec_with_headroom(unique_vertices, 0.0),
            topology: MeshTopology::default(),
        };
        let mut report = mesh.rebuild();
        report.input_vertices = input_vertices;
        report.input_triangles = input_triangles;
        report.welded_vertices = report.input_vertices.saturating_sub(unique_vertices);
        (mesh, report)
    }
}

impl Mesh {
    #[cfg(test)]
    pub fn new(mut positions: Vec<Vec3>, mut triangles: Vec<[u32; 3]>) -> Result<Self, MeshError> {
        validate_positions_and_indices(&positions, &triangles)?;
        reserve_edit_headroom(&mut positions);
        reserve_edit_headroom(&mut triangles);
        let vertex_count = positions.len();
        let mut mesh = Self {
            positions,
            triangles,
            normals: filled_vec_with_headroom(vertex_count, Vec3::ZERO),
            mask: filled_vec_with_headroom(vertex_count, 0.0),
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

        reserve_edit_headroom(&mut self.positions);
        reserve_edit_headroom(&mut self.triangles);

        reserve_for_editing(&mut self.mask, self.positions.len());
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
        self.recompute_normals_without_bvh_refit();

        report.output_vertices = self.positions.len();
        report.output_triangles = self.triangles.len();
        report.boundary_edges = self.topology.boundary_edge_count();
        report.boundary_vertices = self.topology.boundary_vertex_count();
        report.non_manifold_edges = self.topology.non_manifold_edge_count();
        report.non_manifold_vertices = self.topology.non_manifold_vertex_count();
        report
    }

    #[cfg(test)]
    pub fn recompute_normals(&mut self) {
        self.recompute_normals_without_bvh_refit();
        self.topology.bvh.refit(&self.positions, &self.triangles);
    }

    fn recompute_normals_without_bvh_refit(&mut self) {
        self.normals.clear();
        reserve_for_editing(&mut self.normals, self.positions.len());
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
        self.update_deformed_faces(&affected_faces)
    }

    pub(crate) fn update_deformed_faces(&mut self, affected_faces: &[u32]) -> Vec<u32> {
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
            .refit_triangles(&self.positions, &self.triangles, affected_faces);
        normal_vertices
    }

    /// Checks a position-only local edit before normals and BVH bounds are refreshed.
    ///
    /// Fixed-topology brushes use this to reject non-finite positions and collapsed
    /// incident triangles without paying for topology recording on every sample.
    pub(crate) fn validated_deformation_faces(&self, moved_vertices: &[u32]) -> Option<Vec<u32>> {
        let mut affected_faces = Vec::new();
        for &vertex in moved_vertices {
            let index = vertex as usize;
            if !self
                .positions
                .get(index)
                .is_some_and(|position| position.is_finite())
            {
                return None;
            }
            let faces = self.topology.vertex_triangles.get(index)?;
            for &face in faces {
                let triangle = self.triangles.get(face as usize)?;
                if !triangle.contains(&vertex) {
                    return None;
                }
                affected_faces.push(face);
            }
        }
        affected_faces.sort_unstable();
        affected_faces.dedup();
        if affected_faces.iter().all(|&face| {
            let triangle = self.triangles[face as usize];
            triangle_is_valid(&self.positions, triangle)
                && self.topology.face_lookup.get(&sorted_triangle(triangle)) == Some(&face)
        }) {
            Some(affected_faces)
        } else {
            None
        }
    }

    fn apply_edit_delta(&mut self, delta: &MeshEditDelta, after: bool) -> MeshChangeSet {
        let target_vertex_count = if after {
            delta.after_vertex_count
        } else {
            delta.before_vertex_count
        };
        let target_face_count = if after {
            delta.after_face_count
        } else {
            delta.before_face_count
        };
        let mut changes = MeshChangeSet::default();

        for change in &delta.faces {
            if let Some(&triangle) = self.triangles.get(change.index as usize) {
                self.detach_face(change.index, triangle, &mut changes);
            }
        }

        while self.positions.len() < target_vertex_count {
            self.positions.push(Vec3::ZERO);
            self.mask.push(0.0);
            self.normals.push(Vec3::ZERO);
            self.topology.vertex_neighbors.push(SmallVec::new());
            self.topology.vertex_triangles.push(SmallVec::new());
            self.topology.boundary_vertices.push(false);
            self.topology.non_manifold_vertices.push(false);
        }
        self.positions.truncate(target_vertex_count);
        self.mask.truncate(target_vertex_count);
        self.normals.truncate(target_vertex_count);
        self.topology.vertex_neighbors.truncate(target_vertex_count);
        self.topology.vertex_triangles.truncate(target_vertex_count);
        self.topology
            .boundary_vertices
            .truncate(target_vertex_count);
        self.topology
            .non_manifold_vertices
            .truncate(target_vertex_count);

        let mut moved_vertices = Vec::with_capacity(delta.vertices.len());
        for change in &delta.vertices {
            let state = if after { change.after } else { change.before };
            let Some(state) = state else {
                continue;
            };
            let index = change.index as usize;
            if index < target_vertex_count {
                self.positions[index] = state.position;
                self.mask[index] = state.mask;
                moved_vertices.push(change.index);
                changes.dirty_vertices.push(change.index);
            }
        }

        self.triangles.resize(target_face_count, [0, 0, 0]);
        for change in &delta.faces {
            let triangle = if after { change.after } else { change.before };
            let Some(triangle) = triangle else {
                continue;
            };
            if (change.index as usize) < target_face_count {
                self.triangles[change.index as usize] = triangle;
                self.attach_face(change.index, triangle, &mut changes);
                changes.dirty_faces.push(change.index);
            }
        }
        self.triangles.truncate(target_face_count);
        self.topology.bvh.truncate_faces(target_face_count);

        let mut moved_faces = moved_vertices
            .iter()
            .filter_map(|&vertex| self.topology.vertex_triangles.get(vertex as usize))
            .flat_map(|faces| faces.iter().copied())
            .collect::<Vec<_>>();
        moved_faces.sort_unstable();
        moved_faces.dedup();
        self.topology
            .bvh
            .refit_triangles(&self.positions, &self.triangles, &moved_faces);
        changes.dirty_faces.extend(moved_faces);
        self.finish_local_changes(&mut changes);
        changes
    }

    fn detach_face(&mut self, face: u32, triangle: [u32; 3], changes: &mut MeshChangeSet) {
        self.topology
            .bvh
            .remove_face(&self.positions, &self.triangles, face);
        if self.topology.face_lookup.get(&sorted_triangle(triangle)) == Some(&face) {
            self.topology.face_lookup.remove(&sorted_triangle(triangle));
        }
        for vertex in triangle {
            if let Some(faces) = self.topology.vertex_triangles.get_mut(vertex as usize)
                && let Some(position) = faces.iter().position(|&candidate| candidate == face)
            {
                faces.swap_remove(position);
            }
            changes.dirty_vertices.push(vertex);
        }
        for index in 0..3 {
            let edge = edge_key(triangle[index], triangle[(index + 1) % 3]);
            let mut remove_edge = false;
            if let Some(faces) = self.topology.edge_faces.get_mut(&edge) {
                if let Some(position) = faces.iter().position(|&candidate| candidate == face) {
                    faces.swap_remove(position);
                }
                remove_edge = faces.is_empty();
            }
            if remove_edge {
                self.topology.edge_faces.remove(&edge);
                remove_small_value(&mut self.topology.vertex_neighbors[edge.0 as usize], edge.1);
                remove_small_value(&mut self.topology.vertex_neighbors[edge.1 as usize], edge.0);
                changes.record_edge_change(edge, EdgeDelta::Removed);
            }
        }
    }

    fn attach_face(&mut self, face: u32, triangle: [u32; 3], changes: &mut MeshChangeSet) {
        self.topology
            .face_lookup
            .insert(sorted_triangle(triangle), face);
        for vertex in triangle {
            self.topology.vertex_triangles[vertex as usize].push(face);
            changes.dirty_vertices.push(vertex);
        }
        for index in 0..3 {
            let edge = edge_key(triangle[index], triangle[(index + 1) % 3]);
            let is_new = !self.topology.edge_faces.contains_key(&edge);
            self.topology.edge_faces.entry(edge).or_default().push(face);
            if is_new {
                insert_small_sorted(&mut self.topology.vertex_neighbors[edge.0 as usize], edge.1);
                insert_small_sorted(&mut self.topology.vertex_neighbors[edge.1 as usize], edge.0);
                changes.record_edge_change(edge, EdgeDelta::Added);
            }
        }
        self.topology
            .bvh
            .insert_face(&self.positions, &self.triangles, face);
    }

    fn replace_face_local(
        &mut self,
        face: u32,
        triangle: [u32; 3],
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) {
        recorder.record_face(self, face);
        let old = self.triangles[face as usize];
        self.detach_face(face, old, changes);
        self.triangles[face as usize] = triangle;
        self.attach_face(face, triangle, changes);
        changes.dirty_faces.push(face);
    }

    fn append_face_local(
        &mut self,
        triangle: [u32; 3],
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> u32 {
        let face = self.triangles.len() as u32;
        recorder.record_face(self, face);
        self.triangles.push(triangle);
        self.attach_face(face, triangle, changes);
        changes.dirty_faces.push(face);
        face
    }

    fn remove_face_dense(
        &mut self,
        face: u32,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) {
        let last = self.triangles.len().saturating_sub(1) as u32;
        recorder.record_face(self, face);
        let removed = self.triangles[face as usize];
        self.detach_face(face, removed, changes);
        if face != last {
            recorder.record_face(self, last);
            let moved = self.triangles[last as usize];
            self.detach_face(last, moved, changes);
            self.triangles[face as usize] = moved;
            self.attach_face(face, moved, changes);
            changes.dirty_faces.push(face);
        }
        self.triangles.pop();
        self.topology.bvh.truncate_faces(self.triangles.len());
    }

    fn append_vertex_local(
        &mut self,
        position: Vec3,
        mask: f32,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> u32 {
        let vertex = self.positions.len() as u32;
        recorder.record_vertex(self, vertex);
        self.positions.push(position);
        self.mask.push(mask);
        self.normals.push(Vec3::ZERO);
        self.topology.vertex_neighbors.push(SmallVec::new());
        self.topology.vertex_triangles.push(SmallVec::new());
        self.topology.boundary_vertices.push(false);
        self.topology.non_manifold_vertices.push(false);
        changes.dirty_vertices.push(vertex);
        vertex
    }

    fn remove_vertex_dense(
        &mut self,
        vertex: u32,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> Option<(u32, u32)> {
        if !self.topology.vertex_triangles[vertex as usize].is_empty() {
            return None;
        }
        let last = self.positions.len().checked_sub(1)? as u32;
        recorder.record_vertex(self, vertex);
        let remap = if vertex != last {
            recorder.record_vertex(self, last);
            let incident = self.topology.vertex_triangles[last as usize].clone();
            for face in incident {
                let replacement = self.triangles[face as usize]
                    .map(|candidate| if candidate == last { vertex } else { candidate });
                self.replace_face_local(face, replacement, recorder, changes);
            }
            self.positions[vertex as usize] = self.positions[last as usize];
            self.mask[vertex as usize] = self.mask[last as usize];
            self.normals[vertex as usize] = self.normals[last as usize];
            Some((last, vertex))
        } else {
            None
        };
        self.positions.pop();
        self.mask.pop();
        self.normals.pop();
        self.topology.vertex_neighbors.pop();
        self.topology.vertex_triangles.pop();
        self.topology.boundary_vertices.pop();
        self.topology.non_manifold_vertices.pop();
        changes.dirty_vertices.push(vertex);
        remap
    }

    fn finish_local_changes(&mut self, changes: &mut MeshChangeSet) {
        changes.normalize();
        let mut classified = changes.dirty_vertices.clone();
        for &vertex in &changes.dirty_vertices {
            if let Some(neighbors) = self.topology.vertex_neighbors.get(vertex as usize) {
                classified.extend(neighbors.iter().copied());
            }
        }
        classified.retain(|&vertex| (vertex as usize) < self.positions.len());
        classified.sort_unstable();
        classified.dedup();
        for &vertex in &classified {
            let (boundary, non_manifold) = self.classify_vertex(vertex);
            self.topology.boundary_vertices[vertex as usize] = boundary;
            self.topology.non_manifold_vertices[vertex as usize] = non_manifold;
            let normal = self.topology.vertex_triangles[vertex as usize]
                .iter()
                .map(|&face| triangle_cross(&self.positions, self.triangles[face as usize]))
                .sum::<Vec3>()
                .try_normalize()
                .unwrap_or(Vec3::ZERO);
            self.normals[vertex as usize] = normal;
        }
        changes.dirty_vertices.extend(classified);
        changes.normalize();
        changes.vertex_count = self.positions.len();
        changes.face_count = self.triangles.len();
    }

    #[must_use]
    pub fn local_changes_are_valid(&self, changes: &MeshChangeSet) -> bool {
        if self.positions.len() != self.normals.len()
            || self.positions.len() != self.mask.len()
            || self.topology.vertex_neighbors.len() != self.positions.len()
            || self.topology.vertex_triangles.len() != self.positions.len()
        {
            return false;
        }
        for &face in &changes.dirty_faces {
            let Some(&triangle) = self.triangles.get(face as usize) else {
                continue;
            };
            if !triangle_is_valid(&self.positions, triangle)
                || self.topology.face_lookup.get(&sorted_triangle(triangle)) != Some(&face)
            {
                return false;
            }
        }
        for &vertex in &changes.dirty_vertices {
            let index = vertex as usize;
            if index >= self.positions.len() {
                continue;
            }
            if !self.positions[index].is_finite()
                || !self.normals[index].is_finite()
                || !self.mask[index].is_finite()
            {
                return false;
            }
            if self.topology.vertex_triangles[index].iter().any(|&face| {
                let Some(&triangle) = self.triangles.get(face as usize) else {
                    return true;
                };
                !triangle.contains(&vertex)
                    || !triangle_is_valid(&self.positions, triangle)
                    || self.topology.face_lookup.get(&sorted_triangle(triangle)) != Some(&face)
            }) || self.topology.vertex_neighbors[index]
                .iter()
                .any(|&neighbor| {
                    !self
                        .topology
                        .edge_faces
                        .contains_key(&edge_key(vertex, neighbor))
                })
            {
                return false;
            }
        }
        changes
            .added_edges
            .iter()
            .all(|edge| self.topology.edge_faces.contains_key(edge))
            && changes
                .removed_edges
                .iter()
                .all(|edge| !self.topology.edge_faces.contains_key(edge))
    }

    fn classify_vertex(&self, vertex: u32) -> (bool, bool) {
        let index = vertex as usize;
        let incident = &self.topology.vertex_triangles[index];
        if incident.is_empty() {
            return (false, false);
        }
        let mut boundary_degree = 0;
        let mut non_manifold = false;
        for &neighbor in &self.topology.vertex_neighbors[index] {
            match self.topology.edge_faces[&edge_key(vertex, neighbor)].len() {
                1 => boundary_degree += 1,
                2 => {}
                _ => non_manifold = true,
            }
        }
        if boundary_degree != 0 && boundary_degree != 2 {
            non_manifold = true;
        }

        let visited_count = if incident.len() <= INLINE_VERTEX_CLASSIFICATION_FACES {
            let mut visited = SmallVec::<[u32; INLINE_VERTEX_CLASSIFICATION_FACES]>::new();
            let mut stack =
                SmallVec::<[u32; INLINE_VERTEX_CLASSIFICATION_FACES]>::from_slice(&[incident[0]]);
            while let Some(face) = stack.pop() {
                if visited.contains(&face) {
                    continue;
                }
                visited.push(face);
                self.extend_incident_face_stack(vertex, face, &mut stack);
            }
            visited.len()
        } else {
            let mut visited = HashSet::with_capacity(incident.len());
            let mut stack = Vec::with_capacity(incident.len());
            stack.push(incident[0]);
            while let Some(face) = stack.pop() {
                if !visited.insert(face) {
                    continue;
                }
                self.extend_incident_face_stack(vertex, face, &mut stack);
            }
            visited.len()
        };
        if visited_count != incident.len() {
            non_manifold = true;
        }
        (boundary_degree != 0, non_manifold)
    }

    fn extend_incident_face_stack(&self, vertex: u32, face: u32, stack: &mut impl Extend<u32>) {
        let triangle = self.triangles[face as usize];
        for edge_index in 0..3 {
            let a = triangle[edge_index];
            let b = triangle[(edge_index + 1) % 3];
            if a != vertex && b != vertex {
                continue;
            }
            if let Some(faces) = self.topology.edge_faces.get(&edge_key(a, b)) {
                stack.extend(faces.iter().copied());
            }
        }
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

    pub(crate) fn connected_front_facing_vertices(
        &self,
        seed_triangle: u32,
        center: Vec3,
        radius: f32,
        view_direction: Vec3,
        scratch: &mut VertexTraversalScratch,
        result: &mut Vec<u32>,
    ) {
        result.clear();
        let Some(seed) = self.triangles.get(seed_triangle as usize) else {
            return;
        };
        if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return;
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
        scratch.begin(self.positions.len());
        for &vertex in seed {
            scratch.enqueue_once(vertex);
        }
        while let Some(vertex) = scratch.queue.pop_front() {
            let index = vertex as usize;
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
                scratch.enqueue_once(neighbor);
            }
        }
    }

    pub fn remesh_region(
        &mut self,
        vertices: &[u32],
        settings: RemeshSettings,
        recorder: &mut MeshEditRecorder,
    ) -> RemeshOutcome {
        if !settings.target_edge_length.is_finite()
            || settings.target_edge_length <= 0.0
            || settings.iterations == 0
        {
            return RemeshOutcome::default();
        }
        if self.topology.vertex_neighbors.len() != self.positions.len() {
            self.rebuild();
        }
        let mut active = vertices
            .iter()
            .copied()
            .filter(|&vertex| (vertex as usize) < self.positions.len())
            .collect::<HashSet<_>>();
        let core = active.iter().copied().collect::<Vec<_>>();
        for vertex in core {
            active.extend(
                self.topology.vertex_neighbors[vertex as usize]
                    .iter()
                    .copied(),
            );
        }
        if active.is_empty() {
            return RemeshOutcome::default();
        }

        let split_length = settings.target_edge_length * settings.split_threshold.max(1.01);
        let collapse_length = settings.target_edge_length
            * settings
                .collapse_threshold
                .clamp(0.01, settings.split_threshold.min(0.99));
        let mut stats = RemeshStats::default();
        let mut changes = MeshChangeSet::default();

        for _ in 0..settings.iterations {
            stats.iterations += 1;
            let mut phase_changes = MeshChangeSet::default();
            stats.splits += self.split_active_edges_batch(
                &mut active,
                split_length,
                recorder,
                &mut phase_changes,
            );
            changes.merge(phase_changes);

            let mut phase_changes = MeshChangeSet::default();
            stats.collapses += self.collapse_active_edges_batch(
                &mut active,
                collapse_length,
                recorder,
                &mut phase_changes,
            );
            changes.merge(phase_changes);

            if settings.enable_flips {
                let mut phase_changes = MeshChangeSet::default();
                stats.flips += self.flip_active_edges_batch(&active, recorder, &mut phase_changes);
                changes.merge(phase_changes);
            }

            let relaxation = settings.relaxation.clamp(0.0, 1.0);
            if relaxation > 0.0 {
                let mut phase_changes = MeshChangeSet::default();
                stats.relaxed_vertices +=
                    self.relax_active_vertices(&active, relaxation, recorder, &mut phase_changes);
                phase_changes.vertex_count = self.positions.len();
                phase_changes.face_count = self.triangles.len();
                phase_changes.normalize();
                changes.merge(phase_changes);
            }
        }

        changes.vertex_count = self.positions.len();
        changes.face_count = self.triangles.len();
        changes.normalize();
        RemeshOutcome { stats, changes }
    }

    /// Splits a maximal set of long edges whose incident faces do not overlap,
    /// then updates only the affected adjacency and BVH leaves.
    fn split_active_edges_batch(
        &mut self,
        active: &mut HashSet<u32>,
        threshold: f32,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> usize {
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
                let length_squared = self.positions[ai].distance_squared(self.positions[bi]);
                (length_squared > threshold * threshold).then_some((
                    edge,
                    [faces[0], faces[1]],
                    length_squared,
                ))
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

        // Interior edge splits add two faces and three unique edges apiece. Reserve
        // the exact batch growth before the first mutation so a large local batch
        // cannot trigger repeated topology-map reallocations halfway through it.
        self.topology
            .face_lookup
            .reserve(selected.len().saturating_mul(2));
        self.topology
            .edge_faces
            .reserve(selected.len().saturating_mul(3));

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

            let position = self.positions[a as usize].midpoint(self.positions[b as usize]);
            let mask = (self.mask[a as usize] + self.mask[b as usize]) * 0.5;
            let appended = self.append_vertex_local(position, mask, recorder, changes);
            debug_assert_eq!(appended, new_vertex);
            active.insert(new_vertex);
            self.replace_face_local(faces[0], first_a, recorder, changes);
            self.replace_face_local(faces[1], second_a, recorder, changes);
            self.append_face_local(first_b, recorder, changes);
            self.append_face_local(second_b, recorder, changes);
            split_count += 1;
        }
        if split_count != 0 {
            self.finish_local_changes(changes);
        }
        split_count
    }

    /// Collapses a maximal set of one-ring-disjoint short edges in one compacting pass.
    fn collapse_active_edges_batch(
        &mut self,
        active: &mut HashSet<u32>,
        threshold: f32,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> usize {
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
                let length_squared = self.positions[ai].distance_squared(self.positions[bi]);
                (length_squared < threshold * threshold).then_some(((a, b), length_squared))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut blocked = HashSet::new();
        let mut selected = Vec::new();
        for ((keep, remove), _) in candidates {
            if blocked.contains(&keep)
                || blocked.contains(&remove)
                || !self.collapse_candidate_is_safe(keep, remove)
            {
                continue;
            }
            selected.push((keep, remove));
            blocked.insert(keep);
            blocked.insert(remove);
            for &neighbor in self.topology.vertex_neighbors[keep as usize]
                .iter()
                .chain(&self.topology.vertex_neighbors[remove as usize])
            {
                blocked.insert(neighbor);
            }
        }
        if selected.is_empty() {
            return 0;
        }

        let mut deleted_faces = Vec::with_capacity(selected.len().saturating_mul(2));
        for &(keep, remove) in &selected {
            recorder.record_vertex(self, keep);
            self.positions[keep as usize] =
                self.positions[keep as usize].midpoint(self.positions[remove as usize]);
            self.mask[keep as usize] =
                (self.mask[keep as usize] + self.mask[remove as usize]) * 0.5;
            changes.dirty_vertices.push(keep);

            let incident = self.topology.vertex_triangles[remove as usize].clone();
            for face in incident {
                let replacement = self.triangles[face as usize]
                    .map(|vertex| if vertex == remove { keep } else { vertex });
                if replacement[0] == replacement[1]
                    || replacement[1] == replacement[2]
                    || replacement[2] == replacement[0]
                {
                    deleted_faces.push(face);
                } else {
                    self.replace_face_local(face, replacement, recorder, changes);
                }
            }
        }

        deleted_faces.sort_unstable_by(|left, right| right.cmp(left));
        deleted_faces.dedup();
        for face in deleted_faces {
            self.remove_face_dense(face, recorder, changes);
        }

        let mut removed_vertices = selected
            .iter()
            .map(|&(_, remove)| remove)
            .collect::<Vec<_>>();
        removed_vertices.sort_unstable_by(|left, right| right.cmp(left));
        for remove in removed_vertices {
            active.remove(&remove);
            if let Some((from, to)) = self.remove_vertex_dense(remove, recorder, changes)
                && active.remove(&from)
            {
                active.insert(to);
            }
        }
        self.finish_local_changes(changes);
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
        let Some(first_opposite) = opposite_vertex(self.triangles[faces[0] as usize], keep, remove)
        else {
            return false;
        };
        let Some(second_opposite) =
            opposite_vertex(self.triangles[faces[1] as usize], keep, remove)
        else {
            return false;
        };
        if first_opposite == second_opposite {
            return false;
        }
        let keep_neighbors = &self.topology.vertex_neighbors[keep as usize];
        let remove_neighbors = &self.topology.vertex_neighbors[remove as usize];
        let mut common_count = 0;
        let mut found_first = false;
        let mut found_second = false;
        let mut keep_index = 0;
        let mut remove_index = 0;
        while keep_index < keep_neighbors.len() && remove_index < remove_neighbors.len() {
            let keep_neighbor = keep_neighbors[keep_index];
            let remove_neighbor = remove_neighbors[remove_index];
            match keep_neighbor.cmp(&remove_neighbor) {
                std::cmp::Ordering::Less => keep_index += 1,
                std::cmp::Ordering::Greater => remove_index += 1,
                std::cmp::Ordering::Equal => {
                    common_count += 1;
                    if common_count > 2 {
                        return false;
                    }
                    found_first |= keep_neighbor == first_opposite;
                    found_second |= keep_neighbor == second_opposite;
                    keep_index += 1;
                    remove_index += 1;
                }
            }
        }
        if common_count != 2 || !found_first || !found_second {
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

    /// Flips a maximal set of face-disjoint edges and updates their local topology.
    fn flip_active_edges_batch(
        &mut self,
        active: &HashSet<u32>,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> usize {
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
                    || !active.contains(&c)
                    || !active.contains(&d)
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
        candidates.sort_unstable_by(|left, right| {
            right
                .4
                .total_cmp(&left.4)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut used_faces = HashSet::new();
        let mut new_edges = HashSet::new();
        let mut flip_count = 0;
        for (faces, new_edge, first, second, _) in candidates {
            if faces.iter().any(|face| used_faces.contains(face)) || !new_edges.insert(new_edge) {
                continue;
            }
            used_faces.insert(faces[0]);
            used_faces.insert(faces[1]);
            self.replace_face_local(faces[0], first, recorder, changes);
            self.replace_face_local(faces[1], second, recorder, changes);
            flip_count += 1;
        }
        if flip_count != 0 {
            self.finish_local_changes(changes);
        }
        flip_count
    }

    fn relax_active_vertices(
        &mut self,
        active: &HashSet<u32>,
        amount: f32,
        recorder: &mut MeshEditRecorder,
        changes: &mut MeshChangeSet,
    ) -> usize {
        let mut replacements = Vec::new();
        for &vertex in active {
            let index = vertex as usize;
            if index >= self.positions.len()
                || self.topology.vertex_in_protected_neighborhood(vertex)
                || self.topology.vertex_neighbors[index].is_empty()
            {
                continue;
            }
            let neighbors = &self.topology.vertex_neighbors[index];
            let average = neighbors
                .iter()
                .map(|&neighbor| self.positions[neighbor as usize])
                .sum::<Vec3>()
                / neighbors.len() as f32;
            let delta = average - self.positions[index];
            let normal = self.normals[index];
            let tangent_delta = if normal == Vec3::ZERO {
                delta
            } else {
                delta - normal * delta.dot(normal)
            };
            let replacement = self.positions[index] + tangent_delta * amount;
            if replacement.is_finite() {
                replacements.push((vertex, replacement));
            }
        }
        for &(vertex, replacement) in &replacements {
            recorder.record_vertex(self, vertex);
            self.positions[vertex as usize] = replacement;
            changes.dirty_vertices.push(vertex);
        }
        let updated = self.update_deformed_vertices(
            &replacements
                .iter()
                .map(|&(vertex, _)| vertex)
                .collect::<Vec<_>>(),
        );
        changes.dirty_vertices.extend(updated);
        replacements.len()
    }
}

impl MeshBvh {
    fn build(positions: &[Vec3], triangles: &[[u32; 3]]) -> Self {
        if triangles.is_empty() {
            return Self::default();
        }
        let estimated_leaves = triangles.len().div_ceil(BVH_LEAF_SIZE);
        let mut bvh = Self {
            nodes: Vec::with_capacity(editing_capacity(estimated_leaves.saturating_mul(2))),
            leaf_faces: Vec::with_capacity(editing_capacity(estimated_leaves)),
            triangle_leaves: filled_vec_with_headroom(triangles.len(), u32::MAX),
            ..Self::default()
        };
        let centroids = triangles
            .iter()
            .map(|&triangle| triangle_centroid(positions, triangle))
            .collect::<Vec<_>>();
        let mut faces = (0..triangles.len() as u32).collect::<Vec<_>>();
        bvh.root = bvh.build_node(positions, triangles, &centroids, &mut faces, u32::MAX);
        bvh
    }

    fn build_node(
        &mut self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        centroids: &[Vec3],
        faces: &mut [u32],
        parent: u32,
    ) -> u32 {
        let node_index = self.allocate_node();
        if faces.len() <= BVH_LEAF_SIZE {
            let leaf = self.allocate_leaf(faces.to_vec());
            let (min, max, _, _) = triangle_range_bounds(positions, triangles, faces);
            self.nodes[node_index as usize] = BvhNode {
                min,
                max,
                parent,
                leaf,
                height: 0,
                ..BvhNode::default()
            };
            for &triangle in faces.iter() {
                self.triangle_leaves[triangle as usize] = node_index;
            }
            return node_index;
        }

        let (centroid_min, centroid_max) = centroid_range_bounds(centroids, faces);
        let extent = centroid_max - centroid_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let middle = faces.len() / 2;
        faces.select_nth_unstable_by(middle, |&left, &right| {
            centroids[left as usize][axis].total_cmp(&centroids[right as usize][axis])
        });
        let (left_faces, right_faces) = faces.split_at_mut(middle);
        let left = self.build_node(positions, triangles, centroids, left_faces, node_index);
        let right = self.build_node(positions, triangles, centroids, right_faces, node_index);
        let left_node = self.nodes[left as usize];
        let right_node = self.nodes[right as usize];
        self.nodes[node_index as usize] = BvhNode {
            min: left_node.min.min(right_node.min),
            max: left_node.max.max(right_node.max),
            parent,
            left,
            right,
            leaf: u32::MAX,
            height: 1 + left_node.height.max(right_node.height),
        };
        node_index
    }

    #[cfg(test)]
    fn refit(&mut self, positions: &[Vec3], triangles: &[[u32; 3]]) {
        if self.root == u32::MAX || self.triangle_leaves.len() != triangles.len() {
            return;
        }
        self.refit_subtree(self.root, positions, triangles);
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
        if self.root == u32::MAX || self.triangle_leaves.len() != triangles.len() {
            *self = Self::build(positions, triangles);
            return;
        }
        let mut dirty_leaves = HashSet::with_capacity(affected_faces.len());
        for &face in affected_faces {
            let Some(&leaf) = self.triangle_leaves.get(face as usize) else {
                continue;
            };
            if leaf != u32::MAX {
                dirty_leaves.insert(leaf);
            }
        }
        let mut dirty_ancestors = HashSet::with_capacity(dirty_leaves.len());
        for &leaf in &dirty_leaves {
            self.refit_leaf(leaf, positions, triangles);
            let mut ancestor = self.nodes[leaf as usize].parent;
            while ancestor != u32::MAX && dirty_ancestors.insert(ancestor) {
                ancestor = self.nodes[ancestor as usize].parent;
            }
        }
        let mut dirty_ancestors = dirty_ancestors.into_iter().collect::<Vec<_>>();
        dirty_ancestors.sort_unstable_by_key(|&node| self.nodes[node as usize].height);
        for node in dirty_ancestors {
            self.refit_node_from_children(node);
        }
    }

    fn insert_face(&mut self, positions: &[Vec3], triangles: &[[u32; 3]], face: u32) {
        let face_index = face as usize;
        if face_index >= triangles.len() {
            return;
        }
        self.triangle_leaves.resize(triangles.len(), u32::MAX);
        if self.root == u32::MAX {
            let leaf = self.allocate_leaf(vec![face]);
            let node = self.allocate_node();
            let (min, max) = triangle_bounds(positions, triangles[face_index]);
            self.nodes[node as usize] = BvhNode {
                min,
                max,
                leaf,
                height: 0,
                ..BvhNode::default()
            };
            self.triangle_leaves[face_index] = node;
            self.root = node;
            return;
        }

        let (face_min, face_max) = triangle_bounds(positions, triangles[face_index]);
        let mut node = self.root;
        while !self.nodes[node as usize].is_leaf() {
            let current = self.nodes[node as usize];
            let left = self.nodes[current.left as usize];
            let right = self.nodes[current.right as usize];
            let left_cost = aabb_area(left.min.min(face_min), left.max.max(face_max))
                - aabb_area(left.min, left.max);
            let right_cost = aabb_area(right.min.min(face_min), right.max.max(face_max))
                - aabb_area(right.min, right.max);
            node = if left_cost <= right_cost {
                current.left
            } else {
                current.right
            };
        }

        let leaf_index = self.nodes[node as usize].leaf as usize;
        if self.leaf_faces[leaf_index].len() < BVH_LEAF_SIZE {
            self.leaf_faces[leaf_index].push(face);
            self.triangle_leaves[face_index] = node;
            self.refit_leaf(node, positions, triangles);
            self.refit_ancestors(self.nodes[node as usize].parent);
            return;
        }

        let parent = self.nodes[node as usize].parent;
        let old_leaf = self.nodes[node as usize].leaf;
        let mut faces = std::mem::take(&mut self.leaf_faces[old_leaf as usize]);
        faces.push(face);
        self.free_leaves.push(old_leaf);
        let mut centroid_min = Vec3::splat(f32::INFINITY);
        let mut centroid_max = Vec3::splat(f32::NEG_INFINITY);
        for &candidate in &faces {
            let centroid = triangle_centroid(positions, triangles[candidate as usize]);
            centroid_min = centroid_min.min(centroid);
            centroid_max = centroid_max.max(centroid);
        }
        let extent = centroid_max - centroid_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        faces.sort_unstable_by(|&left, &right| {
            triangle_centroid(positions, triangles[left as usize])[axis]
                .total_cmp(&triangle_centroid(positions, triangles[right as usize])[axis])
        });
        let right_faces = faces.split_off(faces.len() / 2);
        let left_node = self.create_leaf_node(positions, triangles, faces, node);
        let right_node = self.create_leaf_node(positions, triangles, right_faces, node);
        let left = self.nodes[left_node as usize];
        let right = self.nodes[right_node as usize];
        self.nodes[node as usize] = BvhNode {
            min: left.min.min(right.min),
            max: left.max.max(right.max),
            parent,
            left: left_node,
            right: right_node,
            leaf: u32::MAX,
            height: 1,
        };
        self.refit_ancestors(parent);
    }

    fn remove_face(&mut self, positions: &[Vec3], triangles: &[[u32; 3]], face: u32) {
        let Some(leaf_node) = self.triangle_leaves.get(face as usize).copied() else {
            return;
        };
        if leaf_node == u32::MAX {
            return;
        }
        let leaf = self.nodes[leaf_node as usize].leaf;
        let Some(position) = self.leaf_faces[leaf as usize]
            .iter()
            .position(|&candidate| candidate == face)
        else {
            return;
        };
        self.leaf_faces[leaf as usize].swap_remove(position);
        self.triangle_leaves[face as usize] = u32::MAX;
        if !self.leaf_faces[leaf as usize].is_empty() {
            self.refit_leaf(leaf_node, positions, triangles);
            self.refit_ancestors(self.nodes[leaf_node as usize].parent);
            return;
        }

        self.free_leaves.push(leaf);
        let parent = self.nodes[leaf_node as usize].parent;
        if parent == u32::MAX {
            self.free_node(leaf_node);
            self.root = u32::MAX;
            return;
        }
        let parent_node = self.nodes[parent as usize];
        let sibling = if parent_node.left == leaf_node {
            parent_node.right
        } else {
            parent_node.left
        };
        let grandparent = parent_node.parent;
        if grandparent == u32::MAX {
            self.root = sibling;
            self.nodes[sibling as usize].parent = u32::MAX;
        } else {
            let grandparent_node = &mut self.nodes[grandparent as usize];
            if grandparent_node.left == parent {
                grandparent_node.left = sibling;
            } else {
                grandparent_node.right = sibling;
            }
            self.nodes[sibling as usize].parent = grandparent;
        }
        self.free_node(leaf_node);
        self.free_node(parent);
        self.refit_ancestors(grandparent);
    }

    fn truncate_faces(&mut self, face_count: usize) {
        self.triangle_leaves.truncate(face_count);
    }

    fn create_leaf_node(
        &mut self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        faces: Vec<u32>,
        parent: u32,
    ) -> u32 {
        let leaf = self.allocate_leaf(faces);
        let node = self.allocate_node();
        let (min, max, _, _) =
            triangle_range_bounds(positions, triangles, &self.leaf_faces[leaf as usize]);
        self.nodes[node as usize] = BvhNode {
            min,
            max,
            parent,
            leaf,
            height: 0,
            ..BvhNode::default()
        };
        for &face in &self.leaf_faces[leaf as usize] {
            self.triangle_leaves[face as usize] = node;
        }
        node
    }

    fn allocate_node(&mut self) -> u32 {
        if let Some(node) = self.free_nodes.pop() {
            self.nodes[node as usize] = BvhNode::default();
            node
        } else {
            let node = self.nodes.len() as u32;
            self.nodes.push(BvhNode::default());
            node
        }
    }

    fn free_node(&mut self, node: u32) {
        self.nodes[node as usize] = BvhNode::default();
        self.free_nodes.push(node);
    }

    fn allocate_leaf(&mut self, faces: Vec<u32>) -> u32 {
        if let Some(leaf) = self.free_leaves.pop() {
            self.leaf_faces[leaf as usize] = faces;
            leaf
        } else {
            let leaf = self.leaf_faces.len() as u32;
            self.leaf_faces.push(faces);
            leaf
        }
    }

    #[cfg(test)]
    fn refit_subtree(
        &mut self,
        node: u32,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
    ) -> (Vec3, Vec3, i32) {
        let current = self.nodes[node as usize];
        let (min, max, height) = if current.is_leaf() {
            let (min, max, _, _) = triangle_range_bounds(
                positions,
                triangles,
                &self.leaf_faces[current.leaf as usize],
            );
            (min, max, 0)
        } else {
            let (left_min, left_max, left_height) =
                self.refit_subtree(current.left, positions, triangles);
            let (right_min, right_max, right_height) =
                self.refit_subtree(current.right, positions, triangles);
            (
                left_min.min(right_min),
                left_max.max(right_max),
                1 + left_height.max(right_height),
            )
        };
        self.nodes[node as usize].min = min;
        self.nodes[node as usize].max = max;
        self.nodes[node as usize].height = height;
        (min, max, height)
    }

    fn refit_leaf(&mut self, node: u32, positions: &[Vec3], triangles: &[[u32; 3]]) {
        let leaf = self.nodes[node as usize].leaf;
        let (min, max, _, _) =
            triangle_range_bounds(positions, triangles, &self.leaf_faces[leaf as usize]);
        self.nodes[node as usize].min = min;
        self.nodes[node as usize].max = max;
    }

    fn refit_ancestors(&mut self, mut node: u32) {
        while node != u32::MAX {
            let current = self.nodes[node as usize];
            if !current.is_live() || current.is_leaf() {
                break;
            }
            self.refit_node_from_children(node);
            node = current.parent;
        }
    }

    fn refit_node_from_children(&mut self, node: u32) {
        let current = self.nodes[node as usize];
        let left = self.nodes[current.left as usize];
        let right = self.nodes[current.right as usize];
        self.nodes[node as usize].min = left.min.min(right.min);
        self.nodes[node as usize].max = left.max.max(right.max);
        self.nodes[node as usize].height = 1 + left.height.max(right.height);
    }

    fn raycast(
        &self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        origin: Vec3,
        direction: Vec3,
        normals: &[Vec3],
    ) -> Option<RayHit> {
        if self.root == u32::MAX {
            return None;
        }
        let mut stack = Vec::from([self.root]);
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
                for &triangle_index in &self.leaf_faces[node.leaf as usize] {
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
        if self.root == u32::MAX {
            return None;
        }
        let mut stack = Vec::from([self.root]);
        let mut best_triangle = None;
        let mut best_distance = f32::INFINITY;
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index as usize];
            if point_aabb_distance_squared(point, node.min, node.max) > best_distance {
                continue;
            }
            if node.is_leaf() {
                for &triangle_index in &self.leaf_faces[node.leaf as usize] {
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

fn editing_capacity(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    len.saturating_add((len / EDIT_GROWTH_DIVISOR).max(MIN_EDIT_GROWTH))
}

fn bounded_hash_editing_capacity(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    len.saturating_add(bounded_hash_edit_growth(len))
}

fn bounded_hash_edit_growth(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (len / EDIT_GROWTH_DIVISOR).clamp(MIN_EDIT_GROWTH, MAX_HASH_EDIT_GROWTH)
}

fn reserve_for_editing<T>(values: &mut Vec<T>, len: usize) {
    let desired = editing_capacity(len);
    if values.capacity() < desired {
        values.reserve_exact(desired.saturating_sub(values.len()));
    }
}

fn reserve_edit_headroom<T>(values: &mut Vec<T>) {
    reserve_for_editing(values, values.len());
}

fn filled_vec_with_headroom<T: Clone>(len: usize, value: T) -> Vec<T> {
    let mut values = Vec::with_capacity(editing_capacity(len));
    values.resize(len, value);
    values
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

fn mesh_vertex_state(mesh: &Mesh, vertex: u32) -> Option<MeshVertexState> {
    let index = vertex as usize;
    mesh.positions
        .get(index)
        .copied()
        .map(|position| MeshVertexState {
            position,
            mask: mesh.mask.get(index).copied().unwrap_or(0.0),
        })
}

fn remove_small_value<const N: usize>(values: &mut SmallVec<[u32; N]>, value: u32)
where
    [u32; N]: smallvec::Array<Item = u32>,
{
    if let Some(position) = values.iter().position(|&candidate| candidate == value) {
        values.remove(position);
    }
}

fn insert_small_sorted<const N: usize>(values: &mut SmallVec<[u32; N]>, value: u32)
where
    [u32; N]: smallvec::Array<Item = u32>,
{
    match values.binary_search(&value) {
        Ok(_) => {}
        Err(position) => values.insert(position, value),
    }
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

fn triangle_bounds(positions: &[Vec3], triangle: [u32; 3]) -> (Vec3, Vec3) {
    let [a, b, c] = triangle.map(|index| positions[index as usize]);
    (a.min(b).min(c), a.max(b).max(c))
}

fn aabb_area(min: Vec3, max: Vec3) -> f32 {
    let extent = (max - min).max(Vec3::ZERO);
    2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
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

    fn remesh(mesh: &mut Mesh, vertices: &[u32], settings: RemeshSettings) -> RemeshOutcome {
        let mut recorder = MeshEditRecorder::new(mesh);
        mesh.remesh_region(vertices, settings, &mut recorder)
    }

    fn assert_edge_faces_match(left: &MeshTopology, right: &MeshTopology) {
        assert_eq!(left.edge_faces.len(), right.edge_faces.len());
        for (edge, left_faces) in &left.edge_faces {
            let mut left_faces = left_faces.to_vec();
            let mut right_faces = right.edge_faces[edge].to_vec();
            left_faces.sort_unstable();
            right_faces.sort_unstable();
            assert_eq!(left_faces, right_faces, "edge {edge:?}");
        }
    }

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
        let active = [0_u32, 1, 4, 5].into_iter().collect::<HashSet<_>>();
        let mut expected = mesh
            .topology
            .edge_faces
            .keys()
            .copied()
            .filter(|&(a, b)| active.contains(&a) && active.contains(&b))
            .collect::<Vec<_>>();
        expected.sort_unstable();

        let mut actual = mesh.topology.active_edges(&active);
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn edge_change_folding_preserves_the_final_sequential_state() {
        let edge = (2, 7);
        let mut changes = MeshChangeSet::default();
        changes.record_edge_change(edge, EdgeDelta::Removed);
        changes.record_edge_change(edge, EdgeDelta::Added);
        changes.record_edge_change(edge, EdgeDelta::Removed);
        changes.normalize();

        assert!(changes.added_edges.is_empty());
        assert_eq!(changes.removed_edges, [edge]);

        let mut next_frame = MeshChangeSet::default();
        next_frame.record_edge_change(edge, EdgeDelta::Added);
        next_frame.normalize();
        changes.merge(next_frame);
        changes.normalize();

        assert!(changes.added_edges.is_empty());
        assert!(changes.removed_edges.is_empty());
    }

    #[test]
    fn absorbing_a_recorder_filters_net_no_op_slots() {
        let mut mesh = octahedron();
        let original = mesh.positions[0];
        let mut stroke = MeshEditRecorder::new(&mesh);
        let mut sample = MeshEditRecorder::new(&mesh);
        sample.record_vertex(&mesh, 0);
        mesh.positions[0] += Vec3::splat(0.25);
        mesh.positions[0] = original;

        stroke.absorb_recorder(sample, &mesh);

        assert!(stroke.vertices.is_empty());
        assert!(stroke.finish(&mesh).is_empty());
    }

    #[test]
    fn high_valence_vertex_classification_uses_the_linear_fallback() {
        const RING_VERTICES: usize = 256;
        let mut positions = Vec::with_capacity(RING_VERTICES + 1);
        positions.push(Vec3::ZERO);
        positions.extend((0..RING_VERTICES).map(|index| {
            let angle = std::f32::consts::TAU * index as f32 / RING_VERTICES as f32;
            Vec3::new(angle.cos(), angle.sin(), 0.0)
        }));
        let triangles = (0..RING_VERTICES)
            .map(|index| {
                [
                    0,
                    index as u32 + 1,
                    ((index + 1) % RING_VERTICES) as u32 + 1,
                ]
            })
            .collect();
        let mesh = Mesh::new(positions, triangles).unwrap();

        assert!(mesh.topology.vertex_triangles[0].len() > INLINE_VERTEX_CLASSIFICATION_FACES);
        assert_eq!(mesh.classify_vertex(0), (false, false));
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
        let mut scratch = VertexTraversalScratch::default();
        let mut selected = Vec::new();
        mesh.connected_front_facing_vertices(
            0,
            Vec3::new(0.3, 0.3, 0.0),
            2.0,
            Vec3::NEG_Z,
            &mut scratch,
            &mut selected,
        );
        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&vertex| vertex < 3));
    }

    #[test]
    fn connected_selection_accepts_consistently_reversed_winding() {
        let mesh = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 2, 1]]).unwrap();
        let mut scratch = VertexTraversalScratch::default();
        let mut selected = Vec::new();
        mesh.connected_front_facing_vertices(
            0,
            Vec3::splat(0.2),
            2.0,
            Vec3::NEG_Z,
            &mut scratch,
            &mut selected,
        );
        assert_eq!(selected.len(), 3);
        let hit = mesh.raycast(Vec3::new(0.2, 0.2, 1.0), Vec3::NEG_Z).unwrap();
        assert!(hit.normal.dot(Vec3::NEG_Z) <= 0.0);
    }

    #[test]
    fn dynamic_remesh_splits_long_interior_edges_and_preserves_validity() {
        let mut mesh = octahedron();
        let original_vertices = mesh.positions.len();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let stats = remesh(
            &mut mesh,
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
    fn first_closed_mesh_split_uses_preallocated_topology_maps() {
        let mut mesh = octahedron();
        let edge_capacity = mesh.topology.edge_faces.capacity();
        let face_capacity = mesh.topology.face_lookup.capacity();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let outcome = remesh(
            &mut mesh,
            &active,
            RemeshSettings {
                target_edge_length: 0.9,
                iterations: 1,
                enable_flips: false,
                relaxation: 0.0,
                ..RemeshSettings::default()
            },
        );

        assert!(outcome.splits > 0);
        assert_eq!(mesh.topology.edge_faces.capacity(), edge_capacity);
        assert_eq!(mesh.topology.face_lookup.capacity(), face_capacity);
    }

    #[test]
    fn local_remesh_work_does_not_grow_with_disconnected_geometry() {
        let mut small = octahedron();
        let component = octahedron();
        let mut positions = component.positions.clone();
        let mut triangles = component.triangles.clone();
        for copy in 1..=256_u32 {
            let vertex_offset = positions.len() as u32;
            let translation = Vec3::X * copy as f32 * 4.0;
            positions.extend(
                component
                    .positions
                    .iter()
                    .map(|position| *position + translation),
            );
            triangles.extend(component.triangles.iter().map(|triangle| {
                [
                    triangle[0] + vertex_offset,
                    triangle[1] + vertex_offset,
                    triangle[2] + vertex_offset,
                ]
            }));
        }
        let mut large = Mesh::new(positions, triangles).unwrap();
        let active = (0..component.positions.len() as u32).collect::<Vec<_>>();
        let settings = RemeshSettings {
            target_edge_length: 0.9,
            iterations: 1,
            enable_flips: false,
            relaxation: 0.0,
            ..RemeshSettings::default()
        };

        let small_outcome = remesh(&mut small, &active, settings);
        let large_outcome = remesh(&mut large, &active, settings);

        assert_eq!(large_outcome.stats, small_outcome.stats);
        assert_eq!(
            large_outcome.changes.dirty_vertices.len(),
            small_outcome.changes.dirty_vertices.len()
        );
        assert_eq!(
            large_outcome.changes.dirty_faces.len(),
            small_outcome.changes.dirty_faces.len()
        );
        assert_eq!(
            large_outcome.changes.added_edges.len(),
            small_outcome.changes.added_edges.len()
        );
        assert_eq!(
            large_outcome.changes.removed_edges.len(),
            small_outcome.changes.removed_edges.len()
        );
        large.validate().unwrap();
    }

    #[test]
    fn dynamic_remesh_batches_safe_short_edge_collapses() {
        let mut mesh = octahedron();
        let original_vertices = mesh.positions.len();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let stats = remesh(
            &mut mesh,
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
        let stats = remesh(
            &mut mesh,
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
    fn topology_delta_restores_live_remesh_exactly() {
        let mut mesh = octahedron();
        let before = mesh.clone();
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let mut recorder = MeshEditRecorder::new(&mesh);
        let outcome = mesh.remesh_region(
            &active,
            RemeshSettings {
                target_edge_length: 0.9,
                enable_flips: true,
                relaxation: 0.0,
                ..RemeshSettings::default()
            },
            &mut recorder,
        );
        assert!(outcome.stats.splits > 0);
        let after = mesh.clone();
        let delta = recorder.finish(&mesh);

        delta.apply_before(&mut mesh);
        assert_eq!(mesh.positions, before.positions);
        assert_eq!(mesh.triangles, before.triangles);
        assert_eq!(mesh.mask, before.mask);
        assert_edge_faces_match(&mesh.topology, &before.topology);
        assert!(
            mesh.raycast(Vec3::new(0.2, 0.2, 3.0), Vec3::NEG_Z)
                .is_some()
        );

        delta.apply_after(&mut mesh);
        assert_eq!(mesh.positions, after.positions);
        assert_eq!(mesh.triangles, after.triangles);
        assert_eq!(mesh.mask, after.mask);
        assert_edge_faces_match(&mesh.topology, &after.topology);
        assert!(
            mesh.raycast(Vec3::new(0.2, 0.2, 3.0), Vec3::NEG_Z)
                .is_some()
        );
    }

    #[test]
    fn remeshing_protects_open_boundaries() {
        let mut mesh = square();
        let before = mesh.clone();
        let active = [0, 1, 2, 3];
        let stats = remesh(
            &mut mesh,
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
        let stats = remesh(
            &mut mesh,
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
        let stats = remesh(
            &mut mesh,
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
