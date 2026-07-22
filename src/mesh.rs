use std::collections::VecDeque;

use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use thiserror::Error;

pub type EdgeKey = (u32, u32);

const TRIANGLE_RELATIVE_EPSILON: f32 = 1.0e-12;
const BVH_LEAF_SIZE: usize = 8;
const RAY_BARYCENTRIC_AMBIGUITY_EPSILON: f64 = 1.0e-10;

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
    #[error("triangle {triangle} is degenerate")]
    DegenerateTriangle { triangle: usize },
    #[error("triangle {triangle} duplicates an earlier face")]
    DuplicateTriangle { triangle: usize },
    #[error("mask value {index} is non-finite or outside the inclusive range 0 to 1")]
    InvalidMask { index: usize },
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
pub struct ClosestSurfacePoint {
    pub triangle: u32,
    pub point: Vec3,
    pub barycentric: Vec3,
    pub distance: f32,
}

/// The result of counting every forward intersection along a ray.
///
/// An edge or vertex hit is deliberately reported as ambiguous instead of
/// assigning ownership to one incident triangle. Callers performing parity
/// tests can retry with another deterministic direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RayIntersectionCount {
    pub intersections: usize,
    pub ambiguous: bool,
}

#[derive(Clone, Debug)]
pub struct MeshBvh {
    root: u32,
    nodes: Vec<BvhNode>,
    leaf_faces: Vec<Vec<u32>>,
    triangle_leaves: Vec<u32>,
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
        let Self {
            positions,
            triangles,
            vertices,
            input_vertices,
            input_triangles,
        } = self;
        drop(vertices);
        let unique_vertices = positions.len();
        let mut mesh = Mesh {
            positions,
            triangles,
            normals: vec![Vec3::ZERO; unique_vertices],
            mask: vec![0.0; unique_vertices],
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
    /// Builds a mesh from already-indexed geometry without repairing or
    /// otherwise rewriting the supplied positions, triangles, or mask.
    ///
    /// This is the installation path for generated geometry whose topology has
    /// already been validated. Unlike [`Mesh::rebuild`], invalid or duplicate
    /// faces are rejected instead of being removed silently.
    pub(crate) fn from_indexed(
        positions: Vec<Vec3>,
        triangles: Vec<[u32; 3]>,
        mask: Vec<f32>,
    ) -> Result<Self, MeshError> {
        validate_positions_and_indices(&positions, &triangles)?;
        if mask.len() != positions.len() {
            return Err(MeshError::MaskCountMismatch {
                actual: mask.len(),
                expected: positions.len(),
            });
        }
        if let Some(index) = mask
            .iter()
            .position(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
        {
            return Err(MeshError::InvalidMask { index });
        }

        let mut unique_faces = HashSet::with_capacity(triangles.len());
        for (triangle_index, &triangle) in triangles.iter().enumerate() {
            if !triangle_is_valid(&positions, triangle) {
                return Err(MeshError::DegenerateTriangle {
                    triangle: triangle_index,
                });
            }
            if !unique_faces.insert(sorted_triangle(triangle)) {
                return Err(MeshError::DuplicateTriangle {
                    triangle: triangle_index,
                });
            }
        }

        let topology = MeshTopology::build(&positions, &triangles);
        let mut mesh = Self {
            normals: vec![Vec3::ZERO; positions.len()],
            positions,
            triangles,
            mask,
            topology,
        };
        mesh.recompute_normals_without_bvh_refit();
        Ok(mesh)
    }

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
        if let Some(index) = self
            .mask
            .iter()
            .position(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
        {
            return Err(MeshError::InvalidMask { index });
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
        }) {
            Some(affected_faces)
        } else {
            None
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

    /// Returns the nearest point on the surface, choosing the lowest triangle
    /// index when multiple faces are exactly equidistant.
    pub fn closest_surface_point(&self, point: Vec3) -> Option<ClosestSurfacePoint> {
        if !point.is_finite() {
            return None;
        }
        self.topology
            .bvh
            .closest_surface_point(&self.positions, &self.triangles, point)
    }

    pub fn nearest_triangle(&self, point: Vec3) -> Option<u32> {
        self.closest_surface_point(point)
            .map(|closest| closest.triangle)
    }

    pub(crate) fn ray_intersection_count(
        &self,
        origin: Vec3,
        direction: Vec3,
    ) -> Option<RayIntersectionCount> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite() {
            return None;
        }
        Some(self.topology.bvh.ray_intersection_count(
            &self.positions,
            &self.triangles,
            origin,
            direction,
        ))
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
}

impl MeshBvh {
    fn build(positions: &[Vec3], triangles: &[[u32; 3]]) -> Self {
        if triangles.is_empty() {
            return Self::default();
        }
        let estimated_leaves = triangles.len().div_ceil(BVH_LEAF_SIZE);
        let mut bvh = Self {
            nodes: Vec::with_capacity(estimated_leaves.saturating_mul(2)),
            leaf_faces: Vec::with_capacity(estimated_leaves),
            triangle_leaves: vec![u32::MAX; triangles.len()],
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

    fn allocate_node(&mut self) -> u32 {
        let node = self.nodes.len() as u32;
        self.nodes.push(BvhNode::default());
        node
    }

    fn allocate_leaf(&mut self, faces: Vec<u32>) -> u32 {
        let leaf = self.leaf_faces.len() as u32;
        self.leaf_faces.push(faces);
        leaf
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
        let mut stack = SmallVec::<[u32; 64]>::new();
        stack.push(self.root);
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

    fn closest_surface_point(
        &self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        point: Vec3,
    ) -> Option<ClosestSurfacePoint> {
        if self.root == u32::MAX {
            return None;
        }
        let mut stack = SmallVec::<[u32; 64]>::new();
        stack.push(self.root);
        let mut best = None;
        let mut best_distance_squared = f32::INFINITY;
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index as usize];
            if point_aabb_distance_squared(point, node.min, node.max) > best_distance_squared {
                continue;
            }
            if node.is_leaf() {
                for &triangle_index in &self.leaf_faces[node.leaf as usize] {
                    let triangle = triangles[triangle_index as usize];
                    let vertices = triangle.map(|index| positions[index as usize]);
                    let (surface_point, barycentric) = closest_point_on_triangle(point, vertices);
                    let distance_squared = point.distance_squared(surface_point);
                    let replaces_best = distance_squared < best_distance_squared
                        || (distance_squared == best_distance_squared
                            && best.is_none_or(|closest: ClosestSurfacePoint| {
                                triangle_index < closest.triangle
                            }));
                    if replaces_best {
                        best_distance_squared = distance_squared;
                        best = Some(ClosestSurfacePoint {
                            triangle: triangle_index,
                            point: surface_point,
                            barycentric,
                            distance: distance_squared.sqrt(),
                        });
                    }
                }
            } else {
                let left = self.nodes[node.left as usize];
                let right = self.nodes[node.right as usize];
                let left_distance = point_aabb_distance_squared(point, left.min, left.max);
                let right_distance = point_aabb_distance_squared(point, right.min, right.max);
                if left_distance < right_distance {
                    if right_distance <= best_distance_squared {
                        stack.push(node.right);
                    }
                    if left_distance <= best_distance_squared {
                        stack.push(node.left);
                    }
                } else {
                    if left_distance <= best_distance_squared {
                        stack.push(node.left);
                    }
                    if right_distance <= best_distance_squared {
                        stack.push(node.right);
                    }
                }
            }
        }
        best
    }

    fn ray_intersection_count(
        &self,
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        origin: Vec3,
        direction: Vec3,
    ) -> RayIntersectionCount {
        if self.root == u32::MAX {
            return RayIntersectionCount::default();
        }

        let mut result = RayIntersectionCount::default();
        let mut stack = SmallVec::<[u32; 64]>::new();
        stack.push(self.root);
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index as usize];
            if ray_aabb(origin, direction, node.min, node.max).is_none() {
                continue;
            }
            if node.is_leaf() {
                for &triangle_index in &self.leaf_faces[node.leaf as usize] {
                    let triangle = triangles[triangle_index as usize];
                    let vertices = triangle.map(|index| positions[index as usize]);
                    match ray_triangle_for_parity(origin, direction, vertices) {
                        ParityRayIntersection::Miss => {}
                        ParityRayIntersection::Ambiguous => result.ambiguous = true,
                        ParityRayIntersection::Hit {
                            distance,
                            minimum_barycentric,
                        } => {
                            result.intersections += 1;
                            result.ambiguous |= distance == 0.0
                                || minimum_barycentric <= RAY_BARYCENTRIC_AMBIGUITY_EPSILON;
                        }
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        result
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum ParityRayIntersection {
    Miss,
    Ambiguous,
    Hit {
        distance: f64,
        minimum_barycentric: f64,
    },
}

/// A parity-specific ray/triangle test evaluated in f64.
///
/// Picking uses a scale-aware parallel threshold for stable viewport behavior,
/// but containment parity must not silently discard a valid shallow crossing.
/// Exact or numerically coplanar rays are marked ambiguous so the caller can
/// retry a deterministic fallback direction.
fn ray_triangle_for_parity(
    origin: Vec3,
    direction: Vec3,
    [a, b, c]: [Vec3; 3],
) -> ParityRayIntersection {
    let origin = origin.as_dvec3();
    let direction = direction.as_dvec3();
    let a = a.as_dvec3();
    let edge_ab = b.as_dvec3() - a;
    let edge_ac = c.as_dvec3() - a;
    let perpendicular = direction.cross(edge_ac);
    let determinant = edge_ab.dot(perpendicular);
    let determinant_scale = edge_ab.length() * perpendicular.length();
    if determinant.abs() <= determinant_scale * f64::EPSILON * 16.0 {
        let normal = edge_ab.cross(edge_ac);
        let origin_offset = origin - a;
        let plane_scale = normal.length()
            * origin_offset
                .length()
                .max(edge_ab.length())
                .max(edge_ac.length());
        return if origin_offset.dot(normal).abs() <= plane_scale * f64::EPSILON * 16.0 {
            ParityRayIntersection::Ambiguous
        } else {
            ParityRayIntersection::Miss
        };
    }

    let inverse = determinant.recip();
    let origin_offset = origin - a;
    let v_weight = origin_offset.dot(perpendicular) * inverse;
    let cross = origin_offset.cross(edge_ab);
    let w_weight = direction.dot(cross) * inverse;
    let u_weight = 1.0 - v_weight - w_weight;
    const BARYCENTRIC_TOLERANCE: f64 = 1.0e-12;
    if u_weight < -BARYCENTRIC_TOLERANCE
        || v_weight < -BARYCENTRIC_TOLERANCE
        || w_weight < -BARYCENTRIC_TOLERANCE
        || u_weight > 1.0 + BARYCENTRIC_TOLERANCE
        || v_weight > 1.0 + BARYCENTRIC_TOLERANCE
        || w_weight > 1.0 + BARYCENTRIC_TOLERANCE
    {
        return ParityRayIntersection::Miss;
    }
    let distance = edge_ac.dot(cross) * inverse;
    if distance < 0.0 || !distance.is_finite() {
        return ParityRayIntersection::Miss;
    }
    ParityRayIntersection::Hit {
        distance,
        minimum_barycentric: u_weight.min(v_weight).min(w_weight),
    }
}

fn point_aabb_distance_squared(point: Vec3, min: Vec3, max: Vec3) -> f32 {
    let below = (min - point).max(Vec3::ZERO);
    let above = (point - max).max(Vec3::ZERO);
    (below + above).length_squared()
}

// Closest-point regions from Real-Time Collision Detection, Christer Ericson.
fn closest_point_on_triangle(point: Vec3, [a, b, c]: [Vec3; 3]) -> (Vec3, Vec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Vec3::X);
    }

    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Vec3::Y);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Vec3::new(1.0 - v, v, 0.0));
    }

    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Vec3::Z);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Vec3::new(1.0 - w, 0.0, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let edge = c - b;
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + edge * w, Vec3::new(0.0, 1.0 - w, w));
    }

    let denominator = (va + vb + vc).recip();
    let v = vb * denominator;
    let w = vc * denominator;
    (a + ab * v + ac * w, Vec3::new(1.0 - v - w, v, w))
}

#[cfg(test)]
fn point_triangle_distance_squared(point: Vec3, triangle: [Vec3; 3]) -> f32 {
    point.distance_squared(closest_point_on_triangle(point, triangle).0)
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
        let mut mesh = square();
        mesh.mask[0] = f32::NAN;
        assert!(matches!(
            mesh.validate(),
            Err(MeshError::InvalidMask { index: 0 })
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
    fn closest_surface_point_reports_face_edge_and_vertex_barycentrics() {
        let mesh = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).unwrap();

        let face = mesh
            .closest_surface_point(Vec3::new(0.25, 0.25, 2.0))
            .unwrap();
        assert_eq!(face.triangle, 0);
        assert!(face.point.abs_diff_eq(Vec3::new(0.25, 0.25, 0.0), 1.0e-6));
        assert!(
            face.barycentric
                .abs_diff_eq(Vec3::new(0.5, 0.25, 0.25), 1.0e-6)
        );
        assert!((face.distance - 2.0).abs() < 1.0e-6);

        let edge = mesh
            .closest_surface_point(Vec3::new(0.75, 0.75, 0.0))
            .unwrap();
        assert!(edge.point.abs_diff_eq(Vec3::new(0.5, 0.5, 0.0), 1.0e-6));
        assert!(
            edge.barycentric
                .abs_diff_eq(Vec3::new(0.0, 0.5, 0.5), 1.0e-6)
        );

        let vertex = mesh
            .closest_surface_point(Vec3::new(-1.0, -1.0, 0.0))
            .unwrap();
        assert_eq!(vertex.point, Vec3::ZERO);
        assert_eq!(vertex.barycentric, Vec3::X);
    }

    #[test]
    fn closest_surface_point_breaks_exact_ties_by_triangle_index() {
        let mesh = Mesh::new(
            vec![
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(0.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(0.0, 1.0, -1.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .unwrap();

        let closest = mesh
            .closest_surface_point(Vec3::new(0.25, 0.25, 0.0))
            .unwrap();
        assert_eq!(closest.triangle, 0);
        assert_eq!(mesh.nearest_triangle(Vec3::new(0.25, 0.25, 0.0)), Some(0));
    }

    #[test]
    fn ray_intersection_count_marks_edge_and_vertex_hits_ambiguous() {
        let mesh = square();
        let interior = mesh
            .ray_intersection_count(Vec3::new(0.5, 0.25, 1.0), Vec3::NEG_Z)
            .unwrap();
        assert_eq!(interior.intersections, 1);
        assert!(!interior.ambiguous);

        let shared_edge = mesh
            .ray_intersection_count(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z)
            .unwrap();
        assert_eq!(shared_edge.intersections, 2);
        assert!(shared_edge.ambiguous);

        let vertex = mesh
            .ray_intersection_count(Vec3::new(1.0, 1.0, 1.0), Vec3::NEG_Z)
            .unwrap();
        assert!(vertex.intersections >= 1);
        assert!(vertex.ambiguous);
    }

    #[test]
    fn ray_intersection_count_keeps_valid_shallow_crossings() {
        let mesh = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).unwrap();
        let direction = Vec3::new(1.0, 0.0, -5.0e-8).normalize();
        let intersection = mesh
            .ray_intersection_count(Vec3::new(0.1, 0.25, 1.0e-8), direction)
            .unwrap();

        assert_eq!(intersection.intersections, 1);
        assert!(!intersection.ambiguous);
    }

    #[test]
    fn indexed_constructor_rejects_faces_instead_of_repairing_them() {
        let duplicate = Mesh::from_indexed(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            vec![[0, 1, 2], [2, 1, 0]],
            vec![0.0, 0.5, 1.0],
        );
        assert!(matches!(
            duplicate,
            Err(MeshError::DuplicateTriangle { triangle: 1 })
        ));

        let mesh = Mesh::from_indexed(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            vec![[0, 2, 1]],
            vec![0.0, 0.5, 1.0],
        )
        .unwrap();
        assert_eq!(mesh.triangles, vec![[0, 2, 1]]);
        assert_eq!(mesh.mask, vec![0.0, 0.5, 1.0]);
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
    fn million_face_mesh_build_pick_and_deform_refresh() {
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
        mesh.validate().unwrap();

        eprintln!(
            "million-face benchmark: build={build_elapsed:?}, pick={pick_elapsed:?}, deform_refresh={deform_refresh_elapsed:?}"
        );
    }
}
