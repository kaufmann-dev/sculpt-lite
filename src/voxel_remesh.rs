use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicU8, Ordering},
};

use glam::Vec3;
use hashbrown::HashMap;
use manifold_rust::{
    linalg::Vec3 as ManifoldVec3,
    manifold::Manifold,
    types::{Box as ManifoldBox, Error as ManifoldError},
};
use thiserror::Error;

use crate::mesh::{Mesh, MeshError};

pub(crate) const MIN_REMESH_RESOLUTION: u32 = 32;
pub(crate) const MAX_REMESH_RESOLUTION: u32 = 192;
pub(crate) const DEFAULT_REMESH_RESOLUTION: u32 = 96;

const EXTRACTION_PADDING_VOXELS: f64 = 2.0;

// Irrational-looking, non-axis-aligned directions make exact edge and vertex hits
// uncommon. The mesh query still identifies those hits and forces both fallbacks.
const PRIMARY_RAY: Vec3 = Vec3::new(0.782_437_4, 0.431_856_3, 0.448_135_2);
const FALLBACK_RAY_A: Vec3 = Vec3::new(-0.327_128_2, 0.864_652_3, 0.382_697_5);
const FALLBACK_RAY_B: Vec3 = Vec3::new(0.238_619_7, -0.397_194_8, 0.886_220_9);

const QUERY_OK: u8 = 0;
const QUERY_CLOSEST_POINT_FAILED: u8 = 1;
const QUERY_RAY_FAILED: u8 = 2;
const QUERY_AMBIGUOUS: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VoxelRemeshSettings {
    pub resolution: u32,
}

impl Default for VoxelRemeshSettings {
    fn default() -> Self {
        Self {
            resolution: DEFAULT_REMESH_RESOLUTION,
        }
    }
}

#[derive(Debug)]
pub(crate) struct VoxelRemeshOutput {
    pub mesh: Mesh,
    pub voxel_size: f32,
    pub source_faces: usize,
    pub output_faces: usize,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum VoxelRemeshError {
    #[error(
        "voxel remesh resolution {resolution} is outside the supported range {MIN_REMESH_RESOLUTION}–{MAX_REMESH_RESOLUTION}"
    )]
    InvalidResolution { resolution: u32 },
    #[error("the source mesh is invalid: {source}")]
    InvalidSource {
        #[source]
        source: MeshError,
    },
    #[error("the source mesh is empty")]
    EmptySource,
    #[error(
        "voxel remeshing requires a two-manifold source ({edges} non-manifold edges, {vertices} non-manifold vertices)"
    )]
    NonManifoldSource { edges: usize, vertices: usize },
    #[error("voxel remeshing requires a closed source ({boundary_edges} boundary edges)")]
    OpenSource { boundary_edges: usize },
    #[error("the source bounds have no finite, non-zero longest axis")]
    DegenerateBounds,
    #[error("the padded extraction bounds cannot be represented by the mesh coordinate type")]
    UnrepresentableBounds,
    #[error("voxel size {voxel_size} is below mesh-coordinate precision at the extraction bounds")]
    InsufficientCoordinatePrecision { voxel_size: f64 },
    #[error("the nearest-surface query failed during signed-distance evaluation")]
    ClosestPointQueryFailed,
    #[error("the ray-containment query failed during signed-distance evaluation")]
    RayContainmentQueryFailed,
    #[error("ray containment remained ambiguous after two deterministic fallback rays")]
    AmbiguousContainment,
    #[error("voxel extraction panicked: {message}")]
    ExtractionPanicked { message: String },
    #[error("voxel extraction failed: {status}")]
    ExtractionFailed { status: String },
    #[error("voxel extraction produced an empty mesh")]
    EmptyOutput,
    #[error("voxel extraction produced an invalid vertex-property layout")]
    InvalidOutputLayout,
    #[error("output vertex {index} has non-finite coordinates")]
    NonFiniteOutputVertex { index: usize },
    #[error(
        "output triangle {triangle} references vertex {index}, but there are only {vertex_count} output vertices"
    )]
    OutputIndexOutOfBounds {
        triangle: usize,
        index: u32,
        vertex_count: usize,
    },
    #[error("output triangle {triangle} is degenerate")]
    DegenerateOutputTriangle { triangle: usize },
    #[error("output edge ({a}, {b}) belongs to {faces} faces instead of exactly two")]
    OutputEdgeUseCount { a: u32, b: u32, faces: u32 },
    #[error("output edge ({a}, {b}) has inconsistent face winding")]
    InconsistentOutputWinding { a: u32, b: u32 },
    #[error("output vertex {vertex} is not referenced by a triangle")]
    UnreferencedOutputVertex { vertex: u32 },
    #[error("output vertex {vertex} has a disconnected incident-face fan")]
    NonManifoldOutputVertex { vertex: u32 },
    #[error("could not reproject the source mask onto output vertex {vertex}")]
    MaskReprojectionFailed { vertex: usize },
    #[error("the validated output mesh could not be installed: {source}")]
    OutputMesh {
        #[source]
        source: MeshError,
    },
}

/// Checks whether a mesh is a supported voxel-remesh source without modifying it.
pub(crate) fn source_eligibility(mesh: &Mesh) -> Result<(), VoxelRemeshError> {
    mesh.validate()
        .map_err(|source| VoxelRemeshError::InvalidSource { source })?;
    if mesh.positions.is_empty() || mesh.triangles.is_empty() {
        return Err(VoxelRemeshError::EmptySource);
    }
    let non_manifold_edges = mesh.topology.non_manifold_edge_count();
    let non_manifold_vertices = mesh.topology.non_manifold_vertex_count();
    if non_manifold_edges != 0 || non_manifold_vertices != 0 {
        return Err(VoxelRemeshError::NonManifoldSource {
            edges: non_manifold_edges,
            vertices: non_manifold_vertices,
        });
    }
    let boundary_edges = mesh.topology.boundary_edge_count();
    if boundary_edges != 0 {
        return Err(VoxelRemeshError::OpenSource { boundary_edges });
    }
    Ok(())
}

/// Rebuilds a closed two-manifold mesh from its signed-distance field.
///
/// The caller is expected to run this on the mesh worker. The source is only
/// borrowed, and a complete validated replacement is returned transactionally.
pub(crate) fn voxel_remesh(
    source: &Mesh,
    settings: VoxelRemeshSettings,
) -> Result<VoxelRemeshOutput, VoxelRemeshError> {
    source_eligibility(source)?;
    let grid = extraction_grid(source, settings)?;
    let query_failure = AtomicU8::new(QUERY_OK);

    let extraction = catch_unwind(AssertUnwindSafe(|| {
        let manifold = Manifold::level_set(
            |point| signed_distance(source, point, &query_failure),
            grid.bounds,
            grid.voxel_size,
        );
        let status = manifold.status();
        let mesh = manifold.get_mesh_gl(-1);
        (status, mesh)
    }))
    .map_err(|payload| VoxelRemeshError::ExtractionPanicked {
        message: panic_payload_message(payload),
    })?;

    match query_failure.load(Ordering::SeqCst) {
        QUERY_OK => {}
        QUERY_CLOSEST_POINT_FAILED => return Err(VoxelRemeshError::ClosestPointQueryFailed),
        QUERY_RAY_FAILED => return Err(VoxelRemeshError::RayContainmentQueryFailed),
        QUERY_AMBIGUOUS => return Err(VoxelRemeshError::AmbiguousContainment),
        _ => return Err(VoxelRemeshError::RayContainmentQueryFailed),
    }

    let (status, extracted) = extraction;
    if status != ManifoldError::NoError {
        return Err(VoxelRemeshError::ExtractionFailed {
            status: status.to_str().to_owned(),
        });
    }
    if extracted.num_vert() == 0 || extracted.num_tri() == 0 {
        return Err(VoxelRemeshError::EmptyOutput);
    }
    if extracted.num_prop < 3
        || extracted.vert_properties.len() % extracted.num_prop as usize != 0
        || extracted.tri_verts.len() % 3 != 0
    {
        return Err(VoxelRemeshError::InvalidOutputLayout);
    }

    let positions = extracted
        .vert_properties
        .chunks_exact(extracted.num_prop as usize)
        .enumerate()
        .map(|(index, properties)| {
            let position = Vec3::new(properties[0], properties[1], properties[2]);
            if position.is_finite() {
                Ok(position)
            } else {
                Err(VoxelRemeshError::NonFiniteOutputVertex { index })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let triangles = extracted
        .tri_verts
        .chunks_exact(3)
        .map(|triangle| [triangle[0], triangle[1], triangle[2]])
        .collect::<Vec<_>>();

    validate_output(&positions, &triangles)?;
    let masks = reproject_masks(source, &positions)?;
    let output_faces = triangles.len();
    let mesh = Mesh::from_indexed(positions, triangles, masks)
        .map_err(|source| VoxelRemeshError::OutputMesh { source })?;

    Ok(VoxelRemeshOutput {
        mesh,
        voxel_size: grid.voxel_size as f32,
        source_faces: source.triangles.len(),
        output_faces,
    })
}

#[derive(Clone, Copy)]
struct ExtractionGrid {
    bounds: ManifoldBox,
    voxel_size: f64,
}

fn extraction_grid(
    mesh: &Mesh,
    settings: VoxelRemeshSettings,
) -> Result<ExtractionGrid, VoxelRemeshError> {
    if !(MIN_REMESH_RESOLUTION..=MAX_REMESH_RESOLUTION).contains(&settings.resolution) {
        return Err(VoxelRemeshError::InvalidResolution {
            resolution: settings.resolution,
        });
    }
    let (min, max) = referenced_surface_bounds(mesh).ok_or(VoxelRemeshError::DegenerateBounds)?;
    let longest_axis = (max - min).max_element() as f64;
    if !longest_axis.is_finite() || longest_axis <= 0.0 {
        return Err(VoxelRemeshError::DegenerateBounds);
    }
    let voxel_size = longest_axis / settings.resolution as f64;
    let padding = voxel_size * EXTRACTION_PADDING_VOXELS;
    let padded_min = ManifoldVec3::new(
        min.x as f64 - padding,
        min.y as f64 - padding,
        min.z as f64 - padding,
    );
    let padded_max = ManifoldVec3::new(
        max.x as f64 + padding,
        max.y as f64 + padding,
        max.z as f64 + padding,
    );
    if !voxel_size.is_finite()
        || voxel_size <= 0.0
        || !manifold_point_fits_mesh_coordinates(padded_min)
        || !manifold_point_fits_mesh_coordinates(padded_max)
    {
        return Err(VoxelRemeshError::UnrepresentableBounds);
    }
    if !voxel_step_is_distinguishable(padded_min, padded_max, voxel_size) {
        return Err(VoxelRemeshError::InsufficientCoordinatePrecision { voxel_size });
    }
    Ok(ExtractionGrid {
        bounds: ManifoldBox::from_points(padded_min, padded_max),
        voxel_size,
    })
}

fn voxel_step_is_distinguishable(
    minimum: ManifoldVec3,
    maximum: ManifoldVec3,
    voxel_size: f64,
) -> bool {
    [
        (minimum.x, maximum.x),
        (minimum.y, maximum.y),
        (minimum.z, maximum.z),
    ]
    .into_iter()
    .all(|(minimum, maximum)| {
        (minimum + voxel_size) as f32 != minimum as f32
            && (maximum - voxel_size) as f32 != maximum as f32
    })
}

fn referenced_surface_bounds(mesh: &Mesh) -> Option<(Vec3, Vec3)> {
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for triangle in &mesh.triangles {
        for &vertex in triangle {
            let &position = mesh.positions.get(vertex as usize)?;
            bounds = Some(bounds.map_or((position, position), |(minimum, maximum)| {
                (minimum.min(position), maximum.max(position))
            }));
        }
    }
    bounds
}

fn manifold_point_fits_mesh_coordinates(point: ManifoldVec3) -> bool {
    [point.x, point.y, point.z]
        .into_iter()
        .all(|coordinate| coordinate.is_finite() && (coordinate as f32).is_finite())
}

fn signed_distance(source: &Mesh, point: ManifoldVec3, failure: &AtomicU8) -> f64 {
    let point = Vec3::new(point.x as f32, point.y as f32, point.z as f32);
    let Some(closest) = source.closest_surface_point(point) else {
        record_query_failure(failure, QUERY_CLOSEST_POINT_FAILED);
        return -1.0;
    };
    let distance = closest.distance as f64;
    // Grid coordinates are generated in f64 and then queried against an f32
    // mesh. A point mathematically on the source surface can therefore retain
    // a few f32 ulps of unsigned distance. Its sign is undefined, and casting
    // rays from it makes every direction correctly report a zero-distance
    // ambiguity. Treat that rounding band as the zero set before ray parity.
    if distance <= surface_roundoff(source, closest.triangle, point) {
        return 0.0;
    }

    let Some(primary) = source.ray_intersection_count(point, PRIMARY_RAY) else {
        record_query_failure(failure, QUERY_RAY_FAILED);
        return -distance;
    };
    let inside = if primary.ambiguous {
        // Both fallbacks are always evaluated so the result cannot depend on
        // which fallback happens to resolve first.
        let fallback_a = source.ray_intersection_count(point, FALLBACK_RAY_A);
        let fallback_b = source.ray_intersection_count(point, FALLBACK_RAY_B);
        match (fallback_a, fallback_b) {
            (Some(a), Some(b)) => match (unambiguous_parity(a), unambiguous_parity(b)) {
                (Some(a_inside), Some(b_inside)) if a_inside == b_inside => a_inside,
                (Some(a_inside), None) => a_inside,
                (None, Some(b_inside)) => b_inside,
                _ => {
                    record_query_failure(failure, QUERY_AMBIGUOUS);
                    false
                }
            },
            _ => {
                record_query_failure(failure, QUERY_RAY_FAILED);
                false
            }
        }
    } else {
        primary.intersections % 2 == 1
    };

    if inside { distance } else { -distance }
}

fn surface_roundoff(source: &Mesh, triangle: u32, point: Vec3) -> f64 {
    let Some(triangle) = source.triangles.get(triangle as usize) else {
        return 0.0;
    };
    let [Some(a), Some(b), Some(c)] =
        triangle.map(|vertex| source.positions.get(vertex as usize).copied())
    else {
        return 0.0;
    };
    let local_scale = a
        .distance_squared(b)
        .max(b.distance_squared(c))
        .max(c.distance_squared(a))
        .max(point.distance_squared(a))
        .sqrt();
    local_scale as f64 * f32::EPSILON as f64 * 8.0
}

fn unambiguous_parity(intersections: crate::mesh::RayIntersectionCount) -> Option<bool> {
    (!intersections.ambiguous).then_some(intersections.intersections % 2 == 1)
}

fn record_query_failure(failure: &AtomicU8, code: u8) {
    let _ = failure.compare_exchange(QUERY_OK, code, Ordering::SeqCst, Ordering::SeqCst);
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EdgeFaces {
    first: u32,
    second: u32,
    count: u32,
    orientation: i32,
}

fn validate_output(positions: &[Vec3], triangles: &[[u32; 3]]) -> Result<(), VoxelRemeshError> {
    if positions.is_empty() || triangles.is_empty() {
        return Err(VoxelRemeshError::EmptyOutput);
    }
    for (index, position) in positions.iter().enumerate() {
        if !position.is_finite() {
            return Err(VoxelRemeshError::NonFiniteOutputVertex { index });
        }
    }

    let mut edge_faces =
        HashMap::<(u32, u32), EdgeFaces>::with_capacity(triangles.len().saturating_mul(3) / 2);
    let mut incident_faces = vec![Vec::<u32>::new(); positions.len()];
    for (triangle_index, &triangle) in triangles.iter().enumerate() {
        for &vertex in &triangle {
            if vertex as usize >= positions.len() {
                return Err(VoxelRemeshError::OutputIndexOutOfBounds {
                    triangle: triangle_index,
                    index: vertex,
                    vertex_count: positions.len(),
                });
            }
        }
        let area_squared = triangle_area_squared(positions, triangle);
        if triangle[0] == triangle[1]
            || triangle[1] == triangle[2]
            || triangle[2] == triangle[0]
            || !area_squared.is_finite()
            || area_squared <= 0.0
        {
            return Err(VoxelRemeshError::DegenerateOutputTriangle {
                triangle: triangle_index,
            });
        }
        for &vertex in &triangle {
            incident_faces[vertex as usize].push(triangle_index as u32);
        }
        for corner in 0..3 {
            let from = triangle[corner];
            let to = triangle[(corner + 1) % 3];
            let (key, orientation) = if from < to {
                ((from, to), 1)
            } else {
                ((to, from), -1)
            };
            let entry = edge_faces.entry(key).or_default();
            match entry.count {
                0 => entry.first = triangle_index as u32,
                1 => entry.second = triangle_index as u32,
                _ => {}
            }
            entry.count = entry.count.saturating_add(1);
            entry.orientation += orientation;
        }
    }

    for (&(a, b), edge) in &edge_faces {
        if edge.count != 2 {
            return Err(VoxelRemeshError::OutputEdgeUseCount {
                a,
                b,
                faces: edge.count,
            });
        }
        if edge.orientation != 0 {
            return Err(VoxelRemeshError::InconsistentOutputWinding { a, b });
        }
    }

    let mut visit_marks = vec![0_u32; triangles.len()];
    let mut generation = 0_u32;
    let mut stack = Vec::new();
    for (vertex, incident) in incident_faces.iter().enumerate() {
        if incident.is_empty() {
            return Err(VoxelRemeshError::UnreferencedOutputVertex {
                vertex: vertex as u32,
            });
        }
        generation = generation.wrapping_add(1);
        if generation == 0 {
            visit_marks.fill(0);
            generation = 1;
        }
        stack.clear();
        stack.push(incident[0]);
        visit_marks[incident[0] as usize] = generation;
        let mut visited = 0;
        while let Some(face) = stack.pop() {
            visited += 1;
            let triangle = triangles[face as usize];
            for corner in 0..3 {
                let a = triangle[corner];
                let b = triangle[(corner + 1) % 3];
                if a != vertex as u32 && b != vertex as u32 {
                    continue;
                }
                let edge = edge_faces[&ordered_edge(a, b)];
                let neighbor = if edge.first == face {
                    edge.second
                } else {
                    edge.first
                };
                if visit_marks[neighbor as usize] != generation {
                    visit_marks[neighbor as usize] = generation;
                    stack.push(neighbor);
                }
            }
        }
        if visited != incident.len() {
            return Err(VoxelRemeshError::NonManifoldOutputVertex {
                vertex: vertex as u32,
            });
        }
    }
    Ok(())
}

fn triangle_area_squared(positions: &[Vec3], triangle: [u32; 3]) -> f64 {
    let [a, b, c] = triangle.map(|vertex| positions[vertex as usize]);
    let ab = [
        b.x as f64 - a.x as f64,
        b.y as f64 - a.y as f64,
        b.z as f64 - a.z as f64,
    ];
    let ac = [
        c.x as f64 - a.x as f64,
        c.y as f64 - a.y as f64,
        c.z as f64 - a.z as f64,
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    cross.into_iter().map(|value| value * value).sum()
}

fn ordered_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn reproject_masks(source: &Mesh, positions: &[Vec3]) -> Result<Vec<f32>, VoxelRemeshError> {
    positions
        .iter()
        .copied()
        .enumerate()
        .map(|(vertex, position)| {
            let closest = source
                .closest_surface_point(position)
                .ok_or(VoxelRemeshError::MaskReprojectionFailed { vertex })?;
            let source_triangle = source
                .triangles
                .get(closest.triangle as usize)
                .ok_or(VoxelRemeshError::MaskReprojectionFailed { vertex })?;
            let [a, b, c] = source_triangle.map(|index| source.mask[index as usize]);
            let value =
                a * closest.barycentric.x + b * closest.barycentric.y + c * closest.barycentric.z;
            if value.is_finite() {
                Ok(value.clamp(0.0, 1.0))
            } else {
                Err(VoxelRemeshError::MaskReprojectionFailed { vertex })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn cube(center: Vec3, half_extent: f32) -> Mesh {
        let positions = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ]
        .map(|position| center + position * half_extent)
        .to_vec();
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        Mesh::new(positions, triangles).unwrap()
    }

    fn append_mesh(target: &mut Mesh, source: &Mesh, reverse: bool) {
        let offset = target.positions.len() as u32;
        target.positions.extend_from_slice(&source.positions);
        target.mask.extend_from_slice(&source.mask);
        target
            .triangles
            .extend(source.triangles.iter().map(|triangle| {
                let mut triangle = triangle.map(|vertex| vertex + offset);
                if reverse {
                    triangle.swap(1, 2);
                }
                triangle
            }));
        target.rebuild();
    }

    fn icosahedron() -> Mesh {
        let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let positions = vec![
            Vec3::new(-1.0, phi, 0.0),
            Vec3::new(1.0, phi, 0.0),
            Vec3::new(-1.0, -phi, 0.0),
            Vec3::new(1.0, -phi, 0.0),
            Vec3::new(0.0, -1.0, phi),
            Vec3::new(0.0, 1.0, phi),
            Vec3::new(0.0, -1.0, -phi),
            Vec3::new(0.0, 1.0, -phi),
            Vec3::new(phi, 0.0, -1.0),
            Vec3::new(phi, 0.0, 1.0),
            Vec3::new(-phi, 0.0, -1.0),
            Vec3::new(-phi, 0.0, 1.0),
        ]
        .into_iter()
        .map(|position| position.normalize())
        .collect();
        let triangles = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        Mesh::new(positions, triangles).unwrap()
    }

    fn test_voxel_size(
        mesh: &Mesh,
        settings: VoxelRemeshSettings,
    ) -> Result<f32, VoxelRemeshError> {
        extraction_grid(mesh, settings).map(|grid| grid.voxel_size as f32)
    }

    #[test]
    fn settings_are_bounded_and_default_to_96() {
        assert_eq!(VoxelRemeshSettings::default().resolution, 96);
        let mesh = cube(Vec3::ZERO, 1.0);
        for resolution in [31, 193] {
            assert!(matches!(
                test_voxel_size(&mesh, VoxelRemeshSettings { resolution }),
                Err(VoxelRemeshError::InvalidResolution { .. })
            ));
        }
    }

    #[test]
    fn grid_bounds_ignore_unreferenced_vertices() {
        let mut mesh = cube(Vec3::ZERO, 1.0);
        mesh.positions.push(Vec3::splat(10_000.0));
        mesh.normals.push(Vec3::ZERO);
        mesh.mask.push(0.0);

        let size = test_voxel_size(&mesh, VoxelRemeshSettings { resolution: 32 }).unwrap();
        assert!((size - 0.0625).abs() < f32::EPSILON);
    }

    #[test]
    fn surface_roundoff_is_translation_invariant() {
        let local = cube(Vec3::ZERO, 1.0);
        let translated = cube(Vec3::splat(1_000_000.0), 1.0);
        let local_point = Vec3::new(0.25, 0.25, 1.0);
        let translated_point = local_point + Vec3::splat(1_000_000.0);
        let local_closest = local.closest_surface_point(local_point).unwrap();
        let translated_closest = translated.closest_surface_point(translated_point).unwrap();

        let local_roundoff = surface_roundoff(&local, local_closest.triangle, local_point);
        let translated_roundoff =
            surface_roundoff(&translated, translated_closest.triangle, translated_point);
        assert!((local_roundoff - translated_roundoff).abs() <= f64::EPSILON);
        assert!(translated_roundoff < 1.0e-4);
    }

    #[test]
    fn grid_rejects_voxels_below_coordinate_precision() {
        let mesh = cube(Vec3::splat(10_000_000.0), 1.0);
        assert!(matches!(
            test_voxel_size(&mesh, VoxelRemeshSettings { resolution: 32 }),
            Err(VoxelRemeshError::InsufficientCoordinatePrecision { .. })
        ));
    }

    #[test]
    fn open_and_non_manifold_sources_are_rejected() {
        let open = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).unwrap();
        assert!(matches!(
            source_eligibility(&open),
            Err(VoxelRemeshError::OpenSource { .. })
        ));

        let mut bow_tie = cube(Vec3::ZERO, 1.0);
        let second = cube(Vec3::new(2.0, 2.0, 2.0), 1.0);
        let offset = bow_tie.positions.len() as u32 - 1;
        bow_tie.positions.extend_from_slice(&second.positions[1..]);
        bow_tie.mask.extend_from_slice(&second.mask[1..]);
        bow_tie.triangles.extend(
            second.triangles.iter().map(|triangle| {
                triangle.map(|vertex| if vertex == 0 { 6 } else { vertex + offset })
            }),
        );
        bow_tie.rebuild();
        assert!(matches!(
            source_eligibility(&bow_tie),
            Err(VoxelRemeshError::NonManifoldSource { .. })
        ));
    }

    #[test]
    fn cube_remesh_is_deterministic_closed_and_reprojects_masks() {
        let mut source = cube(Vec3::ZERO, 1.0);
        source.mask = source
            .positions
            .iter()
            .map(|position| (position.x + 1.0) * 0.5)
            .collect();
        let settings = VoxelRemeshSettings { resolution: 32 };
        let first = voxel_remesh(&source, settings).unwrap();
        let second = voxel_remesh(&source, settings).unwrap();

        assert_eq!(first.mesh.positions, second.mesh.positions);
        assert_eq!(first.mesh.triangles, second.mesh.triangles);
        assert_eq!(first.mesh.mask, second.mesh.mask);
        assert_eq!(first.source_faces, 12);
        assert_eq!(first.output_faces, first.mesh.triangles.len());
        assert_eq!(first.mesh.topology.boundary_edge_count(), 0);
        assert_eq!(first.mesh.topology.non_manifold_edge_count(), 0);
        assert_eq!(first.mesh.topology.non_manifold_vertex_count(), 0);
        for (&position, &mask) in first.mesh.positions.iter().zip(&first.mesh.mask) {
            let closest = source.closest_surface_point(position).unwrap();
            let triangle = source.triangles[closest.triangle as usize];
            let expected = source.mask[triangle[0] as usize] * closest.barycentric.x
                + source.mask[triangle[1] as usize] * closest.barycentric.y
                + source.mask[triangle[2] as usize] * closest.barycentric.z;
            assert!((mask - expected).abs() < 1.0e-6);
        }
        assert!(first.mesh.mask.iter().any(|&mask| mask < 0.1));
        assert!(first.mesh.mask.iter().any(|&mask| mask > 0.9));

        let (min, max) = first.mesh.bounds().unwrap();
        let tolerance = first.voxel_size * 2.0;
        assert!((min.x + 1.0).abs() <= tolerance);
        assert!((max.x - 1.0).abs() <= tolerance);
        assert!((min.y + 1.0).abs() <= tolerance);
        assert!((max.y - 1.0).abs() <= tolerance);
        assert!((min.z + 1.0).abs() <= tolerance);
        assert!((max.z - 1.0).abs() <= tolerance);
    }

    #[test]
    fn curved_cavity_and_disconnected_sources_remesh() {
        let curved = voxel_remesh(&icosahedron(), VoxelRemeshSettings { resolution: 40 }).unwrap();
        assert!(!curved.mesh.triangles.is_empty());
        assert_eq!(curved.mesh.topology.boundary_edge_count(), 0);

        let mut disconnected = cube(Vec3::new(-2.0, 0.0, 0.0), 0.75);
        append_mesh(
            &mut disconnected,
            &cube(Vec3::new(2.0, 0.0, 0.0), 0.75),
            false,
        );
        let disconnected =
            voxel_remesh(&disconnected, VoxelRemeshSettings { resolution: 32 }).unwrap();
        assert_eq!(surface_component_count(&disconnected.mesh), 2);

        let mut cavity = cube(Vec3::ZERO, 1.0);
        append_mesh(&mut cavity, &cube(Vec3::ZERO, 0.45), true);
        let cavity = voxel_remesh(&cavity, VoxelRemeshSettings { resolution: 32 }).unwrap();
        assert_eq!(surface_component_count(&cavity.mesh), 2);
    }

    #[test]
    fn strict_validation_rejects_a_closed_bow_tie_vertex() {
        let mut first = cube(Vec3::ZERO, 1.0);
        let second = cube(Vec3::new(2.0, 2.0, 2.0), 1.0);
        let offset = first.positions.len() as u32 - 1;
        first.positions.extend_from_slice(&second.positions[1..]);
        first.triangles.extend(
            second.triangles.iter().map(|triangle| {
                triangle.map(|vertex| if vertex == 0 { 6 } else { vertex + offset })
            }),
        );
        assert!(matches!(
            validate_output(&first.positions, &first.triangles),
            Err(VoxelRemeshError::NonManifoldOutputVertex { .. })
        ));
    }

    #[test]
    #[ignore = "release-mode voxel-remesh performance probe"]
    fn resolution_96_remesh_probe() {
        let source = icosahedron();
        let started = Instant::now();
        let output = voxel_remesh(&source, VoxelRemeshSettings { resolution: 96 }).unwrap();
        eprintln!(
            "resolution-96 voxel remesh: {} -> {} faces in {:?}",
            source.triangles.len(),
            output.mesh.triangles.len(),
            started.elapsed()
        );
        assert!(!output.mesh.triangles.is_empty());
    }

    fn surface_component_count(mesh: &Mesh) -> usize {
        let mut visited = vec![false; mesh.triangles.len()];
        let mut stack = Vec::new();
        let mut components = 0;
        for seed in 0..mesh.triangles.len() {
            if visited[seed] {
                continue;
            }
            components += 1;
            visited[seed] = true;
            stack.push(seed as u32);
            while let Some(face) = stack.pop() {
                let triangle = mesh.triangles[face as usize];
                for corner in 0..3 {
                    for &neighbor in &mesh.topology.edge_faces
                        [&ordered_edge(triangle[corner], triangle[(corner + 1) % 3])]
                    {
                        if !visited[neighbor as usize] {
                            visited[neighbor as usize] = true;
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }
        components
    }
}
