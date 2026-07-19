use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec3;
use thiserror::Error;

use crate::mesh::{CleanupReport, Mesh, MeshError, TriangleSoupBuilder};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImportReport {
    pub source_triangles: usize,
    pub source_vertices: usize,
    pub unique_vertices: usize,
    pub output_triangles: usize,
    pub welded_vertices: usize,
    pub removed_invalid_faces: usize,
    pub removed_degenerate_faces: usize,
    pub removed_duplicate_faces: usize,
    pub flipped_faces: usize,
    pub boundary_edges: usize,
    pub boundary_vertices: usize,
    pub non_manifold_edges: usize,
    pub non_manifold_vertices: usize,
}

impl ImportReport {
    pub fn has_topology_warnings(self) -> bool {
        self.boundary_edges != 0 || self.non_manifold_edges != 0
    }
}

impl fmt::Display for ImportReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} triangles, {} unique vertices ({} welded); {} boundary edges, {} non-manifold edges",
            self.output_triangles,
            self.unique_vertices,
            self.welded_vertices,
            self.boundary_edges,
            self.non_manifold_edges
        )?;
        let removed = self.removed_invalid_faces
            + self.removed_degenerate_faces
            + self.removed_duplicate_faces;
        if removed != 0 {
            write!(formatter, "; {removed} unusable faces removed")?;
        }
        Ok(())
    }
}

impl From<CleanupReport> for ImportReport {
    fn from(report: CleanupReport) -> Self {
        Self {
            source_triangles: report.input_triangles,
            source_vertices: report.input_vertices,
            unique_vertices: report.output_vertices,
            output_triangles: report.output_triangles,
            welded_vertices: report.welded_vertices,
            removed_invalid_faces: report.removed_invalid_faces,
            removed_degenerate_faces: report.removed_degenerate_faces,
            removed_duplicate_faces: report.removed_duplicate_faces,
            flipped_faces: report.flipped_faces,
            boundary_edges: report.boundary_edges,
            boundary_vertices: report.boundary_vertices,
            non_manifold_edges: report.non_manifold_edges,
            non_manifold_vertices: report.non_manifold_vertices,
        }
    }
}

#[derive(Debug, Error)]
pub enum StlError {
    #[error("failed to read STL `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write STL `{path}`: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid mesh geometry: {0}")]
    InvalidMesh(#[from] MeshError),
    #[error("cannot export more than {} triangles to binary STL", u32::MAX)]
    TooManyTriangles,
    #[error("STL export path must include a file name: `{0}`")]
    MissingFileName(PathBuf),
}

/// Reads either binary or ASCII STL, preserving every finite coordinate exactly as `f32`.
///
/// STL is triangle soup. Equal coordinate bit patterns are welded, bad faces are discarded, and
/// manifold-connected faces are consistently oriented by [`Mesh::from_triangle_soup`].
pub fn load_stl(path: &Path) -> Result<(Mesh, ImportReport), StlError> {
    let file = File::open(path).map_err(|source| StlError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let stl_reader = stl_io::create_stl_reader(&mut reader).map_err(|source| StlError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut mesh_builder = TriangleSoupBuilder::new();
    for triangle in stl_reader {
        let triangle = triangle.map_err(|source| StlError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        mesh_builder.push_triangle(triangle.vertices.map(|vertex| Vec3::from_array(vertex.0)))?;
    }
    let (mesh, cleanup) = mesh_builder.finish();
    Ok((mesh, cleanup.into()))
}

/// Atomically writes a binary STL with normals recomputed from the current face winding.
///
/// The complete file is flushed and synced to a temporary file in the destination directory,
/// then renamed over the destination. A failed export leaves an existing destination untouched.
pub fn save_stl_atomic(path: &Path, mesh: &Mesh) -> Result<(), StlError> {
    mesh.validate()?;
    if mesh.triangles.len() > u32::MAX as usize {
        return Err(StlError::TooManyTriangles);
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| StlError::MissingFileName(path.to_path_buf()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (temporary_path, temporary_file) = create_temporary_file(parent, file_name, path)?;

    let result = (|| -> Result<(), io::Error> {
        let mut writer = BufWriter::new(temporary_file);
        let triangles = mesh.triangles.iter().map(|triangle| {
            let [a, b, c] = triangle.map(|index| mesh.positions[index as usize]);
            let normal = (b - a).cross(c - a).normalize_or_zero();
            stl_io::Triangle {
                normal: stl_io::Normal::new(normal.to_array()),
                vertices: [a, b, c].map(|position| stl_io::Vertex::new(position.to_array())),
            }
        });
        stl_io::write_stl(&mut writer, triangles)?;
        let file = writer.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
        fs::rename(&temporary_path, path)?;
        if let Ok(directory) = File::open(parent) {
            directory.sync_all()?;
        }
        Ok(())
    })();

    if let Err(source) = result {
        let _ = fs::remove_file(&temporary_path);
        return Err(StlError::Write {
            path: path.to_path_buf(),
            source,
        });
    }
    Ok(())
}

fn create_temporary_file(
    parent: &Path,
    file_name: &std::ffi::OsStr,
    destination: &Path,
) -> Result<(PathBuf, File), StlError> {
    for _ in 0..128 {
        let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(file_name);
        temporary_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(StlError::Write {
                    path: destination.to_path_buf(),
                    source,
                });
            }
        }
    }
    Err(StlError::Write {
        path: destination.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique temporary export file",
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn imports_ascii_stl_and_preserves_coordinates_and_scale() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("skull-part.stl");
        fs::write(
            &path,
            b"solid test\n\
              facet normal 0 0 1\n\
                outer loop\n\
                  vertex -12.5 3.25 100\n\
                  vertex 7.75 3.25 100\n\
                  vertex -12.5 9.5 100\n\
                endloop\n\
              endfacet\n\
            endsolid test\n",
        )
        .unwrap();

        let (mesh, report) = load_stl(&path).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.triangles.len(), 1);
        assert_eq!(
            mesh.bounds(),
            Some((Vec3::new(-12.5, 3.25, 100.0), Vec3::new(7.75, 9.5, 100.0)))
        );
        assert_eq!(report.source_triangles, 1);
        assert_eq!(report.source_vertices, 3);
        assert_eq!(report.unique_vertices, 3);
        assert_eq!(report.boundary_edges, 3);
        assert!(report.has_topology_warnings());
    }

    #[test]
    fn binary_round_trip_preserves_edited_vertex_coordinates() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("edited.stl");
        let mesh = Mesh::new(
            vec![
                Vec3::new(-1.25, 2.5, 3.75),
                Vec3::new(4.0, 2.5, 3.75),
                Vec3::new(-1.25, 8.0, 3.75),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();

        save_stl_atomic(&path, &mesh).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 84 + 50);
        let (loaded, report) = load_stl(&path).unwrap();
        assert_eq!(loaded.positions, mesh.positions);
        assert_eq!(loaded.triangles.len(), 1);
        assert_eq!(report.output_triangles, 1);
    }

    #[test]
    fn export_atomically_replaces_an_existing_file() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("existing.stl");
        fs::write(&path, b"old contents").unwrap();
        let mesh = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).unwrap();
        save_stl_atomic(&path, &mesh).unwrap();
        assert_ne!(fs::read(&path).unwrap(), b"old contents");
        assert!(load_stl(&path).is_ok());
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
    }

    #[test]
    fn malformed_stl_is_a_read_error() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("bad.stl");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"not an stl").unwrap();
        drop(file);
        assert!(matches!(load_stl(&path), Err(StlError::Read { .. })));
    }

    #[test]
    fn invalid_mesh_does_not_replace_destination() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("safe.stl");
        fs::write(&path, b"keep me").unwrap();
        let invalid = Mesh {
            positions: vec![Vec3::new(f32::NAN, 0.0, 0.0), Vec3::X, Vec3::Y],
            triangles: vec![[0, 1, 2]],
            normals: vec![Vec3::ZERO; 3],
            mask: vec![0.0; 3],
            topology: Default::default(),
        };
        assert!(matches!(
            save_stl_atomic(&path, &invalid),
            Err(StlError::InvalidMesh(MeshError::NonFiniteVertex { .. }))
        ));
        assert_eq!(fs::read(&path).unwrap(), b"keep me");
    }

    #[test]
    fn import_report_display_surfaces_topology_and_cleanup() {
        let report = ImportReport {
            output_triangles: 12,
            unique_vertices: 8,
            welded_vertices: 28,
            boundary_edges: 3,
            non_manifold_edges: 1,
            removed_degenerate_faces: 2,
            ..ImportReport::default()
        };
        let display = report.to_string();
        assert!(display.contains("12 triangles"));
        assert!(display.contains("3 boundary edges"));
        assert!(display.contains("1 non-manifold edges"));
        assert!(display.contains("2 unusable faces removed"));
    }
}
