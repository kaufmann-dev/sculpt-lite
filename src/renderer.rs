use std::mem::size_of;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use eframe::egui::{Color32, PaintCallback, Pos2, Rect, Stroke, Ui};
use egui_wgpu::wgpu;
use egui_wgpu::wgpu::util::DeviceExt as _;
use glam::Vec3;
use hashbrown::HashMap;
#[cfg(test)]
use hashbrown::HashSet;
use thiserror::Error;

use crate::camera::Camera;
use crate::mesh::{EdgeKey, Mesh, MeshChangeSet};

pub const REQUIRED_DEPTH_BITS: u8 = 32;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Debug, Error)]
pub enum RendererError {
    #[error("the wgpu render state is unavailable; run eframe with Renderer::Wgpu")]
    WgpuUnavailable,
}

/// Egui-side brush cursor drawn over the custom 3D callback.
#[derive(Clone, Copy, Debug)]
pub struct BrushCursor {
    pub center: Pos2,
    pub radius_points: f32,
    pub color: Color32,
    pub active: bool,
}

impl BrushCursor {
    #[must_use]
    pub fn new(center: Pos2, radius_points: f32) -> Self {
        Self {
            center,
            radius_points,
            color: Color32::from_rgb(238, 245, 255),
            active: false,
        }
    }
}

/// Handle used by the app to update and paint a single native sculpt viewport.
///
/// GPU resources live in egui-wgpu's callback resource map. CPU updates are
/// revisioned, so invoking `paint` every frame only uploads data that changed.
pub struct ViewportRenderer {
    shared: Arc<RwLock<RenderInput>>,
    mesh_preparer: MeshGpuPreparer,
}

#[derive(Clone)]
pub(crate) struct MeshGpuPreparer {
    device: wgpu::Device,
}

impl ViewportRenderer {
    pub fn new(creation_context: &eframe::CreationContext<'_>) -> Result<Self, RendererError> {
        let render_state = creation_context
            .wgpu_render_state
            .as_ref()
            .ok_or(RendererError::WgpuUnavailable)?;

        let shared = Arc::new(RwLock::new(RenderInput::default()));
        let resources = ViewportGpu::new(&render_state.device, render_state.target_format);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);

        Ok(Self {
            shared,
            mesh_preparer: MeshGpuPreparer {
                device: render_state.device.clone(),
            },
        })
    }

    /// Uploads the public geometry representation used by the mesh core. Invalid
    /// triangle indices are skipped instead of reaching wgpu validation.
    pub fn update_mesh(&self, mesh: &Mesh) {
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        input.mesh = MeshUpload::from_mesh(mesh);
        input.prepared_gpu = None;
        input.full_vertex_upload = true;
        input.full_topology_upload = true;
        input.dirty_vertices.clear();
        input.dirty_faces.clear();
        input.dirty_edges.clear();
        input.vertex_revision = input.vertex_revision.wrapping_add(1);
        input.topology_revision = input.topology_revision.wrapping_add(1);
    }

    /// Returns a cloneable device-backed preparer for the mesh worker.
    #[must_use]
    pub(crate) fn mesh_preparer(&self) -> MeshGpuPreparer {
        self.mesh_preparer.clone()
    }

    /// Installs CPU geometry and already populated GPU buffers without staging
    /// large transfers on the event thread.
    pub(crate) fn install_prepared_mesh(&self, upload: PreparedMeshUpload) {
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        input.mesh = upload.mesh;
        input.prepared_gpu = Some(upload.gpu);
        input.full_vertex_upload = false;
        input.full_topology_upload = false;
        input.dirty_vertices.clear();
        input.dirty_faces.clear();
        input.dirty_edges.clear();
        input.vertex_revision = input.vertex_revision.wrapping_add(1);
        input.topology_revision = input.topology_revision.wrapping_add(1);
    }

    /// Refreshes only vertices changed by a sculpt sample. The CPU mirror is
    /// updated immediately; adjacent indices are coalesced into compact GPU
    /// writes when the viewport callback is prepared.
    pub fn update_vertices_partial(&self, mesh: &Mesh, changed_vertices: &[u32]) {
        if changed_vertices.is_empty() {
            return;
        }
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        if input.mesh.vertices.len() != mesh.positions.len() {
            input.mesh.vertices = MeshUpload::vertices(&mesh.positions, &mesh.normals, &mesh.mask);
            input.full_vertex_upload = true;
            input.dirty_vertices.clear();
            input.vertex_revision = input.vertex_revision.wrapping_add(1);
            return;
        }

        let mut changed = false;
        for &vertex in changed_vertices {
            let index = vertex as usize;
            let Some(position) = mesh.positions.get(index).copied() else {
                continue;
            };
            input.mesh.vertices[index] = MeshUpload::vertex(
                position,
                mesh.normals.get(index).copied(),
                mesh.mask.get(index).copied(),
            );
            input.dirty_vertices.push(vertex);
            changed = true;
        }
        if changed {
            input.vertex_revision = input.vertex_revision.wrapping_add(1);
        }
    }

    /// Applies a topology edit to the CPU mirror without scanning untouched mesh data.
    pub fn update_mesh_partial(&self, mesh: &Mesh, changes: &MeshChangeSet) {
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        if input.prepared_gpu.is_some() {
            input.mesh = MeshUpload::from_mesh(mesh);
            input.prepared_gpu = None;
            input.full_vertex_upload = true;
            input.full_topology_upload = true;
            input.vertex_revision = input.vertex_revision.wrapping_add(1);
            input.topology_revision = input.topology_revision.wrapping_add(1);
            return;
        }

        let dirty = input.mesh.apply_changes(mesh, changes);
        input.dirty_vertices.extend(dirty.vertices);
        input.dirty_faces.extend(dirty.faces);
        input.dirty_edges.extend(dirty.edges);
        input.vertex_revision = input.vertex_revision.wrapping_add(1);
        input.topology_revision = input.topology_revision.wrapping_add(1);
    }

    /// Updates view and lighting uniforms. Call after camera input and whenever
    /// the viewport aspect ratio changes.
    pub fn update_camera(&self, camera: &Camera, aspect: f32) {
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        input.camera = CameraUniform::from_camera(camera, aspect);
        input.camera_revision = input.camera_revision.wrapping_add(1);
    }

    /// Builds the backend-specific callback for callers that manage painter
    /// ordering themselves.
    #[must_use]
    pub fn paint_callback(&self, rect: Rect, wireframe: bool) -> PaintCallback {
        egui_wgpu::Callback::new_paint_callback(
            rect,
            ViewportPaintCallback {
                shared: Arc::clone(&self.shared),
                wireframe,
            },
        )
    }

    /// Paints a clipped studio viewport and an optional screen-space brush ring.
    pub fn paint(&self, ui: &Ui, rect: Rect, brush_cursor: Option<BrushCursor>, wireframe: bool) {
        let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        painter.rect_filled(rect, 0.0, Color32::from_rgb(30, 34, 41));
        painter.add(self.paint_callback(rect, wireframe));

        if let Some(cursor) = brush_cursor.filter(|cursor| {
            cursor.radius_points.is_finite()
                && cursor.radius_points > 0.0
                && rect.contains(cursor.center)
        }) {
            let color = if cursor.active {
                Color32::from_rgb(255, 176, 74)
            } else {
                cursor.color
            };
            painter.circle_stroke(cursor.center, cursor.radius_points, Stroke::new(1.5, color));
            painter.circle_filled(cursor.center, 1.75, color);
        }
    }
}

#[derive(Default)]
struct RenderInput {
    vertex_revision: u64,
    topology_revision: u64,
    camera_revision: u64,
    mesh: MeshUpload,
    prepared_gpu: Option<PreparedGpuMesh>,
    camera: CameraUniform,
    full_vertex_upload: bool,
    full_topology_upload: bool,
    dirty_vertices: Vec<u32>,
    dirty_faces: Vec<u32>,
    dirty_edges: Vec<u32>,
}

#[derive(Default)]
pub(crate) struct MeshUpload {
    vertices: Vec<GpuVertex>,
    triangle_indices: Vec<u32>,
    edge_indices: Vec<u32>,
    edges: Vec<EdgeKey>,
    edge_slots: HashMap<EdgeKey, u32>,
}

#[derive(Default)]
struct UploadChanges {
    vertices: Vec<u32>,
    faces: Vec<u32>,
    edges: Vec<u32>,
}

pub(crate) struct PreparedMeshUpload {
    mesh: MeshUpload,
    gpu: PreparedGpuMesh,
}

struct PreparedGpuMesh {
    vertices: BufferSlot,
    triangles: BufferSlot,
    edges: BufferSlot,
    triangle_index_count: u32,
    edge_index_count: u32,
}

impl MeshGpuPreparer {
    #[must_use]
    pub(crate) fn prepare_mesh(&self, mesh: &Mesh) -> PreparedMeshUpload {
        let mesh = MeshUpload::from_mesh(mesh);
        let gpu = PreparedGpuMesh {
            vertices: BufferSlot::prepared(
                &self.device,
                wgpu::BufferUsages::VERTEX,
                "sculpt viewport vertices",
                &mesh.vertices,
            ),
            triangles: BufferSlot::prepared(
                &self.device,
                wgpu::BufferUsages::INDEX,
                "sculpt viewport triangle indices",
                &mesh.triangle_indices,
            ),
            edges: BufferSlot::prepared(
                &self.device,
                wgpu::BufferUsages::INDEX,
                "sculpt viewport edge indices",
                &mesh.edge_indices,
            ),
            triangle_index_count: index_count(&mesh.triangle_indices),
            edge_index_count: index_count(&mesh.edge_indices),
        };
        PreparedMeshUpload { mesh, gpu }
    }
}

impl MeshUpload {
    fn from_mesh(mesh: &Mesh) -> Self {
        let vertices = Self::vertices(&mesh.positions, &mesh.normals, &mesh.mask);
        let vertex_count = mesh.positions.len();
        let mut triangle_indices = Vec::with_capacity(mesh.triangles.len().saturating_mul(3));
        for &[a, b, c] in &mesh.triangles {
            if [a, b, c]
                .into_iter()
                .all(|index| (index as usize) < vertex_count)
            {
                triangle_indices.extend_from_slice(&[a, b, c]);
            }
        }
        let mut edges = mesh.topology.edge_faces.keys().copied().collect::<Vec<_>>();
        edges.sort_unstable();
        let mut edge_indices = Vec::with_capacity(edges.len().saturating_mul(2));
        let mut edge_slots = HashMap::with_capacity(edges.len());
        for (slot, &(a, b)) in edges.iter().enumerate() {
            if (a as usize) < vertex_count && (b as usize) < vertex_count {
                edge_indices.extend_from_slice(&[a, b]);
                edge_slots.insert((a, b), slot as u32);
            }
        }
        Self {
            vertices,
            triangle_indices,
            edge_indices,
            edges,
            edge_slots,
        }
    }

    fn apply_changes(&mut self, mesh: &Mesh, changes: &MeshChangeSet) -> UploadChanges {
        let mut dirty = UploadChanges::default();
        self.vertices
            .resize(changes.vertex_count, GpuVertex::default());
        for &vertex in &changes.dirty_vertices {
            let index = vertex as usize;
            let Some(position) = mesh.positions.get(index).copied() else {
                continue;
            };
            self.vertices[index] = Self::vertex(
                position,
                mesh.normals.get(index).copied(),
                mesh.mask.get(index).copied(),
            );
            dirty.vertices.push(vertex);
        }

        self.triangle_indices
            .resize(changes.face_count.saturating_mul(3), 0);
        for &face in &changes.dirty_faces {
            let index = face as usize;
            let Some(&triangle) = mesh.triangles.get(index) else {
                continue;
            };
            self.triangle_indices[index * 3..index * 3 + 3].copy_from_slice(&triangle);
            dirty.faces.push(face);
        }

        for &edge in &changes.removed_edges {
            let Some(slot) = self.edge_slots.remove(&edge) else {
                continue;
            };
            let slot = slot as usize;
            let last = self.edges.len() - 1;
            if slot != last {
                let moved = self.edges[last];
                self.edges[slot] = moved;
                self.edge_indices[slot * 2] = moved.0;
                self.edge_indices[slot * 2 + 1] = moved.1;
                self.edge_slots.insert(moved, slot as u32);
                dirty.edges.push(slot as u32);
            }
            self.edges.pop();
            self.edge_indices.truncate(last * 2);
        }
        for &edge in &changes.added_edges {
            if self.edge_slots.contains_key(&edge) || !mesh.topology.edge_faces.contains_key(&edge)
            {
                continue;
            }
            let slot = self.edges.len() as u32;
            self.edges.push(edge);
            self.edge_indices.extend_from_slice(&[edge.0, edge.1]);
            self.edge_slots.insert(edge, slot);
            dirty.edges.push(slot);
        }
        dirty
    }

    #[cfg(test)]
    fn new(positions: &[Vec3], normals: &[Vec3], masks: &[f32], triangles: &[[u32; 3]]) -> Self {
        let vertices = Self::vertices(positions, normals, masks);

        let mut triangle_indices = Vec::with_capacity(triangles.len().saturating_mul(3));
        let mut edges = HashSet::with_capacity(triangles.len().saturating_mul(3) / 2);
        let vertex_count = positions.len();
        for &[a, b, c] in triangles {
            if [a, b, c]
                .into_iter()
                .all(|index| (index as usize) < vertex_count)
            {
                triangle_indices.extend_from_slice(&[a, b, c]);
                for (left, right) in [(a, b), (b, c), (c, a)] {
                    edges.insert(if left < right {
                        (left, right)
                    } else {
                        (right, left)
                    });
                }
            }
        }
        let mut edge_indices = Vec::with_capacity(edges.len().saturating_mul(2));
        for (a, b) in edges {
            edge_indices.extend_from_slice(&[a, b]);
        }

        Self {
            vertices,
            triangle_indices,
            edge_indices,
            edges: Vec::new(),
            edge_slots: HashMap::new(),
        }
    }

    fn vertices(positions: &[Vec3], normals: &[Vec3], masks: &[f32]) -> Vec<GpuVertex> {
        positions
            .iter()
            .enumerate()
            .map(|(index, &position)| {
                Self::vertex(
                    position,
                    normals.get(index).copied(),
                    masks.get(index).copied(),
                )
            })
            .collect()
    }

    fn vertex(position: Vec3, normal: Option<Vec3>, mask: Option<f32>) -> GpuVertex {
        let normal = normal
            .filter(|normal| normal.is_finite())
            .unwrap_or(Vec3::Y)
            .normalize_or_zero();
        let mask = mask
            .filter(|mask| mask.is_finite())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        GpuVertex {
            position: position.to_array(),
            mask,
            normal: normal.to_array(),
            _padding: 0.0,
        }
    }
}

fn index_count(indices: &[u32]) -> u32 {
    u32::try_from(indices.len()).unwrap_or(u32::MAX)
}

fn coalesced_vertex_ranges(
    dirty_vertices: &mut Vec<u32>,
    vertex_count: usize,
) -> Vec<Range<usize>> {
    const MAX_UNCHANGED_GAP: u32 = 16;

    dirty_vertices.retain(|&vertex| (vertex as usize) < vertex_count);
    dirty_vertices.sort_unstable();
    dirty_vertices.dedup();
    let mut ranges = Vec::<Range<usize>>::new();
    for &vertex in dirty_vertices.iter() {
        let index = vertex as usize;
        if let Some(last) = ranges.last_mut()
            && vertex <= (last.end as u32).saturating_add(MAX_UNCHANGED_GAP)
        {
            last.end = index + 1;
        } else {
            ranges.push(index..index + 1);
        }
    }
    ranges
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    mask: f32,
    normal: [f32; 3],
    _padding: f32,
}

impl GpuVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
    eye: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    material: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
            eye: [0.0, 0.0, 3.0, 1.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
            // Curvature contrast, broad specular strength, and rim strength.
            material: [0.10, 0.055, 0.025, 0.0],
        }
    }
}

impl CameraUniform {
    fn from_camera(camera: &Camera, aspect: f32) -> Self {
        Self {
            view_projection: camera.view_projection(aspect).to_cols_array_2d(),
            eye: camera.eye_position().extend(1.0).to_array(),
            camera_right: camera
                .right()
                .try_normalize()
                .unwrap_or(Vec3::X)
                .extend(0.0)
                .to_array(),
            camera_up: camera
                .up()
                .try_normalize()
                .unwrap_or(Vec3::Y)
                .extend(0.0)
                .to_array(),
            ..Self::default()
        }
    }
}

struct ViewportGpu {
    solid_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertices: BufferSlot,
    triangles: BufferSlot,
    edges: BufferSlot,
    triangle_index_count: u32,
    edge_index_count: u32,
    vertex_revision: u64,
    topology_revision: u64,
    camera_revision: u64,
}

impl ViewportGpu {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt viewport uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<CameraUniform>() as u64),
                    },
                    count: None,
                }],
            });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sculpt viewport camera uniform"),
            contents: bytemuck::bytes_of(&CameraUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sculpt viewport uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sculpt viewport pipeline layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout)],
            immediate_size: 0,
        });

        let solid_pipeline = create_pipeline(
            device,
            &shader,
            &pipeline_layout,
            target_format,
            PipelineSpec {
                label: "sculpt viewport solid pipeline",
                topology: wgpu::PrimitiveTopology::TriangleList,
                vertex_entry: "vs_mesh",
                fragment_entry: "fs_solid",
                depth_write_enabled: true,
            },
        );
        let wire_pipeline = create_pipeline(
            device,
            &shader,
            &pipeline_layout,
            target_format,
            PipelineSpec {
                label: "sculpt viewport wire pipeline",
                topology: wgpu::PrimitiveTopology::LineList,
                vertex_entry: "vs_mesh",
                fragment_entry: "fs_wire",
                depth_write_enabled: false,
            },
        );

        Self {
            solid_pipeline,
            wire_pipeline,
            uniform_buffer,
            uniform_bind_group,
            vertices: BufferSlot::new(wgpu::BufferUsages::VERTEX),
            triangles: BufferSlot::new(wgpu::BufferUsages::INDEX),
            edges: BufferSlot::new(wgpu::BufferUsages::INDEX),
            triangle_index_count: 0,
            edge_index_count: 0,
            vertex_revision: u64::MAX,
            topology_revision: u64::MAX,
            camera_revision: u64::MAX,
        }
    }

    fn upload_vertices(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &MeshUpload) {
        self.vertices
            .write(device, queue, "sculpt viewport vertices", &mesh.vertices);
    }

    fn upload_changed_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshUpload,
        mut dirty_vertices: Vec<u32>,
    ) {
        if self.vertices.buffer.is_none()
            || !buffer_capacity_fits::<GpuVertex>(mesh.vertices.len(), self.vertices.capacity)
        {
            self.upload_vertices(device, queue, mesh);
            return;
        }
        let ranges = coalesced_vertex_ranges(&mut dirty_vertices, mesh.vertices.len());
        let uploaded_vertices = ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum::<usize>();
        if uploaded_vertices > mesh.vertices.len() / 4 || ranges.len() > 2_048 {
            self.upload_vertices(device, queue, mesh);
            return;
        }
        let Some(buffer) = &self.vertices.buffer else {
            return;
        };
        for range in ranges {
            queue.write_buffer(
                buffer,
                (range.start * size_of::<GpuVertex>()) as u64,
                bytemuck::cast_slice(&mesh.vertices[range]),
            );
        }
    }

    fn upload_topology(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &MeshUpload) {
        self.triangles.write(
            device,
            queue,
            "sculpt viewport triangle indices",
            &mesh.triangle_indices,
        );
        self.edges.write(
            device,
            queue,
            "sculpt viewport edge indices",
            &mesh.edge_indices,
        );
        self.triangle_index_count = index_count(&mesh.triangle_indices);
        self.edge_index_count = index_count(&mesh.edge_indices);
    }

    fn upload_changed_topology(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshUpload,
        mut dirty_faces: Vec<u32>,
        mut dirty_edges: Vec<u32>,
    ) {
        let triangle_bytes = (mesh.triangle_indices.len() * size_of::<u32>()) as u64;
        let edge_bytes = (mesh.edge_indices.len() * size_of::<u32>()) as u64;
        if self.triangles.buffer.is_none()
            || self.edges.buffer.is_none()
            || triangle_bytes > self.triangles.capacity
            || edge_bytes > self.edges.capacity
        {
            self.upload_topology(device, queue, mesh);
            return;
        }

        let face_count = mesh.triangle_indices.len() / 3;
        let face_ranges = coalesced_vertex_ranges(&mut dirty_faces, face_count);
        if let Some(buffer) = &self.triangles.buffer {
            for range in face_ranges {
                let indices = range.start * 3..range.end * 3;
                queue.write_buffer(
                    buffer,
                    (indices.start * size_of::<u32>()) as u64,
                    bytemuck::cast_slice(&mesh.triangle_indices[indices]),
                );
            }
        }

        let edge_count = mesh.edge_indices.len() / 2;
        let edge_ranges = coalesced_vertex_ranges(&mut dirty_edges, edge_count);
        if let Some(buffer) = &self.edges.buffer {
            for range in edge_ranges {
                let indices = range.start * 2..range.end * 2;
                queue.write_buffer(
                    buffer,
                    (indices.start * size_of::<u32>()) as u64,
                    bytemuck::cast_slice(&mesh.edge_indices[indices]),
                );
            }
        }
        self.triangle_index_count = index_count(&mesh.triangle_indices);
        self.edge_index_count = index_count(&mesh.edge_indices);
    }

    fn install_prepared_mesh(&mut self, prepared: PreparedGpuMesh) {
        self.vertices = prepared.vertices;
        self.triangles = prepared.triangles;
        self.edges = prepared.edges;
        self.triangle_index_count = prepared.triangle_index_count;
        self.edge_index_count = prepared.edge_index_count;
    }
}

fn buffer_capacity_fits<T>(element_count: usize, capacity: u64) -> bool {
    u64::try_from(element_count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<T>() as u64))
        .is_some_and(|required| required <= capacity)
}

struct PipelineSpec {
    label: &'static str,
    topology: wgpu::PrimitiveTopology,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    depth_write_enabled: bool,
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    target_format: wgpu::TextureFormat,
    spec: PipelineSpec,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(spec.label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(spec.vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[GpuVertex::layout()],
        },
        primitive: wgpu::PrimitiveState {
            topology: spec.topology,
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(spec.depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(spec.fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

struct BufferSlot {
    usage: wgpu::BufferUsages,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
}

impl BufferSlot {
    fn new(usage: wgpu::BufferUsages) -> Self {
        Self {
            usage,
            buffer: None,
            capacity: 0,
        }
    }

    fn prepared<T: Pod>(
        device: &wgpu::Device,
        usage: wgpu::BufferUsages,
        label: &'static str,
        values: &[T],
    ) -> Self {
        let bytes = bytemuck::cast_slice(values);
        if bytes.is_empty() {
            return Self::new(usage);
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: usage | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            usage,
            capacity: buffer.size(),
            buffer: Some(buffer),
        }
    }

    fn write<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        values: &[T],
    ) {
        let bytes = bytemuck::cast_slice(values);
        if bytes.is_empty() {
            return;
        }

        let required = bytes.len() as u64;
        if self.buffer.is_none() || required > self.capacity {
            self.capacity = required
                .checked_next_power_of_two()
                .unwrap_or(required)
                .max(4);
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: self.capacity,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytes);
        }
    }
}

struct ViewportPaintCallback {
    shared: Arc<RwLock<RenderInput>>,
    wireframe: bool,
}

impl egui_wgpu::CallbackTrait for ViewportPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = callback_resources.get_mut::<ViewportGpu>() else {
            return Vec::new();
        };
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());

        if let Some(prepared) = input.prepared_gpu.take() {
            gpu.install_prepared_mesh(prepared);
            input.full_vertex_upload = false;
            input.full_topology_upload = false;
            input.dirty_vertices.clear();
            input.dirty_faces.clear();
            input.dirty_edges.clear();
            gpu.vertex_revision = input.vertex_revision;
            gpu.topology_revision = input.topology_revision;
        }
        if input.vertex_revision != gpu.vertex_revision {
            let full_upload = input.full_vertex_upload || gpu.vertex_revision == u64::MAX;
            let dirty_vertices = std::mem::take(&mut input.dirty_vertices);
            input.full_vertex_upload = false;
            if full_upload {
                gpu.upload_vertices(device, queue, &input.mesh);
            } else {
                gpu.upload_changed_vertices(device, queue, &input.mesh, dirty_vertices);
            }
            gpu.vertex_revision = input.vertex_revision;
        }
        if input.topology_revision != gpu.topology_revision {
            let full_upload = input.full_topology_upload || gpu.topology_revision == u64::MAX;
            let dirty_faces = std::mem::take(&mut input.dirty_faces);
            let dirty_edges = std::mem::take(&mut input.dirty_edges);
            input.full_topology_upload = false;
            if full_upload {
                gpu.upload_topology(device, queue, &input.mesh);
            } else {
                gpu.upload_changed_topology(device, queue, &input.mesh, dirty_faces, dirty_edges);
            }
            gpu.topology_revision = input.topology_revision;
        }
        if input.camera_revision != gpu.camera_revision {
            queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&input.camera));
            gpu.camera_revision = input.camera_revision;
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = callback_resources.get::<ViewportGpu>() else {
            return;
        };
        let (Some(vertex_buffer), Some(index_buffer)) =
            (&gpu.vertices.buffer, &gpu.triangles.buffer)
        else {
            return;
        };

        render_pass.set_pipeline(&gpu.solid_pipeline);
        render_pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..gpu.triangle_index_count, 0, 0..1);

        if self.wireframe
            && gpu.edge_index_count > 0
            && let Some(edge_buffer) = &gpu.edges.buffer
        {
            render_pass.set_pipeline(&gpu.wire_pipeline);
            render_pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(edge_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..gpu.edge_index_count, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn mesh_upload_ignores_invalid_triangles_and_defaults_attributes() {
        let upload = MeshUpload::new(
            &[Vec3::ZERO, Vec3::X, Vec3::Y],
            &[],
            &[],
            &[[0, 1, 2], [0, 2, 99]],
        );

        assert_eq!(upload.vertices.len(), 3);
        assert_eq!(upload.triangle_indices, [0, 1, 2]);
        let edges = upload
            .edge_indices
            .chunks_exact(2)
            .map(|edge| (edge[0], edge[1]))
            .collect::<HashSet<_>>();
        assert_eq!(edges, HashSet::from([(0, 1), (0, 2), (1, 2)]));
        assert_eq!(upload.vertices[0].mask, 0.0);
        assert_eq!(upload.vertices[0].normal, Vec3::Y.to_array());
    }

    #[test]
    fn mesh_upload_emits_each_wire_edge_once() {
        let mesh = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE],
            vec![[0, 1, 2], [2, 1, 3]],
        )
        .unwrap();
        let upload = MeshUpload::from_mesh(&mesh);
        let edges = upload
            .edge_indices
            .chunks_exact(2)
            .map(|edge| (edge[0], edge[1]))
            .collect::<HashSet<_>>();

        assert_eq!(
            edges,
            HashSet::from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
        );
    }

    #[test]
    fn mesh_upload_clamps_mask_values() {
        let upload = MeshUpload::new(
            &[Vec3::ZERO, Vec3::X, Vec3::Y],
            &[Vec3::Z; 3],
            &[-1.0, 0.4, 2.0],
            &[[0, 1, 2]],
        );
        let masks: Vec<_> = upload.vertices.iter().map(|vertex| vertex.mask).collect();
        assert_eq!(masks, [0.0, 0.4, 1.0]);
    }

    #[test]
    fn partial_vertex_ranges_deduplicate_clip_and_coalesce_nearby_writes() {
        let mut dirty = vec![99, 20, 2, 2, 18, 0];
        let ranges = coalesced_vertex_ranges(&mut dirty, 32);

        assert_eq!(dirty, [0, 2, 18, 20]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..21);

        let mut separated = vec![1, 25];
        assert_eq!(coalesced_vertex_ranges(&mut separated, 32), [1..2, 25..26]);
    }

    #[test]
    fn partial_vertex_upload_reallocates_when_adaptive_topology_outgrows_buffer() {
        const CRASHED_BUFFER_BYTES: u64 = 7_510_368;
        let original_vertices = CRASHED_BUFFER_BYTES as usize / size_of::<GpuVertex>();

        assert!(buffer_capacity_fits::<GpuVertex>(
            original_vertices,
            CRASHED_BUFFER_BYTES
        ));
        assert!(!buffer_capacity_fits::<GpuVertex>(
            original_vertices + 1,
            CRASHED_BUFFER_BYTES
        ));
    }

    #[test]
    fn camera_uniform_keeps_studio_lights_in_camera_space() {
        let mut camera = Camera::default();
        camera.yaw = 1.37;
        camera.pitch = -0.42;

        let uniform = CameraUniform::from_camera(&camera, 16.0 / 9.0);
        let right = Vec3::from_array(uniform.camera_right[..3].try_into().unwrap());
        let up = Vec3::from_array(uniform.camera_up[..3].try_into().unwrap());

        assert!(right.abs_diff_eq(camera.right(), 1.0e-6));
        assert!(up.abs_diff_eq(camera.up(), 1.0e-6));
        assert!(right.dot(up).abs() < 1.0e-6);
        assert_eq!(uniform.camera_right[3], 0.0);
        assert_eq!(uniform.camera_up[3], 0.0);
    }

    #[test]
    fn camera_uniform_layout_matches_wgsl_uniform_alignment() {
        assert_eq!(size_of::<CameraUniform>(), 128);
    }

    #[test]
    fn topology_changes_update_only_the_cpu_mirror_delta() {
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
        let mut upload = MeshUpload::from_mesh(&mesh);
        let active = (0..mesh.positions.len() as u32).collect::<Vec<_>>();
        let mut recorder = crate::mesh::MeshEditRecorder::new(&mesh);
        let outcome = mesh.remesh_region(
            &active,
            crate::mesh::RemeshSettings {
                target_edge_length: 0.9,
                enable_flips: true,
                relaxation: 0.0,
                ..crate::mesh::RemeshSettings::default()
            },
            &mut recorder,
        );
        assert!(outcome.stats.splits > 0);
        let dirty = upload.apply_changes(&mesh, &outcome.changes);
        let rebuilt = MeshUpload::from_mesh(&mesh);

        assert!(!dirty.vertices.is_empty());
        assert!(!dirty.faces.is_empty());
        assert_eq!(upload.vertices, rebuilt.vertices);
        assert_eq!(upload.triangle_indices, rebuilt.triangle_indices);
        assert_eq!(
            upload.edges.iter().copied().collect::<HashSet<_>>(),
            rebuilt.edges.iter().copied().collect::<HashSet<_>>()
        );
    }

    #[test]
    #[ignore = "release-mode performance envelope"]
    fn half_million_vertex_deformation_pack() {
        let positions = vec![Vec3::ZERO; 500_000];
        let normals = vec![Vec3::Z; positions.len()];
        let masks = vec![0.0; positions.len()];
        let started = Instant::now();
        let vertices = MeshUpload::vertices(&positions, &normals, &masks);
        let elapsed = started.elapsed();
        assert_eq!(vertices.len(), positions.len());
        eprintln!("half-million vertex GPU pack: {elapsed:?}");
    }
}
