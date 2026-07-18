use std::mem::size_of;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use eframe::egui::{Color32, PaintCallback, Pos2, Rect, Stroke, Ui};
use egui_wgpu::wgpu;
use egui_wgpu::wgpu::util::DeviceExt as _;
use glam::Vec3;
use thiserror::Error;

use crate::camera::Camera;
use crate::mesh::Mesh;

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

        Ok(Self { shared })
    }

    /// Uploads the public geometry representation used by the mesh core. Invalid
    /// triangle indices are skipped instead of reaching wgpu validation.
    pub fn update_mesh(&self, mesh: &Mesh) {
        self.update_mesh_data(&mesh.positions, &mesh.normals, &mesh.mask, &mesh.triangles);
    }

    /// Lower-level upload entry point useful for previews and renderer tests.
    pub fn update_mesh_data(
        &self,
        positions: &[Vec3],
        normals: &[Vec3],
        masks: &[f32],
        triangles: &[[u32; 3]],
    ) {
        let upload = MeshUpload::new(positions, normals, masks, triangles);
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        input.mesh = upload;
        input.full_vertex_upload = true;
        input.dirty_vertices.clear();
        input.vertex_revision = input.vertex_revision.wrapping_add(1);
        input.topology_revision = input.topology_revision.wrapping_add(1);
    }

    /// Refreshes deformable per-vertex data without rebuilding or uploading
    /// the unchanged triangle and wireframe index buffers.
    pub fn update_vertices(&self, mesh: &Mesh) {
        let vertices = MeshUpload::vertices(&mesh.positions, &mesh.normals, &mesh.mask);
        let mut input = self
            .shared
            .write()
            .unwrap_or_else(|error| error.into_inner());
        input.mesh.vertices = vertices;
        input.full_vertex_upload = true;
        input.dirty_vertices.clear();
        input.vertex_revision = input.vertex_revision.wrapping_add(1);
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
    camera: CameraUniform,
    full_vertex_upload: bool,
    dirty_vertices: Vec<u32>,
}

#[derive(Default)]
struct MeshUpload {
    vertices: Vec<GpuVertex>,
    triangle_indices: Vec<u32>,
    edge_indices: Vec<u32>,
}

impl MeshUpload {
    fn new(positions: &[Vec3], normals: &[Vec3], masks: &[f32], triangles: &[[u32; 3]]) -> Self {
        let vertices = Self::vertices(positions, normals, masks);

        let mut triangle_indices = Vec::with_capacity(triangles.len().saturating_mul(3));
        let mut edge_indices = Vec::with_capacity(triangles.len().saturating_mul(6));
        let vertex_count = positions.len();
        for &[a, b, c] in triangles {
            if [a, b, c]
                .into_iter()
                .all(|index| (index as usize) < vertex_count)
            {
                triangle_indices.extend_from_slice(&[a, b, c]);
                edge_indices.extend_from_slice(&[a, b, b, c, c, a]);
            }
        }

        Self {
            vertices,
            triangle_indices,
            edge_indices,
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
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
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
                vertex_entry: "vs_wire",
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
        if self.vertices.buffer.is_none() {
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
        self.triangle_index_count = u32::try_from(mesh.triangle_indices.len()).unwrap_or(u32::MAX);
        self.edge_index_count = u32::try_from(mesh.edge_indices.len()).unwrap_or(u32::MAX);
    }
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
            gpu.upload_topology(device, queue, &input.mesh);
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
        assert_eq!(upload.edge_indices, [0, 1, 1, 2, 2, 0]);
        assert_eq!(upload.vertices[0].mask, 0.0);
        assert_eq!(upload.vertices[0].normal, Vec3::Y.to_array());
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
