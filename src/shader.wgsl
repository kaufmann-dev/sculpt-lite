struct CameraUniform {
    view_projection: mat4x4<f32>,
    eye: vec4<f32>,
    key_light: vec4<f32>,
    material: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) mask: f32,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) mask: f32,
};

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let squared_length = dot(value, value);
    let normalized = value * inverseSqrt(max(squared_length, 1.0e-12));
    return select(fallback, normalized, squared_length > 1.0e-12);
}

fn vertex_common(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.world_position = input.position;
    output.world_normal = input.normal;
    output.mask = input.mask;
    return output;
}

@vertex
fn vs_mesh(input: VertexInput) -> VertexOutput {
    return vertex_common(input);
}

@vertex
fn vs_wire(input: VertexInput) -> VertexOutput {
    var output = vertex_common(input);
    // Pull the line overlay slightly toward the camera without changing the
    // sculpt surface depth values.
    output.clip_position.z -= output.clip_position.w * 0.0001;
    return output;
}

@fragment
fn fs_solid(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var normal = safe_normalize(input.world_normal, vec3<f32>(0.0, 1.0, 0.0));
    if (!front_facing) {
        normal = -normal;
    }

    let light = safe_normalize(camera.key_light.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let fill_light = safe_normalize(vec3<f32>(0.55, 0.15, -0.82), vec3<f32>(0.0, 0.0, -1.0));
    let view = safe_normalize(camera.eye.xyz - input.world_position, vec3<f32>(0.0, 0.0, 1.0));
    let half_vector = safe_normalize(light + view, normal);

    let key = max(dot(normal, light), 0.0);
    let fill = max(dot(normal, fill_light), 0.0);
    let hemisphere = 0.28 + 0.13 * normal.y;
    let specular = pow(max(dot(normal, half_vector), 0.0), 42.0) * camera.material.y;
    let rim = pow(1.0 - max(dot(normal, view), 0.0), 3.0) * 0.16;

    let surface_color = vec3<f32>(0.72, 0.75, 0.79);
    let masked_color = vec3<f32>(0.92, 0.27, 0.16);
    let base = mix(surface_color, masked_color, clamp(input.mask, 0.0, 1.0) * 0.82);

    // Screen-space normal variation gently darkens curved detail, making subtle
    // sculpting changes readable without a heavy cavity post-process.
    let curvature = clamp(
        (length(dpdx(normal)) + length(dpdy(normal))) * camera.material.x,
        0.0,
        0.7,
    );
    let diffuse = hemisphere + key * 0.66 + fill * 0.18;
    let color = base * diffuse * (1.0 - curvature * 0.28)
        + vec3<f32>(specular + rim);
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_wire(_input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.055, 0.065, 0.082, 1.0);
}
