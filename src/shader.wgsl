struct CameraUniform {
    view_projection: mat4x4<f32>,
    eye: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
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
    // Comparisons against NaN are false, so this also rejects invalid STL
    // normals without allowing them to poison the fragment color.
    let valid = squared_length > 1.0e-12 && squared_length < 1.0e20;
    let safe_squared_length = select(1.0, squared_length, valid);
    let normalized = value * inverseSqrt(safe_squared_length);
    return select(fallback, normalized, valid);
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

@fragment
fn fs_solid(input: VertexOutput) -> @location(0) vec4<f32> {
    let view = safe_normalize(
        camera.eye.xyz - input.world_position,
        vec3<f32>(0.0, 0.0, 1.0),
    );

    // Derivatives provide a reliable fallback for meshes whose imported or
    // locally edited vertex normals are zero. Orient the geometric normal to
    // the camera, then align the smooth normal with it; this makes both sides
    // of open STL shells shade consistently without relying on winding.
    var geometric_normal = safe_normalize(
        cross(dpdx(input.world_position), dpdy(input.world_position)),
        view,
    );
    if (dot(geometric_normal, view) < 0.0) {
        geometric_normal = -geometric_normal;
    }
    var normal = safe_normalize(input.world_normal, geometric_normal);
    if (dot(normal, geometric_normal) < 0.0) {
        normal = -normal;
    }

    let camera_right = safe_normalize(camera.camera_right.xyz, vec3<f32>(1.0, 0.0, 0.0));
    let camera_up = safe_normalize(camera.camera_up.xyz, vec3<f32>(0.0, 1.0, 0.0));

    // A camera-relative studio rig keeps the sculpt legible while it rotates:
    // a soft upper-left key, a weaker lower-right fill, and hemispheric fill.
    let key_light = safe_normalize(
        view * 0.78 - camera_right * 0.42 + camera_up * 0.52,
        view,
    );
    let fill_light = safe_normalize(
        view * 0.48 + camera_right * 0.62 - camera_up * 0.12,
        view,
    );
    let key_wrapped = clamp((dot(normal, key_light) + 0.30) / 1.30, 0.0, 1.0);
    let fill_wrapped = clamp((dot(normal, fill_light) + 0.55) / 1.55, 0.0, 1.0);
    let upper_hemisphere = clamp(dot(normal, camera_up) * 0.5 + 0.5, 0.0, 1.0);
    let hemisphere = mix(0.34, 0.45, upper_hemisphere);
    let diffuse = hemisphere + key_wrapped * 0.46 + fill_wrapped * 0.15;

    let surface_color = vec3<f32>(0.48, 0.51, 0.55);
    let masked_color = vec3<f32>(0.78, 0.12, 0.055);
    let base = mix(surface_color, masked_color, clamp(input.mask, 0.0, 1.0) * 0.84);

    // Normal variation adds restrained local-form contrast. The rational
    // response remains bounded at dense silhouettes instead of creating an
    // outline or becoming resolution-dependent without limit.
    let normal_variation = length(dpdx(normal)) + length(dpdy(normal));
    let curvature = normal_variation / (normal_variation + 0.35);
    let form_contrast = 1.0 - curvature * camera.material.x;

    // Broad, low-energy highlights describe the clay surface. Keep the rim
    // base-tinted and weak so silhouettes never blow out to white.
    let half_vector = safe_normalize(key_light + view, normal);
    let specular = pow(clamp(dot(normal, half_vector), 0.0, 1.0), 20.0)
        * camera.material.y;
    let view_facing = clamp(dot(normal, view), 0.0, 1.0);
    let rim = pow(1.0 - view_facing, 2.5) * camera.material.z;
    let highlight_tint = vec3<f32>(0.72, 0.76, 0.82);
    let color = base * diffuse * form_contrast
        + highlight_tint * specular
        + base * rim;

    return vec4<f32>(min(color, vec3<f32>(0.90)), 1.0);
}

@fragment
fn fs_wire(_input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.055, 0.065, 0.082, 1.0);
}
