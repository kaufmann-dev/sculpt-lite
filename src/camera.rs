use std::f32::consts::{FRAC_PI_2, TAU};

use eframe::egui::{Pos2, Rect, Vec2};
use glam::{Mat4, Vec3, Vec4};

const MIN_PITCH_MARGIN: f32 = 0.01;
const ORBIT_RADIANS_PER_POINT: f32 = 0.008;
const FLY_RADIANS_PER_POINT: f32 = 0.0025;
const INITIAL_FLY_SPEED: f32 = 0.15;
const MIN_FLY_SPEED: f32 = 0.002;
const MAX_FLY_SPEED: f32 = 2.0;
const MAX_FLY_DELTA_SECONDS: f32 = 0.1;
const MAX_FLY_WHEEL_POINTS: f32 = 240.0;
const FLY_WHEEL_EXPONENT_PER_POINT: f32 = 0.005;

/// A world-space ray cast from the perspective camera through the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Immutable projection values shared by viewport work and queued brush samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFrame {
    viewport: Rect,
    eye: Vec3,
    right: Vec3,
    up: Vec3,
    view_projection: Mat4,
    inverse_view_projection: Mat4,
    world_units_per_point: f32,
    distance: f32,
}

impl CameraFrame {
    #[must_use]
    pub fn screen_ray(self, pointer: Pos2) -> Option<Ray> {
        if !self.viewport.contains(pointer) {
            return None;
        }
        let ndc_x = ((pointer.x - self.viewport.left()) / self.viewport.width()) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((pointer.y - self.viewport.top()) / self.viewport.height()) * 2.0;
        let near = unproject(
            self.inverse_view_projection,
            Vec4::new(ndc_x, ndc_y, 0.0, 1.0),
        )?;
        let far = unproject(
            self.inverse_view_projection,
            Vec4::new(ndc_x, ndc_y, 1.0, 1.0),
        )?;
        let direction = (far - near).normalize_or_zero();
        if direction == Vec3::ZERO || !direction.is_finite() {
            return None;
        }
        Some(Ray {
            origin: self.eye,
            direction,
        })
    }

    #[must_use]
    pub fn eye(self) -> Vec3 {
        self.eye
    }

    #[must_use]
    pub fn right(self) -> Vec3 {
        self.right
    }

    #[must_use]
    pub fn up(self) -> Vec3 {
        self.up
    }

    #[must_use]
    pub fn view_projection(self) -> Mat4 {
        self.view_projection
    }

    #[must_use]
    pub fn world_units_per_point(self) -> f32 {
        self.world_units_per_point
    }

    #[must_use]
    pub fn distance(self) -> f32 {
        self.distance
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum CameraMode {
    #[default]
    Orbit,
    Fly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FlyMovement {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
}

/// Positive Z is up. `yaw == 0` places the eye on the positive X axis and the
/// orbit camera always looks at `target`.
#[derive(Clone, Debug)]
struct OrbitCamera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.45,
            pitch: 0.2,
            distance: 3.0,
        }
    }
}

impl OrbitCamera {
    fn eye(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let offset = Vec3::new(
            self.yaw.cos() * cos_pitch,
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
        );
        self.target + offset * self.distance
    }

    fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or_zero()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Z).normalize_or_zero()
    }

    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }
}

#[derive(Clone, Copy, Debug)]
struct FlyViewpoint {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

#[derive(Clone, Debug)]
struct FlyCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed_radii_per_second: f32,
    entry_viewpoint: FlyViewpoint,
}

impl FlyCamera {
    fn from_orbit(orbit: &OrbitCamera) -> Self {
        let direction = orbit.forward();
        let entry_viewpoint = FlyViewpoint {
            position: orbit.eye(),
            yaw: direction.y.atan2(direction.x).rem_euclid(TAU),
            pitch: direction.z.clamp(-1.0, 1.0).asin(),
        };
        let mut camera = Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed_radii_per_second: INITIAL_FLY_SPEED,
            entry_viewpoint,
        };
        camera.restore_entry_viewpoint();
        camera
    }

    fn restore_entry_viewpoint(&mut self) {
        self.position = self.entry_viewpoint.position;
        self.yaw = self.entry_viewpoint.yaw;
        self.pitch = self.entry_viewpoint.pitch;
        self.speed_radii_per_second = INITIAL_FLY_SPEED;
    }

    fn forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.cos() * cos_pitch,
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
        )
        .normalize_or_zero()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Z).normalize_or_zero()
    }
}

/// Independent Orbit and no-clip Fly cameras sharing one render and picking frame.
#[derive(Clone, Debug)]
pub struct Camera {
    mode: CameraMode,
    orbit: OrbitCamera,
    fly: FlyCamera,
    fly_entered: bool,
    fov_y_radians: f32,
    bounds_minimum: Vec3,
    bounds_maximum: Vec3,
    scene_radius: f32,
}

impl Default for Camera {
    fn default() -> Self {
        let orbit = OrbitCamera::default();
        let fly = FlyCamera::from_orbit(&orbit);
        Self {
            mode: CameraMode::Orbit,
            orbit,
            fly,
            fly_entered: false,
            fov_y_radians: 45.0_f32.to_radians(),
            bounds_minimum: Vec3::splat(-1.0 / 3.0_f32.sqrt()),
            bounds_maximum: Vec3::splat(1.0 / 3.0_f32.sqrt()),
            scene_radius: 1.0,
        }
    }
}

impl Camera {
    #[must_use]
    pub(crate) fn mode(&self) -> CameraMode {
        self.mode
    }

    pub(crate) fn set_mode(&mut self, mode: CameraMode) {
        if mode == CameraMode::Fly && !self.fly_entered {
            self.fly = FlyCamera::from_orbit(&self.orbit);
            self.fly_entered = true;
        }
        self.mode = mode;
    }

    /// Frames a world-space bounding box, resets both viewpoints, and selects Orbit.
    pub fn fit(&mut self, min: Vec3, max: Vec3) {
        if !min.is_finite() || !max.is_finite() {
            return;
        }

        let low = min.min(max);
        let high = min.max(max);
        self.bounds_minimum = low;
        self.bounds_maximum = high;
        self.orbit.target = (low + high) * 0.5;
        self.scene_radius = ((high - low).length() * 0.5).max(1.0e-4);

        // A bounding sphere is aspect-ratio independent and leaves enough room
        // for the brush cursor at the edges of the object.
        self.orbit.distance = (self.scene_radius / (self.fov_y_radians * 0.5).sin()) * 1.15;
        self.fly = FlyCamera::from_orbit(&self.orbit);
        self.fly_entered = false;
        self.mode = CameraMode::Orbit;
    }

    /// Orbits by an egui pointer delta measured in logical points.
    pub fn orbit(&mut self, delta_points: Vec2) {
        if !delta_points.is_finite() {
            return;
        }
        self.orbit.yaw =
            (self.orbit.yaw - delta_points.x * ORBIT_RADIANS_PER_POINT).rem_euclid(TAU);
        self.orbit.pitch = (self.orbit.pitch + delta_points.y * ORBIT_RADIANS_PER_POINT)
            .clamp(-FRAC_PI_2 + MIN_PITCH_MARGIN, FRAC_PI_2 - MIN_PITCH_MARGIN);
    }

    /// Pans so the object tracks the pointer. `viewport_height_points` is the
    /// height of the custom viewport, not the whole window.
    pub fn pan(&mut self, delta_points: Vec2, viewport_height_points: f32) {
        if !delta_points.is_finite() || !viewport_height_points.is_finite() {
            return;
        }
        let scale = self.orbit_world_units_per_point(viewport_height_points);
        self.orbit.target -= self.orbit.right() * delta_points.x * scale;
        self.orbit.target += self.orbit.up() * delta_points.y * scale;
    }

    /// Zooms Orbit exponentially. Positive deltas zoom in; passing
    /// `egui::InputState::smooth_scroll_delta.y` directly is appropriate.
    pub fn zoom(&mut self, scroll_delta: f32) {
        if !scroll_delta.is_finite() {
            return;
        }

        self.orbit.distance *= (-scroll_delta * 0.0015).exp();
        let radius = self.scene_radius.max(1.0e-4);
        self.orbit.distance = self.orbit.distance.clamp(radius * 0.005, radius * 10_000.0);
    }

    pub(crate) fn fly_look(&mut self, delta_points: Vec2) {
        if !delta_points.is_finite() {
            return;
        }
        self.fly.yaw = (self.fly.yaw - delta_points.x * FLY_RADIANS_PER_POINT).rem_euclid(TAU);
        self.fly.pitch = (self.fly.pitch - delta_points.y * FLY_RADIANS_PER_POINT)
            .clamp(-FRAC_PI_2 + MIN_PITCH_MARGIN, FRAC_PI_2 - MIN_PITCH_MARGIN);
    }

    pub(crate) fn fly_move(&mut self, movement: FlyMovement, delta_seconds: f32) {
        if !delta_seconds.is_finite() {
            return;
        }
        let delta_seconds = delta_seconds.clamp(0.0, MAX_FLY_DELTA_SECONDS);
        let finite_axis = |value: f32| {
            if value.is_finite() {
                value.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };
        let direction = self.fly.forward() * finite_axis(movement.forward)
            + self.fly.right() * finite_axis(movement.right)
            + Vec3::Z * finite_axis(movement.up);
        let direction = direction.normalize_or_zero();
        let displacement = direction
            * self.scene_radius.max(1.0e-4)
            * self.fly.speed_radii_per_second
            * delta_seconds;
        let position = self.fly.position + displacement;
        if position.is_finite() {
            self.fly.position = position;
        }
    }

    pub(crate) fn adjust_fly_speed(&mut self, wheel_points: f32) {
        if !wheel_points.is_finite() {
            return;
        }
        let exponent = wheel_points.clamp(-MAX_FLY_WHEEL_POINTS, MAX_FLY_WHEEL_POINTS)
            * FLY_WHEEL_EXPONENT_PER_POINT;
        self.fly.speed_radii_per_second =
            (self.fly.speed_radii_per_second * exponent.exp()).clamp(MIN_FLY_SPEED, MAX_FLY_SPEED);
    }

    #[must_use]
    pub(crate) fn fly_speed(&self) -> f32 {
        self.fly.speed_radii_per_second
    }

    /// Computes camera data once for all viewport work in the current frame.
    #[must_use]
    pub fn frame(&self, viewport: Rect) -> Option<CameraFrame> {
        if viewport.width() <= 0.0
            || viewport.height() <= 0.0
            || !viewport.width().is_finite()
            || !viewport.height().is_finite()
        {
            return None;
        }

        match self.mode {
            CameraMode::Orbit => {
                let (near, far) = self.orbit_near_far();
                build_frame(
                    viewport,
                    self.orbit.eye(),
                    self.orbit.forward(),
                    self.fov_y_radians,
                    near,
                    far,
                    self.orbit.distance,
                )
            }
            CameraMode::Fly => {
                let (near, far) = self.fly_near_far();
                build_frame(
                    viewport,
                    self.fly.position,
                    self.fly.forward(),
                    self.fov_y_radians,
                    near,
                    far,
                    self.scene_radius,
                )
            }
        }
    }

    fn orbit_world_units_per_point(&self, viewport_height_points: f32) -> f32 {
        if viewport_height_points <= 0.0 {
            return 0.0;
        }
        2.0 * self.orbit.distance * (self.fov_y_radians * 0.5).tan() / viewport_height_points
    }

    fn orbit_near_far(&self) -> (f32, f32) {
        let radius = self.scene_radius.max(1.0e-4);
        let near = (self.orbit.distance - radius * 2.5)
            .max(radius * 0.001)
            .max(1.0e-6);
        let far = (self.orbit.distance + radius * 4.0).max(near * 2.0);
        (near, far)
    }

    fn fly_near_far(&self) -> (f32, f32) {
        let radius = self.scene_radius.max(1.0e-4);
        let center = (self.bounds_minimum + self.bounds_maximum) * 0.5;
        let distance = self.fly.position.distance(center);
        let distance = if distance.is_finite() {
            distance
        } else {
            radius
        };
        let near = (distance - radius * 1.75).max(radius * 0.0001).max(1.0e-6);
        let far = (distance + radius * 4.0).max(radius * 4.0).max(near * 2.0);
        (near, far)
    }
}

fn build_frame(
    viewport: Rect,
    eye: Vec3,
    forward: Vec3,
    fov_y_radians: f32,
    near: f32,
    far: f32,
    reference_distance: f32,
) -> Option<CameraFrame> {
    if !eye.is_finite()
        || forward == Vec3::ZERO
        || !forward.is_finite()
        || !near.is_finite()
        || !far.is_finite()
        || near <= 0.0
        || far <= near
    {
        return None;
    }
    let right = forward.cross(Vec3::Z).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let view = glam::camera::rh::view::look_at_mat4(eye, eye + forward, Vec3::Z);
    let projection = glam::camera::rh::proj::directx::perspective(
        fov_y_radians.clamp(5.0_f32.to_radians(), 120.0_f32.to_radians()),
        (viewport.width() / viewport.height()).max(1.0e-4),
        near,
        far,
    );
    let view_projection = projection * view;
    let inverse_view_projection = view_projection.inverse();
    if right == Vec3::ZERO
        || up == Vec3::ZERO
        || !view_projection.is_finite()
        || !inverse_view_projection.is_finite()
    {
        return None;
    }
    let reference_distance = reference_distance.max(1.0e-4);
    Some(CameraFrame {
        viewport,
        eye,
        right,
        up,
        view_projection,
        inverse_view_projection,
        world_units_per_point: 2.0 * reference_distance * (fov_y_radians * 0.5).tan()
            / viewport.height(),
        distance: reference_distance,
    })
}

fn unproject(inverse_view_projection: Mat4, clip: Vec4) -> Option<Vec3> {
    let world = inverse_view_projection * clip;
    if !world.is_finite() || world.w.abs() <= f32::EPSILON {
        return None;
    }
    Some(world.truncate() / world.w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn viewport() -> Rect {
        Rect::from_min_max(Pos2::new(100.0, 50.0), Pos2::new(900.0, 650.0))
    }

    #[test]
    fn center_screen_ray_points_at_orbit_target() {
        let camera = Camera::default();
        let ray = camera
            .frame(viewport())
            .expect("valid viewport")
            .screen_ray(viewport().center())
            .expect("center is inside viewport");

        assert!(ray.direction.dot(camera.orbit.forward()) > 0.999_99);
        assert_eq!(ray.origin, camera.orbit.eye());
    }

    #[test]
    fn screen_ray_rejects_positions_outside_viewport() {
        let camera = Camera::default();
        assert!(
            camera
                .frame(viewport())
                .expect("valid viewport")
                .screen_ray(Pos2::new(20.0, 20.0))
                .is_none()
        );
    }

    #[test]
    fn fit_centers_and_frames_bounds() {
        let mut camera = Camera::default();
        camera.fit(Vec3::new(-2.0, 1.0, -1.0), Vec3::new(4.0, 5.0, 3.0));

        assert!(
            camera
                .orbit
                .target
                .abs_diff_eq(Vec3::new(1.0, 3.0, 1.0), 1.0e-6)
        );
        assert!(camera.orbit.distance > Vec3::new(3.0, 2.0, 2.0).length());
    }

    #[test]
    fn orbit_drag_right_decreases_yaw() {
        let mut camera = Camera::default();
        let yaw = camera.orbit.yaw;

        camera.orbit(Vec2::new(40.0, 0.0));

        let expected = (yaw - 40.0 * ORBIT_RADIANS_PER_POINT).rem_euclid(TAU);
        assert!((camera.orbit.yaw - expected).abs() < 1.0e-6);
    }

    #[test]
    fn orbit_drag_left_increases_yaw() {
        let mut camera = Camera::default();
        let yaw = camera.orbit.yaw;

        camera.orbit(Vec2::new(-40.0, 0.0));

        assert!((camera.orbit.yaw - (yaw + 40.0 * ORBIT_RADIANS_PER_POINT)).abs() < 1.0e-6);
    }

    #[test]
    fn horizontal_orbit_keeps_world_z_upright() {
        let mut camera = Camera::default();
        let pitch = camera.orbit.pitch;

        assert!(camera.orbit.right().dot(Vec3::Z).abs() < 1.0e-6);
        camera.orbit(Vec2::new(-80.0, 0.0));

        assert!((camera.orbit.pitch - pitch).abs() < 1.0e-6);
        assert!(camera.orbit.right().dot(Vec3::Z).abs() < 1.0e-6);
        assert!(camera.orbit.up().dot(Vec3::Z) > 0.0);
    }

    #[test]
    fn orbit_allows_vertical_rotation_without_crossing_poles() {
        let mut camera = Camera::default();
        camera.orbit(Vec2::new(0.0, 100_000.0));
        assert!(camera.orbit.pitch < FRAC_PI_2);
        camera.orbit(Vec2::new(0.0, -200_000.0));
        assert!(camera.orbit.pitch > -FRAC_PI_2);
    }

    #[test]
    fn horizontal_pan_makes_the_object_track_the_pointer() {
        let mut camera = Camera::default();
        let before = camera.orbit.target;
        let right = camera.orbit.right();
        camera.pan(Vec2::new(40.0, 0.0), 600.0);

        assert!((camera.orbit.target - before).dot(right) < 0.0);
        assert!((camera.orbit.target - before).dot(camera.orbit.up()).abs() < 1.0e-6);
    }

    #[test]
    fn world_scale_matches_vertical_field_of_view() {
        let camera = Camera::default();
        let height = 600.0;
        let visible_world_height = camera.orbit_world_units_per_point(height) * height;
        let expected = 2.0 * camera.orbit.distance * (camera.fov_y_radians * 0.5).tan();
        assert!((visible_world_height - expected).abs() < 1.0e-5);
    }

    #[test]
    fn first_fly_entry_preserves_orbit_viewpoint() {
        let mut camera = Camera::default();
        camera.orbit(Vec2::new(75.0, -30.0));
        let orbit_frame = camera.frame(viewport()).unwrap();
        let orbit_ray = orbit_frame.screen_ray(viewport().center()).unwrap();

        camera.set_mode(CameraMode::Fly);
        let fly_frame = camera.frame(viewport()).unwrap();
        let fly_ray = fly_frame.screen_ray(viewport().center()).unwrap();

        assert!(fly_frame.eye().abs_diff_eq(orbit_frame.eye(), 1.0e-5));
        assert!(fly_ray.direction.dot(orbit_ray.direction) > 0.999_99);
    }

    #[test]
    fn orbit_and_fly_viewpoints_remain_independent() {
        let mut camera = Camera::default();
        let orbit_before = camera.orbit.clone();
        camera.set_mode(CameraMode::Fly);
        camera.fly_look(Vec2::new(80.0, -30.0));
        camera.fly_move(
            FlyMovement {
                forward: 1.0,
                ..FlyMovement::default()
            },
            0.1,
        );
        let fly_before = camera.fly.clone();

        camera.set_mode(CameraMode::Orbit);
        assert!(camera.orbit.eye().abs_diff_eq(orbit_before.eye(), 1.0e-6));
        camera.orbit(Vec2::new(20.0, 10.0));
        camera.set_mode(CameraMode::Fly);

        assert!(camera.fly.position.abs_diff_eq(fly_before.position, 1.0e-6));
        assert_eq!(camera.fly.yaw, fly_before.yaw);
        assert_eq!(camera.fly.pitch, fly_before.pitch);
    }

    #[test]
    fn fly_diagonal_movement_is_normalized() {
        let mut straight = Camera::default();
        straight.set_mode(CameraMode::Fly);
        let start = straight.fly.position;
        straight.fly_move(
            FlyMovement {
                forward: 1.0,
                ..FlyMovement::default()
            },
            0.1,
        );

        let mut diagonal = Camera::default();
        diagonal.set_mode(CameraMode::Fly);
        let diagonal_start = diagonal.fly.position;
        diagonal.fly_move(
            FlyMovement {
                forward: 1.0,
                right: 1.0,
                up: 1.0,
            },
            0.1,
        );

        let straight_distance = straight.fly.position.distance(start);
        let diagonal_distance = diagonal.fly.position.distance(diagonal_start);
        assert!((straight_distance - diagonal_distance).abs() < 1.0e-6);
    }

    #[test]
    fn fly_frame_delta_cap_limits_movement() {
        let movement = FlyMovement {
            forward: 1.0,
            ..FlyMovement::default()
        };
        let mut normal = Camera::default();
        normal.set_mode(CameraMode::Fly);
        let start = normal.fly.position;
        normal.fly_move(movement, 0.1);
        let normal_distance = normal.fly.position.distance(start);

        let mut capped = Camera::default();
        capped.set_mode(CameraMode::Fly);
        let start = capped.fly.position;
        capped.fly_move(movement, 10.0);
        let capped_distance = capped.fly.position.distance(start);

        assert!((capped_distance - normal_distance).abs() < 1.0e-6);
    }

    #[test]
    fn fly_speed_adjustment_stays_within_bounds() {
        let mut camera = Camera::default();
        camera.adjust_fly_speed(f32::INFINITY);
        assert_eq!(camera.fly_speed(), INITIAL_FLY_SPEED);
        for _ in 0..20 {
            camera.adjust_fly_speed(MAX_FLY_WHEEL_POINTS);
        }
        assert_eq!(camera.fly_speed(), MAX_FLY_SPEED);
        for _ in 0..40 {
            camera.adjust_fly_speed(-MAX_FLY_WHEEL_POINTS);
        }
        assert_eq!(camera.fly_speed(), MIN_FLY_SPEED);
    }

    #[test]
    fn non_finite_fly_inputs_do_not_corrupt_the_viewpoint() {
        let mut camera = Camera::default();
        camera.set_mode(CameraMode::Fly);
        let before = camera.fly.clone();

        camera.fly_look(Vec2::new(f32::NAN, 1.0));
        camera.fly_move(
            FlyMovement {
                forward: f32::NAN,
                right: f32::INFINITY,
                up: f32::NEG_INFINITY,
            },
            0.1,
        );
        camera.fly_move(
            FlyMovement {
                forward: 1.0,
                ..FlyMovement::default()
            },
            f32::NAN,
        );
        camera.adjust_fly_speed(f32::NAN);

        assert!(camera.fly.position.abs_diff_eq(before.position, 1.0e-6));
        assert_eq!(camera.fly.yaw, before.yaw);
        assert_eq!(camera.fly.pitch, before.pitch);
        assert_eq!(
            camera.fly.speed_radii_per_second,
            before.speed_radii_per_second
        );
        assert!(camera.frame(viewport()).is_some());
    }

    #[test]
    fn fly_look_wraps_yaw_and_clamps_pitch() {
        let mut camera = Camera::default();
        camera.set_mode(CameraMode::Fly);
        camera.fly_look(Vec2::new(1_000_000.0, 1_000_000.0));
        assert!((0.0..TAU).contains(&camera.fly.yaw));
        assert!(camera.fly.pitch > -FRAC_PI_2);
        camera.fly_look(Vec2::new(-2_000_000.0, -2_000_000.0));
        assert!((0.0..TAU).contains(&camera.fly.yaw));
        assert!(camera.fly.pitch < FRAC_PI_2);
    }

    #[test]
    fn fly_horizontal_look_tracks_mouse_direction() {
        let mut rightward = Camera::default();
        rightward.set_mode(CameraMode::Fly);
        let right = rightward.fly.right();
        rightward.fly_look(Vec2::new(40.0, 0.0));
        assert!(rightward.fly.forward().dot(right) > 0.0);

        let mut leftward = Camera::default();
        leftward.set_mode(CameraMode::Fly);
        let right = leftward.fly.right();
        leftward.fly_look(Vec2::new(-40.0, 0.0));
        assert!(leftward.fly.forward().dot(right) < 0.0);
    }

    #[test]
    fn fitting_resets_both_cameras_and_returns_to_orbit() {
        let mut camera = Camera::default();
        camera.set_mode(CameraMode::Fly);
        camera.fly_move(
            FlyMovement {
                forward: 1.0,
                ..FlyMovement::default()
            },
            0.1,
        );
        camera.adjust_fly_speed(200.0);

        camera.fit(Vec3::splat(-2.0), Vec3::splat(2.0));
        assert_eq!(camera.mode(), CameraMode::Orbit);
        camera.set_mode(CameraMode::Fly);

        assert!(camera.fly.position.abs_diff_eq(camera.orbit.eye(), 1.0e-5));
        assert_eq!(camera.fly_speed(), INITIAL_FLY_SPEED);
    }

    #[test]
    fn fly_projection_and_center_ray_remain_finite_inside_and_outside_bounds() {
        let mut camera = Camera::default();
        camera.fit(Vec3::splat(-1.0), Vec3::splat(1.0));
        camera.set_mode(CameraMode::Fly);
        for position in [Vec3::ZERO, Vec3::new(0.0, 0.0, 1.001), Vec3::splat(100.0)] {
            camera.fly.position = position;
            let frame = camera.frame(viewport()).expect("finite fly frame");
            let ray = frame
                .screen_ray(viewport().center())
                .expect("finite center ray");
            assert!(frame.view_projection().is_finite());
            assert!(ray.origin.is_finite());
            assert!(ray.direction.is_finite());
        }
    }
}
