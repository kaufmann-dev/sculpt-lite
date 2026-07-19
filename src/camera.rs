use std::f32::consts::{FRAC_PI_2, TAU};

use eframe::egui::{Pos2, Rect, Vec2};
use glam::{Mat4, Vec3, Vec4};

const MIN_PITCH_MARGIN: f32 = 0.01;
const ORBIT_RADIANS_PER_POINT: f32 = 0.008;

/// A world-space ray cast from the perspective camera through the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Orbit camera centered on the active sculpting area.
///
/// Positive Y is up. `yaw == 0` places the eye on the positive Z axis and the
/// camera always looks at `target`.
#[derive(Clone, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y_radians: f32,
    scene_radius: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.45,
            pitch: 0.2,
            distance: 3.0,
            fov_y_radians: 45.0_f32.to_radians(),
            scene_radius: 1.0,
        }
    }
}

impl Camera {
    /// Frames an axis-aligned world-space bounding box while preserving the
    /// current viewing direction.
    pub fn fit(&mut self, min: Vec3, max: Vec3) {
        if !min.is_finite() || !max.is_finite() {
            return;
        }

        let low = min.min(max);
        let high = min.max(max);
        self.target = (low + high) * 0.5;
        self.scene_radius = ((high - low).length() * 0.5).max(1.0e-4);

        // A bounding sphere is aspect-ratio independent and leaves enough room
        // for the brush cursor at the edges of the object.
        self.distance = (self.scene_radius / (self.fov_y_radians * 0.5).sin()) * 1.15;
    }

    #[must_use]
    pub fn eye_position(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let offset = Vec3::new(
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            self.yaw.cos() * cos_pitch,
        );
        self.target + offset * self.distance
    }

    #[must_use]
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye_position()).normalize_or_zero()
    }

    #[must_use]
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    #[must_use]
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye_position(), self.target, Vec3::Y)
    }

    #[must_use]
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.near_far();
        glam::camera::rh::proj::directx::perspective(
            self.fov_y_radians
                .clamp(5.0_f32.to_radians(), 120.0_f32.to_radians()),
            aspect.max(1.0e-4),
            near,
            far,
        )
    }

    #[must_use]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    /// Orbits by an egui pointer delta measured in logical points.
    pub fn orbit(&mut self, delta_points: Vec2) {
        self.yaw = (self.yaw + delta_points.x * ORBIT_RADIANS_PER_POINT).rem_euclid(TAU);
        self.pitch = (self.pitch + delta_points.y * ORBIT_RADIANS_PER_POINT)
            .clamp(-FRAC_PI_2 + MIN_PITCH_MARGIN, FRAC_PI_2 - MIN_PITCH_MARGIN);
    }

    /// Pans so the object tracks the pointer. `viewport_height_points` is the
    /// height of the custom viewport, not the whole window.
    pub fn pan(&mut self, delta_points: Vec2, viewport_height_points: f32) {
        let scale = self.world_units_per_pixel(viewport_height_points);
        self.target -= self.right() * delta_points.x * scale;
        self.target += self.up() * delta_points.y * scale;
    }

    /// Zooms exponentially. Positive deltas zoom in; passing
    /// `egui::InputState::smooth_scroll_delta.y` directly is appropriate.
    pub fn zoom(&mut self, scroll_delta: f32) {
        if !scroll_delta.is_finite() {
            return;
        }

        self.distance *= (-scroll_delta * 0.0015).exp();
        let radius = self.scene_radius.max(1.0e-4);
        self.distance = self.distance.clamp(radius * 0.005, radius * 10_000.0);
    }

    /// World-space height represented by one logical viewport point at the
    /// orbit target's depth.
    #[must_use]
    pub fn world_units_per_pixel(&self, viewport_height_points: f32) -> f32 {
        if viewport_height_points <= 0.0 {
            return 0.0;
        }
        2.0 * self.distance * (self.fov_y_radians * 0.5).tan() / viewport_height_points
    }

    /// Returns a perspective ray for a logical screen position. Positions
    /// outside the supplied viewport produce `None`.
    #[must_use]
    pub fn screen_ray(&self, pointer: Pos2, viewport: Rect) -> Option<Ray> {
        if !viewport.contains(pointer) || viewport.width() <= 0.0 || viewport.height() <= 0.0 {
            return None;
        }

        let ndc_x = ((pointer.x - viewport.left()) / viewport.width()) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((pointer.y - viewport.top()) / viewport.height()) * 2.0;
        let aspect = viewport.width() / viewport.height();
        let inverse = self.view_projection(aspect).inverse();

        let near = unproject(inverse, Vec4::new(ndc_x, ndc_y, 0.0, 1.0))?;
        let far = unproject(inverse, Vec4::new(ndc_x, ndc_y, 1.0, 1.0))?;
        let direction = (far - near).normalize_or_zero();
        if direction == Vec3::ZERO || !direction.is_finite() {
            return None;
        }

        Some(Ray {
            origin: self.eye_position(),
            direction,
        })
    }

    fn near_far(&self) -> (f32, f32) {
        let radius = self.scene_radius.max(1.0e-4);
        let near = (self.distance - radius * 2.5)
            .max(radius * 0.001)
            .max(1.0e-6);
        let far = (self.distance + radius * 4.0).max(near * 2.0);
        (near, far)
    }
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
    fn center_screen_ray_points_at_target() {
        let camera = Camera::default();
        let ray = camera
            .screen_ray(viewport().center(), viewport())
            .expect("center is inside viewport");

        assert!(ray.direction.dot(camera.forward()) > 0.999_99);
        assert_eq!(ray.origin, camera.eye_position());
    }

    #[test]
    fn screen_ray_rejects_positions_outside_viewport() {
        let camera = Camera::default();
        assert!(
            camera
                .screen_ray(Pos2::new(20.0, 20.0), viewport())
                .is_none()
        );
    }

    #[test]
    fn fit_centers_and_frames_bounds() {
        let mut camera = Camera::default();
        camera.fit(Vec3::new(-2.0, 1.0, -1.0), Vec3::new(4.0, 5.0, 3.0));

        assert!(camera.target.abs_diff_eq(Vec3::new(1.0, 3.0, 1.0), 1.0e-6));
        assert!(camera.distance > Vec3::new(3.0, 2.0, 2.0).length());
    }

    #[test]
    fn orbit_drag_right_increases_yaw() {
        let mut camera = Camera::default();
        let yaw = camera.yaw;

        camera.orbit(Vec2::new(40.0, 0.0));

        assert!((camera.yaw - (yaw + 40.0 * ORBIT_RADIANS_PER_POINT)).abs() < 1.0e-6);
    }

    #[test]
    fn orbit_drag_left_decreases_yaw() {
        let mut camera = Camera::default();
        let yaw = camera.yaw;

        camera.orbit(Vec2::new(-40.0, 0.0));

        assert!((camera.yaw - (yaw - 40.0 * ORBIT_RADIANS_PER_POINT)).abs() < 1.0e-6);
    }

    #[test]
    fn orbit_allows_vertical_rotation_without_crossing_poles() {
        let mut camera = Camera::default();
        camera.orbit(Vec2::new(0.0, 100_000.0));
        assert!(camera.pitch < FRAC_PI_2);
        camera.orbit(Vec2::new(0.0, -200_000.0));
        assert!(camera.pitch > -FRAC_PI_2);
    }

    #[test]
    fn horizontal_pan_makes_the_object_track_the_pointer() {
        let mut camera = Camera::default();
        let before = camera.target;
        let right = camera.right();
        camera.pan(Vec2::new(40.0, 0.0), 600.0);

        assert!((camera.target - before).dot(right) < 0.0);
        assert!((camera.target - before).dot(camera.up()).abs() < 1.0e-6);
    }

    #[test]
    fn world_scale_matches_vertical_field_of_view() {
        let camera = Camera::default();
        let height = 600.0;
        let visible_world_height = camera.world_units_per_pixel(height) * height;
        let expected = 2.0 * camera.distance * (camera.fov_y_radians * 0.5).tan();
        assert!((visible_world_height - expected).abs() < 1.0e-5);
    }
}
