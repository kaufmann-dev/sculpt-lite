use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use eframe::egui::{Pos2, Vec2};

pub const MAX_DABS_PER_FRAME: usize = 64;

#[derive(Debug)]
pub struct StrokeSampler {
    initial_dab: Option<Pos2>,
    observed_pointer: Pos2,
    path_pointer: Pos2,
    pending_path: VecDeque<Pos2>,
    distance_since_dab: f32,
    elapsed_since_dab: Duration,
    last_advanced_at: Instant,
    released: bool,
}

impl StrokeSampler {
    #[must_use]
    pub fn begin(pointer: Pos2, now: Instant) -> Self {
        Self {
            initial_dab: Some(pointer),
            observed_pointer: pointer,
            path_pointer: pointer,
            pending_path: VecDeque::new(),
            distance_since_dab: 0.0,
            elapsed_since_dab: Duration::ZERO,
            last_advanced_at: now,
            released: false,
        }
    }

    #[must_use]
    pub fn take_initial_dab(&mut self) -> Option<Pos2> {
        self.initial_dab.take()
    }

    pub fn enqueue_pointer(&mut self, pointer: Pos2) {
        if pointer != self.observed_pointer {
            self.observed_pointer = pointer;
            if self.pending_path.back().copied() != Some(pointer) {
                self.pending_path.push_back(pointer);
            }
        }
    }

    #[must_use]
    pub fn drain_spatial_dabs(&mut self, spacing: f32, limit: usize) -> Vec<Pos2> {
        let spacing = spacing.max(1.0);
        let mut dabs = Vec::with_capacity(limit.min(self.pending_path.len()));
        while dabs.len() < limit {
            let Some(target) = self.pending_path.front().copied() else {
                break;
            };
            let segment = target - self.path_pointer;
            let length = segment.length();
            if length <= f32::EPSILON {
                self.path_pointer = target;
                self.pending_path.pop_front();
                continue;
            }

            let distance_to_dab = (spacing - self.distance_since_dab).max(f32::EPSILON);
            if length + f32::EPSILON < distance_to_dab {
                self.distance_since_dab += length;
                self.path_pointer = target;
                self.pending_path.pop_front();
                continue;
            }

            self.path_pointer += segment / length * distance_to_dab;
            self.distance_since_dab = 0.0;
            dabs.push(self.path_pointer);
        }
        dabs
    }

    #[must_use]
    pub fn consume_grab_delta(&mut self, pointer: Pos2) -> Vec2 {
        let delta = pointer - self.observed_pointer;
        self.observed_pointer = pointer;
        self.path_pointer = pointer;
        self.pending_path.clear();
        self.distance_since_dab = 0.0;
        delta
    }

    pub fn advance_to(&mut self, now: Instant) {
        self.elapsed_since_dab = self
            .elapsed_since_dab
            .saturating_add(now.saturating_duration_since(self.last_advanced_at));
        self.last_advanced_at = now;
    }

    pub fn record_spatial_dab(&mut self) {
        self.elapsed_since_dab = Duration::ZERO;
    }

    pub fn record_airbrush_dab(&mut self) {
        self.elapsed_since_dab = Duration::ZERO;
        self.distance_since_dab = 0.0;
    }

    #[must_use]
    pub fn airbrush_due(&self, interval: Duration) -> bool {
        self.elapsed_since_dab >= interval
    }

    #[must_use]
    pub fn airbrush_wait(&self, interval: Duration) -> Duration {
        interval.saturating_sub(self.elapsed_since_dab)
    }

    pub fn release(&mut self) {
        self.released = true;
    }

    #[must_use]
    pub fn is_released(&self) -> bool {
        self.released
    }

    #[must_use]
    pub fn has_pending_path(&self) -> bool {
        !self.pending_path.is_empty()
    }

    #[must_use]
    pub fn pointer(&self) -> Pos2 {
        self.observed_pointer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positions(dabs: &[Pos2]) -> Vec<[f32; 2]> {
        dabs.iter().map(|point| [point.x, point.y]).collect()
    }

    fn sampler(pointer: Pos2) -> StrokeSampler {
        StrokeSampler::begin(pointer, Instant::now())
    }

    #[test]
    fn movement_below_spacing_accumulates_until_one_dab_is_due() {
        let mut sampler = sampler(Pos2::ZERO);
        sampler.enqueue_pointer(Pos2::new(4.0, 0.0));
        assert!(sampler.drain_spatial_dabs(10.0, 64).is_empty());
        sampler.enqueue_pointer(Pos2::new(9.0, 0.0));
        assert!(sampler.drain_spatial_dabs(10.0, 64).is_empty());
        sampler.enqueue_pointer(Pos2::new(12.0, 0.0));

        assert_eq!(
            positions(&sampler.drain_spatial_dabs(10.0, 64)),
            [[10.0, 0.0]]
        );
    }

    #[test]
    fn initial_dab_is_available_exactly_once() {
        let pointer = Pos2::new(12.0, 34.0);
        let mut sampler = sampler(pointer);

        assert_eq!(sampler.take_initial_dab(), Some(pointer));
        assert_eq!(sampler.take_initial_dab(), None);
    }

    #[test]
    fn event_partitioning_does_not_change_spatial_dabs() {
        let mut one_segment = sampler(Pos2::ZERO);
        one_segment.enqueue_pointer(Pos2::new(35.0, 0.0));
        let one_segment = one_segment.drain_spatial_dabs(10.0, 64);

        let mut many_segments = sampler(Pos2::ZERO);
        let mut many_dabs = Vec::new();
        for x in [3.0, 11.0, 17.0, 26.0, 35.0] {
            many_segments.enqueue_pointer(Pos2::new(x, 0.0));
            many_dabs.extend(many_segments.drain_spatial_dabs(10.0, 64));
        }

        assert_eq!(positions(&one_segment), positions(&many_dabs));
        assert_eq!(
            positions(&one_segment),
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
        );
    }

    #[test]
    fn work_budget_retains_the_unprocessed_path() {
        let mut sampler = sampler(Pos2::ZERO);
        sampler.enqueue_pointer(Pos2::new(100.0, 0.0));

        assert_eq!(sampler.drain_spatial_dabs(10.0, 3).len(), 3);
        assert!(sampler.has_pending_path());
        assert_eq!(
            positions(&sampler.drain_spatial_dabs(10.0, 64)),
            [
                [40.0, 0.0],
                [50.0, 0.0],
                [60.0, 0.0],
                [70.0, 0.0],
                [80.0, 0.0],
                [90.0, 0.0],
                [100.0, 0.0],
            ]
        );
        assert!(!sampler.has_pending_path());
    }

    #[test]
    fn releasing_a_stroke_does_not_discard_queued_path() {
        let mut sampler = sampler(Pos2::ZERO);
        sampler.enqueue_pointer(Pos2::new(100.0, 0.0));
        sampler.release();

        assert_eq!(sampler.drain_spatial_dabs(10.0, 2).len(), 2);
        assert!(sampler.is_released());
        assert!(sampler.has_pending_path());
    }

    #[test]
    fn airbrush_waits_for_rate_and_does_not_catch_up() {
        let started = Instant::now();
        let mut sampler = StrokeSampler::begin(Pos2::ZERO, started);
        let interval = Duration::from_millis(100);
        sampler.advance_to(started + Duration::from_millis(99));
        assert!(!sampler.airbrush_due(interval));
        sampler.advance_to(started + Duration::from_millis(100));
        assert!(sampler.airbrush_due(interval));

        sampler.record_spatial_dab();
        assert!(!sampler.airbrush_due(interval));
        assert_eq!(sampler.airbrush_wait(interval), interval);
    }

    #[test]
    fn airbrush_dab_restarts_spatial_spacing_at_current_pointer() {
        let mut sampler = sampler(Pos2::ZERO);
        sampler.enqueue_pointer(Pos2::new(9.0, 0.0));
        assert!(sampler.drain_spatial_dabs(10.0, 64).is_empty());
        sampler.record_airbrush_dab();
        sampler.enqueue_pointer(Pos2::new(11.0, 0.0));

        assert!(sampler.drain_spatial_dabs(10.0, 64).is_empty());
    }

    #[test]
    fn grab_uses_raw_pointer_delta_and_discards_spatial_backlog() {
        let mut sampler = sampler(Pos2::new(2.0, 3.0));
        sampler.enqueue_pointer(Pos2::new(20.0, 3.0));

        assert_eq!(
            sampler.consume_grab_delta(Pos2::new(25.0, 8.0)),
            Vec2::new(5.0, 5.0)
        );
        assert!(!sampler.has_pending_path());
    }
}
