use std::collections::VecDeque;
use std::sync::Arc;

use glam::Vec3;

use crate::mesh::{Mesh, MeshChangeSet, MeshEditDelta};

/// Default memory budget shared by undo and redo entries (512 MiB).
pub const DEFAULT_HISTORY_BUDGET: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionChange {
    pub vertex: u32,
    pub before: Vec3,
    pub after: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskChange {
    pub vertex: u32,
    pub before: f32,
    pub after: f32,
}

/// Exact editable values changed by a fixed-topology operation.
///
/// Changes are sorted by vertex ID. Applying them only refreshes local normals,
/// BVH branches, and renderer vertices; triangle topology is never rebuilt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalEdit {
    pub positions: Vec<PositionChange>,
    pub masks: Vec<MaskChange>,
}

impl LocalEdit {
    #[must_use]
    pub fn new(mut positions: Vec<PositionChange>, mut masks: Vec<MaskChange>) -> Self {
        positions.retain(|change| change.before != change.after);
        masks.retain(|change| change.before != change.after);
        positions.sort_unstable_by_key(|change| change.vertex);
        masks.sort_unstable_by_key(|change| change.vertex);
        Self { positions, masks }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() && self.masks.is_empty()
    }

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.positions.len() * size_of::<PositionChange>()
            + self.masks.len() * size_of::<MaskChange>()
    }

    #[must_use]
    pub fn apply_before(&self, mesh: &mut Mesh) -> Vec<u32> {
        self.apply(mesh, false)
    }

    #[must_use]
    pub fn apply_after(&self, mesh: &mut Mesh) -> Vec<u32> {
        self.apply(mesh, true)
    }

    fn apply(&self, mesh: &mut Mesh, after: bool) -> Vec<u32> {
        let mut moved = Vec::with_capacity(self.positions.len());
        for change in &self.positions {
            let Some(position) = mesh.positions.get_mut(change.vertex as usize) else {
                continue;
            };
            let target = if after { change.after } else { change.before };
            if *position != target {
                *position = target;
                moved.push(change.vertex);
            }
        }

        let mut changed = if moved.is_empty() {
            Vec::new()
        } else {
            mesh.update_deformed_vertices(&moved)
        };
        changed.extend(moved);
        for change in &self.masks {
            let Some(mask) = mesh.mask.get_mut(change.vertex as usize) else {
                continue;
            };
            let target = if after { change.after } else { change.before };
            if *mask != target {
                *mask = target;
                changed.push(change.vertex);
            }
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }
}

/// Editable arrays needed to restore a topology-changing operation exactly.
/// Derived data is rebuilt only on the mesh worker.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshSnapshot {
    positions: Box<[Vec3]>,
    triangles: Box<[[u32; 3]]>,
    mask: Box<[f32]>,
}

impl MeshSnapshot {
    #[must_use]
    pub fn capture(mesh: &Mesh) -> Self {
        Self {
            positions: mesh.positions.clone().into_boxed_slice(),
            triangles: mesh.triangles.clone().into_boxed_slice(),
            mask: mesh.mask.clone().into_boxed_slice(),
        }
    }

    pub fn restore(&self, mesh: &mut Mesh) {
        mesh.positions.clear();
        mesh.positions.extend_from_slice(&self.positions);
        mesh.triangles.clear();
        mesh.triangles.extend_from_slice(&self.triangles);
        mesh.mask.clear();
        mesh.mask.extend_from_slice(&self.mask);
        let _ = mesh.rebuild();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HistoryEntry {
    Local(LocalEdit),
    Topology(Arc<MeshEditDelta>),
}

impl HistoryEntry {
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Local(edit) => edit.byte_len(),
            Self::Topology(edit) => edit.byte_len(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryDirection {
    Undo,
    Redo,
}

#[derive(Debug)]
pub enum HistoryAction {
    Empty,
    Local { changed_vertices: Vec<u32> },
    Topology { changes: MeshChangeSet },
}

#[derive(Debug)]
struct StoredEntry {
    entry: HistoryEntry,
    bytes: usize,
}

impl StoredEntry {
    fn new(entry: HistoryEntry) -> Self {
        let bytes = entry.byte_len();
        Self { entry, bytes }
    }
}

#[derive(Debug)]
pub struct History {
    undo: VecDeque<StoredEntry>,
    redo: Vec<StoredEntry>,
    byte_budget: usize,
    bytes_used: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_BUDGET)
    }
}

impl History {
    #[must_use]
    pub fn new(byte_budget: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            byte_budget,
            bytes_used: 0,
        }
    }

    /// Records a completed edit and invalidates redo even when the new entry is
    /// too large for the configured budget.
    pub fn record(&mut self, entry: HistoryEntry) -> bool {
        self.clear_redo();
        let stored = StoredEntry::new(entry);
        if stored.bytes > self.byte_budget {
            return false;
        }
        self.bytes_used += stored.bytes;
        self.undo.push_back(stored);
        self.trim_to_budget();
        true
    }

    pub fn undo(&mut self, mesh: &mut Mesh) -> HistoryAction {
        self.step(HistoryDirection::Undo, mesh)
    }

    pub fn redo(&mut self, mesh: &mut Mesh) -> HistoryAction {
        self.step(HistoryDirection::Redo, mesh)
    }

    fn step(&mut self, direction: HistoryDirection, mesh: &mut Mesh) -> HistoryAction {
        let stored = match direction {
            HistoryDirection::Undo => self.undo.pop_back(),
            HistoryDirection::Redo => self.redo.pop(),
        };
        let Some(stored) = stored else {
            return HistoryAction::Empty;
        };
        self.bytes_used -= stored.bytes;

        match stored.entry {
            HistoryEntry::Local(edit) => {
                let changed_vertices = match direction {
                    HistoryDirection::Undo => edit.apply_before(mesh),
                    HistoryDirection::Redo => edit.apply_after(mesh),
                };
                self.push_opposite(direction, StoredEntry::new(HistoryEntry::Local(edit)));
                HistoryAction::Local { changed_vertices }
            }
            HistoryEntry::Topology(edit) => {
                let changes = match direction {
                    HistoryDirection::Undo => edit.apply_before(mesh),
                    HistoryDirection::Redo => edit.apply_after(mesh),
                };
                self.push_opposite(
                    direction,
                    StoredEntry::new(HistoryEntry::Topology(Arc::clone(&edit))),
                );
                HistoryAction::Topology { changes }
            }
        }
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.bytes_used = 0;
    }

    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    #[must_use]
    pub fn byte_budget(&self) -> usize {
        self.byte_budget
    }

    fn push_opposite(&mut self, direction: HistoryDirection, stored: StoredEntry) {
        self.bytes_used += stored.bytes;
        match direction {
            HistoryDirection::Undo => self.redo.push(stored),
            HistoryDirection::Redo => self.undo.push_back(stored),
        }
    }

    fn trim_to_budget(&mut self) {
        while self.bytes_used > self.byte_budget {
            if let Some(oldest) = self.undo.pop_front() {
                self.bytes_used -= oldest.bytes;
            } else if !self.redo.is_empty() {
                let oldest = self.redo.remove(0);
                self.bytes_used -= oldest.bytes;
            } else {
                break;
            }
        }
    }

    fn clear_redo(&mut self) {
        for stored in self.redo.drain(..) {
            self.bytes_used -= stored.bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn triangle() -> Mesh {
        Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).expect("valid triangle")
    }

    fn position_edit(before: Vec3, after: Vec3) -> LocalEdit {
        LocalEdit::new(
            vec![PositionChange {
                vertex: 1,
                before,
                after,
            }],
            Vec::new(),
        )
    }

    #[test]
    fn local_undo_and_redo_restore_without_changing_topology() {
        let mut mesh = triangle();
        let triangles = mesh.triangles.clone();
        let before = mesh.positions[1];
        let after = Vec3::new(1.25, -0.5, 0.25);
        mesh.positions[1] = after;
        mesh.update_deformed_vertices(&[1]);
        let mut history = History::default();
        assert!(history.record(HistoryEntry::Local(position_edit(before, after))));

        let HistoryAction::Local { changed_vertices } = history.undo(&mut mesh) else {
            panic!("local undo expected");
        };
        assert_eq!(mesh.positions[1], before);
        assert_eq!(mesh.triangles, triangles);
        assert!(changed_vertices.contains(&1));

        assert!(matches!(
            history.redo(&mut mesh),
            HistoryAction::Local { .. }
        ));
        assert_eq!(mesh.positions[1], after);
        assert_eq!(mesh.triangles, triangles);
    }

    #[test]
    fn mask_edits_are_exact_and_invalidate_redo() {
        let mut mesh = triangle();
        let edit = LocalEdit::new(
            Vec::new(),
            vec![MaskChange {
                vertex: 2,
                before: 0.0,
                after: 0.75,
            }],
        );
        mesh.mask[2] = 0.75;
        let mut history = History::default();
        assert!(history.record(HistoryEntry::Local(edit)));
        history.undo(&mut mesh);
        assert_eq!(mesh.mask[2], 0.0);
        assert!(history.can_redo());

        assert!(history.record(HistoryEntry::Local(position_edit(
            Vec3::X,
            Vec3::new(2.0, 0.0, 0.0),
        ))));
        assert!(!history.can_redo());
    }

    #[test]
    fn topology_delta_undo_and_redo_are_exact() {
        let mut mesh = triangle();
        let before = mesh.positions[1];
        let after = Vec3::new(3.0, 0.0, 0.0);
        let mut recorder = crate::mesh::MeshEditRecorder::new(&mesh);
        recorder.record_vertex(&mesh, 1);
        mesh.positions[1] = after;
        mesh.update_deformed_vertices(&[1]);
        let edit = Arc::new(recorder.finish(&mesh));
        let mut history = History::default();
        assert!(history.record(HistoryEntry::Topology(edit)));

        let HistoryAction::Topology { .. } = history.undo(&mut mesh) else {
            panic!("topology delta expected");
        };
        assert_eq!(mesh.positions[1], before);
        assert!(history.can_redo());

        let HistoryAction::Topology { .. } = history.redo(&mut mesh) else {
            panic!("topology delta expected");
        };
        assert_eq!(mesh.positions[1], after);
    }

    #[test]
    fn history_never_exceeds_its_budget() {
        let edit = position_edit(Vec3::ZERO, Vec3::ONE);
        let one_entry = HistoryEntry::Local(edit.clone()).byte_len();
        let mut history = History::new(one_entry * 2);

        for _ in 0..8 {
            history.record(HistoryEntry::Local(edit.clone()));
            assert!(history.bytes_used() <= history.byte_budget());
        }
    }

    #[test]
    #[ignore = "release-mode performance envelope"]
    fn million_face_local_undo_fits_one_frame() {
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
        let mut mesh = Mesh::new(positions, triangles).unwrap();
        let middle = CELLS / 2;
        let mut changes = Vec::new();
        for y in middle - 10..=middle + 10 {
            for x in middle - 10..=middle + 10 {
                let vertex = (y * row + x) as u32;
                let before = mesh.positions[vertex as usize];
                let after = before + Vec3::Z * 0.01;
                mesh.positions[vertex as usize] = after;
                changes.push(PositionChange {
                    vertex,
                    before,
                    after,
                });
            }
        }
        let moved = changes
            .iter()
            .map(|change| change.vertex)
            .collect::<Vec<_>>();
        mesh.update_deformed_vertices(&moved);
        let mut history = History::default();
        history.record(HistoryEntry::Local(LocalEdit::new(changes, Vec::new())));

        let started = Instant::now();
        let action = history.undo(&mut mesh);
        let elapsed = started.elapsed();
        assert!(matches!(action, HistoryAction::Local { .. }));
        assert!(
            elapsed < Duration::from_millis(17),
            "local undo took {elapsed:?}"
        );
        eprintln!("million-face local undo: {elapsed:?}");
    }
}
