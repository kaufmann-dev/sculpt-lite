use std::collections::VecDeque;

use glam::Vec3;

use crate::mesh::Mesh;

/// Default memory budget for undo and redo snapshots (512 MiB).
pub const DEFAULT_HISTORY_BUDGET: usize = 512 * 1024 * 1024;

/// The editable mesh state needed to restore a sculpt stroke exactly.
///
/// Normals and topology are derived data and are rebuilt on restore. Keeping
/// them out of snapshots makes long sculpting sessions substantially cheaper.
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

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.positions.len() * size_of::<Vec3>()
            + self.triangles.len() * size_of::<[u32; 3]>()
            + self.mask.len() * size_of::<f32>()
    }

    #[cfg(test)]
    #[must_use]
    pub fn matches(&self, mesh: &Mesh) -> bool {
        self.positions.as_ref() == mesh.positions.as_slice()
            && self.triangles.as_ref() == mesh.triangles.as_slice()
            && self.mask.as_ref() == mesh.mask.as_slice()
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

#[derive(Debug)]
struct StoredSnapshot {
    snapshot: MeshSnapshot,
    bytes: usize,
}

impl StoredSnapshot {
    fn capture(mesh: &Mesh) -> Self {
        let snapshot = MeshSnapshot::capture(mesh);
        let bytes = snapshot.byte_len();
        Self { snapshot, bytes }
    }
}

/// Bounded, snapshot-before undo history.
///
/// Call [`History::push_before`] once immediately before a stroke. Undo stores
/// the then-current mesh for redo and restores the saved pre-stroke snapshot.
#[derive(Debug)]
pub struct History {
    undo: VecDeque<StoredSnapshot>,
    redo: Vec<StoredSnapshot>,
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

    /// Saves the current mesh as the state to restore on the next undo.
    ///
    /// Returns `false` when a single snapshot exceeds the entire budget. In
    /// that case existing history is retained and this stroke is not undoable.
    pub fn push_before(&mut self, mesh: &Mesh) -> bool {
        let stored = StoredSnapshot::capture(mesh);
        if stored.bytes > self.byte_budget {
            return false;
        }

        self.clear_redo();
        self.bytes_used += stored.bytes;
        self.undo.push_back(stored);
        self.trim_to_budget();
        true
    }

    /// Removes the most recently saved pre-stroke state.
    ///
    /// This is intended for a stroke which ended without changing geometry or
    /// mask values.
    pub fn discard_latest(&mut self) -> bool {
        let Some(stored) = self.undo.pop_back() else {
            return false;
        };
        self.bytes_used -= stored.bytes;
        true
    }

    pub fn undo(&mut self, mesh: &mut Mesh) -> bool {
        let Some(previous) = self.undo.pop_back() else {
            return false;
        };
        self.bytes_used -= previous.bytes;

        let current = StoredSnapshot::capture(mesh);
        if current.bytes <= self.byte_budget {
            self.bytes_used += current.bytes;
            self.redo.push(current);
        } else {
            self.clear_redo();
        }

        previous.snapshot.restore(mesh);
        self.trim_to_budget();
        true
    }

    pub fn redo(&mut self, mesh: &mut Mesh) -> bool {
        let Some(next) = self.redo.pop() else {
            return false;
        };
        self.bytes_used -= next.bytes;

        let current = StoredSnapshot::capture(mesh);
        if current.bytes <= self.byte_budget {
            self.bytes_used += current.bytes;
            self.undo.push_back(current);
        } else {
            self.clear_undo();
        }

        next.snapshot.restore(mesh);
        self.trim_to_budget();
        true
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

    fn clear_undo(&mut self) {
        for stored in self.undo.drain(..) {
            self.bytes_used -= stored.bytes;
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

    fn triangle() -> Mesh {
        Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]]).expect("valid triangle")
    }

    #[test]
    fn undo_and_redo_restore_exact_editable_state() {
        let mut mesh = triangle();
        let original = MeshSnapshot::capture(&mesh);
        let mut history = History::default();

        assert!(history.push_before(&mesh));
        mesh.positions[1] = Vec3::new(1.25, -0.5, 2.0);
        mesh.triangles[0] = [0, 2, 1];
        mesh.mask[2] = 0.625;
        let _ = mesh.rebuild();
        let edited = MeshSnapshot::capture(&mesh);

        assert!(history.undo(&mut mesh));
        assert!(original.matches(&mesh));
        assert!(history.can_redo());

        assert!(history.redo(&mut mesh));
        assert!(edited.matches(&mesh));
        assert!(history.can_undo());
    }

    #[test]
    fn new_edit_invalidates_redo_and_noop_can_be_discarded() {
        let mut mesh = triangle();
        let mut history = History::default();

        history.push_before(&mesh);
        mesh.positions[0].z = 1.0;
        mesh.rebuild();
        history.undo(&mut mesh);
        assert!(history.can_redo());

        history.push_before(&mesh);
        assert!(!history.can_redo());
        assert!(history.discard_latest());
        assert!(!history.can_undo());
    }

    #[test]
    fn history_never_exceeds_its_budget() {
        let mesh = triangle();
        let snapshot_bytes = MeshSnapshot::capture(&mesh).byte_len();
        let mut history = History::new(snapshot_bytes * 2);

        for _ in 0..5 {
            assert!(history.push_before(&mesh));
            assert!(history.bytes_used() <= history.byte_budget());
        }

        assert!(history.undo.len() <= 2);
    }
}
