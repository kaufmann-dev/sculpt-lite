use glam::Vec3;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::mesh::{Mesh, MeshError};

/// Default memory budget shared by undo and redo entries (512 MiB).
pub const DEFAULT_HISTORY_BUDGET: usize = 512 * 1024 * 1024;

static NEXT_HISTORY_ID: AtomicU64 = AtomicU64::new(1);

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

    /// Returns the heap payload retained by this snapshot for history budgeting.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.positions
            .len()
            .saturating_mul(size_of::<Vec3>())
            .saturating_add(self.triangles.len().saturating_mul(size_of::<[u32; 3]>()))
            .saturating_add(self.mask.len().saturating_mul(size_of::<f32>()))
    }

    /// Reconstructs the editable mesh and all of its derived data.
    ///
    /// Topology reconstruction is intentionally explicit so callers can run it
    /// on the mesh worker before handing a prepared replacement to the UI.
    pub fn restore_mesh(&self) -> Result<Mesh, MeshError> {
        Mesh::from_indexed(
            self.positions.to_vec(),
            self.triangles.to_vec(),
            self.mask.to_vec(),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HistoryEntry {
    Local(LocalEdit),
    /// The complete mesh state on the other side of a topology replacement.
    ///
    /// A newly recorded replacement holds the pre-operation state. Completing
    /// its undo or redo records the inverse snapshot on the opposite stack.
    MeshReplacement(Arc<MeshSnapshot>),
}

impl HistoryEntry {
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Local(edit) => edit.byte_len(),
            Self::MeshReplacement(snapshot) => snapshot.byte_len(),
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
    Local {
        changed_vertices: Vec<u32>,
    },
    /// A whole-mesh state that must be rebuilt away from the UI thread.
    ///
    /// Return the token to [`History::complete_mesh_replacement`] after the
    /// target has been installed, or to [`History::cancel_mesh_replacement`] if
    /// preparation or installation fails.
    MeshReplacement(PendingMeshReplacement),
}

/// A popped whole-mesh history entry awaiting worker preparation.
///
/// While this token is outstanding, its originating [`History`] refuses new
/// records and further undo/redo steps. The token owns the target snapshot so
/// it can travel through a worker job and result without copying mesh arrays.
#[derive(Debug)]
pub struct PendingMeshReplacement {
    history_id: u64,
    transaction_id: u64,
    direction: HistoryDirection,
    target: Arc<MeshSnapshot>,
    bytes: usize,
}

impl PendingMeshReplacement {
    #[must_use]
    pub fn direction(&self) -> HistoryDirection {
        self.direction
    }

    #[must_use]
    pub fn target(&self) -> &Arc<MeshSnapshot> {
        &self.target
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HistoryTransactionError {
    /// The token is stale or belongs to a different history instance.
    #[error("the mesh replacement history token is stale or belongs to another history")]
    InvalidMeshReplacementToken,
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
    history_id: u64,
    next_transaction_id: u64,
    pending_replacement: Option<ActiveMeshReplacement>,
}

#[derive(Clone, Copy, Debug)]
struct ActiveMeshReplacement {
    transaction_id: u64,
    bytes: usize,
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
            history_id: NEXT_HISTORY_ID.fetch_add(1, Ordering::Relaxed),
            next_transaction_id: 1,
            pending_replacement: None,
        }
    }

    /// Records a completed edit and invalidates redo even when the new entry is
    /// too large for the configured budget.
    ///
    /// An oversized mesh replacement clears all history: older local entries
    /// describe the replaced topology and cannot safely cross an unrecorded
    /// replacement. Returning `false` lets the caller report that the completed
    /// replacement is not undoable. Recording is also refused while a mesh
    /// replacement transaction is pending.
    pub fn record(&mut self, entry: HistoryEntry) -> bool {
        if self.pending_replacement.is_some() {
            return false;
        }
        self.clear_redo();
        let stored = StoredEntry::new(entry);
        if stored.bytes > self.byte_budget {
            if matches!(&stored.entry, HistoryEntry::MeshReplacement(_)) {
                self.clear();
            }
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
        if self.pending_replacement.is_some() {
            return HistoryAction::Empty;
        }
        let stored = match direction {
            HistoryDirection::Undo => self.undo.pop_back(),
            HistoryDirection::Redo => self.redo.pop(),
        };
        let Some(stored) = stored else {
            return HistoryAction::Empty;
        };

        match stored.entry {
            HistoryEntry::Local(edit) => {
                self.bytes_used -= stored.bytes;
                let changed_vertices = match direction {
                    HistoryDirection::Undo => edit.apply_before(mesh),
                    HistoryDirection::Redo => edit.apply_after(mesh),
                };
                self.push_opposite(direction, StoredEntry::new(HistoryEntry::Local(edit)));
                HistoryAction::Local { changed_vertices }
            }
            HistoryEntry::MeshReplacement(target) => {
                let transaction_id = self.next_transaction_id;
                self.next_transaction_id = self.next_transaction_id.wrapping_add(1);
                self.pending_replacement = Some(ActiveMeshReplacement {
                    transaction_id,
                    bytes: stored.bytes,
                });
                HistoryAction::MeshReplacement(PendingMeshReplacement {
                    history_id: self.history_id,
                    transaction_id,
                    direction,
                    target,
                    bytes: stored.bytes,
                })
            }
        }
    }

    /// Finishes a worker-prepared replacement undo or redo.
    ///
    /// `inverse` must capture the mesh that was current before the token's
    /// target was installed. It is pushed onto the opposite stack so the
    /// operation can be reversed again. `Ok(false)` means the inverse exceeded
    /// the budget; the replacement remains complete, but the history entries
    /// that would require the missing inverse are cleared. `Err` leaves the
    /// active transaction unchanged.
    pub fn complete_mesh_replacement(
        &mut self,
        pending: PendingMeshReplacement,
        inverse: Arc<MeshSnapshot>,
    ) -> Result<bool, HistoryTransactionError> {
        self.validate_pending(&pending)?;
        self.pending_replacement = None;
        self.bytes_used -= pending.bytes;

        let inverse = StoredEntry::new(HistoryEntry::MeshReplacement(inverse));
        if inverse.bytes > self.byte_budget {
            self.clear_opposite(pending.direction);
            return Ok(false);
        }

        self.push_opposite(pending.direction, inverse);
        self.trim_after_replacement(pending.direction);
        Ok(true)
    }

    /// Cancels a worker replacement and restores the popped entry exactly where
    /// it was. The editable mesh must still contain (or have been restored to)
    /// the state from before [`History::undo`] or [`History::redo`] returned the
    /// token. `Err` leaves the active transaction unchanged.
    pub fn cancel_mesh_replacement(
        &mut self,
        pending: PendingMeshReplacement,
    ) -> Result<(), HistoryTransactionError> {
        self.validate_pending(&pending)?;
        self.pending_replacement = None;
        let restored = StoredEntry {
            entry: HistoryEntry::MeshReplacement(pending.target),
            bytes: pending.bytes,
        };
        match pending.direction {
            HistoryDirection::Undo => self.undo.push_back(restored),
            HistoryDirection::Redo => self.redo.push(restored),
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.pending_replacement = None;
        self.bytes_used = 0;
    }

    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.pending_replacement.is_none() && !self.undo.is_empty()
    }

    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.pending_replacement.is_none() && !self.redo.is_empty()
    }

    #[cfg(test)]
    #[must_use]
    pub fn mesh_replacement_pending(&self) -> bool {
        self.pending_replacement.is_some()
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

    fn validate_pending(
        &self,
        pending: &PendingMeshReplacement,
    ) -> Result<(), HistoryTransactionError> {
        let matches = pending.history_id == self.history_id
            && self.pending_replacement.is_some_and(|active| {
                active.transaction_id == pending.transaction_id && active.bytes == pending.bytes
            });
        if matches {
            Ok(())
        } else {
            Err(HistoryTransactionError::InvalidMeshReplacementToken)
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

    /// Trims around a just-completed replacement without evicting the inverse
    /// snapshot that makes the completed operation reversible.
    fn trim_after_replacement(&mut self, direction: HistoryDirection) {
        while self.bytes_used > self.byte_budget {
            let removed = match direction {
                HistoryDirection::Undo => self
                    .undo
                    .pop_front()
                    .or_else(|| (self.redo.len() > 1).then(|| self.redo.remove(0))),
                HistoryDirection::Redo => {
                    if self.undo.len() > 1 {
                        self.undo.pop_front()
                    } else if self.redo.is_empty() {
                        None
                    } else {
                        Some(self.redo.remove(0))
                    }
                }
            };
            let Some(removed) = removed else {
                break;
            };
            self.bytes_used -= removed.bytes;
        }
    }

    fn clear_redo(&mut self) {
        for stored in self.redo.drain(..) {
            self.bytes_used -= stored.bytes;
        }
    }

    fn clear_opposite(&mut self, direction: HistoryDirection) {
        match direction {
            HistoryDirection::Undo => self.clear_redo(),
            HistoryDirection::Redo => {
                for stored in self.undo.drain(..) {
                    self.bytes_used -= stored.bytes;
                }
            }
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

    fn quad() -> Mesh {
        Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::ONE, Vec3::Y],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .expect("valid quad")
    }

    fn replacement(action: HistoryAction) -> PendingMeshReplacement {
        let HistoryAction::MeshReplacement(pending) = action else {
            panic!("mesh replacement expected");
        };
        pending
    }

    #[test]
    fn snapshot_reports_payload_size_and_restores_a_separate_mesh() {
        let mut source = quad();
        source.mask = vec![0.0, 0.25, 0.5, 1.0];
        let snapshot = MeshSnapshot::capture(&source);

        assert_eq!(
            snapshot.byte_len(),
            source.positions.len() * size_of::<Vec3>()
                + source.triangles.len() * size_of::<[u32; 3]>()
                + source.mask.len() * size_of::<f32>()
        );

        let restored = snapshot.restore_mesh().expect("valid snapshot");
        assert_eq!(restored.positions, source.positions);
        assert_eq!(restored.triangles, source.triangles);
        assert_eq!(restored.mask, source.mask);
        assert_eq!(restored.normals, source.normals);
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
    fn replacement_undo_and_redo_exchange_worker_snapshots() {
        let before = triangle();
        let before_snapshot = Arc::new(MeshSnapshot::capture(&before));
        let mut mesh = quad();
        mesh.mask[2] = 0.625;
        let after_snapshot = Arc::new(MeshSnapshot::capture(&mesh));
        let mut history = History::default();
        assert!(history.record(HistoryEntry::MeshReplacement(Arc::clone(&before_snapshot))));

        let pending = replacement(history.undo(&mut mesh));
        assert_eq!(pending.direction(), HistoryDirection::Undo);
        assert_eq!(pending.target().as_ref(), before_snapshot.as_ref());
        assert!(history.mesh_replacement_pending());
        assert!(!history.can_undo());
        assert!(!history.can_redo());
        assert!(matches!(history.undo(&mut mesh), HistoryAction::Empty));
        assert!(!history.record(HistoryEntry::Local(LocalEdit::default())));

        mesh = pending.target().restore_mesh().expect("worker restore");
        assert!(
            history
                .complete_mesh_replacement(pending, Arc::clone(&after_snapshot))
                .expect("matching transaction")
        );
        assert_eq!(mesh.positions, before.positions);
        assert!(!history.mesh_replacement_pending());
        assert!(history.can_redo());

        let pending = replacement(history.redo(&mut mesh));
        assert_eq!(pending.direction(), HistoryDirection::Redo);
        assert_eq!(pending.target().as_ref(), after_snapshot.as_ref());
        let inverse = Arc::new(MeshSnapshot::capture(&mesh));
        mesh = pending.target().restore_mesh().expect("worker restore");
        assert!(
            history
                .complete_mesh_replacement(pending, inverse)
                .expect("matching transaction")
        );
        assert_eq!(mesh.positions, quad().positions);
        assert_eq!(mesh.mask[2], 0.625);
        assert!(history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn canceled_replacement_returns_to_the_same_stack_and_budget() {
        let snapshot = Arc::new(MeshSnapshot::capture(&triangle()));
        let expected_bytes = snapshot.byte_len();
        let mut mesh = quad();
        let mut history = History::default();
        assert!(history.record(HistoryEntry::MeshReplacement(snapshot)));

        let pending = replacement(history.undo(&mut mesh));
        assert_eq!(history.bytes_used(), expected_bytes);
        assert!(history.mesh_replacement_pending());
        history
            .cancel_mesh_replacement(pending)
            .expect("matching transaction");

        assert_eq!(history.bytes_used(), expected_bytes);
        assert!(history.can_undo());
        assert!(!history.can_redo());
        let pending = replacement(history.undo(&mut mesh));
        history
            .cancel_mesh_replacement(pending)
            .expect("matching transaction");
    }

    #[test]
    fn local_entries_keep_their_order_across_a_replacement() {
        let mut mesh = triangle();
        let original = mesh.positions[1];
        let first = Vec3::new(1.2, 0.0, 0.0);
        mesh.positions[1] = first;
        mesh.update_deformed_vertices(&[1]);

        let mut history = History::default();
        assert!(history.record(HistoryEntry::Local(position_edit(original, first))));
        let before_replacement = Arc::new(MeshSnapshot::capture(&mesh));

        mesh = quad();
        let replacement_state = Arc::new(MeshSnapshot::capture(&mesh));
        assert!(history.record(HistoryEntry::MeshReplacement(Arc::clone(
            &before_replacement
        ))));

        let second_before = mesh.positions[1];
        let second_after = Vec3::new(1.5, -0.25, 0.0);
        mesh.positions[1] = second_after;
        mesh.update_deformed_vertices(&[1]);
        assert!(history.record(HistoryEntry::Local(position_edit(
            second_before,
            second_after,
        ))));

        assert!(matches!(
            history.undo(&mut mesh),
            HistoryAction::Local { .. }
        ));
        assert_eq!(mesh.positions[1], second_before);

        let pending = replacement(history.undo(&mut mesh));
        let inverse = Arc::new(MeshSnapshot::capture(&mesh));
        mesh = pending.target().restore_mesh().expect("worker restore");
        history
            .complete_mesh_replacement(pending, inverse)
            .expect("matching transaction");
        assert_eq!(mesh.positions[1], first);
        assert_eq!(mesh.triangles.len(), 1);

        assert!(matches!(
            history.undo(&mut mesh),
            HistoryAction::Local { .. }
        ));
        assert_eq!(mesh.positions[1], original);
        assert!(matches!(
            history.redo(&mut mesh),
            HistoryAction::Local { .. }
        ));
        assert_eq!(mesh.positions[1], first);

        let pending = replacement(history.redo(&mut mesh));
        let inverse = Arc::new(MeshSnapshot::capture(&mesh));
        assert_eq!(pending.target().as_ref(), replacement_state.as_ref());
        mesh = pending.target().restore_mesh().expect("worker restore");
        history
            .complete_mesh_replacement(pending, inverse)
            .expect("matching transaction");
        assert_eq!(mesh.positions[1], second_before);
        assert_eq!(mesh.triangles.len(), 2);

        assert!(matches!(
            history.redo(&mut mesh),
            HistoryAction::Local { .. }
        ));
        assert_eq!(mesh.positions[1], second_after);
        assert!(!history.can_redo());
    }

    #[test]
    fn oversized_new_replacement_clears_incompatible_history() {
        let snapshot = Arc::new(MeshSnapshot::capture(&quad()));
        let mut mesh = triangle();
        let edit = position_edit(Vec3::X, Vec3::new(1.25, 0.0, 0.0));
        let local_bytes = edit.byte_len();
        assert!(snapshot.byte_len() > local_bytes);
        let mut history = History::new(snapshot.byte_len() - 1);
        assert!(history.record(HistoryEntry::Local(edit)));
        let _ = history.undo(&mut mesh);
        assert!(history.can_redo());

        assert!(!history.record(HistoryEntry::MeshReplacement(snapshot)));
        assert_eq!(history.bytes_used(), 0);
        assert!(!history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn replacement_completion_stays_within_the_shared_budget() {
        let small = Arc::new(MeshSnapshot::capture(&triangle()));
        let large = Arc::new(MeshSnapshot::capture(&quad()));
        assert!(large.byte_len() > small.byte_len());
        let mut mesh = quad();
        let mut history = History::new(small.byte_len());
        assert!(history.record(HistoryEntry::MeshReplacement(small)));

        let pending = replacement(history.undo(&mut mesh));
        assert_eq!(history.bytes_used(), history.byte_budget());
        assert!(
            !history
                .complete_mesh_replacement(pending, large)
                .expect("matching transaction")
        );
        assert!(history.bytes_used() <= history.byte_budget());
        assert!(!history.can_redo());
        assert!(!history.mesh_replacement_pending());
    }

    #[test]
    fn replacement_completion_preserves_its_inverse_while_trimming_other_entries() {
        let target = Arc::new(MeshSnapshot::capture(&triangle()));
        let inverse = Arc::new(MeshSnapshot::capture(&quad()));
        let local = StoredEntry::new(HistoryEntry::Local(position_edit(
            Vec3::X,
            Vec3::new(1.25, 0.0, 0.0),
        )));
        let stored_replacement = StoredEntry::new(HistoryEntry::MeshReplacement(target));
        assert!(stored_replacement.bytes + local.bytes <= inverse.byte_len());

        let mut history = History::new(inverse.byte_len());
        history.bytes_used = stored_replacement.bytes + local.bytes;
        history.redo.push(local);
        history.redo.push(stored_replacement);
        let mut mesh = triangle();

        let pending = replacement(history.redo(&mut mesh));
        assert!(
            history
                .complete_mesh_replacement(pending, inverse)
                .expect("matching transaction")
        );

        assert!(history.can_undo());
        assert!(!history.can_redo());
        assert!(history.bytes_used() <= history.byte_budget());
        assert!(matches!(
            history.undo(&mut mesh),
            HistoryAction::MeshReplacement(_)
        ));
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
