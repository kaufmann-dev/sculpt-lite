# Adaptive symmetry and safety validation could stall a frame

- Fixed: 2026-07-22 11:47:59 UTC (+0000)
- Base commit: `10f95321ccee2be4b50b2ab6a693e7b8081f55c2`

## Symptom

Dense adaptive symmetry dabs could exceed the UI frame budget even though each remesh call had an
edit limit. Both growing symmetry regions were scanned in the same continuation, topology
self-intersection validation ran immediately afterward, and an unsafe deformation could then run
all six safe-strength trials synchronously. An empty prepared pass also reported adequate remesh
support when the helper was called directly because its universal predicate succeeded vacuously.

## Confirmed root cause

The topology edit count bounded mutations but not region discovery, candidate sorting, or geometric
validation. Symmetry doubled those scans in one frame. Deformation intersection testing and its
six-step strength search had no resumable state, so their bounded number of trials still produced an
unacceptably long single event on a sufficiently refined brush region.

The final adaptive support gate already rejected empty passes explicitly and checked every
sparse-but-nonempty pass. Symmetric deformation was also already atomic after support convergence,
so those reported paths did not need another acceptance rule. The empty-pass helper itself still
violated its intended invariant and was unsafe to reuse without the caller's separate guard.

## Fix

- Require a prepared pass to contain at least one vertex before it can report adequate support.
- Round-robin regular remesh slices between mirrored passes so neither side can starve and only one
  growing region scan consumes a UI frame. Keep support-patch insertion and final deformation
  atomic, and retain the same total convergence allowance with proportionally more bounded steps.
- Defer topology publication until its structural and chunked intersection checks succeed; retry or
  roll back the complete dab when validation fails.
- Persist deformation validation and the six safe-strength search trials across frames. Every trial
  still performs the same foldover and self-intersection checks, and deformation commits only after
  every primary and mirrored pass remains adequately supported.

## Verification

- Added regressions for an empty prepared pass, a supported primary pass with an empty mirror, a
  finest-detail symmetric support patch, and substantial mirrored regular remeshing.
- Successful symmetry cases validate the mesh, reject self-intersections, preserve geometric
  symmetry, and restore positions, triangles, and masks exactly through topology undo.
- The dense symmetric release regression keeps every remesh and safety-validation continuation below
  the 8 millisecond UI-frame budget while retaining the full six-step safe-deformation search.
