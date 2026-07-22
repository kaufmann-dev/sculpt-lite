# Fixed adaptive edit cap could deform an unsupported mesh

- Fixed: 2026-07-22 10:13:25 UTC (+0000)
- Base commit: `de8f042`

## Symptom

Adaptive sculpting used a fixed 96-topology-edit limit as both a per-dab runtime bound and an
implicit signal that remeshing was complete. The support-quality check only protected brushes whose
initial vertex selection was empty. A brush that already selected vertices could exhaust the limit
and deform the mesh even when its local topology still did not represent the brush field safely.

Repeated undersized dabs could therefore leave stretched or self-intersecting faces. The limit was
also not generally safe because the required work changes with the source mesh, brush radius,
falloff, strength, symmetry, and adaptive-detail target.

## Confirmed root cause

The sculpt engine had no pending-dab state, so every sample had to remesh and deform in one call.
Once the edit limit was reached, the only available path was to continue with the topology produced
so far. Local remesh validation checked indices, finite values, adjacency, and degenerate faces but
did not reject geometric self-intersections introduced by the topology slice.

While adding slice-level intersection validation, the regression mesh also exposed dimensionally
incorrect tolerances in the coplanar triangle test. Very skinny, disjoint triangles could be
reported as intersecting because a length tolerance was compared directly with 2D cross products.

## Fix

- Treat 96 edits as a bounded topology step, not a quality threshold. Preserve the exact brush
  samples and topology recorder across frames until support converges.
- Evaluate every adaptive pass using the actual brush influence at edge endpoints and the midpoint
  interpolation error. Deformation begins only when that object-, brush-, and strength-dependent
  quality condition passes.
- Pause queued stroke and airbrush dabs while a sample is pending, keep repainting its intermediate
  topology, and do not finish the stroke or mark the document dirty until the sample commits.
- Validate every topology step for local structural correctness and self-intersections. Retry a
  rejected batch with a smaller edit slice; if no safe progress is possible or 32 steps are reached,
  restore the complete pre-sample mesh instead of deforming under-supported topology.
- Scale coplanar segment and point tolerances by edge length, and exclude edges that only touch the
  remesh-region boundary so tangent edges are not refined recursively.

## Verification

- Added regressions proving that a multi-step tiny dab does not deform before support is ready,
  produces no self-intersections, and restores the exact original mesh through topology undo.
- Added a protected-boundary regression proving that an unsupported brush is rolled back without
  moving a vertex.
- Added intersection and remesh-boundary numeric regressions; 94 normal tests pass with 4 large
  probes ignored.
- Release probes measured a million-face adaptive sample at 1.91 milliseconds, million-face local
  remeshing at 4.22 milliseconds, and half-million-vertex GPU packing at 6.97 milliseconds.
