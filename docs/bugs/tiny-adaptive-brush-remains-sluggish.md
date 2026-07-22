# Tiny adaptive brush remains sluggish

- Fixed: 2026-07-22 08:51:48 UTC (+0000)
- Base commit: `a87f5221eff0173db821be01f5dab0b8188af847`

## Symptom

After the undersized-brush support-patch fix, the first dab was fast but repeated adaptive dabs
still took about 5 milliseconds in release mode on the regression mesh. The application remained
within its frame budget, but the smallest brush felt less responsive than fixed-topology sculpting.

## Confirmed root cause

Repeated tiny-brush dabs spent most of their time exhausting the 128-edit adaptive topology budget.
Instrumented dabs performed 47–64 splits and 48–65 flips while remeshing only 22–141 active
vertices. When deformation approached an intersection, the six-step safe-strength search also
created and replayed general mesh-edit deltas for every rejected position-only trial.

Split application separately spent most of its time detaching and reinserting replacement faces in
the BVH and rebuilding topology entries that an edge split leaves unchanged. Paired release probes
on independently built million-face meshes measured the batched split path 0.18–0.44 milliseconds
faster while producing identical mesh arrays and remesh statistics.

## Fix

- Update retained split faces through a specialized topology path, refit their existing BVH leaves
  as one batch, and insert only the genuinely new faces.
- Limit adaptive work to 96 topology edits per dab. A 64-edit trial failed the support-quality
  regression; 96 passed repeated release runs while reducing instrumented five-dab maxima to about
  4.15–4.40 milliseconds.
- Restore rejected deformation trials directly from their captured source positions and refresh
  only the affected faces instead of constructing and replaying general topology edit deltas.
- Retain all six safe-deformation search steps and the existing mesh validity, self-intersection,
  support quality, topology growth, exact undo, and release frame-budget checks.
