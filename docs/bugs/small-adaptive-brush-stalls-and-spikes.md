# Small adaptive brush stalls and creates spikes

- Fixed: 2026-07-22 06:43:14 UTC (+0000)
- Base commit: `88fcfd276bfb44ac29e1a36049cda59466d4e044`

## Symptom

Using adaptive topology with a brush smaller than the vertices of its seed face made a dab miss its frame budget and could pull a lone vertex into a spike connected to distant geometry. Repeated dabs produced a dense, visibly broken patch and made the application progressively slower.

## Confirmed root cause

The empty-brush fallback recursively bisected every long edge in the seed face and its broad one-ring. A single reproduction dab added 320 vertices and took about 10 milliseconds in release mode, but produced only one vertex inside the brush. That vertex remained connected by a 0.325-unit edge, 3.25 times the brush radius and more than 16 times the requested target edge length, so deforming it stretched a large triangle instead of a supported local patch.

## Fix

- Replace broad fallback bisection with a compact concentric support patch inserted directly into the oversized seed face.
- Restrict subsequent split, collapse, and flip candidates to edges intersecting the brush region, prioritize splits nearest the brush center, prevent flips from creating overlong edges, and cap each dab at 128 topology edits.
- Reject an undersized-brush deformation if the generated patch does not provide a bounded edge or falloff transition to every meaningfully influenced vertex.
- Add a five-dab regression covering bounded topology growth, the release frame budget, mesh validity, self-intersections, support quality, and exact topology undo.

The focused first dab now adds 19 vertices instead of 320, and repeated release-mode dabs remain below the 8-millisecond sculpt frame budget.
