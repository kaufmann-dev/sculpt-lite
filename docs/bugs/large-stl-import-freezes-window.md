# Large STL import freezes the window

- Fixed: 2026-07-19 19:05:42 CEST (+0200)
- Pre-fix commit: `97081757ac43cfa622027d51b98b1dad71ce0143`

## Symptom

Opening a large STL took a long time and could make the desktop report that SculptLite was not responding after the mesh worker finished.

## Confirmed root cause

On a 1,749,148-face STL, parsing took about 0.09 seconds while mesh cleanup and topology construction took about 9.48 seconds. The first rendered frame then staged roughly 91 MB of duplicated vertex, triangle, and wireframe data on the event thread, where the thread accumulated about 11 seconds of CPU time and stopped servicing the window.

## Fix

- Weld STL vertices incrementally while reading instead of retaining the complete triangle soup.
- Use compact topology collections, reuse the topology edge map for face orientation, and precompute BVH centroids and internal bounds.
- Emit each wireframe edge once instead of once per adjacent triangle.
- Populate full GPU buffers on the mesh worker with a cloned wgpu device and only swap prepared buffer handles in the render callback.
- Retain revisioned partial vertex writes for interactive sculpt updates.

The same STL completed with about 3 seconds of mesh-worker CPU, the event thread remained responsive after handoff, and the GUI resident set fell from roughly 720 MB to 644 MB.
