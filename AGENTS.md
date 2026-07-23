# Repository Instructions

## Build and verification

Use Rust 1.96.0 from `rust-toolchain.toml`. Run the smallest relevant check after each change, then install the local application:

```sh
cargo fmt --check
cargo check
cargo clippy --all-targets -- -D warnings
cargo test
```

For a release GUI smoke check, build `target/release/sculpt-lite` and run `packaging/linux/smoke-x11.sh target/release/sculpt-lite`.

Run the ignored large-mesh probes documented in `README.md` individually and in release mode after performance-sensitive sculpt, mesh, or renderer changes. Do not run them concurrently; CPU and allocator contention makes their timings misleading.

## Project structure

- `src/app.rs` owns the egui UI, input handling, exact cached document bounds, per-frame render batching, background mesh jobs, and document lifecycle.
- `src/mesh.rs` owns validated mesh data, adjacency, normals, and spatial queries; `src/voxel_remesh.rs` builds explicit whole-object manifold replacements; `src/sculpt.rs` applies and validates fixed-topology brush operations; `src/stroke.rs` retains captured input state while scheduling distance- and time-based brush dabs.
- `src/renderer.rs` and `src/shader.wgsl` own the wgpu viewport; keep CPU/GPU updates revisioned and local, preserve growth headroom within device limits, and avoid blocking the UI thread.
- `src/history.rs`, `src/camera.rs`, and `src/stl.rs` respectively own local and whole-mesh undo/redo, orbit-camera math, and STL I/O.

## Packaging

Build `target/release/sculpt-lite` before running the RPM or AppImage scripts. Keep the version in `Cargo.toml` synchronized with `packaging/rpm/sculpt-lite.spec`, and keep the application ID `app.sculptlite.editor` consistent across native-window and packaging metadata.
