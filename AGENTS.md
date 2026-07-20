# Repository Instructions

## Local user installation

After successfully verifying changes that affect the runtime executable or compile-time assets, replace the current user installation:

```sh
cargo install --locked --force --path . --root "$HOME/.local"
test -x "$HOME/.local/bin/sculpt-lite"
```

The installed binary must remain user-local at `~/.local/bin/sculpt-lite`; do not install it system-wide. If `assets/` or `packaging/linux/` changes, also rerun the launcher-install block in `README.md`.
Skip local installation for documentation-only and test-only changes.

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
- `src/mesh.rs` owns validated mesh data, local topology updates, remeshing, normals, and spatial queries; `src/sculpt.rs` applies and validates brush operations; `src/stroke.rs` retains captured input state while scheduling distance- and time-based brush dabs.
- `src/renderer.rs` and `src/shader.wgsl` own the wgpu viewport; keep CPU/GPU updates revisioned and local, preserve growth headroom within device limits, and avoid blocking the UI thread.
- `src/history.rs`, `src/camera.rs`, and `src/stl.rs` respectively own undo/redo, orbit-camera math, and STL I/O.

## Packaging

Build `target/release/sculpt-lite` before running the RPM or AppImage scripts. Keep the version in `Cargo.toml` synchronized with `packaging/rpm/sculpt-lite.spec`, and keep the application ID `app.sculptlite.editor` consistent across native-window and packaging metadata.
