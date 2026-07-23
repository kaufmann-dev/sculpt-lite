# SculptLite

SculptLite is a native Linux desktop application for importing, sculpting, remeshing, and exporting STL meshes. It provides responsive fixed-topology sculpting brushes with area-weighted surface frames, stabilized plane brushes, low-shrink smoothing, configurable capped, accumulating, and Airbrush strokes, symmetry, undo and redo, an explicit whole-object voxel remesh, a wireframe overlay, and STL export.

## Viewport controls

Choose **Orbit** or **Fly** beside the Frame button, or press `V` to toggle modes. In Orbit, right-drag pans, middle-drag orbits, and the wheel zooms. Fly defaults to **Level** movement, where `W`/`S` and `A`/`D` stay horizontal regardless of look pitch; choose **Free flight** beside the Fly button to make `W`/`S` follow the look direction instead. In either Fly style, hold RMB to capture the pointer, look with the mouse, move down with Shift, move up with Space, hold Ctrl for a temporary 4× speed boost, and use the wheel to adjust the base flight speed. Release RMB or press Escape to release the pointer. Press `F` to frame the mesh and return to Orbit.

With the pointer released, a left click applies one dab and left-drag sculpts from either viewpoint. Choose **Capped** to apply only newly increased brush influence until the button is released, **Accumulate** to build up with every distance-spaced dab while moving, or **Airbrush** to add the same spatial buildup plus timed dabs while held still. Press `A` to cycle these choices. The choice is remembered separately for each sculpt tool and captured when a stroke begins. Grab remains movement-driven: it anchors the affected region on press and keeps moving that region even when the pointer leaves the surface. The usual Shift-to-Smooth and Ctrl-to-Invert modifiers work throughout a stroke.

Draw, Crease, and Pinch derive their working direction from the brush footprint instead of a single triangle. Clay and Flatten additionally stabilize their surface plane over the stroke, while Smooth uses a two-pass correction that loses substantially less volume than basic Laplacian smoothing. Geometry dabs are reduced to a safe partial step when the requested displacement would collapse or reverse a face. When a brush is too small for the local triangle spacing, the status bar warns that the result is undersampled; increase the radius or voxel-remesh a closed mesh at finer resolution for more detail.

## Voxel remeshing

Open **Mesh** in the tool panel to replace the complete object with a uniform voxel-derived surface. Resolution is measured across the longest object axis, ranges from 32 to 192 cells, and defaults to 96; the panel shows the resulting world-space voxel size. Features smaller than a voxel can disappear, and nearby components can merge.

Voxel remeshing requires a closed two-manifold source. Open or non-manifold STL files remain importable, sculptable, and exportable, but the panel explains why remeshing is unavailable. The operation runs on the mesh worker while the original remains intact. A validated manifold replacement and its GPU upload are installed together; failures leave the original mesh unchanged. Masks are reprojected from the nearest source triangles, the camera is preserved, and the status bar reports elapsed time plus old and new face counts. Replacement undo and redo are also prepared on the worker. If the required whole-mesh snapshot exceeds the 512 MiB history budget, remeshing still completes and incompatible history is cleared.

## Run from source

Rust 1.96.0 is selected by `rust-toolchain.toml`. From a checkout, open an STL directly or start with an empty window:

```sh
cargo run -- /path/to/model.stl
# or
cargo run
```

## Install for your user

Install or replace the user-local executable without `sudo`:

```sh
./install
```

The installer builds and installs the executable, application launcher, icon, and AppStream metadata. The executable is installed at `~/.local/bin/sculpt-lite`. Ensure that directory appears before `/usr/bin` in `PATH`, then confirm it is the active copy:

```sh
command -v sculpt-lite
# Expected: /home/your-user/.local/bin/sculpt-lite
```

If necessary, add it for the current shell and open a new terminal afterward:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Run `./install` again after pulling or making later changes; the existing executable and desktop files are replaced.

## Verify a change

```sh
cargo fmt --check
cargo check
cargo clippy --all-targets -- -D warnings
cargo test
```

For a release-binary window smoke check, run:

```sh
cargo build --release
packaging/linux/smoke-x11.sh target/release/sculpt-lite
```

Large-mesh performance probes are ignored by the normal test suite. Run them individually in release mode so parallel tests do not distort their timings:

```sh
cargo test --release --bin sculpt-lite million_face_fixed_sculpt_sample -- --ignored --nocapture --test-threads=1
cargo test --release --bin sculpt-lite million_face_mesh_build_pick_and_deform_refresh -- --ignored --nocapture --test-threads=1
cargo test --release --bin sculpt-lite half_million_vertex_deformation_pack -- --ignored --nocapture --test-threads=1
cargo test --release --bin sculpt-lite resolution_96_remesh_probe -- --ignored --nocapture --test-threads=1
```

## Packages

The Fedora RPM and AppImage scripts consume `target/release/sculpt-lite`. Build the release binary before invoking `packaging/rpm/build-rpm.sh` or `packaging/appimage/build-appimage.sh`.
