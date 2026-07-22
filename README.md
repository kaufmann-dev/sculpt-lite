# SculptLite

SculptLite is a native Linux desktop application for importing, sculpting, and exporting STL meshes. It provides responsive distance-spaced sculpting brushes including Clay and Crease, time-budgeted path processing that preserves captured input, optional Airbrush buildup, symmetry, local adaptive topology that fairly schedules mirrored remeshing and resumes bounded topology and deformation-safety validation across frames before deformation, undo and redo, a wireframe overlay, and STL export.

## Viewport controls

Choose **Orbit** or **Fly** beside the Frame button, or press `V` to toggle modes. In Orbit, right-drag pans, middle-drag orbits, and the wheel zooms. In Fly, hold RMB to capture the pointer, look with the mouse, move with `W`/`S`, strafe with `A`/`D`, move down with Shift, and move up with Space; use the wheel to adjust flight speed. Release RMB or press Escape to release the pointer. Press `F` to frame the mesh and return to Orbit.

With the pointer released, left-drag sculpting and the usual Shift-to-Smooth and Ctrl-to-Invert modifiers work from either viewpoint.

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
cargo test --release --bin sculpt-lite million_face_fixed_and_adaptive_sculpt_samples -- --ignored --nocapture --test-threads=1
cargo test --release --bin sculpt-lite million_face_mesh_build_pick_and_local_remesh -- --ignored --nocapture --test-threads=1
cargo test --release --bin sculpt-lite half_million_vertex_deformation_pack -- --ignored --nocapture --test-threads=1
```

## Packages

The Fedora RPM and AppImage scripts consume `target/release/sculpt-lite`. Build the release binary before invoking `packaging/rpm/build-rpm.sh` or `packaging/appimage/build-appimage.sh`.
