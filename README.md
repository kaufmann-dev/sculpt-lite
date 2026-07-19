# SculptLite

SculptLite is a native Linux desktop application for importing, sculpting, and exporting STL meshes. It provides direct sculpting brushes, symmetry, optional adaptive topology, undo and redo, a wireframe overlay, and STL export.

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
cargo install --locked --force --path . --root "$HOME/.local"
```

The executable is installed at `~/.local/bin/sculpt-lite`. Ensure that directory appears before `/usr/bin` in `PATH`, then confirm it is the active copy:

```sh
command -v sculpt-lite
# Expected: /home/your-user/.local/bin/sculpt-lite
```

If necessary, add it for the current shell and open a new terminal afterward:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Run the same `cargo install` command after pulling or making later changes.

### Add the application launcher

`cargo install` installs only the executable. Install or refresh the user-level launcher, icon, and metainfo from the repository root with:

```sh
install -Dm0644 assets/sculpt-lite.png \
  "$HOME/.local/share/icons/hicolor/512x512/apps/app.sculptlite.editor.png"
install -Dm0644 packaging/linux/app.sculptlite.editor.metainfo.xml \
  "$HOME/.local/share/metainfo/app.sculptlite.editor.metainfo.xml"
install -d "$HOME/.local/share/applications"
desktop-file-install \
  --dir="$HOME/.local/share/applications" \
  --set-key=Exec --set-value="$HOME/.local/bin/sculpt-lite %f" \
  --set-key=TryExec --set-value="$HOME/.local/bin/sculpt-lite" \
  packaging/linux/app.sculptlite.editor.desktop
update-desktop-database "$HOME/.local/share/applications"
gtk-update-icon-cache --force --ignore-theme-index \
  "$HOME/.local/share/icons/hicolor"
```

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

## Packages

The Fedora RPM and AppImage scripts consume `target/release/sculpt-lite`. Build the release binary before invoking `packaging/rpm/build-rpm.sh` or `packaging/appimage/build-appimage.sh`.
