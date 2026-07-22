# Tool panel grows across frames

- Fixed: 2026-07-22 18:05:19 CEST (+0200)
- Base commit: `ff72f475fa718bc3e2245650df4686776173259e`

## Symptom

After launch, the fixed left tool panel continuously widened until it occupied the application
window and hid the viewport.

## Confirmed root cause

The two buttons in the Mask section divided the available width after subtracting a hard-coded
6-point gap, while their enclosing `ui.horizontal` used egui's default 8-point item spacing. The
row therefore requested 2 points more than its available width on every frame. A non-resizable egui
panel still persists its content's measured width when configured with `default_size`, so the excess
became the next frame's width and created an unbounded feedback loop.

A headless reproduction grew from 240 to 260 points in 10 frames. Isolating the tool grid, symmetry
combo box, and Mask row showed that only the Mask row reproduced the growth.

## Fix

- Calculate equal two-column item widths with the active layout spacing instead of a hard-coded
  value.
- Reuse the same width helper for the tool grid's explicitly configured spacing.
- Add a repeated-frame egui regression that verifies the panel remains 240 points wide.

## Verification

- `cargo fmt --check`
- `cargo check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
