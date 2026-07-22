# Fly look horizontal direction is inverted

- Fixed: 2026-07-22 18:09:56 CEST (+0200)
- Base commit: `1a4812e7dd5112ecaf5b4471a3ab933b27831f95`

## Symptom

In Fly camera mode, moving the mouse right while holding the right mouse button turned the view
left, and moving the mouse left turned the view right.

## Confirmed root cause

egui reports rightward mouse motion as a positive horizontal delta. The Fly camera added that delta
to its yaw, but its coordinate basis defines camera-right as `forward × Z`, so increasing yaw turns
the forward vector toward camera-left. A runtime camera probe measured a `-0.097843` component along
the original right vector after a positive 40-point horizontal input.

## Fix

- Subtract the horizontal mouse delta from Fly yaw so positive input turns toward camera-right and
  negative input turns toward camera-left.
- Keep the vertical Fly look direction and Orbit controls unchanged.
- Add a regression that checks the resulting forward vector for both rightward and leftward input.

## Verification

- The runtime probe now measures a `0.097843` component toward camera-right.
- `cargo fmt --check`
- `cargo check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
