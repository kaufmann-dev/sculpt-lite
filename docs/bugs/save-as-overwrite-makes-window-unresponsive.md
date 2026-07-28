# Save As overwrite makes the window unresponsive

- Fixed: 2026-07-28 18:08:35 CEST (+0200)
- Base commit: `74f6b2b1e05dbecdeef276e73703341237122bc7`

## Symptom

Choosing an existing STL in Export As could make GNOME report that SculptLite was not responding.
Waiting allowed the overwrite and export to finish successfully.

## Confirmed root cause

STL serialization and atomic replacement already ran on the mesh worker, but the synchronous
`rfd::FileDialog::save_file` call ran on the event thread. A runtime reproduction captured the main
thread parked in `pollster::Signal::wait` through rfd and `SculptLiteApp::export_as` while the save
and overwrite dialog was open. Because winit could not process window events until the dialog
returned, GNOME offered to force-quit the otherwise healthy application.

## Fix

- Run export destination selection on a dedicated `sculptlite-file-dialog` thread.
- Deliver the selected path back through a channel and request an egui repaint when the dialog
  completes.
- Keep an in-app modal active while choosing a destination, prevent closing and conflicting mesh
  actions, and restore the unsaved-action prompt if the dialog is canceled or stops unexpectedly.
- Preserve the existing worker-based STL serialization, atomic overwrite, recovery snapshot, and
  error handling.

## Verification

- A blocking-dialog regression proves file-dialog work does not block its caller.
- A release runtime probe captured the event thread continuing through winit/wgpu while only
  `sculptlite-file-dialog` waited inside rfd.
