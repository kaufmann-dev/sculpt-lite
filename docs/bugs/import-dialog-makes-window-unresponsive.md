# Import dialog makes the window unresponsive

- Fixed: 2026-07-28 18:33:11 CEST (+0200)
- Base commit: `77828ab0a4ed3dc2dcb6317d9670534b37dc3c98`

## Symptom

Opening the Import STL file chooser could make GNOME report that SculptLite was not responding
while the chooser remained open.

## Confirmed root cause

The synchronous `rfd::FileDialog::pick_file` call ran on the event thread. A runtime reproduction
captured the main thread parked in `pollster::Signal::wait` through rfd and
`SculptLiteApp::perform_action` while the import dialog was open. The mesh worker was idle, but
winit could not process window events until the chooser returned.

## Fix

- Generalize the asynchronous file-dialog task used by Export As to support Import STL.
- Run import source selection on the dedicated `sculptlite-file-dialog` thread and deliver the
  selected path through a channel.
- Keep the existing in-app modal and action guards active for either import or export selection.
- Start the existing background import job only after the event thread receives the selected path.

## Verification

- A blocking-dialog regression proves file-dialog work does not block its caller.
- A release runtime probe captured the event thread continuing through winit/wgpu while only
  `sculptlite-file-dialog` waited inside `rfd::FileDialog::pick_file`.
- All native file chooser calls now run inside the asynchronous file-dialog task.
