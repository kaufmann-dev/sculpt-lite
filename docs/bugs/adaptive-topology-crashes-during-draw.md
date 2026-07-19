# Adaptive topology crashes during draw

- Fixed: 2026-07-19 19:47:50 CEST (+0200)
- Pre-fix commit: `d8fa097fcda3624298cb7964c6a8fc94a0cf7adb`

## Symptom

Enabling adaptive topology and drawing on an imported mesh terminated SculptLite.
Both observed crashes were fatal wgpu validation errors from `Queue::write_buffer`.

## Confirmed root cause

Imported meshes use prepared GPU vertex buffers sized exactly for the imported vertex count.
Live adaptive remeshing can add vertices, but the incremental vertex uploader checked only whether
a buffer existed before writing changed ranges. It did not check whether the grown CPU vertex array
still fit that buffer. The observed write began at byte 7,510,368, exactly at the end of a
7,510,368-byte destination buffer, and attempted to append another 1,568 bytes.

## Fix

- Check the complete CPU vertex array size against GPU buffer capacity before any partial vertex
  upload.
- Route an outgrown buffer through the existing full upload path, which reallocates with spare
  power-of-two capacity before writing.
- Add a regression test using the exact crashed buffer size and verifying that one added vertex
  requires reallocation.
