#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    printf 'usage: %s EXECUTABLE\n' "${0##*/}" >&2
    exit 2
fi

readonly executable="$1"

if [[ ! -x "$executable" ]]; then
    printf 'error: executable not found: %s\n' "$executable" >&2
    exit 1
fi

if [[ "${SCULPT_LITE_INSIDE_XVFB:-0}" != 1 ]]; then
    command -v xvfb-run >/dev/null 2>&1 || {
        printf 'error: xvfb-run is required\n' >&2
        exit 1
    }
    exec env SCULPT_LITE_INSIDE_XVFB=1 xvfb-run -a -s '-screen 0 1280x800x24' "$0" "$executable"
fi

command -v xdotool >/dev/null 2>&1 || {
    printf 'error: xdotool is required\n' >&2
    exit 1
}

log_file="$(mktemp "${TMPDIR:-/tmp}/sculpt-lite-x11.XXXXXX.log")"
readonly log_file
app_pid=''

# This function is invoked indirectly by the traps below.
# shellcheck disable=SC2317
cleanup() {
    if [[ -n "$app_pid" ]] && kill -0 "$app_pid" 2>/dev/null; then
        kill "$app_pid" 2>/dev/null || true
        wait "$app_pid" 2>/dev/null || true
    fi
    rm -f -- "$log_file"
}
trap cleanup EXIT INT TERM

env -u WAYLAND_DISPLAY \
    XDG_SESSION_TYPE=x11 \
    WINIT_UNIX_BACKEND=x11 \
    WGPU_BACKEND=vulkan \
    "$executable" >"$log_file" 2>&1 &
app_pid=$!

window_id=''
for ((attempt = 0; attempt < 150; attempt += 1)); do
    if ! kill -0 "$app_pid" 2>/dev/null; then
        wait "$app_pid" || true
        printf 'error: SculptLite exited before creating an X11 window\n' >&2
        sed -n '1,200p' "$log_file" >&2
        exit 1
    fi
    window_id="$(xdotool search --name '^SculptLite$' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$window_id" ]]; then
        break
    fi
    sleep 0.1
done

if [[ -z "$window_id" ]]; then
    printf 'error: SculptLite did not create an X11 window within 15 seconds\n' >&2
    sed -n '1,200p' "$log_file" >&2
    exit 1
fi

kill -TERM "$app_pid"
for ((attempt = 0; attempt < 50; attempt += 1)); do
    if ! kill -0 "$app_pid" 2>/dev/null; then
        if wait "$app_pid"; then
            app_status=0
        else
            app_status=$?
        fi
        app_pid=''
        if [[ $app_status -ne 0 && $app_status -ne 143 ]]; then
            printf 'error: SculptLite returned unexpected status %d after SIGTERM\n' \
                "$app_status" >&2
            sed -n '1,200p' "$log_file" >&2
            exit 1
        fi
        printf 'SculptLite X11 smoke check passed\n'
        exit 0
    fi
    sleep 0.1
done

printf 'error: SculptLite did not exit after SIGTERM\n' >&2
sed -n '1,200p' "$log_file" >&2
exit 1
