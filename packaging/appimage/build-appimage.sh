#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
umask 022

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly script_dir
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
readonly repo_root
readonly app_id='app.sculptlite.editor'
readonly binary="${SCULPT_LITE_BINARY:-$repo_root/target/release/sculpt-lite}"
readonly source_date_epoch="${SOURCE_DATE_EPOCH:-0}"

if [[ $# -gt 1 ]]; then
    printf 'usage: %s [OUTPUT.AppImage]\n' "${0##*/}" >&2
    exit 2
fi

if [[ ! "$source_date_epoch" =~ ^[0-9]+$ ]]; then
    printf 'error: SOURCE_DATE_EPOCH must be a non-negative integer\n' >&2
    exit 1
fi

if [[ ! -x "$binary" ]]; then
    printf 'error: release binary not found: %s\n' "$binary" >&2
    exit 1
fi

package_version="$(awk -F '"' '/^version = "/ { print $2; exit }' "$repo_root/Cargo.toml")"
readonly package_version
if [[ -z "$package_version" ]]; then
    printf 'error: could not read the package version from Cargo.toml\n' >&2
    exit 1
fi

output="${1:-$repo_root/dist/SculptLite-$package_version-x86_64.AppImage}"
mkdir -p -- "$(dirname -- "$output")"
output_path="$(realpath -m -- "$output")"
readonly output_path

appimage_tool_name="${APPIMAGETOOL:-appimagetool}"
if ! appimage_tool="$(command -v "$appimage_tool_name")"; then
    printf 'error: appimagetool is required; set APPIMAGETOOL to its path\n' >&2
    exit 1
fi
readonly appimage_tool

appimage_runtime="${APPIMAGE_RUNTIME:-}"
if [[ -z "$appimage_runtime" || ! -f "$appimage_runtime" ]]; then
    printf 'error: set APPIMAGE_RUNTIME to a pinned x86_64 type-2 runtime file\n' >&2
    exit 1
fi
appimage_runtime="$(realpath -- "$appimage_runtime")"
readonly appimage_runtime

for required_file in \
    "$repo_root/assets/sculpt-lite.png" \
    "$repo_root/packaging/linux/$app_id.desktop" \
    "$repo_root/packaging/linux/$app_id.metainfo.xml" \
    "$script_dir/AppRun"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'error: required packaging input not found: %s\n' "$required_file" >&2
        exit 1
    fi
done

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sculpt-lite-appimage.XXXXXX")"
readonly work_dir
trap 'rm -rf -- "$work_dir"' EXIT INT TERM
readonly app_dir="$work_dir/SculptLite.AppDir"

install -Dm0755 "$binary" "$app_dir/usr/bin/sculpt-lite"
install -Dm0755 "$script_dir/AppRun" "$app_dir/AppRun"
install -Dm0644 "$repo_root/packaging/linux/$app_id.desktop" \
    "$app_dir/usr/share/applications/$app_id.desktop"
install -Dm0644 "$repo_root/packaging/linux/$app_id.metainfo.xml" \
    "$app_dir/usr/share/metainfo/$app_id.appdata.xml"
install -Dm0644 "$repo_root/assets/sculpt-lite.png" \
    "$app_dir/usr/share/icons/hicolor/512x512/apps/$app_id.png"

ln -s "usr/share/applications/$app_id.desktop" "$app_dir/$app_id.desktop"
ln -s "usr/share/icons/hicolor/512x512/apps/$app_id.png" "$app_dir/$app_id.png"
ln -s "$app_id.png" "$app_dir/.DirIcon"

command -v ldconfig >/dev/null 2>&1 || {
    printf 'error: ldconfig is required to resolve runtime libraries\n' >&2
    exit 1
}

declare -a dependency_roots=("$binary")
readonly required_runtime_sonames=(
    libX11.so.6
    libXcursor.so.1
    libXi.so.6
    libXrandr.so.2
    libwayland-client.so.0
    libxkbcommon-x11.so.0
    libxkbcommon.so.0
)
for soname in "${required_runtime_sonames[@]}"; do
    library_path="$(
        ldconfig -p | awk -v soname="$soname" \
            '$1 == soname && !found { path = $NF; found = 1 } END { if (found) print path }'
    )"
    if [[ -z "$library_path" || ! -f "$library_path" ]]; then
        printf 'error: required runtime library not found: %s\n' "$soname" >&2
        exit 1
    fi
    dependency_roots+=("$library_path")
done

ldd_output="$work_dir/ldd.txt"
: >"$ldd_output"
for dependency_root in "${dependency_roots[@]}"; do
    if ! ldd "$dependency_root" >>"$ldd_output"; then
        printf 'error: could not inspect dependency: %s\n' "$dependency_root" >&2
        exit 1
    fi
done

mapfile -t dynamic_libraries < <(
    {
        printf '%s\n' "${dependency_roots[@]:1}"
        awk '/=> \// { print $3 } /^\// { print $1 }' "$ldd_output"
    } | LC_ALL=C sort -u
)
for library in "${dynamic_libraries[@]}"; do
    library_name="${library##*/}"
    case "$library_name" in
        ld-linux* | libc.so.* | libdl.so.* | libgcc_s.so.* | libm.so.* | libpthread.so.* | \
            libresolv.so.* | librt.so.* | libstdc++.so.* | libutil.so.* | libEGL.so.* | \
            libGL.so.* | libGLES*.so.* | libGLX.so.* | libOpenGL.so.* | libvulkan.so.*)
            continue
            ;;
    esac
    install -Dm0644 "$library" "$app_dir/usr/lib/$library_name"
done

if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$app_dir/usr/share/applications/$app_id.desktop"
fi
if command -v appstreamcli >/dev/null 2>&1; then
    appstreamcli validate --no-net "$app_dir/usr/share/metainfo/$app_id.appdata.xml"
fi
if command -v xmllint >/dev/null 2>&1; then
    xmllint --noout "$app_dir/usr/share/metainfo/$app_id.appdata.xml"
fi

while IFS= read -r -d '' entry; do
    touch -h -d "@$source_date_epoch" "$entry"
done < <(find "$app_dir" -print0)

rm -f -- "$output_path"
ARCH=x86_64 \
APPIMAGE_EXTRACT_AND_RUN=1 \
SOURCE_DATE_EPOCH="$source_date_epoch" \
    "$appimage_tool" --runtime-file "$appimage_runtime" --comp zstd "$app_dir" "$output_path"
chmod 0755 "$output_path"
printf '%s\n' "$output_path"
