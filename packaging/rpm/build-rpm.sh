#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
umask 022

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly script_dir
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
readonly repo_root
readonly app_id='app.sculptlite.editor'
readonly spec_file="$script_dir/sculpt-lite.spec"
readonly binary="${SCULPT_LITE_BINARY:-$repo_root/target/release/sculpt-lite}"

if [[ $# -gt 1 ]]; then
    printf 'usage: %s [OUTPUT_DIRECTORY]\n' "${0##*/}" >&2
    exit 2
fi

if [[ "$(uname -m)" != x86_64 ]]; then
    printf 'error: RPM builds are supported only on x86_64\n' >&2
    exit 1
fi
if [[ ! -x "$binary" ]]; then
    printf 'error: release binary not found: %s\n' "$binary" >&2
    exit 1
fi

for required_command in appstreamcli desktop-file-validate rpmbuild rpmspec xmllint; do
    command -v "$required_command" >/dev/null 2>&1 || {
        printf 'error: required command not found: %s\n' "$required_command" >&2
        exit 1
    }
done

cargo_version="$(awk -F '"' '/^version = "/ { print $2; exit }' "$repo_root/Cargo.toml")"
readonly cargo_version
spec_version="$(rpmspec -q --qf '%{VERSION}\n' "$spec_file" | head -n 1)"
readonly spec_version
if [[ -z "$cargo_version" || "$cargo_version" != "$spec_version" ]]; then
    printf 'error: Cargo version %s does not match RPM version %s\n' \
        "${cargo_version:-<missing>}" "${spec_version:-<missing>}" >&2
    exit 1
fi

output_directory="${1:-$repo_root/dist/rpm}"
mkdir -p -- "$output_directory"
output_path="$(cd -- "$output_directory" && pwd -P)"
readonly output_path

if [[ -n "${SOURCE_DATE_EPOCH:-}" ]]; then
    source_date_epoch="$SOURCE_DATE_EPOCH"
elif command -v git >/dev/null 2>&1 && git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1; then
    source_date_epoch="$(git -C "$repo_root" log -1 --format=%ct)"
else
    source_date_epoch=0
fi
if [[ ! "$source_date_epoch" =~ ^[0-9]+$ ]]; then
    printf 'error: SOURCE_DATE_EPOCH must be a non-negative integer\n' >&2
    exit 1
fi
readonly source_date_epoch

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sculpt-lite-rpm.XXXXXX")"
readonly work_dir
trap 'rm -rf -- "$work_dir"' EXIT INT TERM
readonly top_dir="$work_dir/rpmbuild"
readonly source_dir="$top_dir/SOURCES"
mkdir -p -- "$source_dir" "$top_dir/BUILD" "$top_dir/BUILDROOT" "$top_dir/RPMS" "$top_dir/SRPMS"

install -m0755 "$binary" "$source_dir/sculpt-lite"
install -m0644 "$repo_root/packaging/linux/$app_id.desktop" "$source_dir/$app_id.desktop"
install -m0644 "$repo_root/packaging/linux/$app_id.metainfo.xml" \
    "$source_dir/$app_id.metainfo.xml"
install -m0644 "$repo_root/assets/sculpt-lite.png" "$source_dir/$app_id.png"

while IFS= read -r -d '' source_file; do
    touch -h -d "@$source_date_epoch" "$source_file"
done < <(find "$source_dir" -print0)

rpmspec --parse "$spec_file" >/dev/null
SOURCE_DATE_EPOCH="$source_date_epoch" rpmbuild \
    --define "_topdir $top_dir" \
    --define "_buildhost sculptlite.invalid" \
    --define "_binary_filedigest_algorithm 8" \
    --define "_source_filedigest_algorithm 8" \
    --define "clamp_mtime_to_source_date_epoch 1" \
    --define "use_source_date_epoch_as_buildtime 1" \
    -bb "$spec_file"

mapfile -d '' -t rpm_files < <(find "$top_dir/RPMS/x86_64" -maxdepth 1 -type f -name '*.rpm' -print0)
if [[ ${#rpm_files[@]} -ne 1 ]]; then
    printf 'error: expected one binary RPM, found %d\n' "${#rpm_files[@]}" >&2
    exit 1
fi

readonly rpm_output="$output_path/${rpm_files[0]##*/}"
install -m0644 "${rpm_files[0]}" "$rpm_output"
printf '%s\n' "$rpm_output"
