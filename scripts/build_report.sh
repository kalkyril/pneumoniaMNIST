#!/usr/bin/env bash
# Compile the report without leaving LaTeX intermediates in the repository.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
report_name=PneumoniaMNIST_MatteoLugli
report_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/texture-report.XXXXXX")"
trap 'rm -rf "$report_build_dir"' EXIT
latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir="$report_build_dir" "$report_name.tex"
if command -v gs >/dev/null 2>&1; then
    # Text remains vector-based; 240 dpi is ample for the small report panels.
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
        -dCompatibilityLevel=1.7 -dAutoRotatePages=/None \
        -dDownsampleColorImages=true -dColorImageResolution=240 \
        -dColorImageDownsampleType=/Bicubic \
        -dDownsampleGrayImages=true -dGrayImageResolution=240 \
        -dGrayImageDownsampleType=/Bicubic \
        -sOutputFile="$report_build_dir/optimized.pdf" \
        "$report_build_dir/$report_name.pdf"
    cp "$report_build_dir/optimized.pdf" "$report_name.pdf"
else
    cp "$report_build_dir/$report_name.pdf" "$report_name.pdf"
fi
printf 'Report written to %s/%s.pdf\n' "$repo_root" "$report_name"
