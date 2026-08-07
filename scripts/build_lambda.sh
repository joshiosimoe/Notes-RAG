#!/usr/bin/env bash
# Build the indexer Lambda deployment package.
#
# Cross-compiles for the Lambda runtime from any host: every dependency ships a
# manylinux wheel, so --only-binary makes this a download-and-unpack rather than
# a build, and no Docker is required.
#
# boto3 is deliberately excluded - the python3.12 runtime provides a current one.
# pysqlite3-binary is deliberately INCLUDED and is not optional: the runtime's
# stdlib sqlite3 has no loadable-extension support, so sqlite-vec cannot load
# without it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build/lambda"
ZIP="$ROOT/build/lambda.zip"

rm -rf "$BUILD" "$ZIP"
mkdir -p "$BUILD"

uv pip install \
  --python-platform x86_64-manylinux2014 \
  --python-version 3.12 \
  --only-binary :all: \
  --target "$BUILD" \
  --link-mode=copy \
  --quiet \
  "sqlite-vec>=0.1.6" \
  "pysqlite3-binary>=0.5" \
  "PyYAML>=6.0"

cp -r "$ROOT/src/notes_rag" "$BUILD/notes_rag"
find "$BUILD" -name '__pycache__' -type d -prune -exec rm -rf {} +

# No `zip` binary on the build host, and Python's zipfile lets us make the
# archive byte-reproducible: fixed entry order plus a pinned date_time/mode
# per entry, so identical source trees produce identical zip bytes even
# though `cp -r` stamps notes_rag/ with the current wall-clock mtime on every
# run. Terraform (Task 7) hashes this zip for source_code_hash - a hash that
# drifts on every rebuild with no source change would make every `apply`
# look like a redeploy.
python3 - "$BUILD" "$ZIP" <<'PY'
import sys, zipfile
from pathlib import Path

build, archive_path = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p for p in build.rglob("*") if p.is_file())
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        zi = zipfile.ZipInfo(path.relative_to(build).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        archive.writestr(zi, path.read_bytes())
print(f"packed {len(files)} files")
PY

echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
