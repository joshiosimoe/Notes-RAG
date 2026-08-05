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

# No `zip` binary on the build host, and Python's zipfile is deterministic
# because we control the entry order - the same tree produces the same archive.
python3 - "$BUILD" "$ZIP" <<'PY'
import sys, zipfile
from pathlib import Path

build, archive_path = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p for p in build.rglob("*") if p.is_file())
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(build).as_posix())
print(f"packed {len(files)} files")
PY

echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
