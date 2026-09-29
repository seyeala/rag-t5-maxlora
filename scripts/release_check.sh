#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORK="$(mktemp -d)"
cleanup() {
  rm -rf "$WORK" "$ROOT/build"
}
trap cleanup EXIT

DIST="$WORK/dist"
TESTS="$WORK/tests"
VENV="$WORK/venv"
mkdir -p "$DIST"
cp -a tests "$TESTS"

python -m build --outdir "$DIST"
WHEEL="$(find "$DIST" -maxdepth 1 -name '*.whl' -print -quit)"
SDIST="$(find "$DIST" -maxdepth 1 -name '*.tar.gz' -print -quit)"
test -n "$WHEEL"
test -n "$SDIST"
python - "$WHEEL" "$SDIST" <<'PY'
import sys
import tarfile
import zipfile

wheel, sdist = sys.argv[1:3]
with zipfile.ZipFile(wheel) as archive:
    wheel_files = archive.namelist()
with tarfile.open(sdist) as archive:
    sdist_files = archive.getnames()

forbidden = (
    "data/generated/",
    "outputs/",
    "local_plans",
    "ml_persistent",
    ".venv/",
    "__pycache__",
)
assert not any(
    any(marker in name for marker in forbidden)
    for name in wheel_files + sdist_files
)
required = (
    "rag_t5/models/inference.py",
    "rag_t5/models/export.py",
    "rag_t5/eval/instruction.py",
    "rag_t5/eval/sst2.py",
    "rag_t5/synth/gen_qa_offline.py",
)
for name in required:
    assert name in wheel_files, name
print("artifact inspection passed")
PY

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "$WHEEL[data,demo,dev]"
cd "$WORK"
python - <<'PY'
import importlib.metadata as metadata
import rag_t5
import rag_t5.eval.instruction
import rag_t5.eval.sst2
import rag_t5.models.export
import rag_t5.models.inference
import rag_t5.synth.gen_qa_offline
import cli.generate
import cli.gen_synth
print("installed version", metadata.version("rag-t5-maxlora"))
PY

python -m rag_t5.eval.instruction --help >/dev/null
python -m rag_t5.eval.sst2 --help >/dev/null
python -m rag_t5.models.export --help >/dev/null
python -m cli.generate --help >/dev/null
python -m cli.gen_synth --help >/dev/null
smoke-model --help >/dev/null
train-model --help >/dev/null
CUDA_VISIBLE_DEVICES="" pytest -q "$TESTS"
python -m pip check
echo "release check passed"
