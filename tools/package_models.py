"""Build the model archive of a release: setsuna_models_<edition>.zip, its SHA-256 file and a manifest.

    python tools/package_models.py <folder with the checkpoints> <output folder>

Every checkpoint the engine can load is required (file names from setsuna/engine.py); the script stops if one is
missing and leaves out anything else in the folder. Upload the .zip and .zip.sha256 as assets of a GitHub release of
this repository; `setsuna download-models` and JupiterScope's model download fetch them from there.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from setsuna import engine as E  # noqa: E402

BANDS_OF = {"jGECO1a": ("LOW", "MID")}                          # every other specialist: LOW, MID, HIGH
PV_BANDS = {"GCaMP6f": ("LOW", "MID"), "GCaMP8s": E.BANDS}
ALLCT_BANDS = {"GCaMP6f": ("LOW", "MID"), "GCaMP8s": E.BANDS, "GCaMP6s": E.BANDS, "ALL": E.BANDS}


def expected():
    names = []
    for ind in E.CORE_INDICATORS:
        names += [E.SPEC_FILE_FMT.format(ind=ind, band=b, seed=s) for b in BANDS_OF.get(ind, E.BANDS) for s in E.SEEDS]
    names += [E.GEN_FILE_FMT.format(band=b, seed=s) for b in E.BANDS for s in E.SEEDS]
    for ind, bands in PV_BANDS.items():
        names += [E.PV_FILE_FMT.format(ind=ind, band=b, seed=s) for b in bands for s in E.SEEDS]
    names += [E.PV_GEN_FILE_FMT.format(band=b, seed=s) for b in E.BANDS for s in E.SEEDS]
    names += [E.SP_FILE_FMT.format(band=b, seed=s) for b in E.BANDS for s in E.SEEDS]
    for ind, bands in ALLCT_BANDS.items():
        names += [E.ALLCT_FILE_FMT.format(ind=ind, band=b, seed=s) for b in bands for s in E.SEEDS]
    return names


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = expected()
    missing = [n for n in names if not (src / n).is_file()]
    if missing:
        sys.exit("missing %d checkpoint(s), e.g. %s" % (len(missing), ", ".join(missing[:5])))
    out.mkdir(parents=True, exist_ok=True)
    edition = E.EDITION_TAG.lower()
    zpath = out / ("setsuna_models_%s.zip" % edition)
    manifest = {"edition": E.EDITION_TAG, "files": {n: {"sha256": sha256(src / n), "bytes": (src / n).stat().st_size} for n in names}}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(src / n, n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
    (out / (zpath.name + ".sha256")).write_text("%s  %s\n" % (sha256(zpath), zpath.name))
    print("%s: %d checkpoints, %.1f MB" % (zpath, len(names), zpath.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
