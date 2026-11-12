"""Build the model archive of a release: setsuna_models_<edition>.zip and its SHA-256 file.

    python tools/package_models.py <folder with the checkpoints> <output folder>

Every checkpoint the engine can load is required (setsuna.models.expected_files); the script stops if one is missing
and leaves out anything else in the folder. The archive holds the checkpoints and a SHA256SUMS.txt, and is
reproducible (sorted names, fixed timestamps, stored uncompressed): the same checkpoints always give the same SHA-256.
It is the archive both `setsuna` and JupiterScope's "Get SETSUNA models" download. Upload the .zip and .zip.sha256
as assets of a GitHub release of this repository and set MODEL_RELEASE, MODEL_ARCHIVE and MODEL_SHA256 in
setsuna/models.py (and the matching constants in JupiterScope) to it.
"""
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from setsuna import engine as E  # noqa: E402
from setsuna.models import expected_files  # noqa: E402

STAMP = (2026, 11, 12, 0, 0, 0)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sorted(expected_files())
    missing = [n for n in names if not (src / n).is_file()]
    if missing:
        sys.exit("missing %d checkpoint(s), e.g. %s" % (len(missing), ", ".join(missing[:5])))
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / ("setsuna_models_%s.zip" % E.EDITION_TAG)
    sums = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            data = (src / n).read_bytes()
            sums.append("%s  %s" % (hashlib.sha256(data).hexdigest(), n))
            z.writestr(zipfile.ZipInfo(n, date_time=STAMP), data)
        z.writestr(zipfile.ZipInfo("SHA256SUMS.txt", date_time=STAMP), "\n".join(sums) + "\n")
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (out / (zpath.name + ".sha256")).write_text("%s  %s\n" % (digest, zpath.name))
    print("%s: %d checkpoints, %.1f MB, sha256 %s" % (zpath, len(names), zpath.stat().st_size / 1e6, digest))


if __name__ == "__main__":
    main()
