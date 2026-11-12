"""Download SETSUNA's trained models from the GitHub Releases of the setsuna repository.

Each release carries one archive of checkpoints (setsuna_models_<edition>.zip) and its SHA-256 checksum
(setsuna_models_<edition>.zip.sha256). download() fetches the archive of a release (default: the latest), checks it and
unpacks it into the model folder (default: $SETSUNA_MODEL_DIR, else ~/.setsuna/models). JupiterScope's model download
uses the same release assets.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "ryhattori/setsuna"
API = "https://api.github.com/repos/%s/releases/%s"

#: The models this version of the package was built and tested with: the archive attached to GitHub release
#: MODEL_RELEASE, verified against MODEL_SHA256 (a download that does not match is refused).
MODEL_RELEASE = "v1.0.0"
MODEL_ARCHIVE = "setsuna_models_C168.zip"
MODEL_SHA256 = "8baaf0d08ddf62ce10e18ca993c8e41f3a3c83bc401f4805edca40b6ec2ee2f5"
#: Set to 1 to turn off the download on first use (offline machines; install with `setsuna download-models`).
NO_DOWNLOAD_ENV = "SETSUNA_NO_DOWNLOAD"


def expected_files() -> list:
    """Every checkpoint of the model archive (the engine's file names, per family, band and seed)."""
    from . import engine as E
    bands_of = {"jGECO1a": ("LOW", "MID")}
    pv_bands = {"GCaMP6f": ("LOW", "MID"), "GCaMP8s": E.BANDS}
    allct_bands = {"GCaMP6f": ("LOW", "MID"), "GCaMP8s": E.BANDS, "GCaMP6s": E.BANDS, "ALL": E.BANDS}
    names = []
    for ind in E.CORE_INDICATORS:
        names += [E.SPEC_FILE_FMT.format(ind=ind, band=b, seed=s) for b in bands_of.get(ind, E.BANDS) for s in E.SEEDS]
    names += [E.GEN_FILE_FMT.format(band=b, seed=s) for b in E.BANDS for s in E.SEEDS]
    for ind, bands in pv_bands.items():
        names += [E.PV_FILE_FMT.format(ind=ind, band=b, seed=s) for b in bands for s in E.SEEDS]
    names += [E.PV_GEN_FILE_FMT.format(band=b, seed=s) for b in E.BANDS for s in E.SEEDS]
    names += [E.SP_FILE_FMT.format(band=b, seed=s) for b in E.BANDS for s in E.SEEDS]
    for ind, bands in allct_bands.items():
        names += [E.ALLCT_FILE_FMT.format(ind=ind, band=b, seed=s) for b in bands for s in E.SEEDS]
    return names


def installed(model_dir) -> bool:
    """True when ``model_dir`` holds every checkpoint of the archive."""
    return all((Path(model_dir) / n).is_file() for n in expected_files())


def ensure_models(quiet: bool = False) -> Path:
    """The model folder, downloading this version's models into it first if they are not installed anywhere."""
    from . import engine
    for d in engine.model_dir_candidates():
        if installed(d):
            return d
    if os.environ.get(NO_DOWNLOAD_ENV, "").strip() not in ("", "0"):
        raise FileNotFoundError("SETSUNA's models are not installed and %s is set; install them with "
                                "`setsuna download-models` (release %s)" % (NO_DOWNLOAD_ENV, MODEL_RELEASE))
    if not quiet:
        print("SETSUNA's models are not installed: downloading release %s (%s, about 64 MB) ..." % (MODEL_RELEASE, MODEL_ARCHIVE),
              flush=True)
    return download(quiet=quiet)


def _request(url: str, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "setsuna-model-download"})
    token = os.environ.get("GITHUB_TOKEN", "").strip()       # only needed while the repository is private
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return urllib.request.urlopen(req, timeout=60)


def release_assets(tag: str = "latest") -> tuple[str, dict]:
    """(tag, {asset name: API url}) of a release."""
    path = "latest" if tag == "latest" else "tags/" + tag
    try:
        with _request(API % (REPO, path)) as r:
            rel = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"no release {tag!r} found in {REPO} (or the repository is not public yet; set "
                               "GITHUB_TOKEN to a token with read access)") from None
        raise
    return rel["tag_name"], {a["name"]: a["url"] for a in rel.get("assets", [])}


def download(model_dir: str | Path | None = None, tag: str | None = None, quiet: bool = False) -> Path:
    """Fetch, verify and unpack the model archive of release ``tag`` (default: MODEL_RELEASE, the models of this version)
    into ``model_dir``; returns the folder."""
    tag = tag or MODEL_RELEASE
    from . import engine
    dst = Path(model_dir) if model_dir else engine.runtime_model_dir()
    if not model_dir and os.environ.get(engine.MODEL_DIR_ENV, "").strip():
        dst = Path(os.environ[engine.MODEL_DIR_ENV]).expanduser()
    tag, assets = release_assets(tag)
    zips = [n for n in assets if n.startswith("setsuna_models_") and n.endswith(".zip")]
    if not zips:
        raise RuntimeError(f"release {tag} of {REPO} has no setsuna_models_*.zip asset")
    name = MODEL_ARCHIVE if (tag == MODEL_RELEASE and MODEL_ARCHIVE in zips) else zips[0]
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / name
        if not quiet:
            print(f"downloading {name} ({tag}) ...", flush=True)
        with _request(assets[name], "application/octet-stream") as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        want = None
        if name + ".sha256" in assets:
            with _request(assets[name + ".sha256"], "application/octet-stream") as r:
                want = r.read().decode().split()[0].lower()
        if want or (tag == MODEL_RELEASE and name == MODEL_ARCHIVE):
            h = hashlib.sha256()
            with open(zpath, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if tag == MODEL_RELEASE and name == MODEL_ARCHIVE:
                want = MODEL_SHA256          # the pinned checksum of this version's models
            if h.hexdigest() != want:
                raise RuntimeError(f"checksum mismatch for {name}: the download is incomplete or corrupted")
        with zipfile.ZipFile(zpath) as z:
            sums = {}
            if "SHA256SUMS.txt" in z.namelist():       # every checkpoint is checked before any is written
                for line in z.read("SHA256SUMS.txt").decode().splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        sums[parts[1]] = parts[0].lower()
            members = [m for m in z.namelist() if m.endswith(".pt")]
            for m in members:
                if m in sums and hashlib.sha256(z.read(m)).hexdigest() != sums[m]:
                    raise RuntimeError(f"checksum mismatch for {m} inside {name}")
            for m in members:
                target = dst / Path(m).name          # flat: every checkpoint in the model folder itself
                with z.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    if not quiet:
        print(f"models of release {tag} installed in {dst}")
    return dst
