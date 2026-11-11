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
import urllib.request
import zipfile
from pathlib import Path

REPO = "ryhattori/setsuna"
API = "https://api.github.com/repos/%s/releases/%s"


def _request(url: str, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "setsuna-model-download"})
    token = os.environ.get("GITHUB_TOKEN", "").strip()       # only needed while the repository is private
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return urllib.request.urlopen(req, timeout=60)


def release_assets(tag: str = "latest") -> tuple[str, dict]:
    """(tag, {asset name: API url}) of a release."""
    path = "latest" if tag == "latest" else "tags/" + tag
    with _request(API % (REPO, path)) as r:
        rel = json.load(r)
    return rel["tag_name"], {a["name"]: a["url"] for a in rel.get("assets", [])}


def download(model_dir: str | Path | None = None, tag: str = "latest", quiet: bool = False) -> Path:
    """Fetch, verify and unpack the model archive of release ``tag`` into ``model_dir``; returns the folder."""
    from . import engine
    dst = Path(model_dir) if model_dir else engine.runtime_model_dir()
    if not model_dir and os.environ.get(engine.MODEL_DIR_ENV, "").strip():
        dst = Path(os.environ[engine.MODEL_DIR_ENV]).expanduser()
    tag, assets = release_assets(tag)
    zips = [n for n in assets if n.startswith("setsuna_models_") and n.endswith(".zip")]
    if not zips:
        raise RuntimeError(f"release {tag} of {REPO} has no setsuna_models_*.zip asset")
    name = zips[0]
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / name
        if not quiet:
            print(f"downloading {name} ({tag}) ...", flush=True)
        with _request(assets[name], "application/octet-stream") as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if name + ".sha256" in assets:
            with _request(assets[name + ".sha256"], "application/octet-stream") as r:
                want = r.read().decode().split()[0].lower()
            h = hashlib.sha256()
            with open(zpath, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != want:
                raise RuntimeError(f"checksum mismatch for {name}: the download is incomplete or corrupted")
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                if m.endswith("/"):
                    continue
                target = dst / Path(m).name          # flat: every checkpoint in the model folder itself
                with z.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    if not quiet:
        print(f"models of release {tag} installed in {dst}")
    return dst
