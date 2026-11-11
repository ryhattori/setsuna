"""JupiterScope's folders outside a dataset: the runtime folder and the models folder.

The runtime folder holds what JupiterScope downloads or remembers (SETSUNA's models, the remembered
folders and panel settings, caches): ``$JUPITERSCOPE_RUNTIME_DIR``,
else ``.jupiterscope_runtime`` beside the package when that exists, else ``~/.jupiterscope_runtime``.
The models folder holds the segmentation models, the GT sets and the spine and bouton models:
``~/jupiterscope_seg``.

Jupiter2p, as JupiterScope was called before, named them ``.jupiter2p_runtime`` and
``jupiter2p_seg``. An installation that has one of those (and not the new one) keeps using it, so
nothing is downloaded, trained or set again; nothing is moved.
"""
from __future__ import annotations

import os
from pathlib import Path

RUNTIME_DIR_ENV = "JUPITERSCOPE_RUNTIME_DIR"
RUNTIME_DIR_NAME, FORMER_RUNTIME_DIR_NAME = ".jupiterscope_runtime", ".jupiter2p_runtime"
MODELS_DIR_NAME, FORMER_MODELS_DIR_NAME = "jupiterscope_seg", "jupiter2p_seg"
#: The folder that holds the package (a checkout's root).
CHECKOUT_DIR = Path(__file__).resolve().parents[1]


def _first_existing(folders, default: Path) -> Path:
    return next((folder for folder in folders if folder.exists()), default)


def runtime_dir(beside_package: bool = True) -> Path:
    """The runtime folder. ``beside_package=False`` skips the folder beside the package."""
    env = os.environ.get(RUNTIME_DIR_ENV)
    if env:
        return Path(env).expanduser()
    beside = (CHECKOUT_DIR / RUNTIME_DIR_NAME, CHECKOUT_DIR / FORMER_RUNTIME_DIR_NAME) if beside_package else ()
    home = Path.home()
    return _first_existing((*beside, home / RUNTIME_DIR_NAME, home / FORMER_RUNTIME_DIR_NAME),
                           home / RUNTIME_DIR_NAME)


def models_dir() -> Path:
    """The models folder (it may not exist yet)."""
    home = Path.home()
    return _first_existing((home / MODELS_DIR_NAME, home / FORMER_MODELS_DIR_NAME), home / MODELS_DIR_NAME)
