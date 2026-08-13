"""RSL-RL version compatibility checks."""

from __future__ import annotations

import importlib.metadata as metadata

from packaging import version


MIN_RSL_RL_VERSION = "3.1.0"
MAX_RSL_RL_VERSION = "3.3.0"


def check_rsl_rl_version() -> str:
    """Check that the installed rsl-rl-lib version matches this repository's wrapper contract."""
    try:
        installed_version = metadata.version("rsl-rl-lib")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit("rsl-rl-lib is not installed in the active Python environment.") from exc

    parsed_version = version.parse(installed_version)
    if parsed_version < version.parse(MIN_RSL_RL_VERSION) or parsed_version > version.parse(MAX_RSL_RL_VERSION):
        raise SystemExit(
            "Unsupported rsl-rl-lib version.\n"
            f"Installed version: {installed_version}\n"
            f"Supported range: >= {MIN_RSL_RL_VERSION}, <= {MAX_RSL_RL_VERSION}\n"
            "This repository's custom VecEnv wrapper uses the RSL-RL 3.3 TensorDict observation API."
        )
    return installed_version
