from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import build_meta as _setuptools_build_meta


_ROOT = Path(__file__).resolve().parent


def _clean_root_build_state() -> None:
    shutil.rmtree(_ROOT / "build", ignore_errors=True)
    for path in _ROOT.glob("*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def get_requires_for_build_wheel(config_settings=None):
    _clean_root_build_state()
    return _setuptools_build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _clean_root_build_state()
    return _setuptools_build_meta.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _clean_root_build_state()
    return _setuptools_build_meta.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _clean_root_build_state()
    return _setuptools_build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _clean_root_build_state()
    return _setuptools_build_meta.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _clean_root_build_state()
    return _setuptools_build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _clean_root_build_state()
    return _setuptools_build_meta.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _clean_root_build_state()
    return _setuptools_build_meta.build_editable(wheel_directory, config_settings, metadata_directory)