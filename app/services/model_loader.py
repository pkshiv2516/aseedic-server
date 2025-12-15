from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_decision_forests as tfdf
try:
    import tf_keras as keras  # TF >= 2.16
except Exception:
    from tensorflow import keras  # TF 2.15

from app.config import Config


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_artifacts(model_type):
    Config.ARTIFACTS_DIR = Config.resolve_artifacts_dir(model_type)
    print(Config.ARTIFACTS_DIR)
    Config.CAT_CLASS_FILES = {
        "Headquarters Location": Config.ARTIFACTS_DIR / "headquarters_location_classes.json",
        "Funding Status": Config.ARTIFACTS_DIR / "funding_status_classes.json",
        "Last Funding Type": Config.ARTIFACTS_DIR / "last_funding_type_classes.json",
    }
    mapping = _read_json(Config.ARTIFACTS_DIR / "original_to_safe_columns.json")
    feature_order = _read_json(Config.ARTIFACTS_DIR / "feature_order.json")
    meta = _read_json(Config.ARTIFACTS_DIR / "training_meta.json")
    industries_classes = _read_json(Config.ARTIFACTS_DIR / "industries_classes.json")

    cat_maps = {}
    for orig_col, p in Config.CAT_CLASS_FILES.items():
        if p.exists():
            cat_maps[orig_col] = _read_json(p)

    return mapping, feature_order, meta, industries_classes, cat_maps


@lru_cache(maxsize=1)
def load_model(model_type):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=all, 1=hide INFO, 2=hide INFO+WARNING, 3=hide all but FATAL
    import logging
    import tensorflow as tf
    # Quiet Python-side TF logs
    tf.get_logger().setLevel(logging.ERROR)
    # Quiet absl logs (used by TF/TF-DF)
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    Config.MODEL_DIR = Config.resolve_model_dir(model_type)
    print(Config.MODEL_DIR)
    import sys, contextlib, ctypes

    @contextlib.contextmanager
    def hard_silence_stderr():
        libc = ctypes.CDLL(None)
        libc.fflush(None)              # flush C buffers
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)      # duplicate fd 2 (stderr)
        try:
            os.dup2(devnull, 2)        # redirect fd 2 to /dev/null
            yield
        finally:
            libc.fflush(None)
            os.dup2(old_stderr_fd, 2)  # restore
            os.close(old_stderr_fd)
            os.close(devnull)

    with hard_silence_stderr():
        model = keras.models.load_model(str(Config.MODEL_DIR))
    # Touch signatures once here to fail-fast if wrong
    _ = model.signatures["serving_default"]
    return model


def tf_to_np_dtype(tf_dtype):
    return tf_dtype.as_numpy_dtype