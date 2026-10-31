"""Load models only at the commit recorded in models.lock (example for Fig. 5)."""
import json
from pathlib import Path

from transformers import AutoModel

LOCK = json.loads((Path(__file__).parent / "models.lock").read_text())


def load(model_id, cls=AutoModel, **kwargs):
    if model_id not in LOCK:
        raise KeyError(f"{model_id} is not in models.lock; run lock_models.py {model_id}")
    return cls.from_pretrained(model_id, revision=LOCK[model_id], **kwargs)
