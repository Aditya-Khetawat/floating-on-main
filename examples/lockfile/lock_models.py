"""Create or refresh models.lock: map each Hugging Face repository id to its current commit hash.

Usage:  python lock_models.py org/model-a org/model-b ...   (requires: pip install huggingface_hub)

Example code for the lockfile workflow of Fig. 5; it is not part of the study pipeline.
Re-run it only when you intend to update a model, then review the diff and run your
regression tests before committing the new lockfile.
"""
import json
import os
import sys

from huggingface_hub import HfApi

LOCK = "models.lock"


def main(repo_ids):
    lock = json.load(open(LOCK)) if os.path.exists(LOCK) else {}
    api = HfApi()
    for repo_id in repo_ids:
        lock[repo_id] = api.model_info(repo_id).sha
        print(f"{repo_id} -> {lock[repo_id]}")
    with open(LOCK, "w") as fh:
        json.dump(dict(sorted(lock.items())), fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
