import json
import os
import subprocess
import sys

import pytest

from pincheck.core import analyze_notebook, analyze_source
from pincheck.cli import main as cli_main, violations
from pincheck.manifests import dependency_spec, scan_repository_metadata

HASH = "0123456789abcdef0123456789abcdef01234567"


def one(src, **kw):
    found = analyze_source(src, **kw)
    assert found is not None and len(found) == 1, found
    return found[0]


def test_hub_literal_unpinned():
    f = one('from transformers import AutoModel\nm = AutoModel.from_pretrained("bert-base-uncased")\n')
    assert (f.api, f.id_class, f.model_id, f.revision) == ("from_pretrained", "hub", "bert-base-uncased", "absent")
    assert violations(f) == ["unpinned"]


def test_hash_named_and_configurable_revisions():
    assert one(f'AutoModel.from_pretrained("org/m", revision="{HASH}")').revision == "hash"
    assert one('AutoModel.from_pretrained("org/m", revision="main")').revision == "named_ref"
    assert one('AutoModel.from_pretrained("org/m", revision=cfg.rev)').revision == "indirect"


def test_multiline_call_revision_found():
    src = 'm = AutoModel.from_pretrained(\n    "org/m",\n    revision="v1.0",\n)\n'
    f = one(src)
    assert f.revision == "named_ref" and f.line == 1


def test_constant_propagation_and_ablation():
    src = 'MODEL_ID = "BAAI/bge-small-en-v1.5"\nm = SentenceTransformer(MODEL_ID)\n'
    assert one(src).id_class == "hub"
    assert one(src, constant_propagation=False).id_class == "indirect"


def test_ambiguous_constant_stays_indirect():
    src = 'M = "a/b"\nif x:\n    M = "c/d"\nAutoModel.from_pretrained(M)\n'
    assert one(src).id_class == "indirect"


def test_local_paths_are_classified_local():
    for arg in ['"./ckpt"', '"/models/x"', '"model.safetensors"', '"./out/" + name', 'f"./runs/{i}"']:
        f = one(f"AutoModel.from_pretrained({arg})")
        assert f.id_class == "local" and violations(f) == []


def test_pipeline_rules():
    assert one('from transformers import pipeline\np = pipeline("summarization")').pipeline_default
    assert one('import transformers\np = transformers.pipeline("ner", model="org/m")').id_class == "hub"
    # a project's own pipeline function or a pipeline object is not a model load
    assert analyze_source('from transformers import AutoModel\ndef pipeline(x): ...\npipeline("rag")') == []
    assert analyze_source('from transformers import pipeline\nself.pipeline(text)') == []
    # the legacy rule frozen for the RQ1 evaluation still counts the project function
    assert len(analyze_source('from transformers import AutoModel\npipeline("rag")', pipeline_rule="legacy")) == 1


def test_remote_code_rule():
    f = one('AutoModel.from_pretrained(name, trust_remote_code=True)')
    assert f.trust_remote_code and set(violations(f)) == {"unpinned", "remote-code"}
    f = one(f'AutoModel.from_pretrained("org/m", trust_remote_code=True, revision="{HASH}")')
    assert violations(f) == []


def test_langchain_model_kwargs_revision():
    f = one('HuggingFaceEmbeddings(model_name="org/m", model_kwargs={"revision": "main"})')
    assert f.api == "HuggingFaceEmbeddings" and f.revision == "named_ref"


def test_download_apis_keyword_identifier():
    f = one(f'hf_hub_download(repo_id="org/m", filename="x.bin", revision="{HASH}")')
    assert (f.api, f.id_class, f.revision) == ("hf_hub_download", "hub", "hash")


def test_docstrings_and_definitions_are_not_sites():
    src = ('def from_pretrained(cls, name):\n    """Example:\n    >>> AutoModel.from_pretrained("gpt2")\n    """\n'
           '# AutoModel.from_pretrained("gpt2")\n')
    assert analyze_source(src) == []


def test_unparseable_source_returns_none():
    assert analyze_source("def broken(:\n") is None


def test_notebook_magics_ignored():
    nb = {"cells": [{"cell_type": "code", "source": ["!pip install transformers\n",
                                                     "m = AutoModel.from_pretrained('gpt2')\n"]},
                    {"cell_type": "markdown", "source": ["from_pretrained('x')"]}]}
    found = analyze_notebook(json.dumps(nb))
    assert len(found) == 1 and found[0].model_id == "gpt2"


def test_dependency_spec():
    text = "torch>=2.0\ntransformers==4.44.2\nsentence_transformers\n"
    assert dependency_spec(text, "transformers") == "exact"
    assert dependency_spec(text, "torch") == "range"
    assert dependency_spec(text, "sentence-transformers") == "unpinned"
    assert dependency_spec(text, "huggingface-hub") is None


def test_repository_metadata():
    sig = scan_repository_metadata([(".github/workflows/test_ci.yml", None), ("Dockerfile", None),
                                    ("poetry.lock", None), ("requirements.txt", "transformers==4.4\n")])
    assert sig["ci"] and sig["docker"] and sig["lockfile"] and not sig["tests"]
    assert sig["deps"]["transformers"] == "exact"


def test_cli_exit_codes_and_ignore(tmp_path, capsys):
    bad = tmp_path / "app.py"
    bad.write_text('from transformers import AutoModel\nm = AutoModel.from_pretrained("gpt2")\n')
    assert cli_main([str(tmp_path)]) == 1
    assert cli_main([str(tmp_path), "--fail-on", "none"]) == 0
    assert cli_main([str(tmp_path), "--fail-on", "remote-code"]) == 0
    bad.write_text('m = AutoModel.from_pretrained("gpt2")  # pincheck: ignore\n')
    assert cli_main([str(tmp_path)]) == 0
    good = tmp_path / "app.py"
    good.write_text(f'm = AutoModel.from_pretrained("gpt2", revision="{HASH}")\n')
    capsys.readouterr()
    assert cli_main([str(tmp_path), "--format", "json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["call_sites"] == 1
    with pytest.raises(SystemExit):
        cli_main([str(tmp_path), "--fail-on", "nonsense"])


def test_cli_as_module(tmp_path):
    (tmp_path / "x.py").write_text('AutoModel.from_pretrained("gpt2")\n')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-m", "pincheck.cli", str(tmp_path)], capture_output=True, text=True,
                       cwd=root)
    assert r.returncode == 1 and "unpinned" in r.stdout
