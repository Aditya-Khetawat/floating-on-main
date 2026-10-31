"""Core static analysis of model-loading call sites.

PinCheck parses Python source into an abstract syntax tree (AST), finds calls to
seven Hugging Face model-loading interfaces, resolves the model identifier
(including names bound to a single string constant in the same file), and
classifies the ``revision`` argument.

Identifier classes (``Finding.id_class``)
    hub       string literal (or resolved constant) that looks like a hub id
    indirect  expression that cannot be resolved statically
    local     path, file name or path-prefixed expression (not a hub load)
    none      no identifier argument (e.g. ``pipeline("summarization")``)

Revision classes (``Finding.revision``)
    hash       literal 40-character hexadecimal commit id (immutable)
    named_ref  other literal such as "main" or "v1.0" (mutable)
    indirect   revision passed through a variable ("configurable" in the paper)
    absent     no revision argument ("none" in the paper)
"""
from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

LOAD_APIS = frozenset({
    "from_pretrained", "pipeline", "SentenceTransformer", "CrossEncoder",
    "hf_hub_download", "snapshot_download", "HuggingFaceEmbeddings",
})
_ID_KEYWORDS = ("pretrained_model_name_or_path", "model_name_or_path", "model",
                "repo_id", "model_name", "model_id")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*(/[\w.\-]+)?$")
_LOCAL_SUFFIX = re.compile(r"\.(bin|pt|pth|safetensors|onnx|gguf|json)$")
_PATH_PREFIXES = (".", "/", "~")

REVISION_HASH = "hash"
REVISION_NAMED = "named_ref"
REVISION_CONFIG = "indirect"
REVISION_ABSENT = "absent"


@dataclass
class Finding:
    """One model-loading call site."""
    file: str
    line: int
    api: str
    id_class: str
    model_id: Optional[str]
    revision: str
    trust_remote_code: bool
    pipeline_default: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _callee_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def string_constants(tree: ast.AST) -> Dict[str, str]:
    """Names assigned exactly one distinct string constant anywhere in the file."""
    values: Dict[str, set] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, set()).add(node.value.value)
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            values.setdefault(node.target.id, set()).add(node.value.value)
    return {k: next(iter(v)) for k, v in values.items() if len(v) == 1}


def resolve(node: Optional[ast.AST], consts: Dict[str, str]) -> Tuple[str, Optional[str]]:
    """Resolve an argument to ("literal", value), ("indirect", None) or ("none", None)."""
    if node is None:
        return "none", None
    if (isinstance(node, ast.BinOp) and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str) and node.left.value.startswith(_PATH_PREFIXES)):
        return "literal", node.left.value
    if (isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant)
            and str(node.values[0].value).startswith(_PATH_PREFIXES)):
        return "literal", node.values[0].value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "literal", node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return "literal", consts[node.id]
    return "indirect", None


def classify_identifier(kind: str, value: Optional[str]) -> str:
    if kind in ("none", "indirect"):
        return kind
    v = value.strip()
    if v.startswith(_PATH_PREFIXES) or "\\" in v or _LOCAL_SUFFIX.search(v) or not _HUB_ID.match(v):
        return "local"
    return "hub"


def _classify_revision(node: ast.AST, consts: Dict[str, str]) -> str:
    kind, value = resolve(node, consts)
    if kind == "literal" and _HEX40.match(value.strip()):
        return REVISION_HASH
    if kind == "literal":
        return REVISION_NAMED
    return REVISION_CONFIG


def _imports_transformers_pipeline(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("transformers"):
            if any(a.name == "pipeline" and a.asname in (None, "pipeline") for a in node.names):
                return True
    return False


def analyze_source(source: str, path: str = "<string>", constant_propagation: bool = True,
                   pipeline_rule: str = "import") -> Optional[List[Finding]]:
    """Return all model-loading call sites in ``source`` (``None`` if it does not parse).

    ``pipeline_rule="import"`` (default, used for RQ2-RQ5) counts ``pipeline(...)``
    only when it is imported from transformers or called as ``transformers.pipeline``.
    ``pipeline_rule="legacy"`` reproduces the version frozen for the held-out
    evaluation (RQ1): a bare ``pipeline(...)`` in any file that mentions transformers.
    ``constant_propagation=False`` is the ablation reported in Table II.
    """
    try:
        with warnings.catch_warnings():  # e.g. invalid escape sequences in analyzed code
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    consts = string_constants(tree) if constant_propagation else {}
    tf_pipeline = _imports_transformers_pipeline(tree)
    findings: List[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node.func)
        if name not in LOAD_APIS:
            continue
        if name == "pipeline":
            is_attr = isinstance(node.func, ast.Attribute)
            is_tf_attr = (is_attr and isinstance(node.func.value, ast.Name)
                          and node.func.value.id == "transformers")
            if pipeline_rule == "legacy":
                if "transformers" not in source or (is_attr and not is_tf_attr):
                    continue
            elif (is_attr and not is_tf_attr) or (not is_attr and not tf_pipeline):
                continue
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if name == "pipeline":
            id_node = kwargs.get("model") or (node.args[1] if len(node.args) > 1 else None)
        elif name == "HuggingFaceEmbeddings":
            id_node = kwargs.get("model_name") or kwargs.get("model")
        else:
            id_node = node.args[0] if node.args else next(
                (kwargs[k] for k in _ID_KEYWORDS if k in kwargs), None)
        if name == "from_pretrained" and id_node is None and not node.args and not kwargs:
            continue
        kind, value = resolve(id_node, consts)
        id_class = classify_identifier(kind, value)
        revision = REVISION_ABSENT
        if "revision" in kwargs:
            revision = _classify_revision(kwargs["revision"], consts)
        model_kwargs = kwargs.get("model_kwargs")
        if revision == REVISION_ABSENT and isinstance(model_kwargs, ast.Dict):
            for k, v in zip(model_kwargs.keys, model_kwargs.values):
                if isinstance(k, ast.Constant) and k.value == "revision":
                    revision = _classify_revision(v, consts)
        trc = kwargs.get("trust_remote_code")
        trust_remote = isinstance(trc, ast.Constant) and trc.value is True
        findings.append(Finding(
            file=path, line=getattr(node.func, "end_lineno", node.lineno), api=name,
            id_class=id_class, model_id=value if id_class == "hub" else None,
            revision=revision, trust_remote_code=trust_remote,
            pipeline_default=(name == "pipeline" and id_class == "none")))
    return findings


def notebook_to_source(raw: str) -> Optional[str]:
    """Concatenate notebook code cells; IPython magics and shell lines become blank lines."""
    try:
        nb = json.loads(raw)
    except ValueError:
        return None
    lines: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        for line in src.splitlines():
            lines.append("" if line.lstrip().startswith(("%", "!")) else line)
        lines.append("")
    return "\n".join(lines)


def analyze_notebook(raw: str, path: str = "<notebook>", **kwargs) -> Optional[List[Finding]]:
    src = notebook_to_source(raw)
    return None if src is None else analyze_source(src, path, **kwargs)
