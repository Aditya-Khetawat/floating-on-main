"""Line-based pattern-matching baseline used in RQ1 (Section V-A).

It matches the seven loading interfaces with a regular expression on every
non-comment line, takes the first quoted argument on that line as the model
identifier, and marks a site as pinned if the word "revision" appears later on
the same line. This mirrors grep-style detection common in repository mining.
"""
import re

from pincheck.core import classify_identifier

API_RE = re.compile(r"\b(from_pretrained|pipeline|SentenceTransformer|CrossEncoder|"
                    r"hf_hub_download|snapshot_download|HuggingFaceEmbeddings)\s*\(")
STR_RE = re.compile(r"""["']([^"']+)["']""")


def detect(source: str):
    """Return dicts with line, api, id_class, model_id and revision ("pinned"/"absent")."""
    out = []
    for lineno, line in enumerate(source.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        for m in API_RE.finditer(line):
            api = m.group(1)
            if api == "pipeline" and "transformers" not in source:
                continue
            rest = line[m.end():]
            if api == "pipeline":
                mm = re.search(r"model\s*=\s*[\"']([^\"']+)", rest)
                val = mm.group(1) if mm else None
                if val:
                    id_class = classify_identifier("literal", val)
                else:
                    id_class = "none" if re.match(r"\s*[\"'][^\"']*[\"']\s*\)", rest) else "indirect"
            else:
                s = STR_RE.search(rest)
                val = s.group(1) if s and rest.lstrip().startswith(("'", '"')) else None
                if val:
                    id_class = classify_identifier("literal", val)
                else:
                    id_class = "none" if rest.lstrip().startswith(")") else "indirect"
            out.append({"line": lineno, "api": api, "id_class": id_class,
                        "model_id": val if id_class == "hub" else None,
                        "revision": "pinned" if "revision" in rest else "absent"})
    return out
