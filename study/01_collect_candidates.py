"""Step 1: collect candidate repositories from the GitHub Search API.

Runs the 12 keyword queries and 10 topic queries described in Section IV and
writes output/candidates.json. GitHub search results change over time, so a
re-run will not return exactly the same list; the frozen list used in the paper
(collected 5 October 2026) is data/candidates.json.

Unauthenticated search is limited to 10 requests per minute. Set GITHUB_TOKEN to
raise the limit.
"""
import json
import os
import time

import requests

from common import out_path

FILTERS = "stars:>=20 pushed:>=2025-01-01 fork:false archived:false"
KEYWORD_QUERIES = [
    "rag chatbot transformers language:python",
    "semantic search sentence-transformers language:python",
    "sentiment analysis huggingface language:python",
    "fastapi transformers language:python",
    "streamlit huggingface language:python",
    "text classification bert language:python",
    "question answering transformers language:python",
    "embeddings retrieval huggingface language:python",
    "summarization transformers app language:python",
    "llm inference huggingface language:python",
    "named entity recognition transformers language:python",
    "gradio huggingface language:python",
]
TOPICS = ["huggingface", "transformers", "sentence-transformers", "bert", "huggingface-transformers",
          "rag", "nlp", "semantic-search", "text-classification", "sentiment-analysis"]
FIELDS = ["full_name", "stargazers_count", "size", "default_branch", "created_at", "pushed_at", "language"]


def search(query: str, page: int, headers: dict):
    params = {"q": f"{query} {FILTERS}", "sort": "stars", "order": "desc", "per_page": 100, "page": page}
    while True:
        r = requests.get("https://api.github.com/search/repositories", params=params, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json().get("items", [])
        print("rate limited or error", r.status_code, "- waiting 65 s")
        time.sleep(65)


def main():
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    queries = KEYWORD_QUERIES + [f"topic:{t} language:python" for t in TOPICS]
    repos = {}
    for q in queries:
        for page in (1, 2):
            items = search(q, page, headers)
            for it in items:
                repos.setdefault(it["full_name"], {k: it[k] for k in FIELDS} | {"query": q})
            print(f"{q!r} page {page}: {len(items)} results, {len(repos)} unique")
            if len(items) < 100:
                break
            time.sleep(7)
        time.sleep(7)
    json.dump(list(repos.values()), open(out_path("candidates.json"), "w"), indent=1)
    print("wrote output/candidates.json with", len(repos), "repositories")


if __name__ == "__main__":
    main()
