#!/usr/bin/env python3
"""Eval harness.

    python evals/run.py --fast     # retrieval + routing. No generation, ~2s total.
    python evals/run.py --full     # also grades answers. Calls the model, slow.
    python evals/run.py --fast -v  # per-case detail

Split by cost on purpose. Retrieval and routing are where the failures have actually
been, they need no generation, and they run in seconds -- so they can gate every change.
Answer grading costs a generation per case (~37s each on qwen2.5:7b) and is worth
running only when the answer path changes.

Note what is deliberately NOT graded: prose quality. Judging it needs a judge model,
and with a 7B judge that mostly measures the judge. Structure is graded instead --
did it abstain, do the citations resolve.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_harness_base as harness  # noqa: E402
import rag_service  # noqa: E402

CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")
ABSTAIN_MARKERS = ("don't know", "do not know", "does not contain", "no document matched",
                   "not contain", "no index found", "unavailable", "i don't")


def load_cases():
    with open(CASES_FILE) as f:
        return json.load(f)["cases"]


def load_holdout():
    with open(CASES_FILE) as f:
        return json.load(f).get("holdout", [])


def basename(path):
    return os.path.basename(path)


def retrieve_sources(query, k=4):
    """Top-k source basenames for a query, straight from Qdrant.

    Deliberately not rag_service.retrieve(): that filters on MIN_SCORE, which would
    hide exactly the retrieval failures this is meant to measure.
    """
    emb = rag_service.llm.embeddings.create(
        model=rag_service.EMBED_MODEL, input=[rag_service.SEARCH_PREFIX + query]
    )
    points = rag_service.qdrant().query_points(
        rag_service.COLLECTION, query=emb.data[0].embedding, limit=k, with_payload=True
    ).points
    return [(basename(p.payload.get("file", "")), p.score) for p in points]


def check_fast(cases, verbose=False):
    """Retrieval recall and routing accuracy. No generation."""
    rows = []
    for case in cases:
        expect = case["expect"]
        row = {"id": case["id"]}

        if "sources" in expect:
            t = time.time()
            hits = retrieve_sources(case["query"])
            row["_secs"] = time.time() - t
            got = [h for h, _ in hits]
            row["sources"] = got
            row["retrieval_ok"] = any(s in got for s in expect["sources"])
            row["top_score"] = hits[0][1] if hits else 0.0
        else:
            row["retrieval_ok"] = None  # not measured for this case

        if "must_retrieve" in expect:
            t = time.time()
            decision = harness.needs_documents(case["query"])
            row["_route_secs"] = time.time() - t
            row["routing_ok"] = decision == expect["must_retrieve"]
            row["routed_docs"] = decision
            # Whether the *model* would have called the tool, given the router said yes.
            # Scoring routing alone cannot catch the failure this column exists for: the
            # router was right every time while qwen2.5:7b answered "no tool" anyway, so
            # routing accuracy read 1.00 on a completely broken path.
            row["model_ok"] = _model_would_call(case["query"], decision)
        else:
            row["routing_ok"] = None
            row["model_ok"] = None

        rows.append(row)
        if verbose:
            _print_case(row, case)
    return rows


def _model_would_call(query, wants_docs, tools=None):
    """Would the main model call answer_docs, given the router's decision?

    Returns None when it cannot be determined (no bridge, or the router abstained), so it
    never turns an infrastructure problem into a routing failure.
    """
    if wants_docs is None:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=str(harness.client.base_url).rstrip("/"),
            api_key=harness.client.api_key or "ollama",
            timeout=180,
        )
        tool = {
            "type": "function",
            "function": {
                "name": harness.DOCS_TOOL_NAME,
                "description": (
                    "Answer a question using the indexed documents. Returns a cited answer, "
                    "or says plainly that the documents do not cover it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
        resp = client.chat.completions.create(
            model=harness.MODEL,
            messages=[
                {"role": "system", "content": harness.SYSTEM},
                {"role": "user", "content": query},
            ],
            tools=[tool],
        )
        return bool(resp.choices[0].message.tool_calls)
    except Exception as e:
        print(f"  model_would_call unavailable ({type(e).__name__}: {e})")
        return None


def check_full(cases, bridge, verbose=False):
    """Abstention, citation validity and answer content. Calls the MCP tool."""
    rows = []
    for case in cases:
        expect = case["expect"]
        row = {"id": case["id"]}
        if not expect.get("must_retrieve") or bridge is None:
            rows.append(row)
            continue

        t = time.time()
        try:
            from mcp_client import result_to_text

            answer = result_to_text(bridge.call("answer_docs", {"query": case["query"]}))
        except Exception as e:
            answer = f"ERROR: {e}"
        row["_full_secs"] = time.time() - t
        row["answer"] = answer

        low = answer.lower()
        if "must_abstain" in expect:
            row["abstain_ok"] = expect["must_abstain"] == any(m in low for m in ABSTAIN_MARKERS)
        if expect.get("must_cite"):
            cited = set(re.findall(r"\[(\d+)\]", answer))
            listed = set(re.findall(r"^\s*\[(\d+)\]", answer, re.M))
            row["cite_ok"] = bool(cited) and cited <= listed and bool(listed)
        if "answer_contains" in expect:
            needle = expect["answer_contains"].lower()
            row["contains_ok"] = needle in low
        rows.append(row)
        if verbose:
            _print_case(row, case)
    return rows


def _print_case(row, case):
    print(f"    {row['id']:<18} {row.get('_route_secs', 0):.2f}s "
          f"route={row.get('routed_docs')!s:<5} "
          f"recall={row.get('retrieval_ok')!s:<5} "
          f"top={row.get('top_score', 0):.2f}")


def summarise(rows, keys):
    """Mean of each truthy metric, ignoring cases that did not measure it."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def report(title, rows, keys, labels):
    """Print pass rates plus latency. Latency keys are per-mode so a retrieval
    measurement is never averaged together with a generation one -- that bug
    reported 0.54s for answers that actually take ~37s."""
    s = summarise(rows, keys)
    print(f"\n  {title}")
    for k, label in zip(keys, labels):
        if k in s:
            n = sum(1 for r in rows if r.get(k) is not None)
            print(f"    {label:<34} {s[k]:.2f}  ({n} cases)")
    for key, label in (("_secs", "retrieval latency p50"),
                       ("_full_secs", "answer latency p50"),
                       ("_route_secs", "router latency p50")):
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"    {label:<34} {sorted(vals)[len(vals) // 2]:.2f}s")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="retrieval + routing only")
    ap.add_argument("--full", action="store_true", help="also grade generated answers")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not (args.fast or args.full):
        args.full = True

    cases = load_cases()
    print(f"\n  {len(cases)} cases from {basename(CASES_FILE)}")
    if not rag_service.available():
        print(f"  no index: collection {rag_service.COLLECTION!r} not found at "
              f"{rag_service.QDRANT_URL}. Run the indexer first.")
        return 2

    rows = check_fast(cases, args.verbose)
    fast = report("fast  tuning set", rows,
                  ["retrieval_ok", "routing_ok", "model_ok"],
                  ["retrieval recall@4", "routing accuracy", "model would call"])

    # The tuning set is contaminated by definition: the router prompt was changed
    # against it. The holdout is the number to trust.
    holdout = load_holdout()
    hold = None
    if holdout:
        print(f"\n  {len(holdout)} held-out cases (never tuned against)")
        hrows = check_fast(holdout, args.verbose)
        hold = report("fast  HOLDOUT", hrows,
                      ["retrieval_ok", "routing_ok", "model_ok"],
                      ["retrieval recall@4", "routing accuracy", "model would call"])

    failed = False
    if args.full:
        bridge, _ = harness.connect(harness.MCP_URL, harness.LOCAL_NAMES)
        if bridge is None:
            print("\n  full  skipped: mcp server unreachable")
        else:
            report("full  tuning set", check_full(cases, bridge, args.verbose),
                   ["abstain_ok", "cite_ok", "contains_ok"],
                   ["abstention accuracy", "citation validity", "answer contains"])
            if holdout:
                report("full  HOLDOUT", check_full(holdout, bridge, args.verbose),
                       ["abstain_ok", "cite_ok", "contains_ok"],
                       ["abstention accuracy", "citation validity", "answer contains"])

    # Gate on the holdout, not the tuning set: a tuned set always looks good. The bar
    # is not 1.0 -- routing is a judgement call and the residual errors are mostly
    # false positives, which are harmless (they only add an option). Drop below 0.75
    # means it is losing real searches and the prompt needs work.
    MIN_ROUTING = 0.75
    if hold and hold.get("routing_ok") is not None and hold["routing_ok"] < MIN_ROUTING:
        print(f"\n  FAIL  holdout routing {hold['routing_ok']:.2f} < {MIN_ROUTING:.2f}")
        failed = True
    elif hold:
        print(f"\n  ok    holdout routing {hold['routing_ok']:.2f} >= {MIN_ROUTING:.2f}")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
