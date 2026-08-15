#!/usr/bin/env python
"""
scripts/assistant/manage_evidence.py
────────────────────────────────────
Tooling for the patient evidence corpus.

The corpus is the one part of the patient assistant that cannot be written by a
model: every entry is quoted to a patient under a named source, so an entry
nobody opened the page for is a fabricated citation. This script exists to make
adding a *real* entry cheap and to make an invalid one impossible to commit.

Commands
────────
    validate   check a corpus file: schema, trusted domains, duplicates
    stats      coverage — how many entries, which tiers, which topics, reviewed
    template   print a blank entry to fill in
    add        append an entry from command-line arguments, validating first
    review     mark an existing entry clinician_reviewed
    check-topics
               report which topics have no coverage, given a list

Typical use
───────────
    # 1. see the shape
    python scripts/assistant/manage_evidence.py template

    # 2. open the real page, transcribe it, add it
    python scripts/assistant/manage_evidence.py add \\
        --doc-id NHS-ANAEMIA-IRON \\
        --title "Iron deficiency anaemia" \\
        --source-name NHS --source-tier 2 \\
        --url https://www.nhs.uk/conditions/iron-deficiency-anaemia/ \\
        --topics anaemia haemoglobin tiredness \\
        --text "Iron deficiency anaemia is caused by a lack of iron..."

    # 3. after a clinician has checked it
    python scripts/assistant/manage_evidence.py review --doc-id NHS-ANAEMIA-IRON

    # 4. confirm the whole file still loads
    python scripts/assistant/manage_evidence.py validate

``add`` refuses an entry whose URL is outside the allowlist, and refuses to
guess a retrieval date: ``--retrieved-on`` defaults to today only because you
are running it at the moment you read the page.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.abspath("."))

import yaml

from src.assistant.evidence import (
    CorpusError, PATIENT_CORPUS_PATH, TIER_NAMES, corpus_stats, load_corpus,
)

TEMPLATE = """\
  - doc_id: SOURCE-TOPIC-DETAIL
    topics: [topic_one, topic_two]
    title: The page title, as printed
    source_name: NHS
    source_tier: 2          # 1 guideline · 2 government · 3 institution
                            # 4 literature · 5 database · 6 general reference
    url: https://www.nhs.uk/conditions/example/
    retrieved_on: "YYYY-MM-DD"
    review_status: unreviewed
    verbatim: false
    keywords: [optional, ranking, hints]
    text: >-
      The patient-facing content, in plain language. A faithful summary of what
      that page says. Set verbatim: true only if this is a literal quotation.
"""


def _load_raw(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _header(path: Path) -> str:
    """The leading comment block, which explains why the corpus is what it is.

    Worth preserving across rewrites: it is where the rule against writing
    entries from memory is recorded, and a `yaml.safe_dump` round-trip would
    otherwise silently delete it.
    """
    if not path.exists():
        return ""
    lines: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "\n".join(lines).rstrip()


def _write(path: Path, raw: dict, header: str = "") -> None:
    """Write the corpus, re-attaching the header.

    ``header`` is passed in rather than read from ``path``: the validation step
    writes a candidate file that does not exist yet, and reading its header
    raised FileNotFoundError before every add.
    """
    body = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=88)
    path.write_text((header + "\n\n" if header else "") + body,
                    encoding="utf-8")


def cmd_validate(args) -> int:
    path = Path(args.path)
    try:
        corpus = load_corpus(path, refresh=True)
    except CorpusError as exc:
        print(f"INVALID  {path}\n  {exc}")
        return 1
    print(f"OK  {path}")
    print(f"  version           {corpus['version']}")
    print(f"  documents         {len(corpus['documents'])}")
    print(f"  trusted domains   {len(corpus['trusted_domains'])}")
    return 0


def cmd_stats(args) -> int:
    path = Path(args.path)
    try:
        s = corpus_stats(path)
    except CorpusError as exc:
        print(f"INVALID  {exc}")
        return 1
    print(f"Corpus: {path}")
    print(f"  version              {s['version']}")
    print(f"  documents            {s['n_documents']}")
    print(f"  clinician reviewed   {s['n_clinician_reviewed']}"
          f" / {s['n_documents']}")
    print(f"  trusted domains      {s['n_trusted_domains']}")
    if s["by_tier"]:
        print("  by tier:")
        for tier, n in s["by_tier"].items():
            print(f"    {tier}  {TIER_NAMES.get(tier, '?'):42} {n}")
    print(f"  topics ({len(s['topics'])}): "
          + (", ".join(s["topics"]) if s["topics"] else "none"))
    if s["n_documents"] == 0:
        print("\n  The corpus is empty, so the assistant declines every "
              "substantive\n  question. That is correct behaviour, not a bug — "
              "see the header of\n  the corpus file for why it ships this way.")
    return 0


def cmd_template(args) -> int:
    print(TEMPLATE)
    return 0


def cmd_add(args) -> int:
    path = Path(args.path)
    raw = _load_raw(path)
    header = _header(path)
    docs = raw.setdefault("documents", []) or []
    raw["documents"] = docs

    if any(d.get("doc_id") == args.doc_id for d in docs):
        print(f"ERROR  doc_id {args.doc_id!r} already exists; use `review` or "
              f"edit the file directly")
        return 1

    entry = {
        "doc_id": args.doc_id,
        "topics": [t.lower() for t in args.topics],
        "title": args.title,
        "source_name": args.source_name,
        "source_tier": int(args.source_tier),
        "url": args.url,
        "retrieved_on": args.retrieved_on or date.today().isoformat(),
        "review_status": "unreviewed",
        "verbatim": bool(args.verbatim),
        "text": args.text,
    }
    if args.keywords:
        entry["keywords"] = [k.lower() for k in args.keywords]

    docs.append(entry)

    # Validate before writing, by loading a candidate copy.
    tmp = path.with_suffix(".candidate.yaml")
    try:
        _write(tmp, raw, header)
        load_corpus(tmp, refresh=True)
    except CorpusError as exc:
        print(f"REJECTED  {exc}")
        return 1
    finally:
        if tmp.exists():
            tmp.unlink()

    _write(path, raw, header)
    print(f"Added {args.doc_id} to {path}")
    print("  review_status is `unreviewed`. Run `review --doc-id "
          f"{args.doc_id}` once a clinician has checked it.")
    return 0


def cmd_review(args) -> int:
    path = Path(args.path)
    raw = _load_raw(path)
    for d in raw.get("documents") or []:
        if d.get("doc_id") == args.doc_id:
            d["review_status"] = "clinician_reviewed"
            if args.reviewer:
                d["reviewed_by"] = args.reviewer
            d["reviewed_on"] = date.today().isoformat()
            _write(path, raw, _header(path))
            print(f"Marked {args.doc_id} clinician_reviewed")
            return 0
    print(f"ERROR  no document with doc_id {args.doc_id!r}")
    return 1


def cmd_check_topics(args) -> int:
    from src.assistant.evidence import retrieve

    path = Path(args.path)
    missing = []
    for topic in args.topics:
        if not retrieve(topic, path=path).ok:
            missing.append(topic)
    covered = len(args.topics) - len(missing)
    print(f"Coverage: {covered}/{len(args.topics)}")
    if missing:
        print("  no source on file for:")
        for t in missing:
            print(f"    - {t}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Manage the patient evidence corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=str(PATIENT_CORPUS_PATH),
                   help="corpus file (default: the shipped patient corpus)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("template").set_defaults(func=cmd_template)

    a = sub.add_parser("add", help="append a new entry")
    a.add_argument("--doc-id", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--source-name", required=True)
    a.add_argument("--source-tier", required=True, type=int, choices=sorted(TIER_NAMES))
    a.add_argument("--url", required=True)
    a.add_argument("--text", required=True)
    a.add_argument("--topics", nargs="+", required=True)
    a.add_argument("--keywords", nargs="*", default=[])
    a.add_argument("--retrieved-on", default=None,
                   help="YYYY-MM-DD; defaults to today")
    a.add_argument("--verbatim", action="store_true",
                   help="set when `text` is a literal quotation")
    a.set_defaults(func=cmd_add)

    r = sub.add_parser("review", help="mark an entry clinician_reviewed")
    r.add_argument("--doc-id", required=True)
    r.add_argument("--reviewer", default=None)
    r.set_defaults(func=cmd_review)

    c = sub.add_parser("check-topics", help="report topics with no coverage")
    c.add_argument("topics", nargs="+")
    c.set_defaults(func=cmd_check_topics)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
