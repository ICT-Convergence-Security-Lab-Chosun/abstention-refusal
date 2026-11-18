#!/usr/bin/env python3
"""Generate and validate controlled A/FP/NC/UNK entity pools for v2 splits."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from instruction_common import write_manifest
from v2_common import CONTROL_CONDITIONS, entity_key, read_jsonl, write_jsonl


CATEGORIES = [
    "historical figure",
    "scientist",
    "city or landmark",
    "country",
    "animal species",
    "plant",
    "physical object or invention",
    "historical event",
    "company or product",
    "book or film",
    "abstract concept",
    "natural phenomenon",
    "food or drink",
    "sport",
]

ENTITY_PROMPT = (
    "List {n} well-known, distinct entities in the category '{category}'. "
    "Use real, widely documented entities. Reply only as a JSON array of strings."
)

QUESTION_PROMPT = (
    'You are building a controlled stimulus set. For the entity "{entity}" '
    "(category: {category}), write FOUR short questions, all about this same entity, "
    "matched in length and natural phrasing, one per category:\n"
    "1. A: ANSWERABLE, a factual question with a clear, well-known correct answer.\n"
    "2. FP: FALSE PREMISE, a fluent question that presupposes a specific false claim.\n"
    "3. NC: NO CONTEXT, a question that explicitly depends on a passage/document/text not provided.\n"
    "4. UNK: UNKNOWABLE, a genuinely unrecorded trivial/private detail.\n"
    "Each one sentence. Reply only JSON: {{\"A\":\"...\",\"FP\":\"...\",\"NC\":\"...\",\"UNK\":\"...\"}}"
)

CHECK_PROMPTS = {
    "A": "Is this question answerable from well-known general knowledge, with a clear correct answer?",
    "FP": "Does this question presuppose something that is actually false?",
    "NC": "Does answering this require a specific passage/document/context that is not included in the question itself?",
    "UNK": "Is the answer genuinely unknowable, unrecorded, or undocumented anywhere?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_raw", required=True)
    parser.add_argument("--output_valid", required=True)
    parser.add_argument("--exclude_jsonl", nargs="*", default=[])
    parser.add_argument("--target_valid_entities", type=int, default=100)
    parser.add_argument("--entities_per_category", type=int, default=16)
    parser.add_argument("--api_base", default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--generator_model", default="openai/gpt-4.1")
    parser.add_argument("--validator_model", default="openai/gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--skip_generation", action="store_true", help="Validate output_raw instead of calling the generator.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def chat(api_base: str, api_key: str, model: str, prompt: str, temperature: float, max_tokens: int, retries: int = 4) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except Exception as err:
            last_error = err
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"API call failed: {last_error}")


def parse_json(text: str) -> Any:
    candidate = text.strip()
    if "```" in candidate:
        candidate = re.sub(r"^```(?:json)?", "", candidate.strip(), flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate.strip()).strip()
    start_obj = candidate.find("{")
    start_arr = candidate.find("[")
    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        start, end = start_arr, candidate.rfind("]")
    else:
        start, end = start_obj, candidate.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except Exception:
        return None


def excluded_entities(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = entity_key(row)
            if key:
                out.add(key)
    return out


def generate_entities(args: argparse.Namespace, api_key: str, excluded: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for category in CATEGORIES:
        raw = chat(
            args.api_base,
            api_key,
            args.generator_model,
            ENTITY_PROMPT.format(n=args.entities_per_category, category=category),
            args.temperature,
            400,
        )
        parsed = parse_json(raw)
        if isinstance(parsed, list):
            for entity in parsed:
                key = str(entity).strip().lower()
                if key and key not in excluded:
                    pairs.append((str(entity).strip(), category))
    seen = set()
    deduped = []
    for entity, category in pairs:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            deduped.append((entity, category))
    return deduped


def generate_questions(args: argparse.Namespace, api_key: str, entities: list[tuple[str, str]]) -> list[dict[str, Any]]:
    def one(pair: tuple[str, str]) -> dict[str, Any] | None:
        entity, category = pair
        raw = chat(
            args.api_base,
            api_key,
            args.generator_model,
            QUESTION_PROMPT.format(entity=entity, category=category),
            args.temperature,
            500,
        )
        parsed = parse_json(raw)
        if not isinstance(parsed, dict) or any(cond not in parsed for cond in CONTROL_CONDITIONS):
            return None
        return {"entity": entity, "category": category, **{cond: str(parsed[cond]).strip() for cond in CONTROL_CONDITIONS}}

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        return [row for row in pool.map(one, entities) if row]


def validate_questions(args: argparse.Namespace, api_key: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    def check(task: tuple[int, str, str]) -> tuple[int, str, bool]:
        idx, condition, question = task
        prompt = f"{CHECK_PROMPTS[condition]} Answer only YES or NO.\n\nQuestion: {question}"
        raw = chat(args.api_base, api_key, args.validator_model, prompt, 0.0, 16)
        return idx, condition, raw.strip().lower().startswith("y")

    tasks = [(idx, condition, row[condition]) for idx, row in enumerate(rows) for condition in CONTROL_CONDITIONS]
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = list(pool.map(check, tasks))
    by_idx: dict[int, dict[str, bool]] = {}
    counts: Counter = Counter()
    for idx, condition, passed in results:
        by_idx.setdefault(idx, {})[condition] = passed
        counts[(condition, passed)] += 1
    kept = [row for idx, row in enumerate(rows) if all(by_idx.get(idx, {}).get(condition) for condition in CONTROL_CONDITIONS)]
    return kept, counts


def main() -> None:
    args = parse_args()
    raw_path = Path(args.output_raw)
    valid_path = Path(args.output_valid)
    if (raw_path.exists() or valid_path.exists()) and not args.overwrite and not args.skip_generation:
        raise FileExistsError("Output exists. Pass --overwrite or --skip_generation.")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key env var: {args.api_key_env}")
    excluded = excluded_entities(args.exclude_jsonl)
    if args.skip_generation:
        raw_rows = read_jsonl(raw_path)
    else:
        entities = generate_entities(args, api_key, excluded)
        print(f"[ENTITIES] generated={len(entities)} excluded={len(excluded)}", flush=True)
        raw_rows = generate_questions(args, api_key, entities)
        write_jsonl(raw_path, raw_rows)
    valid_rows, counts = validate_questions(args, api_key, raw_rows)
    if len(valid_rows) < args.target_valid_entities:
        raise RuntimeError(
            f"Only {len(valid_rows)} valid controlled entities; "
            f"target is {args.target_valid_entities}. Increase --entities_per_category and rerun."
        )
    valid_rows = valid_rows[: args.target_valid_entities]
    write_jsonl(valid_path, valid_rows)
    write_manifest(
        valid_path.with_suffix(".manifest.json"),
        {
            "stage": "generate_controlled_pool",
            "output_raw": str(raw_path),
            "output_valid": str(valid_path),
            "exclude_jsonl": args.exclude_jsonl,
            "target_valid_entities": args.target_valid_entities,
            "n_raw_entities": len(raw_rows),
            "n_valid_entities": len(valid_rows),
            "validation_counts": {f"{condition}:{passed}": count for (condition, passed), count in counts.items()},
            "generator_model": args.generator_model,
            "validator_model": args.validator_model,
        },
    )
    print(f"[DONE] valid_entities={len(valid_rows)} output={valid_path}")


if __name__ == "__main__":
    main()

