"""Check reviewed US V4 silver against corrected, source-held evaluation."""

import collections
import json
import re
from pathlib import Path

from active_sources import V4_SILVER
from build_contact_snippets import lines
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4, main as verify_evaluation
from build_us_v4_org_paragraphs import OUT as ORG_PARAGRAPHS_PATH
from build_us_v4_org_paragraphs import main as verify_org_paragraphs
from build_us_v4_reviewed_additions import OUT as ADDITIONS_PATH
from build_us_v4_reviewed_additions import main as verify_additions
from build_us_v4_long_additions import OUT as LONG_ADDITIONS_PATH
from build_us_v4_long_additions import main as verify_long_additions
from build_us_v4_historical_additions import OUT as HISTORICAL_PATH
from build_us_v4_historical_additions import main as verify_historical
from build_us_v4_historical_acronyms import OUT as ACRONYMS_PATH
from build_us_v4_historical_acronyms import main as verify_acronyms
from check_holdout_overlap import collisions, contact_collisions
from silver_dedupe import NearTextIndex
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/detector-shared-v4.toml"
KNOWN_NEAR_TEXT = {
    ("2024-29610-contact-1", "2024-31193-contact-1"),
    ("raw-us-org-2024-30611-2", "2024-29917-contact-2"),
}


def config_array(name, source):
    """Read the pinned single-line TOML arrays without an extra Python package."""
    match = re.search(r"^" + re.escape(name) + r" = (\[.*\])$", source, re.M)
    if not match:
        raise ValueError(f"missing {name} in US V4 config")
    return json.loads(match[1])


def main():
    """Fail if source exposure, review data, or the intended draw mix changes."""
    for path in (EVALUATION_PATH, ORG_PARAGRAPHS_PATH, ADDITIONS_PATH,
                 LONG_ADDITIONS_PATH, HISTORICAL_PATH, ACRONYMS_PATH):
        if not path.exists() or not path.with_suffix(".manifest.json").exists():
            raise ValueError(f"US V4 frozen input is missing: {path}")
    verify_evaluation()
    verify_org_paragraphs()
    verify_additions()
    verify_long_additions()
    verify_historical()
    verify_acronyms()
    config = CONFIG.read_text()
    paths = [f"data/interim/silver/{name}.jsonl" for name, _ in V4_SILVER]
    if config_array("silver", config) != paths:
        raise ValueError("US V4 silver configuration differs from reviewed source list")
    weights = config_array("silver_repeats", config)
    if (weights != [16, 1, 4, 16, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16,
                    16, 16, 16, 32, 16, 16, 16, 16] or
            'synthetic_per_epoch = 60000' not in config or
            'source_backed_hierarchy = true' not in config or
            'exclude_gold = "data/interim/review/us-eval-exclusions-v4.jsonl"' not in config):
        raise ValueError("US V4 data mixture changed without review")
    rows = [row for path in paths for row in lines(ROOT / path)]
    evaluation, changed = exclusion_rows_v4()
    if len(rows) != 1670 or len(evaluation) != 343 or len(changed) != 8:
        raise ValueError("US V4 training or evaluation membership changed")
    train_keys = [document_key(row) for row in rows]
    eval_keys = {document_key(row) for row in evaluation}
    if (None in train_keys or len(set(train_keys)) != 641 or
            set(train_keys) & eval_keys):
        raise ValueError("US V4 training shares an evaluation source")
    near = NearTextIndex()
    near_pairs = set()
    for row in rows:
        prior = near.prior(row["text"])
        if prior:
            near_pairs.add((row["id"], prior))
        near.add(row["id"], row["text"])
    if near_pairs != KNOWN_NEAR_TEXT:
        raise ValueError(f"US V4 near-duplicate texts changed: {near_pairs}")
    samples = [(row["id"], row["text"], row["entities"]) for row in rows]
    if collisions(evaluation, samples):
        raise ValueError("US V4 labeled entities overlap evaluation")
    contacts = contact_collisions(evaluation, samples)
    unexpected = {
        name: {kind: matches for kind, matches in kinds.items()
               if kind != "phone" or any(
                   re.sub(r"\D", "", surface) != "711" for _, surface in matches)}
        for name, kinds in contacts.items()
    }
    unexpected = {name: kinds for name, kinds in unexpected.items() if kinds}
    if unexpected:
        raise ValueError(f"US V4 contact labels overlap evaluation: {unexpected}")
    per_source = collections.Counter(document_key(row) for row in rows)
    print(f"US V4 source split passed: {len(rows)} real documents from "
          f"{len(per_source)} source documents; 0 held-out source or substantive label overlaps")


if __name__ == "__main__":
    main()
