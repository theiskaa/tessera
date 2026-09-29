"""Rebuild and check the US detector data without starting model training."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDERS = (
    "build_office_eval.py",
    "freeze_staff_challenge.py",
    "freeze_park_address_challenge.py",
    "freeze_ky_superintendent_challenge.py",
    "build_corrected_dev.py",
    "build_eval_exclusions.py",
    "build_agency_roster.py",
    "build_silver.py",
    "build_contact_snippets.py",
    "build_reviewed_paragraphs.py",
    "build_park_contact_silver.py",
    "build_or_district_silver.py",
    "build_house_staff_silver.py",
    "build_house_office_silver.py",
    "freeze_r23_org_packet.py",
    "freeze_r23_org_packet_remaining.py",
    "build_r23_reviewed_org.py",
    "freeze_federal_org_packet.py",
    "build_federal_org_reviewed.py",
    "build_2025_year_contacts_silver.py",
    "build_2025_year_acronym_silver.py",
    "build_2025_year_contact_expansion_gold.py",
    "freeze_2025_year_contact_expansion_train_candidates.py",
    "build_2025_year_contact_expansion_silver.py",
    "build_eval_disjoint_silver.py",
    "build_2025_targeted_contact_gold.py",
    "freeze_2025_targeted_contact_train_candidates.py",
    "build_2025_targeted_contact_silver.py",
)


def run(*command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    for builder in BUILDERS:
        run(sys.executable, ROOT / "bench/us" / builder)
    run("cargo", "build", "-p", "trainer")
    trainer = ROOT / "target/debug/trainer"
    config = ROOT / "configs/detector-shared.toml"
    run(trainer, "check-silver", "--config", config)
    run(trainer, "generate", "--config", config)
    run(sys.executable, ROOT / "bench/us/check_ready.py")


if __name__ == "__main__":
    main()
