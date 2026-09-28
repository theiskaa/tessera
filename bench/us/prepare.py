"""Rebuild and check the US detector data without starting model training."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDERS = (
    "build_office_eval.py",
    "freeze_staff_challenge.py",
    "build_eval_exclusions.py",
    "build_agency_roster.py",
    "build_silver.py",
    "build_house_staff_silver.py",
)


def run(*command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    for builder in BUILDERS:
        run(sys.executable, ROOT / "bench/us" / builder)
    run("cargo", "build", "-p", "trainer")
    trainer = ROOT / "target/debug/trainer"
    config = ROOT / "configs/detector-shared.toml"
    run(trainer, "generate", "--config", config)
    run(trainer, "check-silver", "--config", config)
    run(sys.executable, ROOT / "bench/us/check_ready.py")


if __name__ == "__main__":
    main()
