"""
ANTIGRAVITY Benchmark Runner & Scoreboard Generator.

Usage:
    python -m eval.bench --suite all --iter 00
    python -m eval.bench --suite tier1 --iter 01
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval.tiers import EvaluationSuite, TierResult

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("artifacts")


class BenchmarkRunner:
    """
    Executes benchmark suites, outputs markdown scoreboards, and writes metrics artifacts.
    """

    def __init__(self, iteration: str = "00"):
        self.iteration = iteration
        self.suite = EvaluationSuite()
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def run(self, suite_name: str = "all") -> Dict[str, Any]:
        """Runs the requested evaluation suites."""
        results: List[TierResult] = []

        if suite_name in ("all", "dev", "tier1"):
            results.append(self.suite.run_tier1_analytic())
        if suite_name in ("all", "dev", "tier2_dev"):
            results.append(self.suite.run_tier2_digital_twin(split="dev"))
        if suite_name in ("all", "tier2_holdout"):
            results.append(self.suite.run_tier2_digital_twin(split="holdout"))
        if suite_name in ("all", "dev", "tier3"):
            results.append(self.suite.run_tier3_metamorphic())
        if suite_name in ("all", "dev", "tier4"):
            results.append(self.suite.run_tier4_adversarial())
        if suite_name in ("all", "dev", "tier5"):
            results.append(self.suite.run_tier5_physical_proxies())
        if suite_name in ("all", "dev", "tier6"):
            results.append(self.suite.run_tier6_human_retest())
        if suite_name in ("all", "dev", "tier7"):
            results.append(self.suite.run_tier7_privacy_airgap())
        if suite_name in ("all", "dev", "tier8"):
            results.append(self.suite.run_tier8_golden_file())

        metrics_payload = {
            "iteration": self.iteration,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suites": {},
            "all_passed": all(r.is_passed for r in results),
        }

        # Print Markdown Scoreboard
        print("\n### ANTIGRAVITY Benchmark Scoreboard (Iteration " + self.iteration + ")\n")
        print("| suite | N | MAE_cm | bias_cm | P95_cm | silent_fail | refusal | runtime_s | PASS |")
        print("|---|---|---|---|---|---|---|---|---|")

        for r in results:
            pass_str = "PASS" if r.is_passed else "FAIL"
            if r.status_note and "NOT_RUN" in r.status_note:
                pass_str = "NOT_RUN"

            print(
                f"| {r.tier_name:<30} | {r.num_tests:>3} | {r.mae_cm:>6.3f} | {r.bias_cm:>7.3f} | "
                f"{r.p95_cm:>6.3f} | {r.silent_failure_rate:>11.1%} | {r.refusal_rate:>7.1%} | "
                f"{r.runtime_seconds:>9.3f} | {pass_str:>4} |"
            )

            metrics_payload["suites"][r.tier_name] = {
                "num_tests": r.num_tests,
                "num_passed": r.num_passed,
                "mae_cm": round(r.mae_cm, 4),
                "bias_cm": round(r.bias_cm, 4),
                "p95_cm": round(r.p95_cm, 4),
                "silent_failure_rate": round(r.silent_failure_rate, 4),
                "refusal_rate": round(r.refusal_rate, 4),
                "runtime_seconds": round(r.runtime_seconds, 4),
                "is_passed": r.is_passed,
                "status_note": r.status_note,
                "details": r.details,
            }

        print("")

        # Persist metrics artifact
        artifact_path = ARTIFACTS_DIR / f"metrics_iter_{self.iteration}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"Artifact written: {artifact_path}\n")

        return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="ANTIGRAVITY Benchmark Suite Runner")
    parser.add_argument("--suite", default="all", choices=["all", "dev", "tier1", "tier2_dev", "tier2_holdout", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8"])
    parser.add_argument("--iter", default="00", help="Iteration index (e.g. 00, 01)")
    args = parser.parse_args()

    runner = BenchmarkRunner(iteration=args.iter)
    runner.run(suite_name=args.suite)


if __name__ == "__main__":
    main()
