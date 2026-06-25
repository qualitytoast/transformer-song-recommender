import json
import os
import time
from dataclasses import asdict


class ExperimentTracker:
    """Lightweight, dependency-free experiment tracker.
    Each run gets runs/<run_id>/ with its config, per-epoch metrics, and a summary."""

    def __init__(self, cfg, root="runs"):
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(root, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.start = time.time()

        # Snapshot the exact config that defines this run.
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        print(f"[TRACK] Logging run '{self.run_id}' -> {self.dir}")

    def log_epoch(self, epoch, **metrics):
        # One JSON object per line -> easy to append live and load later.
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

    def finish(self, **summary):
        summary["runtime_sec"] = round(time.time() - self.start, 1)
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[TRACK] Run '{self.run_id}' done -> {summary}")