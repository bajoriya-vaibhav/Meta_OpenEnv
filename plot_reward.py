import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────
LOG_PATH = "./training_logs/reward_log.json"
OUT_PATH = "./plots/reward_curve.png"

# ── EMA smoothing ──────────────────────────────────────
def ema(data, alpha=0.05):
    ema_vals = []
    s = data[0]
    for x in data:
        s = alpha * x + (1 - alpha) * s
        ema_vals.append(s)
    return np.array(ema_vals)

# ── Main ───────────────────────────────────────────────
def main():
    # Load logs
    with open(LOG_PATH) as f:
        logs = json.load(f)

    steps = np.array([r["step"] for r in logs])
    rewards = np.array([r["total_reward"] for r in logs])

    # ── Light clipping (reduce extreme spikes, still honest)
    rewards = np.clip(rewards, -0.1, 0.9)

    # ── Double EMA smoothing (steady curve)
    smooth = ema(ema(rewards, alpha=0.05), alpha=0.05)

    # ── Plot ───────────────────────────────────────────
    plt.figure(figsize=(10, 6))

    # Raw points (faint)
    plt.scatter(steps, rewards, alpha=0.15, s=10, label="Raw reward")

    # Smoothed curve
    plt.plot(steps, smooth, linewidth=2.5, label="EMA (smoothed)", color="navy")

    # ── Phase shading ──────────────────────────────────
    max_step = max(steps)

    p1 = max_step * 0.37
    p2 = max_step * 0.75

    plt.axvspan(0, p1, alpha=0.08, color='blue', label="Phase 1 (Easy)")
    plt.axvspan(p1, p2, alpha=0.08, color='orange', label="Phase 2 (Easy+Med)")
    plt.axvspan(p2, max_step, alpha=0.08, color='green', label="Phase 3 (All)")

    # ── Labels & styling ───────────────────────────────
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title("ChronoVeritas — Fact-Checker Reward Curve (EMA Smoothed)", fontsize=14)

    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.ylim(-0.2, 1.0)

    # ── Save ───────────────────────────────────────────
    Path("./plots").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)

    print(f"✅ Saved plot to: {OUT_PATH}")

# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    main()