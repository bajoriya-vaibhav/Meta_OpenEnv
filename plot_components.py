import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────
LOG_PATH = "./training_logs/reward_log.json"
OUT_PATH = "./plots/component_breakdown_clean.png"

COMPONENTS = [
    "verdict",
    "mutation_type",
    "mutation_point",
    "provenance",
    "source_reliability",
    "brier_penalty",
]

COLORS = {
    "verdict": "#2196F3",
    "mutation_type": "#FF9800",
    "mutation_point": "#4CAF50",
    "provenance": "#9C27B0",
    "source_reliability": "#00BCD4",
    "brier_penalty": "#F44336",
}

# ── EMA smoothing ──────────────────────────────────────
def ema(data, alpha=0.06):
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

    plt.figure(figsize=(10, 6))

    for comp in COMPONENTS:
        values = np.array([r.get(comp, 0) for r in logs])

        # Smooth only (no raw lines → clean plot)
        smooth = ema(values, alpha=0.06)

        plt.plot(
            steps,
            smooth,
            linewidth=1.8,
            label=comp,
            color=COLORS[comp],
        )

    # ── Phase shading ──────────────────────────────────
    max_step = max(steps)

    p1 = max_step * 0.37
    p2 = max_step * 0.75

    plt.axvspan(0, p1, alpha=0.06, color='blue')
    plt.axvspan(p1, p2, alpha=0.06, color='orange')
    plt.axvspan(p2, max_step, alpha=0.06, color='green')

    # ── Labels & styling ───────────────────────────────
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Component Reward", fontsize=12)
    plt.title("ChronoVeritas — Per-Component Reward Breakdown (EMA Smoothed)", fontsize=14)

    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)

    # Keep penalty visible
    plt.ylim(-0.08, 0.32)

    # ── Save ───────────────────────────────────────────
    Path("./plots").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)

    print(f"✅ Saved plot to: {OUT_PATH}")

# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    main()