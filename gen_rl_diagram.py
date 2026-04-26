import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import os

fig, ax = plt.subplots(figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')

C = {
    'bg_section' : '#161b22',
    'data_fill'  : '#0f2033',
    'data_border': '#388bfd',
    'model_fill' : '#1a1030',
    'model_border': '#a371f7',
    'reward_fill': '#0f2010',
    'reward_brd' : '#3fb950',
    'grpo_fill'  : '#2a1010',
    'grpo_border': '#f78166',
    'eval_fill'  : '#0f1e30',
    'eval_border': '#58a6ff',
    'pen_fill'   : '#2a1e08',
    'pen_border' : '#e3b341',
    'text_main'  : '#e6edf3',
    'text_sub'   : '#8b949e',
    'arrow'      : '#6e7681',
}

def box(ax, x, y, w, h, label, sublabel='',
        fill='#0f2033', border='#388bfd', fontsize=10, subfontsize=8, radius=0.22):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=fill, edgecolor=border, linewidth=2.0, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + h*0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C['text_main'], zorder=4)
        ax.text(x, y - h*0.20, sublabel, ha='center', va='center',
                fontsize=subfontsize, color=C['text_sub'], zorder=4,
                linespacing=1.4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C['text_main'], zorder=4)

def section_bg(ax, x, y, w, h, title, border_color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.4",
                          facecolor='#0d1117', edgecolor=border_color,
                          linewidth=1.5, alpha=1.0, zorder=1,
                          linestyle='--')
    ax.add_patch(rect)
    ax.text(x + 0.25, y + h - 0.12, title, ha='left', va='top',
            fontsize=9, fontweight='bold', color=border_color, zorder=5,
            alpha=0.9)

def arrow(ax, x1, y1, x2, y2, label='', color=C['arrow'], lw=1.8, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=6, shrinkB=6),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my, label, fontsize=7.5, color=color, ha='left', va='center')

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(11.0, 15.7, 'ChronoVeritas — GRPO RL Training Loop',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color=C['text_main'],
        path_effects=[pe.withStroke(linewidth=5, foreground='#0d1117')])

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Task Generation
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.3, 13.2, 21.4, 2.1, '📦  Task Generation Pipeline', C['data_border'])

box(ax, 2.1, 14.25, 2.6, 1.1, '🌱  Seed Facts',
    '8 domains · 40+ verified claims',
    fill=C['data_fill'], border=C['data_border'], fontsize=10, subfontsize=8.5)

box(ax, 5.6, 14.25, 3.0, 1.1, '⚙️  Mutator',
    'distortion · fabrication\nomission · context_shift',
    fill=C['data_fill'], border=C['data_border'], fontsize=10, subfontsize=8.5)

box(ax, 9.6, 14.25, 3.0, 1.1, '🕸️  Spreader',
    'multi-tier corpus · noise docs\nprovenance chain injection',
    fill=C['data_fill'], border=C['data_border'], fontsize=10, subfontsize=8.5)

box(ax, 13.6, 14.25, 3.0, 1.1, '📋  TaskSpec',
    'claim · corpus · ground truth\ncorpus_tiers · corpus_ids',
    fill=C['data_fill'], border=C['data_border'], fontsize=10, subfontsize=8.5)

arrow(ax, 3.4, 14.25, 4.1, 14.25, color=C['data_border'])
arrow(ax, 7.1, 14.25, 8.1, 14.25, color=C['data_border'])
arrow(ax, 11.1, 14.25, 12.1, 14.25, color=C['data_border'])

# TaskSpec → GRPO loop (left)
arrow(ax, 13.0, 13.7, 7.5, 12.6, color=C['data_border'], lw=1.4, rad=-0.1)
# TaskSpec → Eval (right)
arrow(ax, 15.2, 13.7, 18.5, 12.7, color=C['eval_border'], lw=1.4, rad=0.1)

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN LEFT: GRPO Training Loop
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.3, 0.8, 14.5, 12.1, '⚡  GRPO Training Loop', C['grpo_border'])

# Prompt
box(ax, 7.5, 12.1, 4.2, 0.95, '📝  Prompt Formation',
    'system instruction · claim · corpus docs (chronologically sorted)',
    fill=C['model_fill'], border=C['model_border'], fontsize=10, subfontsize=8.5)

# LLM subgraph
section_bg(ax, 2.5, 9.7, 9.0, 1.9, '🧠  Qwen2.5-7B-Instruct  ·  4-bit QLoRA (Unsloth)', C['model_border'])
box(ax, 7.0, 10.6, 8.5, 1.3, 'Generate  G = 8  independent completions',
    'temperature = 1.1  ·  max_new_tokens = 200  ·  top_p = 0.95',
    fill=C['model_fill'], border=C['model_border'], fontsize=11, subfontsize=9)

# Completions
box(ax, 7.0, 9.0, 8.5, 0.95, '📤  JSON Completions  ( one per sample )',
    '{ verdict · mutation_type · mutation_doc_id · provenance_chain · confidence }',
    fill=C['model_fill'], border=C['model_border'], fontsize=10, subfontsize=8.5)

arrow(ax, 7.5, 11.625, 7.0, 11.25, color=C['model_border'])
arrow(ax, 7.0, 9.95,  7.0, 9.475, color=C['model_border'])

# ── Reward components ─────────────────────────────────────────────────────────
section_bg(ax, 0.4, 5.35, 13.6, 3.1, '🎯  compute_reward()  ·  train_grpo.py', C['reward_brd'])

rw, rh = 2.8, 0.85
pos_x = [1.7, 4.6, 7.5, 10.4]
pos = [('format  +0.05',       'valid JSON · all required fields'),
       ('verdict  +0.30',      'true / false / misleading / unverifiable'),
       ('mutation_type  +0.18','exact mutation class match'),
       ('mutation_point +0.18','exact doc (½× adjacent in timeline)')]
neg = [('provenance F1  +0.18','multiset F1 vs ground-truth chain'),
       ('source_rel  +0.11',   'avg tier score of cited docs'),
       ('hallucination  −0.12','−0.12 × min(n_fake × 0.25, 1.0)'),
       ('Brier  −0.08',        '(conf − correct)²  ×1.5 if over-confident+wrong')]

for x, (lbl, sub) in zip(pos_x, pos):
    box(ax, x, 7.8, rw, rh, lbl, sub,
        fill=C['reward_fill'], border=C['reward_brd'], fontsize=8.5, subfontsize=7.2)
    arrow(ax, 7.0, 8.525, x, 8.225, color=C['reward_brd'], lw=0.9)

for x, (lbl, sub) in zip(pos_x, neg):
    box(ax, x, 6.7, rw, rh, lbl, sub,
        fill=C['pen_fill'], border=C['pen_border'], fontsize=8.5, subfontsize=7.2)
    arrow(ax, 7.0, 8.525, x, 7.125, color=C['pen_border'], lw=0.9)

# Total reward
box(ax, 7.0, 5.7, 4.0, 0.85, '∑  Total Reward',
    'clamped ∈ [−0.20 ,  +1.00]',
    fill=C['reward_fill'], border=C['reward_brd'], fontsize=11, subfontsize=9)

for x in pos_x:
    arrow(ax, x, 7.375, 7.0, 6.125, color=C['reward_brd'], lw=0.8)
    arrow(ax, x, 6.275, 7.0, 6.125, color=C['pen_border'], lw=0.8)

# ── GRPO mechanics ────────────────────────────────────────────────────────────
box(ax, 7.0, 4.45, 6.0, 0.9, '📊  Group-Relative Advantage Estimation',
    'Âᵢ = ( rᵢ  −  μ_group ) / σ_group     for each of G = 8 completions',
    fill=C['grpo_fill'], border=C['grpo_border'], fontsize=10, subfontsize=8.5)

box(ax, 7.0, 3.3, 6.0, 0.9, '∇  Clipped Policy Gradient   (PPO-style)',
    'clip ratio ε = 0.2  ·  max_grad_norm = 0.1  ·  β_KL = 0.04',
    fill=C['grpo_fill'], border=C['grpo_border'], fontsize=10, subfontsize=8.5)

box(ax, 7.0, 2.15, 6.0, 0.9, '🔄  LoRA Adapter Weight Update',
    'rank r = 8  ·  alpha α = 16  ·  targets: q/k/v/o/gate/up/down  ·  lr = 5e-5',
    fill=C['grpo_fill'], border=C['grpo_border'], fontsize=10, subfontsize=8.5)

arrow(ax, 7.0, 5.275, 7.0, 4.9,  color=C['grpo_border'], lw=2.0)
arrow(ax, 7.0, 4.0,   7.0, 3.75, color=C['grpo_border'], lw=2.0)
arrow(ax, 7.0, 2.85,  7.0, 2.6,  color=C['grpo_border'], lw=2.0)

# Feedback: LoRA → LLM (curved back up left side)
ax.annotate('', xy=(2.5, 10.6), xytext=(4.0, 2.15),
            arrowprops=dict(arrowstyle='->', color=C['model_border'], lw=2.0,
                            connectionstyle='arc3,rad=-0.45',
                            shrinkA=8, shrinkB=8), zorder=2)
ax.text(0.85, 6.5, 'updated\nweights', ha='center', va='center',
        fontsize=8.5, color=C['model_border'], style='italic', rotation=90,
        path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN RIGHT: Evaluation Episode
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(ax, 15.2, 0.8, 6.5, 12.1, '🔬  Evaluation Episode  ·  env/environment.py', C['eval_border'])

ew = 5.2
box(ax, 18.5, 12.15, ew, 0.95, '🔁  env.reset(task_id)',
    'loads claim + corpus + ground truth  ·  resets step + token budgets',
    fill=C['eval_fill'], border=C['eval_border'], fontsize=10, subfontsize=8.5)

box(ax, 18.5, 10.75, ew, 1.55, '👁️  Observation',
    'claim  ·  corpus_metadata  ·  retrieved_docs\nagent_timeline  ·  flagged_contradictions\nstep_budget_remaining  ·  token_budget_remaining',
    fill=C['eval_fill'], border=C['eval_border'], fontsize=10, subfontsize=8.5)

box(ax, 18.5, 9.2, ew, 1.0, '🎮  Agent chooses Action',
    'search  ·  fetch_doc  ·  add_timeline_event\nflag_contradiction  ·  set_mutation_point  ·  submit_verdict',
    fill=C['eval_fill'], border=C['eval_border'], fontsize=10, subfontsize=8.5)

box(ax, 18.5, 7.5, ew, 1.3, '⚡  PBRS Step Reward  ( non-terminal )',
    'Φ(s) = 0.25·exploration + 0.15·authority + 0.20·contradiction_density\n       + 0.20·hypothesis_grounding + 0.20·evidence_coherence\nstep_reward = 0.15 × ( Φ_after − Φ_before )',
    fill=C['eval_fill'], border=C['eval_border'], fontsize=9.5, subfontsize=8)

box(ax, 18.5, 5.8, ew, 1.4, '🧮  UnifiedGrader  ( terminal )',
    'verdict · mutation_type · mutation_point · provenance F1\nsource_reliability · timeline Kendall-τ · efficiency\nearly_detection · reconciliation  +  hallucination & Brier penalties',
    fill=C['eval_fill'], border=C['eval_border'], fontsize=9.5, subfontsize=8)

arrow(ax, 18.5, 11.675, 18.5, 11.525, color=C['eval_border'], lw=2.0)
arrow(ax, 18.5, 10.0,   18.5, 9.7,   color=C['eval_border'], lw=2.0)
arrow(ax, 18.5, 8.7,    18.5, 8.15,  color=C['eval_border'], lw=2.0, label='  non-terminal')

# PBRS loop back to Observation
ax.annotate('', xy=(21.35, 10.75), xytext=(21.35, 8.15),
            arrowprops=dict(arrowstyle='->', color=C['eval_border'], lw=1.6,
                            connectionstyle='arc3,rad=-0.3',
                            shrinkA=6, shrinkB=6), zorder=2)
ax.text(21.65, 9.45, 'loop\nnext step', ha='left', va='center',
        fontsize=7.5, color=C['eval_border'], style='italic')

arrow(ax, 18.5, 6.85, 18.5, 6.5, color=C['eval_border'], lw=2.0, label='  submit_verdict')

# Periodic eval dashed arrow
ax.annotate('', xy=(15.2, 7.5), xytext=(10.0, 4.45),
            arrowprops=dict(arrowstyle='->', color='#6e7681', lw=1.4,
                            connectionstyle='arc3,rad=-0.25',
                            linestyle='dashed',
                            shrinkA=8, shrinkB=8), zorder=2)
ax.text(12.9, 6.3, 'periodic eval', ha='center', va='center',
        fontsize=8.5, color='#6e7681', style='italic',
        path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    (C['data_border'],   'Data Generation'),
    (C['model_border'],  'LLM / Model'),
    (C['reward_brd'],    'Positive Rewards'),
    (C['pen_border'],    'Penalties'),
    (C['grpo_border'],   'GRPO Mechanics'),
    (C['eval_border'],   'Evaluation Episode'),
]
lx = 0.5
for col, lbl in legend_items:
    ax.add_patch(plt.Rectangle((lx, 0.22), 0.38, 0.38,
                               facecolor=col, edgecolor=col, linewidth=1.5, zorder=3))
    ax.text(lx + 0.52, 0.41, lbl, va='center', fontsize=8.5, color=C['text_sub'])
    lx += 3.6

os.makedirs('plots/ema', exist_ok=True)
plt.tight_layout(pad=0.2)
plt.savefig('plots/ema/rl_training_loop.png', dpi=180, bbox_inches='tight',
            facecolor='#0d1117')
print('DONE: plots/ema/rl_training_loop.png')
