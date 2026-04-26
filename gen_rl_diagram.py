import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(22, 16))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
ax.set_xlim(0, 22); ax.set_ylim(0, 16); ax.axis('off')

C = dict(
    data_fill='#eff6ff',  data_brd='#1d6fe8',
    model_fill='#f5f0ff', model_brd='#7c3aed',
    rew_fill='#f0faf0',   rew_brd='#16a34a',
    pen_fill='#fffbeb',   pen_brd='#d97706',
    grpo_fill='#fff5f5',  grpo_brd='#dc2626',
    eval_fill='#eff8ff',  eval_brd='#0284c7',
    txt='#111827',        sub='#4b5563',
    sec_bg='#f8fafc',
)

def bg(x, y, w, h, title, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.35",
        fc=C['sec_bg'], ec=col, lw=1.6, ls='--', zorder=1))
    ax.text(x+0.2, y+h-0.1, title, ha='left', va='top',
            fontsize=8.5, fontweight='bold', color=col, zorder=5)

def bx(x, y, w, h, t1, t2='', fc='#eff6ff', ec='#1d6fe8', fs=10, ss=8):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.22",
        fc=fc, ec=ec, lw=2.0, zorder=3))
    if t2:
        ax.text(x, y+h*0.14, t1, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=C['txt'], zorder=4)
        ax.text(x, y-h*0.22, t2, ha='center', va='center',
                fontsize=ss, color=C['sub'], zorder=4)
    else:
        ax.text(x, y, t1, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=C['txt'], zorder=4)

def arr(x1, y1, x2, y2, col='#6b7280', lw=1.8, rad=0.0, lbl=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=col, lw=lw,
            connectionstyle=f'arc3,rad={rad}', shrinkA=6, shrinkB=6), zorder=2)
    if lbl:
        ax.text((x1+x2)/2+0.1, (y1+y2)/2, lbl,
                fontsize=7, color=col, ha='left', va='center')

# Title
ax.text(11, 15.75, 'ChronoVeritas  -  GRPO RL Training Loop',
        ha='center', va='center', fontsize=18, fontweight='bold', color=C['txt'])

# Row 1: Data generation
bg(0.3, 13.2, 21.4, 2.1, 'Task Generation Pipeline', C['data_brd'])
for xc, t1, t2 in [
    (2.1,  'Seed Facts',  '8 domains, 40+ verified claims'),
    (5.6,  'Mutator',     'distortion / fabrication\nomission / context_shift'),
    (9.6,  'Spreader',    'multi-tier corpus + noise docs\nprovenance chain injection'),
    (13.5, 'TaskSpec',    'claim  .  corpus  .  ground truth\ncorpus_tiers  .  corpus_ids'),
]:
    bx(xc, 14.25, 2.8, 1.1, t1, t2,
       fc=C['data_fill'], ec=C['data_brd'], fs=10, ss=8.5)

arr(3.5,  14.25, 4.2,  14.25, C['data_brd'])
arr(7.0,  14.25, 8.1,  14.25, C['data_brd'])
arr(11.0, 14.25, 12.1, 14.25, C['data_brd'])
arr(12.9, 13.7,  7.5,  12.55, C['data_brd'], lw=1.4, rad=-0.1)
arr(15.0, 13.7,  18.5, 12.7,  C['eval_brd'], lw=1.4, rad=0.1)

# Left column: GRPO loop
bg(0.3, 0.8, 14.5, 12.1, 'GRPO Training Loop', C['grpo_brd'])

bx(7.5, 12.1, 4.2, 0.9, 'Prompt Formation',
   'system instruction  .  claim  .  corpus docs (sorted chronologically)',
   fc=C['model_fill'], ec=C['model_brd'], fs=10, ss=8.5)

bg(2.5, 9.7, 9.0, 1.9,
   'Qwen2.5-7B-Instruct   4-bit QLoRA   Unsloth + TRL', C['model_brd'])
bx(7.0, 10.6, 8.5, 1.3, 'Generate  G = 8  independent completions',
   'temperature=1.1  .  max_new_tokens=200  .  top_p=0.95',
   fc=C['model_fill'], ec=C['model_brd'], fs=11, ss=9)

bx(7.0, 9.05, 8.5, 0.9, 'JSON Completions  ( one per sample )',
   '{ verdict  .  mutation_type  .  mutation_doc_id  .  provenance_chain  .  confidence }',
   fc=C['model_fill'], ec=C['model_brd'], fs=10, ss=8.5)

arr(7.5, 11.655, 7.0, 11.25, C['model_brd'])
arr(7.0, 9.95,   7.0, 9.5,   C['model_brd'])

# Reward section
bg(0.4, 5.35, 13.6, 3.1, 'compute_reward()   train_grpo.py', C['rew_brd'])

pos = [('format  +0.05',        'valid JSON, all required fields'),
       ('verdict  +0.30',       'true/false/misleading/unverifiable'),
       ('mutation_type  +0.18', 'exact mutation class'),
       ('mutation_point  +0.18','exact doc (0.5x adjacent)')]
neg = [('provenance F1  +0.18', 'multiset F1 vs GT chain'),
       ('source_rel  +0.11',    'avg tier score of cited docs'),
       ('hallucination  -0.12', '-0.12 x min(n_fake x 0.25, 1.0)'),
       ('Brier  -0.08',         '(conf-correct)^2  x1.5 if over-conf+wrong')]

xs = [1.7, 4.6, 7.5, 10.4]
for x, (t1, t2) in zip(xs, pos):
    bx(x, 7.75, 2.75, 0.85, t1, t2, fc=C['rew_fill'], ec=C['rew_brd'], fs=8.5, ss=7.2)
    arr(7.0, 8.6, x, 8.18, C['rew_brd'], lw=0.9)
for x, (t1, t2) in zip(xs, neg):
    bx(x, 6.65, 2.75, 0.85, t1, t2, fc=C['pen_fill'], ec=C['pen_brd'], fs=8.5, ss=7.2)
    arr(7.0, 8.6, x, 7.08, C['pen_brd'], lw=0.9)

bx(7.0, 5.7, 4.0, 0.85, 'Total Reward', 'clamped in [-0.20 ,  +1.00]',
   fc=C['rew_fill'], ec=C['rew_brd'], fs=11, ss=9)
for x in xs:
    arr(x, 7.32, 7.0, 6.13, C['rew_brd'], lw=0.8)
    arr(x, 6.22, 7.0, 6.13, C['pen_brd'], lw=0.8)

bx(7.0, 4.45, 6.0, 0.9, 'Group-Relative Advantage Estimation',
   'A_i = ( r_i  -  mean_group ) / std_group   for each of G=8 completions',
   fc=C['grpo_fill'], ec=C['grpo_brd'], fs=10, ss=8.5)
bx(7.0, 3.3,  6.0, 0.9, 'Clipped Policy Gradient  (PPO-style)',
   'clip ratio e=0.2  .  max_grad_norm=0.1  .  beta_KL=0.04',
   fc=C['grpo_fill'], ec=C['grpo_brd'], fs=10, ss=8.5)
bx(7.0, 2.15, 6.0, 0.9, 'LoRA Adapter Weight Update',
   'rank r=8  .  alpha=16  .  targets: q/k/v/o/gate/up/down proj  .  lr=5e-5',
   fc=C['grpo_fill'], ec=C['grpo_brd'], fs=10, ss=8.5)

arr(7.0, 5.275, 7.0, 4.9,  C['grpo_brd'], lw=2.0)
arr(7.0, 4.0,   7.0, 3.75, C['grpo_brd'], lw=2.0)
arr(7.0, 2.85,  7.0, 2.6,  C['grpo_brd'], lw=2.0)

ax.annotate('', xy=(2.5, 10.6), xytext=(4.0, 2.15),
    arrowprops=dict(arrowstyle='->', color=C['model_brd'], lw=2.0,
        connectionstyle='arc3,rad=-0.45', shrinkA=8, shrinkB=8), zorder=2)
ax.text(0.85, 6.5, 'updated\nweights', ha='center', va='center',
        fontsize=8.5, color=C['model_brd'], style='italic', rotation=90)

# Right column: Eval episode
bg(15.2, 0.8, 6.5, 12.1, 'Evaluation Episode   env/environment.py', C['eval_brd'])
ew = 5.2

bx(18.5, 12.15, ew, 0.9,  'env.reset( task_id )',
   'loads claim + corpus + GT  .  resets step & token budgets',
   fc=C['eval_fill'], ec=C['eval_brd'], fs=10, ss=8.5)
bx(18.5, 10.8,  ew, 1.5,  'Observation',
   'claim  .  corpus_metadata  .  retrieved_docs\nagent_timeline  .  flagged_contradictions\nstep_budget_remaining  .  token_budget_remaining',
   fc=C['eval_fill'], ec=C['eval_brd'], fs=10, ss=8.5)
bx(18.5, 9.2,   ew, 1.0,  'Agent chooses Action',
   'search  .  fetch_doc  .  add_timeline_event\nflag_contradiction  .  set_mutation_point  .  submit_verdict',
   fc=C['eval_fill'], ec=C['eval_brd'], fs=10, ss=8.5)
bx(18.5, 7.5,   ew, 1.3,  'PBRS Step Reward  ( non-terminal )',
   'phi(s) = 0.25*exploration + 0.15*authority + 0.20*contradiction_density\n        + 0.20*hypothesis_grounding + 0.20*evidence_coherence\nstep_reward = 0.15 x ( phi_after - phi_before )',
   fc=C['eval_fill'], ec=C['eval_brd'], fs=9.5, ss=8)
bx(18.5, 5.8,   ew, 1.4,  'UnifiedGrader  ( terminal )',
   'verdict  .  mutation_type  .  mutation_point  .  provenance F1\nsource_reliability  .  timeline Kendall-tau  .  efficiency\nearly_detection  .  reconciliation  .  hallucination  .  Brier',
   fc=C['eval_fill'], ec=C['eval_brd'], fs=9.5, ss=8)

arr(18.5, 11.675, 18.5, 11.525, C['eval_brd'], lw=2.0)
arr(18.5, 10.05,  18.5, 9.7,    C['eval_brd'], lw=2.0)
arr(18.5, 8.7,    18.5, 8.15,   C['eval_brd'], lw=2.0, lbl='  non-terminal')
arr(18.5, 6.85,   18.5, 6.5,    C['eval_brd'], lw=2.0, lbl='  submit_verdict')

ax.annotate('', xy=(21.35, 10.8), xytext=(21.35, 8.15),
    arrowprops=dict(arrowstyle='->', color=C['eval_brd'], lw=1.6,
        connectionstyle='arc3,rad=-0.3', shrinkA=6, shrinkB=6), zorder=2)
ax.text(21.65, 9.45, 'loop\nnext step', ha='left', va='center',
        fontsize=7.5, color=C['eval_brd'], style='italic')

ax.annotate('', xy=(15.2, 7.5), xytext=(10.0, 4.45),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.4,
        connectionstyle='arc3,rad=-0.25', linestyle='dashed',
        shrinkA=8, shrinkB=8), zorder=2)
ax.text(12.9, 6.3, 'periodic eval', ha='center', va='center',
        fontsize=8.5, color='#9ca3af', style='italic')

# Legend
leg = [(C['data_brd'], 'Data Generation'), (C['model_brd'], 'LLM / Model'),
       (C['rew_brd'],  'Positive Rewards'), (C['pen_brd'],  'Penalties'),
       (C['grpo_brd'], 'GRPO Mechanics'),   (C['eval_brd'], 'Eval Episode')]
lx = 0.5
for col, lbl in leg:
    ax.add_patch(plt.Rectangle((lx, 0.22), 0.38, 0.38,
                               fc=col, ec=col, lw=1.5, zorder=3))
    ax.text(lx+0.52, 0.41, lbl, va='center', fontsize=8.5, color=C['sub'])
    lx += 3.6

os.makedirs('plots/ema', exist_ok=True)
plt.tight_layout(pad=0.2)
plt.savefig('plots/ema/rl_training_loop.png', dpi=180,
            bbox_inches='tight', facecolor='#ffffff')
print('DONE: plots/ema/rl_training_loop.png')
