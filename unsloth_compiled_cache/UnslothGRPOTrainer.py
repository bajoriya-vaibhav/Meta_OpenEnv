"""
2025.3.17
2025.3.19
4.47.1
0.14.0
__UNSLOTH_VERSIONING__

PATCHED: This file is intentionally a no-op redirect to trl's native GRPOTrainer.
The generated compute_loss in the original cache expected inputs as a dict with
prompt_ids/prompt_mask keys, but trl 0.14.0's GRPOTrainer.compute_loss expects
inputs as a list-of-dicts. Using the native trl implementation is correct and safe.
"""

# Re-export everything from trl's native GRPOTrainer unchanged.
# This means unsloth loads this file (as it always does) but gets the real
# trl 0.14.0 implementation without the incompatible compute_loss override.
from trl.trainer.grpo_trainer import *
from trl import GRPOConfig, GRPOTrainer

# Prevent unsloth_zoo from re-patching by marking as already patched
import sys
sys.modules.setdefault("UnslothGRPOTrainer", sys.modules[__name__])
