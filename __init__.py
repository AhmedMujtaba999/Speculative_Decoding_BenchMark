"""specdec - benchmark and explain speculative decoding in vLLM.

Modules
-------
prompts  : the workload (4 categories, from "output copies the prompt" to "free writing")
theory   : closed-form expected speedup (Leviathan et al., 2023) + inverse helpers
metrics  : read vLLM's spec-decode counters and turn them into acceptance stats
"""

__version__ = "0.2.0"
