"""DFlash2 entry point for the shared DSpark/DFlash2 offline evaluator.

Both entry points intentionally use the same dataset discovery, deterministic
sample slicing, verifier loop, proposal budget, timing fields, and acceptance
statistics. ``accepted_draft_length`` excludes the anchor; ``acceptance_length``
is the same value plus one, matching the existing DSpark report.
"""

from dspark_offline_eval import main


if __name__ == "__main__":
    main()
