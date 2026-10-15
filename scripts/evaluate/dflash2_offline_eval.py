"""DFlash2 entry point for the shared offline evaluator.

The shared loader selects the implementation from the saved checkpoint config.
Dataset selection, token verification, EOS handling, and statistics therefore
match the DSpark/MMuse evaluation entry point, including multi-device workers.
"""

from dspark_offline_eval import main

if __name__ == "__main__":
    main()
