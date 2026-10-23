"""CPU model tests do not use the POSIX-only hidden-state file transport."""

import sys
from types import ModuleType

if sys.platform == "win32":
    # The package eagerly imports the Linux training transport. Leave this
    # module without flock: accidentally using the transport must still fail.
    sys.modules.setdefault("fcntl", ModuleType("fcntl"))
