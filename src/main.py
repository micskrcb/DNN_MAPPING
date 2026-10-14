"""Notice: Legacy single-chip mapping framework has been archived to legacy/single_chip/.

For the active multi-chip many-core research implementation, use:
    python src/run_multi_chip.py --help

To execute the legacy single-chip code:
    cd legacy/single_chip && python main.py --config config/example.py
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    legacy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legacy", "single_chip"))
    print("=" * 60)
    print("NOTICE: The legacy single-chip GCN PPO mapper has moved to:")
    print(f"  {legacy_dir}")
    print("For active multi-chip placement, use: python src/run_multi_chip.py")
    print("=" * 60)

    # Forward execution to legacy/single_chip/main.py if arguments provided
    script_path = os.path.join(legacy_dir, "main.py")
    cmd = [sys.executable, script_path] + sys.argv[1:]
    sys.exit(subprocess.call(cmd, cwd=legacy_dir))
