"""Quick test run of the pipeline with non-interactive matplotlib."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
_original_show = plt.show
plt.show = lambda *a, **kw: plt.close("all")

import sys
sys.argv = ["main.py", "--skip-eda"]

# Patch input to auto-answer 'n' for interactive prediction
import builtins
_original_input = builtins.input
builtins.input = lambda *a, **kw: "n"

from main import main
main()
