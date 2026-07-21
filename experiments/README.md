# experiments/

Experiment drivers (the `run_*.py`, `dissociation.py`, `midscale*.py`, `analysis_*.py`,
`port_*.py`, ...). They import the core modules (`pcn_config`, `encoder_encoder_pcn`,
`train_coco64`, ...) which live at the repo root, and they read/write result files relative
to the **run directory**.

Run them from the repo root with the root on the path:

```
PYTHONPATH=. python experiments/run_coupling_scale.py
```

This is what the cluster jobs already do (`export PYTHONPATH=$HOME/encoder_encoder_PCN`).
Result JSONs land in the run directory (repo root), where the paper's Appendix A expects them.
