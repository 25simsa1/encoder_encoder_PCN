# Regression tests

These run on CPU and need no GPU. They never build the full ~28.7 GiB model, so
they fit on a laptop. Each script exits non-zero on failure.

```
python tests/test_mechanisms.py        # suspect mechanisms in isolation, tiny dims
python tests/test_tiny_integration.py  # real train_step/pass_next on a tiny graph
python tests/test_realdim.py           # scale-only suspects at TRUE dims, in isolation
```

## What is covered

- transpose and flatten predict_prev round-trips
- the num_units in {48,12,3} mask-resize matmul, both zero mask and real mask,
  including the full 192 to 48 to 12 to 3 chain at real sizes
- conv2d_transpose output_shape across the real 572 backbone, plus Conv2DBackpropFilter
  at the largest conv dims
- AddNorm gamma and beta updates inside a transformer
- a full TransformerPCNLayer update sweep, including seq=3 attention at d_model=4096
- the real train_step traversal on a tiny graph that reproduces every structural
  pattern (conv stack with a skip head, transformer then transpose-resize then
  transformer, mask resize, and a shared-state recon head)

All of the above pass. The plumbing is sound at both tiny and real per-layer dims.

## What is NOT covered (needs a GPU with >= 40 GB, ideally 80 GB)

Run `python run_instrumented.py` on that host. Still open are

- aggregate peak memory, expected near 38 GiB at batch 1
- whether gc.collect() actually recovers update-loop transients (saw-tooth) or
  memory grows monotonically (a retained-reference leak worth fixing)
- numeric stability of the full coupled system over many steps

## Known gotcha

TransformerPCNLayer.__call__ does not set its sub-layer states (it omits
set_state=True). train_step is unaffected because pass_next sets states directly,
but the notebook unit-test cells that call the transformer through __call__ would
hit a None state. Their saved outputs are stale.
