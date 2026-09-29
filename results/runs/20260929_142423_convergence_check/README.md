# Convergence sanity check (2026-09-29)

Basic training-convergence sanity check for the MR-HGAT `none` variant on the real 666-community
graph, run after both architecture fixes in `src/D_mrhgat/model.py` (`hid` routed into the
`compet` channel's destination attention; leave-one-out outflow `hid_out_loo` feeding the `comp`
channel). Uses a genuine spatial train/eval split (`masks(d, 0)`), baseline-initialized `fit`
with early stopping (epochs=400, patience=20), to confirm the loss decreases steadily and
converges with no NaN/blowup.

This is only a convergence check, not the formal protocol simulation-validation gates.
