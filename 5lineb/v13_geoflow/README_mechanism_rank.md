# Independent mechanism ranking

This experiment retains the M1 network, its three vectors and concatenation,
two independent sigmoid gates, two GATv2 attentions, separate message maps,
destination head, candidates, fixed split, CE, Adam L2 and early stopping.
The competition gate means origin pressure associated with outward substitution.
No destination-pressure reinterpretation is used. No gate labels are swapped.

## Input audit

The fetched ruler copy differs from the original byte hash only by a final newline; removing that newline reproduces the saved SHA256 exactly. All numeric parameters are unchanged.

The nine columns, in model order, are RA, SA_per_10000 and TC for TH, SH and
CHC. Every indicator is a benefit. The existing frozen Yeo Johnson and z-score
ruler is applied unchanged. RA is a resource index, not measured visit capacity.
The original pressure is log1p of visible nonself incoming trips divided by
positive raw RA, separately for CHC, SH and TH. Its standardized coordinates
and service flags continue to enter the origin gate unchanged.

The original local flow vector uses fixed internal trips and current visible
outflow. The internal count remains visible while internal destinations remain
training targets. This existing limitation is disclosed, not silently repaired.
All new flow-dependent references use current visible nonself messages.

## Weak references

Let x be the frozen nine-dimensional supply input, d the unchanged standardized
road impedance, and a_ij = sigmoid(-d_ij). Zero supply denotes the frozen
reference mean, not a clinical adequacy threshold. All formulas are weak
operational references and do not measure causal displacement or service loss.

For component k, deficit_ik = max(-x_ik,0)/(1+max(-x_ik,0)) and
advantage_ijk = max(x_jk-x_ik,0)/(1+max(x_jk-x_ik,0)).
The compensation reference is a_ij times the mean of deficit_ik*advantage_ijk.
This requires a relative local deficit and an advantage in the same dimension.
Equal weighting is an explicit modeling convention, not an estimated importance.

For competition, L_i averages log1p(I_i/RA_it) over tiers present at the origin.
P_i = L_i/(L_i+median_positive_training_L). The denominator scale is fixed from
the training graph. V_i = mean sigmoid(-x_ik) measures relative vulnerability.
O_i is the current visible outflow, and U_i = O_i/(O_i+F_ii).
A_ij = mean sigmoid(x_jk) * a_ij represents a feasible external alternative.
The competition reference is P_i*V_i*U_i*A_ij. The pressure already contains
resource availability, so V adds correlated supply evidence; specificity must
be tested rather than assumed. The references are not independent ground truth.
No positive RA or undefined outward share makes the competition reference
unavailable and excludes that edge from its loss, never encoding absence as zero.

## Matching and losses

Compensation pairs share an origin, topology and training-fitted impedance
decile. Competition pairs share a destination, topology and impedance decile,
allowing origin pressure to vary while holding the substitute destination fixed.
Four independently shuffled within-stratum adjacent pairings are formed using
a separate deterministic RNG. Proxy separation must exceed 0.1 training SD.
At most 8192 pairs per mechanism enter each epoch. References are NumPy arrays
outside autograd. The losses are mean softplus(-(q_high-q_low)) for each gate.
Missing references and masked messages are excluded. No balancing, exclusivity,
orthogonality, HSIC or calibration loss is introduced.

The candidate grid uses 0.01, 0.1 and 1 for E1 and E2 separately. E3 uses the
three diagonal pairs (0.01,0.01), (0.1,0.1), (1,1), not a full Cartesian grid.
All settings use seeds 0,1,2 and the same split. Weights are independent loss
fields and both component losses are recorded even when disabled.

## Evaluation and decision

Saved gates and references distinguish validation held-out edges, test held-out
edges and visible non-test edges. Hidden gate flow inputs are zero. Diagnostic
pressure uses the corresponding permissible context, never the held-out counts.
Quantile cutpoints, reference SDs and matching bins are fitted on training only.
Stability uses unchanged gate order and reports every seed pair. Correlations,
MAD, MSE and ordering agreement are unweighted over the declared OD population.
These differ from older flow-weighted, optionally permuted diagnostic summaries.

Validation acceptance, fixed before completed E3 results, requires all
three seeds to meet both gates' matched rank agreement above 0.55, own-reference
agreement exceeding the cross-reference agreement by 0.02, and exceeding the
other gate on the same reference by 0.02. Gate variance must exceed 0.0001 and
neither tail below 0.05 or above 0.95 may contain 95 percent of the edges.
The top versus bottom training-defined quantile response must exceed 0.02 for
deficit and advantage at the compensation gate, pressure and vulnerability at
the competition gate. Mean validation CE may exceed fresh E0 by at most 0.05.
These are exploratory operational thresholds, not statistical significance tests.
Only an eligible E3 proceeds to E4 with the existing weight-10 consistency term.
No test metric selects weights. Failure of specificity is reported explicitly.

## Run

```bash
python 5lineb/v13_geoflow/run_od_identifiability_v13.py --config 5lineb/v13_geoflow/configs/outpatient_d0.25_od_identifiability_rank_v13.yaml
python 5lineb/v13_geoflow/summarize_od_identifiability_v13.py results_od_mechanism/outpatient_d0.25_od_identifiability_rank_v13
python -m pytest 5lineb/v13_geoflow/tests -q
```

Each seed saves a checkpoint, config, training curve, predictions, gate values,
references, response diagnostics and exact score replay verification. Root files
save the split, proxy rules and source/data hashes. Existing M1/M2 outputs are
not overwritten. Existing ablation results remain historical comparisons, while
fresh E0 is the same-environment comparator for the new ranking runs.

## Archived checkpoints

Per-seed binary outputs are preserved in `archives/<setting>_seed<n>.zip` under the new result directory. Extract each archive into that result directory before replaying or rebuilding summaries. The archive restores the setting and seed subdirectories. Root tables and figures are directly readable.
