# Mechanism identifiability audit

## Scope and decision

This audit concerns the saved outpatient d=0.25 M1 and M2 experiments with seeds 0, 1 and 2. It changes documentation only. The operational definitions follow `Manuscript0423_Section_Keywords_and_Key_Sentences.md` in the manuscript `major` directory, especially the definitions of outflow compensation and inflow competition in sections 2.2 and 3.4.

Existing inputs support weak references for relative supply compensation and destination demand pressure. They do not identify realized compensation, competitive erosion, effective supply or causal effects. Destination prediction and gate reproducibility require separate assessment.

## Fixed supply inputs

`src/D_mrhgat/routing_data.py` selects columns by `SUPPLY_COLUMNS`. The model order is the following order, regardless of their physical order in `supply_v13.xlsx`.

| Position | Column | Direction | Frozen Yeo Johnson lambda | Frozen mean | Frozen population standard deviation |
| --- | --- | --- | --- | --- | --- |
| 0 | `RA_TH` | Larger resource availability | −7.805753105 | 0.019717925 | 0.046100701 |
| 1 | `SA_TH_per_10000` | Larger spatial accessibility | −4.797224603 | 0.031869771 | 0.074551647 |
| 2 | `TC_TH` | Larger travel convenience | −55.476644868 | 0.002705197 | 0.006326830 |
| 3 | `RA_SH` | Larger resource availability | −15.955884621 | 0.005457667 | 0.017670335 |
| 4 | `SA_SH_per_10000` | Larger spatial accessibility | −10.995244415 | 0.007884771 | 0.025530117 |
| 5 | `TC_SH` | Larger travel convenience | −88.026926975 | 0.000988694 | 0.003201104 |
| 6 | `RA_CHC` | Larger resource availability | Identity | 2.984158880 | 1.995015738 |
| 7 | `SA_CHC_per_10000` | Larger spatial accessibility | −0.254861568 | 0.962057232 | 0.409035308 |
| 8 | `TC_CHC` | Larger travel convenience | −4.633773300 | 0.080695476 | 0.046485852 |

The exact parameters are in `重做实验/step3_od_demand_outpatient/results/node_typing/ruler.json` under `main.frozen`. `5lineb/common/ruler.py` applies the saved transform to each listed column and subtracts the saved mean before dividing by the saved standard deviation. No sign reversal, refitting, clipping or PCA projection occurs in the model input. The input matrix contains nine standardized indicators.

The definitions come from `重做实验/step3_od_demand_outpatient/supply_indicators_revised.py`. RA sums facility resource weights `0.5 + 0.5 × capacity / positive tier median` within each community life circle. Hospital capacity uses beds and CHC capacity uses a service score. SA for CHC divides life circle capacity by life circle population. SA for SH and TH sums facility capacity divided by population in the facility service catchment. The exported SA columns multiply those ratios by 10,000. TC is TOPSIS closeness using population weighted facility counts in walking time bands 0 to 5, 5 to 10 and 10 to 15 minutes with weights 0.5, 0.3 and 0.2. Every band is treated as a benefit criterion. TC is a convenience score, not elapsed travel time. These indicators are not the manuscript Table 2 count ratio, density and accessibility formulas.

The saved workbook contains 666 communities and exactly matches the corresponding nine columns in `community_supply_indicators_2018_mobile_users.csv`. Every raw indicator is nonnegative. The frozen transform preserves the stated directions. RA is a weighted resource index with tier specific units and cannot be treated as a measured visit capacity.

## Tensor orientation and visible contexts

`W[i,j]` means trips from origin `i` to destination `j`. The graph applies the fixed flow threshold of 2 before splitting. The thresholded matrix contains 1,469,546 trips including 26,194 internal trips, with 126,108 nonself edges. The fixed test and validation masks contain 25,316 and 10,059 edges. The remaining 90,733 edges form the available training graph.

Let `V` denote the current nonself message mask and `C = W × V`. Training draws about half of the available training edges each epoch. Validation uses all available training edges with both test and validation edges absent. Final testing uses all nonself edges outside the test mask, including validation edges. A reference used during optimization must use the current training view.

### Exact definition of phi_i

`MechanismNet.set_context` constructs two columns from `log1p(W[i,i])` and `log1p(sum_j C[i,j])`, with self entries excluded from the sum. Each column is centered across all 666 nodes and divided by its population standard deviation, floored at `1e−12`. The self column comes from the diagonal of the fixed thresholded `W`, not from `C` and not from the unthresholded OD matrix. The outgoing column and its standardization change with the message view.

The base channel reads the supply embedding and these two phi columns without neighbor aggregation. The same phi enters both node propensities. Self trips never enter external messages or pressure. Self trips remain visible through phi while the internal choice remains in the training CE target. This is a property of the saved protocol. Validation and test CE, CPC and NRMSE use nonself choices.

### Exact definition of inflow

`run_od_relation_v13.pressure_context` calculates `I[v] = sum_u C[u,v]` with the diagonal excluded. For tiers CHC, SH and TH, it computes `L[v,t] = log1p(I[v] / RA[v,t])` where raw `RA[v,t] > 0`. It standardizes each tier over nodes having that tier using population standard deviation floored at `1e−12`. A node without that tier receives zero in its standardized pressure coordinate. Three explicit service presence flags follow the pressure coordinates.

The six columns of `model.inflow` therefore have order CHC pressure, SH pressure, TH pressure, CHC presence, SH presence and TH presence. `inflow[v]` describes incoming demand at node `v`. It is not the set of incoming messages or a capacity share. The pressure denominator uses raw RA. The numerator includes all visible nonself incoming trips and has no tier specific trip allocation.

`forward_context` recomputes pressure and phi for every view. A masked edge contributes no message, no attention denominator, no edge flow feature and no outgoing or incoming flow sum. Changes to the sums also change the associated standardization across nodes. The gate flow coordinate is `log1p(C[i,j])`, which becomes zero for a pair without a message. Dense diagnostic gates are still calculated for such pairs and must not be confused with transmitted messages. The message share matrix is passed through the interface but the mechanism model does not use it to weight aggregation.

Each GATv2 attention normalizes over visible destinations of the same origin. The external vectors accumulate at `oi` using `index_add(0, oi, ...)`. A node without outgoing messages has zero external vectors. Compensation and competition retain independent attention parameters and independent message mappings.

## Competition endpoint and aggregation meaning

The manuscript defines competition as external arrivals reducing service opportunities for residents of the destination community. For a candidate `i → j`, the relevant destination load is `I[j]`. Its pressure feature must therefore be read as `model.inflow[di]` for a destination competition reference. The current `gate_inputs` concatenates `model.inflow[oi]` in both gates and reads the propensity at `s[oi]`. Both node propensities also use their own node pressure. Origin pressure describes competition at the origin and does not directly encode pressure faced by residents of destination `j` for this edge.

The endpoint correction concerns the edge reference and a possible gate conditioning input. The existing competition vector still aggregates outgoing destination messages into the origin representation `h[i]`. It represents pressure associated with destinations connected to origin residents. It is not an estimate of service opportunity lost by residents of `j`, and replacing one pressure input cannot establish that interpretation. An incoming aggregation rewrite would change the specified architecture and is outside the proposed scope. The prediction head reads representations at both endpoints but does not resolve the mechanism attribution.

## Saved prediction and stability evidence

The following values were recomputed from saved per seed and stability files. Prediction values are means with sample standard deviations across three seeds. Gate correlations use the correspondence selected on visible training edges and applied to the test context. The reported stability population is the visible nonself edges outside the test mask, including validation edges. It is not the held out test edge population.

| Model | Test CE | CPC | NRMSE | Compensation correlation mean and minimum | Competition correlation mean and minimum | Gate vector mean absolute difference | Dominant gate agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 3.811934 ± 0.014908 | 0.685590 ± 0.001557 | 4.093904 ± 0.055535 | 0.284758 and −0.649252 | 0.279046 and −0.801701 | 0.192167 | 0.608544 |
| M2 weight 1 | 3.808232 ± 0.008172 | 0.683829 ± 0.005441 | 4.199275 ± 0.170607 | 0.733793 and 0.717705 | −0.088807 and −0.607674 | 0.213389 | 0.703658 |
| M2 weight 10 | 3.796097 ± 0.020341 | 0.687926 ± 0.007790 | 4.084987 ± 0.168152 | 0.658651 and 0.602714 | −0.014326 and −0.522478 | 0.191541 | 0.778632 |
| M2 weight 100 | 3.820035 ± 0.007285 | 0.684317 ± 0.004449 | 4.112432 ± 0.023739 | −0.013973 and −0.314799 | 0.380924 and −0.009847 | 0.260844 | 0.399810 |

M2 adds the mean squared Euclidean difference between the two gate vectors on edges shared by two independently drawn training views. Its prediction CE uses the first view. It constrains agreement within a seed across views and does not impose agreement across seeds.

In M1 the unpermuted competition correlations are −0.801701, −0.809286 and 0.952465 for seed pairs 0 and 1, 0 and 2, and 1 and 2. Correspondence matching cannot consistently establish a fixed scientific meaning. Compensation and competition also use different value inputs, so exchanging diagnostic columns is not an architectural symmetry or semantic validation. Node propensity correlations after matching have means 0.350682 and 0.315810 for M1, 0.178018 and −0.307486 for M2 weight 1, 0.741723 and −0.292140 for weight 10, and 0.613184 and 0.654902 for weight 100.

M1 CE is 4.798130 when external vectors are zeroed at the same checkpoints and 4.068353 with a retrained local baseline. Retraining with one external channel gives CE 3.821239 for compensation and 3.808091 for competition. External aggregation helps prediction, while the gain can be carried by either external channel. M1 has a flow weighted simultaneous activation share of 0.007195 to 0.067496 using both gates above 0.5. Independent gates permit simultaneous activation without identifying the mechanisms.

## Proposed weak references

The existing frozen ruler supplies the basic supply scalar `B = X × U × weights`. Its effective coefficients in model column order are approximately 0.214314, 0.214278, 0.214113, 0.072772, 0.072227, 0.072778, 0.298857, 0.177651 and 0.289116. All are positive in this saved ruler. This makes a larger value a directionally consistent relative supply reference for these inputs. The scalar can be negative and is not an absolute adequacy or visit capacity measurement. It must not replace the learned base channel.

A concrete continuous compensation reference can use `D[i,j] = max(B[j] − B[i], 0)` and `r_comp[i,j] = D[i,j] / (1 + D[i,j])`. It expresses the relative supply improvement potentially available through an external destination. It has no absolute local scarcity threshold, service quality observation or measured resident benefit.

A concrete competition reference can average `L[j,t]` over tiers with positive raw RA at destination `j` to obtain `A[j]`, then use `r_compete[i,j] = A[j] / (1 + A[j])`. A destination with no positive RA has an unavailable reference and must be excluded from its auxiliary term with an explicit validity mask. There are 44 such communities in the saved supply input. Zero service cannot mean zero competition. The unstandardized nonnegative pressure values make this reference increase with visible incoming demand and decrease with resource availability. Averaging across tiers defines a weak index and does not combine tier capacities in common physical units.

These are proposed operational references, not implemented targets or verified mechanisms. They use fixed supply, the frozen ruler and the current visible training context. They require no test or validation OD counts, fitted gates, prediction residuals, inferred hard mechanism classes, manuscript supply change outcomes or test metric optimization. Apply any future reference term to valid visible training edges and detach the reference from model gradients. Recompute pressure for each training view. For diagnostic use in a test context, disclose that validation edges are visible under the saved protocol. A destination can have both a positive relative supply gap and high pressure, so both references and both sigmoid gates can be high on the same OD.

The references can anchor operational directions. Without independent service opportunity evidence, stronger interpretation as measured compensation or competitive erosion remains unsupported. Their scale, reference weight and application to node propensities require explicit specification in the subsequent experiment. No experimental setting or scientific result is established by this proposal.

## Exact proposed implementation file list

The following narrowly scoped files are proposed for subsequent work. No implementation file is changed by this audit.

| File | Proposed responsibility |
| --- | --- |
| `5lineb/v13_geoflow/model_od_mechanism_v13.py` | Explicit reference compatible gate parameterization and destination pressure conditioning while retaining the three vectors, concatenation, two independent sigmoid gates, GATv2 and message mappings |
| `5lineb/v13_geoflow/run_od_mechanism_v13.py` | Fixed input reference construction, valid training view masks, configured auxiliary terms and reference diagnostics with separate result directories |
| `5lineb/v13_geoflow/run_od_relation_v13.py` | Minimum generic auxiliary loss accumulation and logging needed by the mechanism experiment, preserving prediction CE and all existing protocols |
| `5lineb/v13_geoflow/tests/test_od_mechanism_v13.py` | Direction, endpoint, masking, independent activation, unchanged disabled reference behavior and gradient checks |
| `5lineb/v13_geoflow/configs/outpatient_d0.25_od_identifiability_v13.yaml` | New experiment settings with a distinct result name and declared reference construction and weights |
| `5lineb/v13_geoflow/README_identifiability.md` | Audited definitions and the eventual evidence summary |

If the required tables and figures need a separate renderer, the only additional proposed code file is `5lineb/v13_geoflow/summarize_od_identifiability_v13.py`. It must read existing and new experiments without overwriting their results. Further config variants must be specified before implementation. New artifacts belong under a new `results_od_mechanism` experiment directory.

Preserve prediction loss, candidate sets, fixed splits, CE, CPC, NRMSE, early stopping, checkpoint replay, both independent attentions and both message mappings. Preserve the local baseline, compensation and competition vectors and their concatenation. Do not introduce balancing, mutual exclusion, forced orthogonality or artificial hard OD labels.

## Read only verification

All four saved fixed splits equal the current reconstructed splits. All 24 checkpoints reproduce saved scores and dense gate arrays exactly, and their CE, CPC and NRMSE agree within `1e−12`. All existing result file hashes remained unchanged during replay. The 17 existing mechanism and relation tests pass.

The M2 manifests match all four recorded code files and all five recorded input files. The M1 manifest matches the model, `model_v13.py` and every recorded input. Its hashes for `run_od_mechanism_v13.py` and `run_od_relation_v13.py` differ from the current files. Exact replay establishes current numerical reproducibility of those saved checkpoints. It does not establish byte identity between the current runners and the runners recorded at M1 training time.
