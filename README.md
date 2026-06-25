# Replication Study — Setup & Instructions

## 1. Paper selected

**Haben, G., Habchi, S., Papadakis, M., Cordy, M., Le Traon, Y. (2024).**
*"The Importance of Discerning Flaky from Fault-triggering Test Failures:
A Case Study on the Chromium CI."*
**ASE 2024** (39th IEEE/ACM International Conference on Automated Software
Engineering, Sacramento, Oct 27 – Nov 1, 2024).

- **Conference rank:** ASE is **CORE A\*** (confirmed against the current
  CORE rankings portal, ICORE2026 listing, June 2026). This exceeds the
  assignment's minimum bar of rank B.
- **Why this paper:** it has a genuinely public, code-complete replication
  package on GitHub (not just a PDF appendix), the methodology is fully
  specified in the paper text (Sections 5.2 and 6), and the computational
  footprint (CountVectorizer + Random Forest on tabular/text data) is light
  enough to run on a laptop in minutes-to-an-hour — no GPU, no multi-day
  training. This matters because plenty of A\*/A papers have replication
  packages that are technically "available" but practically unusable
  (needing GPU clusters, proprietary data, or undocumented Docker images);
  this one is not one of those.
- **Replication package:** https://github.com/GuillaumeHaben/ChromiumFlakyFailures
  (code) + a figshare-hosted dataset linked from that repo's README — see
  step 2 below.
- **Paper PDF:** https://arxiv.org/pdf/2302.10594 (this is the arXiv preprint
  of the ASE 2024 paper; same content, freely accessible, useful for citing
  exact numbers without needing ACM Digital Library access).

### What the paper studies
Flaky tests intermittently pass/fail on the same code version and waste
developer time. The paper asks: if you use a state-of-the-art flaky-test
predictor on the Chromium CI, how often does it wrongly mark a **real bug**
(fault-triggering failure) as "flaky" and tell developers to ignore it? It
trains Random Forest classifiers (vocabulary/bag-of-words features from test
source code) across three research questions:

- **RQ1** — train on whole tests, evaluate on individual failures.
- **RQ2** — train on individual failures directly instead of whole tests.
- **RQ3** — RQ2 plus dynamic execution features (flake rate, run duration).

Headline finding: despite 99%+ precision, the RQ1 model misses **76.2%** of
real regression faults (classifies them as flaky). Adding execution features
in RQ3 improves MCC from 0.20 (RQ1) to 0.42, but the authors say performance
is still "not actionable" for production use.

---

## 2. Environment setup

```bash
python3 -m venv ./.venv
source ./.venv/bin/activate        # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

Tested and confirmed installable together as of June 2026 with:
pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, scipy
(modern versions — see `requirements.txt`; the authors' own `requirements.txt`
pins versions from 2022/2023 that no longer install cleanly on current Python,
so this project re-implements their methodology rather than running their
original notebook verbatim. This is itself worth a sentence in your report's
"Setup" section — environment rot is one of the most common real-world
threats to replicability, and the ACM badging guide explicitly anticipates it).

## 3. Getting the data

The four data files are hosted on **figshare**:

> https://figshare.com/s/4dbab50a216a1b3a3172

Download these four files and place them in `data/` (already created):

- `dataset.35past.Linux10k.json`
- `dataset.pass.json`
- `nft-121.json`
- `nft-123.json`


## 4. Running the pipeline

```bash
# Step 1 — ALWAYS run this first on real data. It tells you whether the
# real column names/label encoding match what reproduce.py assumes.
python inspect_data.py

# Step 2 — run all three RQs (or one at a time with --rq 1 / 2 / 3)
python reproduce.py --rq all
```

Outputs:
- `results/reproduced_results.csv` — precision/recall/F1/MCC/FPR per RQ
- `figures/rq{1,2,3}_confusion_matrix.png`

`reproduce.py` is a from-scratch re-implementation of the methodology
described in paper Section 5.2 (not the authors' original notebook), with
every modelling decision commented and tied back to the specific paper
section/table it comes from. I did this deliberately rather than just
patching their notebook, for two reasons:

1. Their `requirements.txt` pins `pandas==0.25.3`, deprecated
   `sklearn.metrics.plot_confusion_matrix`, and other APIs that no longer
   exist — patching all of that line-by-line is more error-prone than
   writing clean equivalent code against the paper's stated methodology.
2. For an FCA-adjacent course used to rigorous documentation, re-deriving
   the pipeline from the *written methodology* (rather than trusting
   undocumented notebook cells) is the more defensible thing to put in a
   report — you can cite exactly which paragraph each modelling choice
   comes from.

**Important — documented assumptions you should verify the moment you have
real data** (also flagged inline in `reproduce.py`'s docstring):

- Label encoding: `0=flaky, 1=fault-revealing, 2=passing`.
- The mapping between the two passing-test snapshot builds
  (`121238`, `123038`) and the paper's "build b₈₀₀₀" boundary is not fully
  explicit in the README — the script unions both by default and exposes a
  config flag (`PASS_BUILD_STRATEGY`) to switch.
- Exact grid-search ranges (number of trees, SMOTE strategy, k-best
  features) are described qualitatively in the paper ("tuned using a grid
  search approach") but not given as exact values — `reproduce.py` uses a
  reasonable small grid; widen `GRID` in the script if your hardware allows
  for a closer match.


## 6. What's in this folder

```
project-directory/
├── README.md                     <- this file
├── requirements.txt
├── inspect_data.py               <- run first, on real data
├── reproduce.py                  <- main RQ1/RQ2/RQ3 pipeline
├── original_paper_results.csv    <- ground-truth numbers from the paper (Tables 4 & 5)
├── independent_replicate.py      <- independent replication script
├── run_extensions.py             <- independent replication script with modifications
├── visualize_results.py          <- Data visualization script
├── data/                         <- put the 4 downloaded JSON files here
├── results/                      <- reproduce.py writes its CSV output here
└── figures/                      <- confusion matrix PNGs land here
```

## 7. Experimental Redesign & Independent Architecture (Task 5)

To satisfy clean-room validation requirements, an alternative replication
strategy was designed entirely from scratch without using any of the
authors' original pipeline mechanics, over-sampling strategies, or
classification engines. 

### Architectural Pipeline Design
The clean-room architecture completely decouples the replication from the
original implementation by swapping out the authors' basic word-count
bag-of-words representation and balanced random forest structure.
Instead, it pairs a character-level structural text parsing engine with an
un-sampled, native loss-weighted Gradient Boosting Machine.

### Feature Engineering Divergences
1. **Textual Space:** Instead of counting exact whitespace-separated words
(which is highly vulnerable to code formatting styles and completely empty
string data discovered during inspection), the independent pipeline extracts
character $n$-grams ranging from 3 to 5 characters.
This helps isolate key runtime substrings like `wait`, `lock`, `async`, and
`time` embedded across varying language syntaxes.
2. **Numerical & Execution State Engine:** `runDuration` values are parsed
and normalized via a robust scaler to isolate severe execution timeout behavior
from common runtime distributions.
Historical test-level context is independently compiled using a strict trailing
historical build matrix ($w=35$):
$$flakeRate(t, n) = \frac{1}{w} \sum_{x=n-w}^{n-1} flake(t,x)$$

### Class-Imbalance Management Strategy
The target database suffers from a highly skewed class distribution, where target
anomalies represent only ~1% of all failure rows. Rather than synthetically
synthesizing rows in the feature space (such as using SMOTE, which can add noise
to sparse text vector patterns), this architecture uses an internal **Cost-Sensitive
Loss Scaling** approach within the gradient-boosting tree configuration. The
splitting loss evaluation dynamically applies an inverse balance multiplier $W$ to
scale gradients on the minority class instances:
$$W_1 = \frac{N_{\text{majority}}}{N_{\text{minority}}}$$

## 8. Custom Independent Source Code Development

A completely clean-room script implementing the alternative architecture has been fully
written and deployed inside the root execution directory.

### Execution Instructions
```bash
# Execute the alternative clean-room pipeline using the independent architecture
python independent_replicate.py
```

### Example Output
```text
================================================================================
RUNNING HIST-GRADIENT BOOSTING PIPELINE
================================================================================
 -> Clean Train Matrix Size: 428017 rows
 -> Clean Evaluation Deck Size: 30999 rows

Fitting Class-Weighted Gradient Boosting Framework...
Executing Inference across Evaluation Holdout Deck...

----------------------------------------
RESULTS SUMMARY
----------------------------------------
  Pipeline_Type: Clean-Room GradientBoosting (TF-IDF)
  Fault_Precision: 0.0830
  Fault_Recall: 0.5502
  MCC: 0.1956
  Missed_Fault_Rate (FPR-inverted): 0.4498
  TN (True Flaky): 28812
  FP (Flaky Mislabeled as Bug): 1878
  FN (Bug Mislabeled as Flaky): 139
  TP (True Bug caught): 170

```

## 9. Execute Independent Replication and Extensions

An automated exploration framework was built to evaluate how the clean-room
gradient-boosting pipeline holds up under altered algorithmic constraints.

### Extension Design & Parameter Perturbation
The extension focuses on a major limitation highlighted in the baseline paper:
the tradeoff between high precision and an unacceptable missed-fault rate (FPR).
By perturbing the learning constraints, we test whether tweaking model capacity
can lower the severe fault blind spots:

1. **Iteration Capacity Sweep (`max_iter` $\in [50, 100, 200]$):** Scales the
 number of boosting rounds to check if deeper tree ensembles catch subtle
character patterns without over-fitting the sparse TF-IDF feature space.
2. **Loss Weight Skew Adjustment (`balance_factor` $\in [1.0, 1.5, 2.0]$):**
Multiplies the native cost-sensitive balance class weight by a shifting coefficient.
Shifting the loss penalty heavier toward the minority class forces the gradient
booster to prioritize minimizing False Positives (Real Bugs flagged as flaky).

### Extension Execution Instructions

```bash
# Execute the parameter sweep across both replication and extension test scenarios
python run_extensions.py
```

### Example output
```text
================================================================================
EXTENSION MATRIX EXPERIMENTS & ENSEMBLE PERTURBATIONS
================================================================================

Executing Baseline Replication Run...
 -> Baseline Complete: MCC=0.1487, Missed_Fault_Rate=0.5152

Launching Extension Matrix Parameter Perturbations...
 -> Testing Extension_Variant_1: Max_Iter=50, Weight_Skew=1.0...
 -> Testing Extension_Variant_2: Max_Iter=50, Weight_Skew=1.5...
 -> Testing Extension_Variant_3: Max_Iter=50, Weight_Skew=2.0...
 -> Testing Extension_Variant_5: Max_Iter=100, Weight_Skew=1.5...
 -> Testing Extension_Variant_6: Max_Iter=100, Weight_Skew=2.0...
 -> Testing Extension_Variant_7: Max_Iter=200, Weight_Skew=1.0...
 -> Testing Extension_Variant_8: Max_Iter=200, Weight_Skew=1.5...
 -> Testing Extension_Variant_9: Max_Iter=200, Weight_Skew=2.0...

----------------------------------------
EXTENSION MATRIX PROCESS RUNS COMPLETE
----------------------------------------
           Scenario  Max_Iter  Balance_Factor  Fault_Precision  Fault_Recall    MCC  Missed_Fault_Rate                                                                Raw_Log
Clean-Room Baseline       100             1.0           0.0597        0.4848 0.1487             0.5152 TN_Flaky=21271, FP_Mislabeled=1763, FN_MissedBug=119, TP_BugCaught=112
Extension_Variant_1        50             1.0           0.0597        0.4848 0.1487             0.5152 TN_Flaky=21271, FP_Mislabeled=1763, FN_MissedBug=119, TP_BugCaught=112
Extension_Variant_2        50             1.5           0.0538        0.6667 0.1657             0.3333  TN_Flaky=20325, FP_Mislabeled=2709, FN_MissedBug=77, TP_BugCaught=154
Extension_Variant_3        50             2.0           0.0469        0.6407 0.1477             0.3593  TN_Flaky=20025, FP_Mislabeled=3009, FN_MissedBug=83, TP_BugCaught=148
Extension_Variant_5       100             1.5           0.0538        0.6667 0.1657             0.3333  TN_Flaky=20325, FP_Mislabeled=2709, FN_MissedBug=77, TP_BugCaught=154
Extension_Variant_6       100             2.0           0.0469        0.6407 0.1477             0.3593  TN_Flaky=20025, FP_Mislabeled=3009, FN_MissedBug=83, TP_BugCaught=148
Extension_Variant_7       200             1.0           0.0597        0.4848 0.1487             0.5152 TN_Flaky=21271, FP_Mislabeled=1763, FN_MissedBug=119, TP_BugCaught=112
Extension_Variant_8       200             1.5           0.0538        0.6667 0.1657             0.3333  TN_Flaky=20325, FP_Mislabeled=2709, FN_MissedBug=77, TP_BugCaught=154
Extension_Variant_9       200             2.0           0.0469        0.6407 0.1477             0.3593  TN_Flaky=20025, FP_Mislabeled=3009, FN_MissedBug=83, TP_BugCaught=148
```

## 10. Data Visualization Assets (Task 8)

### Visualization Execution Instructions

```bash
# Generate the parameter tradeoffs and comparative confusion matrix plots
python visualize_results.py
```

## Generated Visual Assets
- Algorithmic Sensitivity Tradeoff Chart (figures/extension_optimization_tradeoff.png):

This plot graphs the algorithmic compensation sweet spot discovered.
It visualizes how shifting minority class cost weights isolates real bugs while
balancing precision degradation. It explicitly maps out the optimization point where
Matthews Correlation is maximized (MCC = 0.1657) and Missed Faults drop down to 33.33%.

- Structural Confusion Matrix Comparison (figures/confusion_matrix_comparison.png):

A side-by-side verification plot illustrating the paradigm shift between label targets:
- Left Frame (Trial Run 1): Illustrates the fatal majority-class over-fit, where a
flaky-centric label setup leads to high accuracy but creates massive blind spots for bugs.
- Right Frame (Clean-Room Run): Illustrates the bug-centric configuration, mapping out
true regressions caught vs. flaky false alerts in continuous integration downsampled environments