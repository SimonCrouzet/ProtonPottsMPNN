# pH-sensitive binder design with Proton-PottsMPNN

A **PottsMPNN with an explicit protonation-state alphabet**, for designing **pH-switchable** binders.
Histidine is `HIS-P` (charged, +1) vs `HIS-S` (neutral); acids are `ASP-P`/`GLU-P` (protonated, neutral
COOH) vs `ASP-D`/`GLU-D` (deprotonated, −1). Because the learned Potts energy is protonation-aware, the
design engine can **pin protonated centres** and redesign around them so that binding **switches with pH**.

This is the code for *"pH-sensitive binder design with Proton-PottsMPNN"* (Jacobsen et al., 2026). It
extends **PottsMPNN** — the Potts-energy inverse-folding
model of **Birnbaum & Keating** ([github.com/KeatingLab/PottsMPNN](https://github.com/KeatingLab/PottsMPNN);
[PNAS 2026, 10.1073/pnas.2535494123](https://www.pnas.org/doi/10.1073/pnas.2535494123)) — by adding explicit
protonation-state tokens so that a single energy function scores alternative protonation assignments on a
fixed backbone, which is what makes pH-conditioned design possible.

![Figure 1 — Proton-PottsMPNN and pH-conditioned binder design](figures/Figure_1.png)

> **Figure 1.** **(a)** Local geometric features predict per-residue protonation states, encoded as sequence
> tokens to train Proton-PottsMPNN; backbone node/edge embeddings feed a shared encoder, then an
> autoregressive decoder (token likelihoods) and a Potts head (single-site fields + pairwise couplings).
> **(b)** Protonation-conditioned redesign for a fixed `ASP-P` centre (pink) vs `ASP-D` (blue): neighbours are
> mutated to balance global Potts energy against the selective gap `E_selective = E_P − E_D`, weighted by λ.
> **(c)** Flow cytometry of enriched yeast-display binders incubated with PD-L1 at pH 5.0 vs 7.4.

The project has two halves, and this folder is self-contained for both:

1. **Label** — a FLAML labeller assigns a protonation state to every titratable residue of every training
   structure (the supervision the model learns from).
2. **Design** — the trained Potts model drives a block-descent optimiser that places protonated centres and
   redesigns their neighbourhood.

---

## Install (once) — with uv

```bash
cd ProtonPottsMPNN
./install.sh                     # uv venv (Python 3.12) + uv pip install -e ./foundry + extras
./install.sh --clear             # rebuild an existing venv from scratch
VENV_DIR=~/venvs/ppm ./install.sh   # put the venv elsewhere (default .venv)
source .venv/bin/activate
```

`install.sh` runs `uv venv --clear` then a single `uv pip install -e ./foundry -r requirements-extra.txt`
(the one resolution keeps both the foundry core — torch, lightning, atomworks[ml] — and the extras — the
FLAML stack, jupyter, propka), verifies `import mpnn` resolves inside this folder, and registers the venv as
a Jupyter kernel `ProtonPottsMPNN (.venv)`. Scripts just `import mpnn`; there are no `sys.path` hacks.

- **Python 3.12** is required (`mpnn`/`foundry` pin `>=3.12,<3.13`).
- **Line endings** are LF (`.gitattributes`), so a checkout on Windows or `/mnt/d` runs `install.sh` and the SLURM
  scripts unchanged.
- **HBPLUS** is an external C binary (not pip-installable) used by the **labeller** and the **fold scoring**
  to read H-bond geometry. Point `HBPLUS_PATH` at your build (`export HBPLUS_PATH=/path/to/hbplus`). It is
  **not** needed to run the design engine (which only reads the trained checkpoint).
- Let the install finish uninterrupted — a killed `uv pip install` can leave the venv half-written (package
  metadata present, module files missing). If imports fail oddly, repair in place with
  `uv pip install --python .venv/bin/python --reinstall -e ./foundry -r requirements-extra.txt`.

| part | needs mpnn/torch | needs HBPLUS | needs FLAML stack | needs external oracle weights |
|------|:---:|:---:|:---:|:---:|
| **label a PDB** (`labeller/`) | ✅ | ✅ | ✅ | — |
| **design a binder** (`inference/`) | ✅ | — | — | — |
| **score a fold** (`scoring/`) | ✅ | ✅ (pH-bonds) | — | — |
| **benchmarks** (`benchmarks/`) | ✅ | — | — | ProteinMPNN arm only |
| **train** (`training/`, reference) | ✅ | ✅ | — | — |

---

## Two examples

### 1. Label a PDB with the protonation pipeline

Assign a protonation state to every His/Asp/Glu in a structure — the labels the Potts model is trained on and
the design engine consumes. This runs the **transformation pipeline** `prepare_potts_input(..., extended_vocab="v6")`
— the *same* code path the training data and `_build_context` use: it strips hydrogens, runs HBPLUS, applies
the FLAML labeller, and attaches a per-residue `protonation_label` token (`-P` protonated · `-S` neutral His ·
`-D` deprotonated acid · `-A` ambiguous).

```bash
HBPLUS_PATH=/path/to/hbplus python labeller/label_pdb.py     # -> labeller/outputs/protonation_labels.csv
```
```python
import numpy as np
from mpnn.potts_inference import prepare_potts_input

out = prepare_potts_input("inference/examples/pdl1_seed_binder.pdb", extended_vocab="v6")
ca  = out["atom_array"][out["atom_array"].atom_name == "CA"]     # one CA per residue, token order
labels = ca.get_annotation("protonation_label")                  # per-residue token: HIS-S / ASP-D / GLU-P / …
```
```text
chain  res_id res_name token
    A      32      HIS  HIS-S     # neutral at rest
    A      57      GLU  GLU-D     # deprotonated
    B      52      HIS  HIS-S
    …                             # 21 titratable residues, all neutral/deprotonated
```
Note the read-out: the apo seed binder carries **no** strongly-protonated residue at rest — which is exactly
why *design* (below) **pins** `HIS-P`/`ASP-P`/`GLU-P` centres deliberately rather than reading them off the
input. The pipeline emits the discrete token (the model-input form); for the raw labeller probabilities
(`p_protonated`, `sd`) call `mpnn.transforms.ev6.EV6Predictor` directly — the labeller `prepare_potts_input` wraps.

### 2. Design a pH-switch binder

`PottsMPNNPHEngine` (in the `mpnn` package at `inference_engines/potts_mpnn_ph.py`) places protonated centres
and redesigns their neighbourhood with **Potts-head block descent** — the *exact* optimiser used in our
internal design campaigns.

```bash
python inference/design_ph.py       # -> inference/outputs/ (see below)
# interactively:  jupyter lab inference/design_ph.ipynb   (pick the "ProtonPottsMPNN (.venv)" kernel)
```
```python
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria

engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6")   # 30-token v6 model
crit = PHDesignCriteria(
    method="block_descent", backend="potts",          # the internal-campaign optimiser
    combined_lambda=0.3,                               # Eq (6): O = (1−λ)·zscore(H_stab) + λ·zscore(Σ sel)
    seed_source="native",                              # start from the input sequence (the default "inverse" needs initial_sequences)
    center_types=["HIS-P", "ASP-P", "GLU-P"],          # composition to place …
    placement_by="scan_potts",                         # … placement chooses the positions
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},  # v6 has no HID/HIE
    block_size=3, temperature=0.05, neighbour_k=16, max_mutations=20,
    record_trajectory=True,
)
design_set = engine.run_ph_redesign(atom_array=aa, binder_chain="A", criteria_list=[crit], seed=0)
```

**Direction.** `selective_energy` = Σ(e_P − e_D) over the pinned centres, in the bound complex, and the
optimiser **lowers** it → the bound state prefers the protonated centre → binding is **stronger at low pH**
and weakens once the centre loses its proton. The opposite direction is not available here; use
`run_switch_design` (below) and swap `on` / `off`.

**Interface region.** `PHDesignCriteria.interface_distance` (default 6 Å, CA–CA) sets the `interface`
placement region — raise it when loops reach over the target, otherwise the region can come out empty (a
warning is logged).

**Plain rows.** `design_set.to_rows(reference_sequence=…)` → one dict per design (`canonical_sequence`,
`centers`, `n_mutations`, `selective_energy`, `potts_energy`, …).

The notebook opens up the internals: **placement** (which state goes where, and how many centres — acids to
the core, `HIS-P` to the surface) and the **block-descent trajectory** (stability + selectivity vs. step).
`_build_context` featurises through `prepare_potts_input` (the same transform pipeline the labeller/training
use) — the engine reads a path *or* an AtomArray.

**Many designs → the Pareto front.** `run_ph_redesign` takes a *list* of criteria, so N designs is just a
`combined_lambda` sweep from 0 (pure stability) to 1 (pure selectivity), fanned across a CPU fork pool
(`n_jobs`). The notebook runs `N_DESIGNS` of them and plots each in **(stability, selectivity)** space with the
**Pareto front** marked (`pareto_front.png`). Set `N_DESIGNS = 20` for a denser front.

**Reproducible.** The same `seed` returns the same designs for any `n_jobs` (results come back in task
order). Two designs that share an id but differ in sequence are both kept, the second as `<id>~2`.

**Fold the Pareto designs.** It then picks `N_FOLD` designs off the Pareto front and folds each with the
**packaged RF3 engine** (`fold_rf3.py`, from_target templating: the target chain is templated, the binder is
folded from its designed sequence), scoring charge clashes + pH-sensitive H-bonds at the pinned centres. RF3
**code** ships here, but the **weights (~3 GB) do not** — set `RF3_CKPT=/abs/rf3_*.ckpt` and run on a **GPU**
(with the RF3 atom-embedding cache) to fold. This path was **exercised end-to-end** (input build → checkpoint
load → RF3 forward → a 2-chain binder+target `.cif`) with an internal checkpoint; on a GPU node with the
embedding cache it produces a physical fold. Without RF3 the notebook still runs: it **exports** the selected
designs (`pareto_fold_manifest.json`) ready to fold elsewhere and scores a shipped real RF3 fold as the demo.

**Every design is saved with its protonation states.** Outputs in `inference/outputs/`:

| file | what |
|------|------|
| `designs.fasta` | 1-letter canonical sequence (RF3-foldable) |
| `designs_states.fasta` | the **3-letter + protonation-state** sequence (`… ASP-P … HIS-P …`) |
| `designs.tsv` / `designs.json` | both sequence forms + energies + pinned centres |
| `trajectory.tsv` | the binder sequence (1-letter **and** 3-letter/protonation) **at every optimisation step** |
| `sweep_designs.tsv` | the N sweep designs (λ, stability, selectivity, both sequences) |
| `pareto_fold_manifest.json` | the `N_FOLD` Pareto designs selected to fold (sequence, binder/target chains, centres) |
| `pareto_fold_scores.csv` | per-fold clash / pH-bond scores (**only when RF3 is available**) |
| `placement_scan.png` · `optimisation_trajectory.png` · `pareto_front.png` | the figures |

> Point it at your own backbone by editing `PDB` / `BINDER_CHAIN` (or `inference/examples/example_meta.json`).
> The checkpoint's `extended_vocab` **must** be `"v6"` or the 30-token weight load fails.

---

## Switch design with explicit states (`mpnn.ph`)

`run_ph_redesign` pins protonated centres on the **binder**. `PottsMPNNPHEngine.run_switch_design` takes
**conditions** instead (say pH 7.4 and pH 6.5) that name the protonation state of **any residue on either
chain** — so a binder that binds at 7.4 but not 6.5, and the reverse, differ only by which condition is `on`,
and titratable residues on the **target** switch just like the binder's.

**Three terms, each optional** (weight 0 = off), lower is better:

| term | what it scores |
|------|----------------|
| `stability` | Potts energy of the binder alone in each condition → `max` (worst condition, default) or `mean` |
| `potency` | binding energy in the `on` condition |
| `switch` | how much weaker binding is in `off` than in `on`; `off_margin` stops rewarding the gap once it is wide enough |

**Binding energy** (`binding_model`):

| model | binding energy |
|-------|----------------|
| `state_binding` (default) | complex − binder alone − receptor alone → only the **interface** contribution counts |
| `complex_gap` | the complex energy |
| `linked_equilibrium` | experimental — averages over the protonation states the free partners populate at each condition's pH; `beta` is uncalibrated |

**One run, or a Pareto sweep — your choice.** One active term, or one explicit weight vector → an ordinary
single-objective run. `search: "pareto"` with two or more active terms sweeps weight vectors and returns the
**Pareto set** (`front`, `selected`, hypervolume).

```python
config = {
    "conditions": {                                           # state of any residue, either chain
        "on":  {"ph": 7.4, "states": {"A:12": "deprotonated", "B:409": "deprotonated"}},
        "off": {"ph": 6.5, "states": {"A:12": "protonated",   "B:409": "protonated"}},
    },
    "terms": {"stability": {"weight": 0.3}, "potency": {"weight": 0.3},
              "switch": {"weight": 0.4, "off_margin": 2.0}},
    "designable": {"chains": ["A"],                           # residues named in a condition are never redesigned
                   "near": {"sites": ["B:409"], "k": 16, "max_mutations": 20}},   # optional: only around B:409
    "binding_model": "state_binding",      # | "complex_gap" | "linked_equilibrium"
    "search": "single",                    # | "pareto"
    "target_reduce": "max",                # several target states: "max" = worst state | "mean"
}
run = engine.run_switch_design(atom_array=aa, binder_chain="A", config=config)
best = run.result.records[0]
run.describe(best, "A")                    # canonical sequence + token names of the binder
run.to_rows("A")                           # one dict per design: sequence, n_mutations, term_<name>, front, …
run.site_report(best)                      # per condition and site: complex vs free-partner preference
```

**Redesign around a residue.** `designable.near` keeps only the binder positions the model couples to the
named sites — the same neighbourhood `neighbour_k` / `max_mutations` define in `run_ph_redesign`, except that
a site can now be a **target** residue (`k` caps each site's neighbours, `max_mutations` the total).

**Free-state reference.** `run.site_report(best)` → one row per condition and titratable site:

| column | meaning |
|--------|---------|
| `state_token` | the state the condition imposes (`HIS-S`, `ASP-P`, …) |
| `gap_complex` · `gap_free` | protonated − deprotonated energy of the site in the complex · in the free partners |
| `binding_gap` | their difference; **negative** = binding favours the protonated state → its apparent pKa **rises** on binding |
| `apparent_pka_free` · `_bound` · `_shift` | `linked_equilibrium` only — pKa implied by the protonated fraction at the condition's pH |

All values are in model units (`beta` is uncalibrated).

**Several target states.** `engine.run_switch_design_ensemble(structures={"human": aa_h, "mouse": aa_m},
binder_chain="A", config=config)` designs **one binder against several complexes**. The binder must be the
same in every complex (same residues, same sequence) and the sites named in the conditions must exist in each,
with the same numbering. Every term is evaluated in every state, then combined by `target_reduce`:

| `target_reduce` | a term takes the value of … |
|-----------------|-----------------------------|
| `max` (default) | the **worst** state → the design must hold up in all of them |
| `mean` | the **average** over states |

The reduction is per term, so the worst state for `potency` need not be the worst for `switch`. Each record
keeps the per-state values (`record.per_state`, `term_<name>@<state>` columns in `to_rows`), and `site_report`
covers every state.

Unknown config keys raise instead of being ignored. Designed positions take only non-titratable residues
unless `allow_bare_titratable` lists a parent (bare His/Asp/Glu mean "state unspecified"). Tests:

```bash
cd foundry/models/mpnn && PYTHONPATH=src python -m pytest tests/ph --confcutdir=tests/ph -q
```

---

## Layout

| folder | what it holds |
|--------|---------------|
| [`foundry/`](foundry/) | a full verbatim copy of the `ph/foundry` monorepo. The `mpnn` package is the deliverable: the model (`model/pottsmpnn.py`), transforms, the **deployed FLAML labeller** (`transforms/ev6/`), the **design engine** (`inference_engines/potts_mpnn_ph.py`), and `prepare_potts_input` (`potts_inference.py`). `models/{rf3,rfd3,rfd3na}` come along but their multi-GB weights are not shipped. |
| [`labeller/`](labeller/) | the FLAML protonation labeller: `label_pdb.py` (example above), the train path `01…05_*.py`, `pr_curve.py` (AUPR), and the trained models. |
| [`inference/`](inference/) | `design_ph.ipynb` / `.py` (example above) + `design_placement_scan.py` (the engine's per-design placement plot) + `fold_rf3.py` (fold a design with the packaged RF3 engine; needs `RF3_CKPT` + GPU) + `examples/` (a PD-L1 seed binder + a real RF3 fold). |
| [`scoring/`](scoring/) | fold read-outs: `charge_clash` (geometry) + `annotate` (pH-sensitive H-bonds / salt bridges via HBPLUS+PLIP). `score_example.py` is a runnable demo. |
| [`benchmarks/`](benchmarks/) | PKAD (pKa), MegaScale/FireProt (stability), binding AP, within-backbone, and `placement_by_class.py` (library-wide placement propensity) — all data in-folder. |
| [`training/`](training/) | PottsMPNN + H-bond-head SLURM launchers — **reference** (cluster-specific; the trainer itself is `mpnn.train`). |
| [`checkpoints/`](checkpoints/) | the v6 design checkpoint (21 MB). |
| [`.github/`](.github/workflows/ph-tests.yml) | CI: ruff + the `mpnn.ph` tests on every pull request. |

### Where the core code lives

| component | path |
|-----------|------|
| **PottsMPNN implementation** (encoder → Potts field + pairwise couplings) | [`model/pottsmpnn.py`](foundry/models/mpnn/src/mpnn/model/pottsmpnn.py) · loss: [`loss/potts_loss.py`](foundry/models/mpnn/src/mpnn/loss/potts_loss.py) · train entry: [`train.py`](foundry/models/mpnn/src/mpnn/train.py) |
| **Transformation pipeline** (structure → model input: parse, HBPLUS/SASA features, protonation-vocab encoding) | [`potts_inference.py`](foundry/models/mpnn/src/mpnn/potts_inference.py) (`prepare_potts_input`) · transforms: [`transforms/`](foundry/models/mpnn/src/mpnn/transforms/) (FLAML labeller in [`transforms/ev6/`](foundry/models/mpnn/src/mpnn/transforms/ev6/)) |
| **Inference / design engine** (placement + block-descent pH-redesign) | [`inference_engines/potts_mpnn_ph.py`](foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py) (`PottsMPNNPHEngine`, `PHDesignCriteria`) |
| **Switch design with explicit states** (conditions on either chain, terms, binding models, Pareto, several target states) | [`ph/`](foundry/models/mpnn/src/mpnn/ph/) · entry points `run_switch_design` / `run_switch_design_ensemble` in `potts_mpnn_ph.py` · tests: [`tests/ph/`](foundry/models/mpnn/tests/ph/) |

### Everything else, one command each

| I want to… | run | out |
|------------|-----|-----|
| **label a PDB** | `HBPLUS_PATH=… python labeller/label_pdb.py` | `labeller/outputs/protonation_labels.csv` |
| **design a binder** | `python inference/design_ph.py` | placement / trajectory / Pareto PNGs + `designs*.fasta` / `.tsv` (with protonation states) |
| **switch design, explicit states** | `engine.run_switch_design(atom_array=aa, binder_chain="A", config=config)` (see above) | ranked designs: `run.to_rows("A")`, `run.site_report(best)` |
| **test `mpnn.ph`** | `cd foundry/models/mpnn && PYTHONPATH=src python -m pytest tests/ph --confcutdir=tests/ph -q` | pass / fail |
| **retrain the labeller** | `python labeller/05_train_one.py HIS features 1200` | `labeller/models/automl_feature_HIS/` |
| **labeller AUPR** | `python labeller/pr_curve.py` | `labeller/{his,acid}_pr.png` (AP HIS 0.70, acids 0.33) |
| **score a fold** | `HBPLUS_PATH=… python scoring/score_example.py` | pH-bond / charge-clash counts |
| **placement by class** (library-wide) | `python benchmarks/placement_by_class.py` | `benchmarks/placement_by_class.png` |
| **pKa benchmark** | `PROTON_ROOT=$PWD python -m mpnn.scripts.eval_pkad --checkpoints checkpoints/…/epoch-0125.ckpt` | `pkad_*.csv` + scatter |
| **stability benchmark** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/stability_benchmark.py` | ΔΔG CSVs (GPU recommended) |
| **3-way bar plot** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/benchmark_barplot.py` | recovery + MegaScale + FireProt figure |
| **binding AP** | `python benchmarks/binding_ap.py` | `benchmarks/results/summary_global_ap.png` |
| **within-backbone** | `python benchmarks/within_backbone.py` | `within_backbone_inverse_outcomes.png` |

Benchmark scripts derive the package root from their location (override with `PROTON_ROOT=/abs/path`). The
model-quality benchmarks (PKAD, stability) compute from raw structures; the design benchmarks (binding,
within-backbone) read shipped parquets (regenerating them needs the Boltz-2 / RF3 oracles).

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE). It covers the
Proton-PottsMPNN code authored here (`labeller/`, `inference/`, `scoring/`, `benchmarks/`, `training/`,
`checkpoints/`, the docs, and the pH-design additions to the `mpnn` package). The bundled [`foundry/`](foundry/)
is a verbatim copy of IPD's rc-foundry and keeps its own **BSD 3-Clause License**
([`foundry/LICENSE.md`](foundry/LICENSE.md), © 2025 Institute for Protein Design, University of Washington).

If you use this in academic work, please cite the manuscript *"pH-sensitive binder design with
Proton-PottsMPNN"* (Jacobsen et al., 2026).

---

## Changes in this fork

This fork ([SimonCrouzet/ProtonPottsMPNN](https://github.com/SimonCrouzet/ProtonPottsMPNN)) is **modified by
Simon Crouzet** from the published release
([christian-creator/ProtonPottsMPNN](https://github.com/christian-creator/ProtonPottsMPNN), commit `09682ab`).
The text above is the original README **plus** what was added for these changes; `git log 09682ab..` lists
each one.

**What changed**

| area | change |
|------|--------|
| **`mpnn.ph`** | explicit states on both sides, either direction, several target states, opt-in Pareto search — see *Switch design with explicit states* |
| **`PHDesignCriteria`** | `interface_distance` (the interface placement cutoff, default 6 Å) |
| **results** | the same designs for any `n_jobs`; ids that collide on different sequences are kept (`~2`); `to_rows` |
| **setup** | LF line endings, `install.sh` (`--clear`, `VENV_DIR`, repo-root check), HBPLUS check that fails fast, `ipdb` import removed, `propka` imported where used, `import mpnn.train` works again |
| **tests · CI** | `foundry/models/mpnn/tests/ph/` (run on synthetic Potts tables) and a ruff + tests workflow |

**Where the original text above no longer holds**

| the original says | now |
|--------------------|-----|
| Install: `install.sh` runs `uv venv --clear`; kernel `ProtonPottsMPNN (.venv)` (also in example 2) | it **refuses to overwrite** an existing venv unless given `--clear`; `VENV_DIR` sets where it goes; the kernel is `ProtonPottsMPNN (venv)` |
| HBPLUS is **not** needed to run the design engine (and the install table) | it **is**: `prepare_potts_input` runs HBPLUS on every structure it featurises. Only `mpnn.ph` itself needs just torch + numpy |
| example 2 snippet | it needs `seed_source="native"` (the snippet above now has it): the default `"inverse"` needs `initial_sequences` |
| `foundry/` is "a full verbatim copy" (also under License) | this fork edits the `mpnn` package inside it; the rest is unchanged |
| pKa benchmark: `… eval_pkad --checkpoints …` with `PROTON_ROOT` | `python -m mpnn.scripts.eval_pkad --ckpt_dir checkpoints/potts_v6_afdb_edge_his0.3_acid0.06 --pkad_csv benchmarks/data/PKAD/PKAD-R-v1.0_2026-04-22T16_0755.441Z.csv --pdb_dir benchmarks/data/PKAD/pdb_cache --extended_vocab v6` — there is no `--checkpoints` flag, `--extended_vocab` defaults to `v4`, and the default paths point at the original cluster |
| the design benchmarks (binding, within-backbone) read shipped parquets | only **binding AP** does (`mb_gbind_benchmark.parquet`). `within_backbone.py` needs `egfr_pdl1_*.parquet`, `optimized_sweep_scores.parquet` and `curated_designs.parquet`; `placement_by_class.py` needs `data/placement_scan.parquet`. None of them is in the repo |
