# BO coursework: getting started

You write a Bayesian-optimisation algorithm that spends a fixed budget (in $) on simulated cell-culture experiments across three scales
(24-well plates, mini bioreactors and pilot reactors). It tries to find the clone (out of 30) and recipe with the highest product
titer **at Pilot scale**.

Everything you need is in **[project.ipynb](project.ipynb)**: the rules, the problem description, the cell where you write your
algorithm, and plots comparing it with two reference algorithms. You don't need to look at anything outside this `student/`
folder.

## 1. Install uv

[uv](https://docs.astral.sh/uv/) installs Python and every package for you.

- macOS / Linux:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Windows (PowerShell):

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Close and reopen your terminal afterwards.

## 2. Install the project

In a terminal, go to the **repository root** (the folder that contains `pyproject.toml`) and run:

```bash
uv sync
```

This downloads Python 3.14 if you don't have it, then installs everything into a `.venv/` folder. You only need to do this once,
and again whenever the course repository is updated.

## 3. Open the notebook

- **VS Code** (recommended): open the repository folder, open `student/project.ipynb`, click *Select Kernel** in the top right and
  choose the `.venv` Python environment.
- **Jupyter Lab**: from the repository root, run `uv run --with jupyterlab jupyter lab` and open `student/project.ipynb`.

Run the cells from top to bottom.

*if you don't see the kernel, restart VS Code or Jupyter Lab and try again.

## Files

| File | What it is |
| --- | --- |
| `project.ipynb` | The notebook you work in. |
| `algorithms/my_algorithm.py` | Your algorithm. The notebook writes this file, so edit your code in the notebook, not here. |
| `algorithms/reference_screen_then_confirm.py` | Reference: a screening funnel without a model (plate, then mini bioreactors, then Pilot). A good place to see how to build valid batches and pick scales. |
| `algorithms/reference_mfbo.py` | Reference: simplified multi-fidelity Bayesian optimisation. |
| `algorithms/saved/` | Versions of your algorithm you saved with `save_version("name")`, to compare against in the notebook. |
| `results/` | Saved results of every run, reused so unchanged algorithms don't have to run again. Safe to delete. |

## The rules in short

- Budget: $25,000 per run (`lab.budget`). Costs per experiment: plate well $10, mini bioreactor $500, pilot reactor $2,000.
- 5 minutes per run (one seed). At 5 minutes your algorithm is stopped and graded on what it found up to then.
- At most 300 experiments per run (`lab.max_experiments`, wells + reactors, including the initial design), besides the
  budget. A batch runs completely or not at all.
- Every `lab(x)` call runs one scale with exactly `lab.batch_sizes[scale]` rows: a 24-well plate, 4 mini bioreactors or 1 pilot
  reactor.
- In the bioreactors (MBR and Pilot) you choose, per reactor: the clone, temperature, pH, starting glucose and glutamine, two
  feeds (day 5 and day 10), and a new temperature and pH from each feed on (a temperature/pH shift).
- On a plate only the clone varies per well: temperature, pH and the starting medium are shared, and there are no feeds and no
  temperature/pH shifts.
- Only Pilot-scale titers count. MTP and MBR are cheaper reactors you can use to explore.
- Allowed imports only: `numpy`, `scipy`, `random`, `sobol_seq`, `time`, `datetime`, `typing`.
- Grading runs on a **different, unseen lab**: other clones, other seeds, a different initial design, and possibly a different
  budget, experiment limit, costs and batch sizes. Read every setting from the `lab` object and don't tune to the practice lab.
- Only use the `lab` object's public interface: anything starting with `_` (e.g. `lab._simulate`) is rejected at submission, and the
  lab is read-only. Reading files or the lab's configuration, or hard-coding clone names, is flagged for the GTA.
- Put the full names of everyone in your group in `STUDENT_NAMES`.

## How you're scored

The score of one run looks at the best titer measured **at Pilot scale** at three moments: when 33 %, 66 % and 100 % of the
budget has been spent (`lab.score_checkpoints`, in $). It is the sum of the three:

```text
score = best Pilot titer by 33 % of budget + best Pilot titer by 66 % + best Pilot titer by 100 %
```

A checkpoint with no Pilot result yet counts **0**. So run at least one Pilot experiment before the first checkpoint (with the
default budget, before $8,250 is spent), and keep improving afterwards. Finding a good recipe early is worth up to three times as
much as finding it at the very end. The notebook warns you when a run missed a checkpoint. A run stopped at the 5-minute mark is
scored on what it had by then.

Your final score is this averaged over several seeds. The ranking uses only the score.

## How the clones are made

You can't look inside the lab, but knowing roughly how it is built helps you choose a sensible model. Each of the 30 clones
(`celltype_1` … `celltype_30`) is a set of parameters of a cell-culture model (growth, death, nutrient uptake, by-products and
product formation). Every parameter is drawn **at random, uniformly**, from a fixed range, independently for each clone. The ones
that matter most:

| Parameter | Meaning | Range |
| --- | --- | --- |
| optimal temperature | growth is fastest here and drops off on both sides | 30–40 °C |
| optimal pH | growth is fastest here and drops off on both sides | 6.0–8.0 |
| maximum growth rate | how fast the clone can grow under ideal conditions | 0.02–0.06 1/h |
| product yields | how much product the cells make (per cell, and while growing) | vary several-fold between clones |
| nutrient affinity and yields | how glucose and glutamine are used, and how much lactate and ammonia build up | vary between clones |

What this means for you:

- **Clones really differ, and each has its own best temperature and pH.** A condition that is good for one clone can be poor for
  another, so clone and conditions interact.
- **The scales behave differently.** The same recipe gives a different titer on a plate, in a mini bioreactor and in a pilot
  reactor, but good recipes tend to stay good.
- **Measurements are noisy** (a few percent).

Please **don't fit your algorithm to this description or to the practice lab.** The grading lab draws new clones the same way
but with a different random seed, and the ranges or reactor settings may change too. An algorithm that hard-codes these ranges,
particular clones or the practice lab's best recipe will do worse there. Use this section to understand the problem, and let your
algorithm learn from the experiments it runs.

## Submitting

Hand in **only** `algorithms/my_algorithm.py`. Before you do, re-run the `%%writefile` cell so the file is up to date, and check
that the `check_my_algorithm` cell reports no problems.
