# BO Coursework – FAQ

General principle: **read everything from the `lab` object and never hard-code values.** The grading lab may differ from the development lab, and your algorithm should adapt dynamically.

---

## Lab setup

### Will the hidden grading lab still have 30 clones, or can `len(lab.celltypes)` change?
It can change. In practice you won't know the range, so your algorithm should adapt dynamically (e.g. choose more screening plates if it decides more are needed) rather than hard-coding a value. For this coursework, the grading lab will use **between 20 and 40 clones**.

### How many initial experiments will be provided?
The grading lab may use a different number of plates and MBR batches, so handle `initial_x` / `initial_y` of **any length**.

### What is the range for pH and temperature?
Read the bounds from `lab.bounds`. Don't hard-code them.

### Will MTP (wells per batch), MBR (reactors per batch), or Pilot (reactors per batch) change?
They won't change for this coursework. In practice they could (e.g. different machinery might have one more micro-bioreactor), so don't rely on fixed values.

---

## Costs and budget

### What is the cost per experiment?
The relative ordering **MTP ≪ MBR < Pilot** will remain unchanged, but don't rely on the exact numbers. Use `lab.batch_cost(reactor)`.

### What are the budget and experiment limits?
Read both from the lab:
- Budget: `lab.budget`
- Maximum allowed experiments: `lab.max_experiments`

Plan your Pilot runs from `lab.score_checkpoints` (in dollars). The time budget is fixed at **5 minutes**.

---

## Scoring and examination

### Is the examination one run per seed checked at 33%, 66%, and 100% of the budget, or three separate budgets?
It stays at the **33%, 66%, and 100%** marks of a single run. You can read the checkpoints from `lab.score_checkpoints` (in dollars).

⚠️ Make sure you have run a **Pilot-scale experiment before each checkpoint**, otherwise your score for that checkpoint is **0**.

---

## Pilot runs

### Can we change pH and temperature mid-experiment in a Pilot run, and is the effect non-linear (i.e. does the previous set-point affect performance at the current one)?
Yes, you can change pH and temperature at the **two feeding events**. You don't know whether the effect is linear or non-linear.

---
