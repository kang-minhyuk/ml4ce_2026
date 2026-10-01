# ML4CE-RL: Reinforcement learning for a home energy microgrid

You will train **one policy** that controls the heat pump and EV battery of five households at once, so that together they spend less, stay comfortable, and wear out the battery less. You compare two ready-made algorithms (CEM and REINFORCE), then write **your own** and submit it. Your algorithm is graded automatically on data you have never seen.

## 1. Start the coursework

You only need [Docker](https://docs.docker.com/get-docker/) — no Python install, no manual setup.

```bash
git clone https://github.com/OptiMaL-PSE-Lab/ml4ce_2026.git
cd ml4ce_2026/RL_coursework
docker compose up --build
```

The first build downloads the base image and Python packages, so it's slower than later starts.

Open <http://localhost:8888> (token `ml4ce-2026`) and open `RL_coursework.ipynb`. The file browser shows:

- `README.md` and `RL_coursework.ipynb`
- `algorithms/` — where you edit your implementation
- `data/training/` — read-only
- `outputs/` — created automatically for trained policies, plots and submissions

**Your work is always safe on your machine**, not just inside the container: everything in the file browser, including any extra file you create, is written straight to your local `RL_coursework/` folder. Stop anytime with `Ctrl+C` or `docker compose down` and resume later with `docker compose up` — nothing on disk is lost.

The one exception is **variables in memory**: stopping the container ends the Python kernel, so a break resets anything you haven't saved to a file. Use *Kernel → Restart & Run All* to rebuild that state, and save anything expensive (trained policies, plots) to `outputs/` as soon as you compute it rather than keeping it only in a variable.

## 2. Your workflow

| Step | What to do | Where |
|---|---|---|
| **1. Explore** | Run the cells that create the environment, reset it and take a few actions. | Notebook, section 1 |
| **2. Compare** | Train the provided **CEM** and **REINFORCE** and evaluate them. This is your benchmark to beat. | Notebook, section 3.1 |
| **3. Write yours** | Fill in `AlgorithmConfig` and `train()` in the blank cell. Running the cell saves it as `algorithms/student_algorithm.py`. | Notebook, section 3.2 |
| **4. Test** | Train and evaluate your algorithm with the cells that follow. Iterate. | Notebook, section 3.2 |
| **5. Submit** | Set your group id in the last cell and run it. It writes `outputs/submissions/<your group>.py` and checks that `train` and `AlgorithmConfig` exist. | Notebook, section 3.3 |

Hand in that one file: `outputs/submissions/G00.py` (with your own group id).

### Training budget
50,000 environment steps or 5 minutes (300 s), whichever comes first. The notebook's `MAX_STEPS` and `MAX_TIME` use these same values.

## 3. What your `train()` must do

```python
def train(env, agent_policy, config, seed):
    set_seed(seed)  # makes your run reproducible
    ...  # your learning algorithm
    return agent_policy, {}  # the trained policy + a dict of saved paths
```

- Improve the `agent_policy` you are given and **return it**. One `PolicyNetwork` is shared by all agents.
- **Stop by yourself** before `config.max_steps` environment steps or `config.max_time` seconds. Count steps in your own loop (as `cem.py` does).
- Save anything you want to keep (checkpoints, plots) under `outputs/`.

## 4. How you are graded

1. Your algorithm is trained **3 times** with different seeds, each in its own isolated container.
2. Each of the 3 trained policies is tested on **10 fixed episodes of unseen data**, always using its **mean action** (no random exploration).
3. The score of one policy is the mean over those episodes of the **total reward, summed over all agents**. Your group's score is the mean of the 3 policies. Higher is better, and groups are ranked by it.
4. If a seed fails (crash, out of time, NaN weights), it is left out of your mean **and your score is penalised**, so make sure every seed runs.

## 5. Key ideas

- **Shared policy (parameter sharing).** One network is used by every agent. Agents still behave differently because their observations differ: besides time-varying signals, each agent observes its own heat pump, PV and EV battery parameters (capacities, efficiencies, charging limits, COP, consumption). CEM and REINFORCE both update this one policy using the experience of all agents.
- **Joint rollouts.** The dynamics depend on everyone's actions, so an episode is always run with all agents acting together. Rewards come back per agent. Your algorithm decides how to combine them (for example by summing them).
- **Why it matters.** Evaluating agents one at a time would give misleading rewards, because the other agents would behave differently.

## 6. If something goes wrong

- **The first build is slow.** It downloads the Jupyter base image, CPU-only PyTorch and the coursework dependencies. Later builds reuse Docker's cache.
- **Port 8888 is busy.** Use another port: `JUPYTER_PORT=8899 docker compose up`, then open <http://localhost:8899>.
- **`export_submission` complains.** Your file must define both `AlgorithmConfig` and `train`.
- **The repository was updated.** Save your algorithm, run `git pull`, and then run `docker compose build`.
- **My variables are gone after a break.** Expected — see the note in [Section 1](#1-start-the-coursework). Files are safe; kernel memory isn't. *Restart & Run All*.
