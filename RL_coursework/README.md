# ML4CE-RL: Reinforcement learning for a home energy microgrid

You will train **one policy** that controls the heat pump and EV battery of five households at once, so that together they spend less, stay comfortable, and wear out the battery less. You compare two ready-made algorithms (CEM and REINFORCE), then write **your own** and submit it. Your algorithm is graded automatically on data you have never seen.

## 1. Start in 3 steps

You only need [Docker](https://docs.docker.com/get-docker/). No Python installation. Put the `compose.student.yaml` file you were given in an empty folder and run, from that folder:

```bash
docker login <registry>           # once, only if your instructor says the registry is private
mkdir -p student-work outputs     # your workspace and your results
docker compose -f compose.student.yaml pull   # downloads the course image (a few GB, first time only)
docker compose -f compose.student.yaml up
```

Your instructor gives you the registry address and, if needed, a user name and a token for `docker login`.

Then open <http://localhost:8888> and enter the token `ml4ce-2026`. Open `RL_coursework.ipynb`.

Stop with `Ctrl+C`, then run `docker compose -f compose.student.yaml down`.

Your files stay on your computer:
- `student-work/` holds the notebook, `algorithms/` and the training data.
- `outputs/` holds trained policies, learning curves and your submission.

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

- **`pull access denied` or `unauthorized`.** Run `docker login <registry>` again and check the user name and token your instructor gave you.
- **Permission errors in `student-work/` or `outputs/`.** These folders were created by Docker as root. Delete them and create them yourself with `mkdir -p` before starting.
- **Port 8888 is busy.** Use another port: `JUPYTER_PORT=8899 docker compose -f compose.student.yaml up`, then open <http://localhost:8899>.
- **`export_submission` complains.** Your file must define both `AlgorithmConfig` and `train`.
- **A new course version was announced.** Run `docker compose -f compose.student.yaml pull` again (and use the new compose file or image tag if you were given one). Your existing `student-work/` is **not** updated automatically. To get the new notebook, first copy your own algorithm somewhere safe, then move the old `student-work/RL_coursework.ipynb` and `student-work/algorithms/` aside and start again.