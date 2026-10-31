# Start here: the project, one step at a time

This guide walks through the project in the order you will actually do it. Each step says
**what** you are doing, **why** it matters for the research question, **which file** does it,
and **what to check** before moving on. Do not skip the checks; most of the marks are for controls.

---

## The question in one paragraph

An agent "knows" deleting is permanent if (a) you can read that information out of its hidden
states, or (b) it says so when asked. It "uses" that knowledge if it behaves more carefully when
an action is irreversible. The project measures all three on the same items and reports the
difference between knowing and acting as one number: **Gap = K - A**.

---

## Step 0 - Setup and smoke test (day 1-2)

1. Apply for Llama-3.1 access on Hugging Face **today**. Approval can take days.
2. Install (README "Setup"). Run `python -m tests.run_all`. All tests must pass.
3. Run `python scripts/demo_synthetic.py`. It runs the whole pipeline on fake data and makes
   all the figures with a SYNTHETIC watermark. Open `results/figures_demo/` to see what each
   result will look like. **Never report anything from `results/demo/`.**
4. On the GPU: `python scripts/00_smoke_test.py --model qwen7b`. It checks that we read the
   right tokens, that hidden states have the right shape, and that steering hooks work.
   If it fails, the library version probably changed something. Fix this before anything else.

## Step 1 - The sandbox and the labels (week 1-2)

**What.** `irrev/sandbox/env.py` is a fake workspace (files, trash, backups, version history,
email outbox, wallet) with 23 tools. Four settings decide reversibility: trash on/off, version
history on/off, email undo window or not, transfers instant / pending / pending with a fee.

**How labels are made.** `irrev/labeling/bfs.py`: save the state, apply the action, then search
all action sequences of length up to 3 for a way back. If the exact state comes back, the label is
*reversible*. If only an approximate state comes back (version counter moved, fee lost), it's
*partially reversible*. Otherwise it's *irreversible*. The search only uses information the agent
would have, so it can't "retype" text it never saw.

**Why minimal pairs.** `irrev/data/gen_sandbox.py` makes groups where the *same* action
(`delete_file docs/a.txt`) appears in states that differ by one setting, and gets different labels.
If a probe only learned "delete = dangerous", it would fail on these. This is the core of the design.

**Check.**
- `python -m irrev.data.gen_sandbox` prints class counts and the minimal-pair share (needs >= 30%; we get ~60%).
- Open `data/sandbox_pairs.jsonl` and read three minimal pairs yourself. The state texts should
  differ in exactly one settings line.
- Discuss as a team: do you agree with every label rule? (e.g. "backup_file over an older backup is
  irreversible because the old backup is lost.") Write disagreements down for the Limitations section.

## Step 2 - ScienceWorld labels (week 2-3)

**What.** `irrev/data/scienceworld_label.py` uses the same labeller. ScienceWorld has no
save/restore, so we restore by reloading the task and replaying the action history. That is slow,
so we only label a sample of states.

**Check.**
- Run a pilot first: `--tasks boil --variations 1 --states-per-episode 2`, and time it.
  Then scale up so the total dataset exceeds 2,000 pairs.
- Look at the `volatile` field. ScienceWorld time passes on every step, so some states change on
  their own. Report results with and without volatile states.
- The real API might differ from what the code expects (`get_possible_objects`, the "Ambiguous"
  prompt). If something breaks, read `scienceworld/scienceworld.py` in the installed package.

Then run `python -m irrev.data.build ...` to merge everything and assign the **scenario-level**
test split (20%). Never split by individual action, or the probe memorises scenarios.

**Quality control.** `python -m irrev.data.agreement sample ...` makes a 100-row sheet with the
automatic labels hidden. Two people label it **separately**. Then `... score` gives Cohen's kappa.

## Step 3 - Hidden states (week 3)

`python -m irrev.models.extract --model qwen7b`. One forward pass per pair. It saves the hidden
state of the **last token of the action** at every layer, and the mean over action tokens
(ablation 1). It takes minutes, not hours. Do it once, then never touch the model for RQ1.

## Step 4 - Probes (RQ1, week 3-4)

`python -m irrev.probes.run --model qwen7b --task binary` trains one logistic regression per layer,
plus every control the brief requires:

| Control | What it rules out | What you want to see |
|---|---|---|
| majority class | nothing learned | 0.5 balanced accuracy |
| lexical (action text only) | probe just reads the verb | probe >> lexical **on minimal pairs** |
| lexical (state + action) | probe just reads setting words | probe > this too (be honest if not) |
| shuffled labels | probe overfits anything | ~0.5 |
| control task (verb -> random label) | probe memorises verbs | selectivity = probe - control, clearly > 0 |
| layer 0 | information already in embeddings | low |

Then run the ablations (the `probes` stage of `scripts/cluster/job.sh` does all of them).

**Important finding from testing the pipeline:** the *state + action* bag-of-words already scores
~0.68 on minimal pairs, because the settings text contains words like "cannot be cancelled".
This is the stricter bar your probe must clear, and the report must mention it.

## Step 5 - Ask the model (RQ2, week 4)

`python -m irrev.stated.ask` then `python -m irrev.analysis.compare_rq2`. You get the stated accuracy,
kappa between stated answer and probe, and a per-pair CSV. Look at where the probe is right and the
stated answer is wrong (or the reverse); those make good examples for the paper.

## Step 6 - Agent behaviour (RQ3, week 4-6)

`data/agent_tasks.jsonl`: 65 task pairs in 5 families. In each pair, the same tempting shortcut is
reversible in one version and irreversible in the other, and something in the workspace makes
care sensible (an "only copy", a "duplicate" that isn't one, a draft with an internal note, an
invoice that might already be paid).

Rehearse without a GPU: `python scripts/demo_agent.py --scripted reckless --pair cleanup_03`.

Run `python -m irrev.agent.run --model qwen7b` (and `--careful` for ablation 5),
`python -m irrev.agent.knowledge` (K), then `python -m irrev.analysis.rq3_gap` for the headline.

**Before you look at results**, read `docs/preregistration.md` together, agree on it, and commit
it to Git. The commit date is your proof that the analysis was decided in advance.

**If the agent never takes the shortcut** (nothing to measure), the brief says to make the shortcut
more tempting and add tasks. Edit the instructions in `irrev/agent/tasks.py`, and record the change.

## Step 7 - Second model, then bonus (week 6-8)

Repeat Steps 3-6 for Llama-3.1-8B. That gives you the **minimum publishable result**: probes with
all controls on two models and two testbeds, the stated comparison, the gap on 65 pairs, and the
dataset. Only then do RQ4 (steering) and RQ5 (uncertainty).

Steering: `scripts/cluster/job.sh steer qwen7b` (see the runbook, phase E). Causal evidence = careful behaviour rises with strength
along the probe direction, not along random directions, and task success doesn't collapse.

## Step 8 - Write up (week 8-10)

`docs/report/main.tex` is a drafted ACL paper. Methods and related work are written; every number
is a red placeholder, and results paragraphs offer alternatives. Keep only the one your data supports.
Fill `docs/data_card.md`, follow `docs/presentation_outline.md`, and each write one paragraph in
`docs/contribution_log.md`.
