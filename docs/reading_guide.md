# Reading guide

What to take from each paper **for this project**. These are reading prompts, not summaries to
cite: open every paper yourself, check its title, authors and venue, and write your own notes
(brief, Section 3). Use the template at the bottom for Group B summaries.

## Group A (everyone, week 1)

**Grinsztajn et al. 2021, "There Is No Turning Back" (NeurIPS).**
How do they define and estimate reversibility without a simulator? Compare their definition with
our "restore the exact state within H steps". Look for: the precedence estimator idea, and how they
use reversibility to avoid or penalise irreversible actions. Our Related Work contrasts *designing*
reversibility-aware agents with *measuring* whether LLM agents already are.

**Alain & Bengio 2016, linear classifier probes.**
Why probe every layer; what probe accuracy means and does not mean. Justifies our per-layer plot.

**Hewitt & Liang 2019, control tasks (EMNLP-IJCNLP).** Read carefully.
What a control task is, why high accuracy alone is uninformative, the definition of selectivity,
and how probe capacity (regularisation) affects selectivity. Our control task gives each action
verb a random fixed label. Be ready to explain this in the presentation.

**Ruan et al. 2024, ToolEmu (ICLR).**
How test cases are written (underspecification, potential risky actions), and why we do NOT run
their emulator (it calls GPT-4 by default; the authors put the cost at about USD 1.2 per case).
We reuse only the 144 curated case descriptions.

## Group B (divided, week 2)

- **Krakovna et al., stepwise relative reachability**: reachability as a side-effect measure; the
  relationship between "irreversible" and "reduces reachability". Cite the current title.
- **Turner et al. 2020, attainable utility preservation (AIES)**: a different notion of side
  effects: losing the ability to do other things. Useful contrast with exact-state reversibility.
- **Yin et al., SafeAgentBench**: what "unsafe" covers, how they score it, and the numbers *in the
  version you read* (they differ across arXiv versions).
- **Li et al. 2023, Othello-GPT (ICLR)**: the representation was found to be non-linear
  (later work found a linear one in a different basis). Why a failed linear probe is not proof of absence.
- **Rimsky et al. 2024, CAA (ACL)**: how the steering vector is built (mean difference of
  activations on contrastive pairs), which layers, how they choose strengths, how they check
  that steering doesn't just break the model. Our RQ4 follows this.
- **Tang et al. 2025 (COLM)**: closest work. Exactly what gap they measure, how, on which agents.
  Our paper must state clearly how ours differs (reversibility specifically, exact labels,
  minimal pairs, internal representations, steering).
- **Schmied et al. (ICLR 2026)**: greediness / knowing-doing gap in LLM decision making;
  a possible explanation for taking fast irreversible shortcuts.
- **Wang et al. 2024, LLMs as text-based world simulators (ACL short)**: the ByteSized32 state
  prediction data used in RQ5; the action vs tick state split.

## Also search arXiv (week 1, and again before submission)
Queries: "reversibility LLM agent", "irreversible actions language model agent",
"probing agent hidden states safety", "knowing-doing gap LLM agents". If someone has published
this exact experiment, tell the professor immediately (brief, Section 3).

## Summary template (one page)
1. Citation (checked against the PDF): authors, title, venue, year, arXiv version.
2. The question they ask, in one sentence.
3. Method in three bullets.
4. Main result (with the number and the table it comes from).
5. What we reuse or contrast with, and where in our paper it goes.
6. One limitation or open question relevant to us.
