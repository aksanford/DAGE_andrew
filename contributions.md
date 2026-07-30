# Contributions

**Group:** [DAGE (Data Analysis is Great and Excellent)]
**Group Members:** [Dasey Dang, Andrew Sanford, Grant Mooslin, Emily Huffaker]
**Project:** [DAGE Saves the Day - Predicting Survey Results from Open-Text Responses]
**Repository:** [https://github.com/grantmooslin/DAGE]

---

## How to use this file

Each member of the group completes one section below. Fill in every bullet. Delete the
instructions in *italics* as you go, and delete any unused student sections at the bottom
if your group has fewer than five members.

Three rules:

1. **Components are not co-owned.** No two students may not claim the same component. If you and a partner pair-programmed something, decide who owned it and acknowledge the two person effort. Each person needs their own entry below.
2. **Everything here must be checkable.** We will follow your file paths and click your links. A claim we cannot verify does not count.
3. **Link to permanent URLs, not moving ones.** See the note on line numbers below.

> **Reminder — Markdown link syntax.** Write links as `[link text](https://example.com)`. The visible words go in the square brackets and the URL goes in the parentheses, with no space between the two. For example, `[Permalink to load.py](https://github.com/...)` renders as [Permalink to load.py](https://github.com/...). Bare URLs work too, but named links are easier to read.

### A note on line numbers and links

Line numbers go stale the moment someone edits the file above yours. So do links to a branch. Use GitHub **permalinks**, which pin to a specific commit and never move:

> Open the file on GitHub → click the line number (or drag to select a range) → press **`y`** to convert the URL to a permalink → copy.

A permalink looks like this. Note the 40-character commit SHA in the path:

```
https://github.com/ORG/REPO/blob/a3f2c1e9d4b7.../analysis/model.qmd#L112-L168
```

Not like this (this one rots):

```
https://github.com/ORG/REPO/blob/main/analysis/model.qmd#L112-L168
```

### The data science process

The last bullet in each section asks which portion of the data science process your work contributes to. Name the stage and be specific about your part in it ("data acquisition and ingestion," "cleaning and validation," "exploratory analysis," "modeling," "evaluation," "visualization and communication," "infrastructure and reproducibility.") If your component spans two stages, say so, and say which one it mostly lives in.

---

## Example (delete this section before submitting)

### Jane Doe (`jdoe-wisc`)

- **The component I "owned" and that I summarize here is best described as** the modeling and training or fit step — the KNN classifier that takes the cleaned feature matrix and predicts the target class, together with the grid search that chose the value of *K*.
- **You can find this contribution in a file called** `src/models/knn.py` **at lines** 1–98, and in `manuscript.qmd` at lines 112–168 where the model is described and the grid-search results table is generated. [Permalink to `knn.py`](https://github.com/ORG/REPO/blob/a3f2c1e.../src/models/knn.py#L1-L98)
- **Owning this component means** I chose KNN over the two other classifiers we considered (write-up in PR #21); I designed the *K*-grid (odd values from 3 to 51) and the 5-fold stratified cross-validation used to score each candidate; I wrote the fit-and-predict loop; and I debugged the scaling bug that was letting one high-variance feature dominate the distance metric. I did not build the feature matrix — Marcus did the wrangling, and he covers that in his section.
- **The commits or PRs that are most relevant are** [#21 — model selection write-up](https://github.com/ORG/REPO/pull/21), [#24 — KNN + grid search over *K*](https://github.com/ORG/REPO/pull/24), and [a3f2c1e — scale features before distance calc](https://github.com/ORG/REPO/commit/a3f2c1e).
- **The portion(s) of the [data science process](https://adamrossnelson.github.io/integsci375-public/readings/data_science_processes.html) that this effort contributes to is** stage 5, **Select + Apply** — reviewing candidate techniques, selecting KNN, and applying it with a grid search over *K* to produce the predictive model the manuscript reports on. It also has a foot in stage 6, **Check + Recheck**: the 5-fold cross-validation inside the grid search is where each candidate *K* is validated against held-out folds before the final *K* is chosen.

---

## Student 1: [Grant Mooslin] (`grantmooslin`)

- **The component I "owned" and that I summarize here is best described as** *[One or two sentences. Name the thing. "The ingestion layer," "the model comparison and cross-validation," "the figure pipeline." Not "I helped with the analysis."]*
- **You can find this contribution in a file called** `filename.qmd` **at lines** XX–YY. *[Add a permalink. List more than one file if the component genuinely spans them.]*
- **Owning this component means** *[What did owning it actually involve — designing it? writing it? debugging it? deciding between alternatives? Use verbs. If someone else contributed to it, say what they did and where they claim it.]*
- **The commits or PRs that are most relevant are** *[Two to four actual links. Not SHAs typed out as text — links we can click.]*
- **The portion of the data science process that this effort contributes to is** *[Name the stage. Say why your part of it mattered to the project's findings.]*

---

## Student 2: Emily Huffaker (`emilyhuffaker`)

- **The component I "owned" and that I summarize here is best described as** the neural-network activation-function comparison. I tested ReLU, tanh, Leaky ReLU, and GELU to determine which hidden-layer activation produced the strongest validation macro-F1 score when predicting dominant transportation mode from participants’ open-ended responses.

- **You can find this contribution in** `scripts/activation_functions_test.py`, where the activation functions are tested while holding the model architecture, Adam optimizer, data split, and training settings constant. The results are summarized in `memos/memos_emily/Emily_memo.pdf`, and the sources used in the memo were added to `references.bib`. [Permalink to activation-function script](https://github.com/emilyhuffaker/DAGE/blob/bf9a2c4f81ef2fba6d16339b107d047a3b3c3cde/scripts/activation_functions_test.py), [permalink to research memo](https://github.com/emilyhuffaker/DAGE/blob/63b878727fa64b2fdbde56dc8744a6693785bed6/memos/memos_emily/Emily_memo.pdf), and [permalink to bibliography]((https://github.com/emilyhuffaker/DAGE/blob/main/references.bib)).

- **Owning this component means** I adapted the group’s neural-network pipeline to compare four hidden-layer activation functions, kept the remaining modeling choices constant for a fair comparison, calculated validation macro-F1 and accuracy, and created a visual summary of the results. I also wrote the research memo and added the supporting sources to the project bibliography.

- **The commits or PRs that are most relevant are** [activation-function analysis commit](https://github.com/emilyhuffaker/DAGE/commit/bf9a2c4f81ef2fba6d16339b107d047a3b3c3cde), [research memo commit](https://github.com/emilyhuffaker/DAGE/commit/63b878727fa64b2fdbde56dc8744a6693785bed6), and [bibliography update commit](https://github.com/emilyhuffaker/DAGE/commit/1dd08fa2c827386abcf0c3a2661d262882cb9bf1).

- **The portions of the data science process that this effort contributes to are** stage 5, **Select + Apply**, and stage 6, **Check + Recheck**. The work primarily involved applying and comparing several neural-network activation functions, then evaluating their performance on held-out validation data using macro-F1 because the outcome classes were imbalanced. It also contributes to **visualization and communication** through the comparison figure, research memo, and supporting references.

---

## Student 3: [Andrew Sanford] (`aksanford`)

- **The component I "owned" and that I summarize here is best described as** the cleaning script/notebook that prepares raw data for modelling. This included joining the 3 separate raw data files, basic cleaning and engineering our categorical outcome variables based upon a comparison of composite scores from the PRCA and transportation questions.
- **You can find this contribution in a file called** `clean_data.qmd` within the DAGE/scripts/ directory. [Permalink to 'clean_data.qmd'] (https://github.com/grantmooslin/DAGE/blob/main/scripts/clean_data.qmd)
- **Owning this component means** that I authored the entirety of this component individually.
- **The commits or PRs that are most relevant are** [#11 - Add cleaning and EDA scripts](https://github.com/grantmooslin/DAGE/pull/11) where the component was added to the repo.
- **The portions of the data science process that this effort contributes to are** stage 1, **data acquisition and ingestion** and stage 2, **cleaning and validation** - This script is an essential first step for all team members to execute and generate usable data files for modelling. 

---

## Student 4: [Dasey Dang] (`daseydang`)

- **The component I "owned" and that I summarize here is best described as** the neural network modelling script used to predict dominant transportation mode from open-ended responses.
- **You can find this contribution in a file called** `nn_model.py` within the `DAGE/scripts/` directory.
- **Owning this component means** that I adapted and implemented the modelling workflow using `aksanford`’s `proj_bible_example` as a reference.
- **The commits or PRs that are most relevant are** [insert commit or PR link].
- **The portion of the data science process that this effort contributes to is** stage 4, **modelling**, and stage 5, **evaluation**.

---

## Group sign-off

By adding your name below, each member affirms that the account of their own contribution is accurate, and that they have read the other four sections and believe them to be accurate as well.

- [ ] [Full Name] (`github-username`) — [date]
- [ ] [Full Name] (`github-username`) — [date]
- [ ] [Full Name] (`github-username`) — [date]
- [ ] [Full Name] (`github-username`) — [date]
- [ ] [Full Name] (`github-username`) — [date]
