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

## Student 1: [Grant Mooslin] (`grantmooslin`)

- The component I worked on can be best described as the doing EDA on the words for the given classes in the dataset.
- You can find this contribution in a file called `eda_words.qmd` at lines 1-50.
- Owning this component means I created the word frequency analysis and visualizations to understand the most common words associated with each class.
- The commits or PRs that are most relevant are the commits that created the `eda_words.qmd` file and the commits that added the visualizations.
- The portion of the data science process that this effort contributes to is stage 3, **Explore + Visualize** — creating visualizations to understand the data and identify patterns.
- Advised future tests of the models based on the words that were most indicative of each class.
- tested the model utilizing feature abalation when removing the most common words on "bus" and "walking" which found out that reduced the accuracy of the model down to random chance.

---

## Student 2: [Emily Huffaker] (`emilyhuffaker')

-   **The component I "owned" and that I summarize here is best described as**
- **You can find this contribution in a file called** `filename.qmd` **at lines** XX–YY.
- **Owning this component means**
- **The commits or PRs that are most relevant are**
- **The portion of the data science process that this effort contributes to is**

---

## Student 3: [Andrew Sanford] (`aksanford`)

- **The component I "owned" and that I summarize here is best described as** the cleaning script/notebook that prepares raw data for modelling. This included joining the 3 separate raw data files, basic cleaning and engineering our categorical outcome variables based upon a comparison of composite scores from the PRCA and transportation questions.
- **You can find this contribution in a file called** `clean_data.qmd` within the DAGE/scripts/ directory. [Permalink to 'clean_data.qmd'] (https://github.com/grantmooslin/DAGE/blob/main/scripts/clean_data.qmd)
- **Owning this component means** that I authored the entirety of this component individually.
- **The commits or PRs that are most relevant are** [#11 - Add cleaning and EDA scripts](https://github.com/grantmooslin/DAGE/pull/11) where the component was added to the repo.
- **The portions of the data science process that this effort contributes to are** stage 1, **data acquisition and ingestion** and stage 2, **cleaning and validation** - This script is an essential first step for all team members to execute and generate usable data files for modelling. 

---

## Student 4: [Dasey Dang] (`daseydang`)

- **The component I "owned" and that I summarize here is best described as** the neural-network modeling pipeline used to predict transportation-use category from participants’ open-ended transportation responses. I developed the final binary classification workflow, including text vectorization, train-validation-test splitting, model training, class weighting, threshold selection, and evaluation using accuracy, precision, recall, F1, and confusion matrices.
- **You can find this contribution in files called** `nn_publicvsother.py` and `nn_publicvsrideshare.py` within the `DAGE/scripts/nn_trans` directory, as well as in the Dominant Transportation Methods subsection of index.qmd
- **Owning this component means** I adapted the group’s initial neural-network approach into a working end-to-end modeling script, using `aksanford`’s `proj_bible_example` as a reference. I implemented the text-vectorization and embedding pipeline, created reproducible stratified 70/15/15 data splits, converted the transportation outcome into a binary target, addressed class imbalance through class weights, and evaluated the model on held-out validation and test data. I also embedded the executable model code and dynamically generated results into the Quarto manuscript, including labeled and captioned accuracy curves, a performance table, a confusion matrix, and cross-referenced written results. Additionally, I debugged local and Colab execution issues involving file paths, TensorFlow environments, Quarto rendering, Python environment selection, and model-output handling.
- **The commits or PRs that are most relevant are** [#23 - Added NN for Public Transportation VS. RS/Other](https://github.com/grantmooslin/DAGE/commit/1479c42fbdb7189509e9cab1f595228cb3c2d6db) where the component was added to the repo.
- **The portion of the data science process that this effort contributes to is** stage 5, **Select + Apply**, and stage 6, **Check + Recheck**. The work involved selecting and applying a neural-network text-classification approach, then evaluating whether the model generalized to held-out validation and test data using macro/F1, accuracy, class-specific performance, and confusion matrices. It also contributes to reproducibility and implementation through the executable modeling script and the Colab/Devin workflow used to run the complete pipeline. Embedding the executable workflow and results into index.qmd also contributed to reproducibility by ensuring that the reported metrics and visualizations were generated directly from the modeling code rather than entered manually.

---

## Group sign-off

By adding your name below, each member affirms that the account of their own contribution is accurate, and that they have read the other four sections and believe them to be accurate as well.

- [X] [Grant Mooslin] (`grantmooslin`) — [8/6/26]
- [X] [Dasey Dang] (`daseydang`) — [8/6/26]
- [X] [Emily Huffaker] (`emilyhuffaker`) — [8/6/26]
- [X] [Andrew Sanford] (`aksanford`) — [8/6/26]
