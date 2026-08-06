# [DAGE Saves the Day]

**Group:** DAGE
**Course:** PSYCH 755, Summer 2026

> **Communication apprehension:**
How accurately can semantic features from participants’ open-ended responses predict their dominant type of communication apprehension?\
>**Transportation use:**
How accurately can semantic features from participants’ descriptions of their ideal travel predict whether their transportation use is public-transit dominant, rideshare dominant, or has no dominant mode? 

## Members

| Name | GitHub username |
|---|---|
| Grant Mooslin | grantmooslin |
| Emily Huffaker | emilyhuffaker |
| Andrew Sanford | aksanford |
| Dasey Dang | daseydang |

## Project Structure + Contents

| Path | Description |
|---|---|
| `index.qmd` | The primary manuscript. Start here. |
| `contributions.md` | Who owned what. |
| `memos/` | Individual research memos, one per member. |
| `references.bib` | Shared BibTeX file for the manuscript and memos. |
| `data_org/` | Houses original/raw data for the project |
| `data_clean/` | Houses cleaned or otherwise modified data for the project |
| `scripts/` | Analysis scripts and neural network models |
| `results/` | Model outputs and training histories |

## Environment Setup

### Prerequisites

- **Python 3.11** (required for TensorFlow compatibility)
- **R** (for data cleaning and EDA scripts)
- **Quarto** (for rendering the manuscript)

### Installing Python Dependencies

```bash
# Install Python packages from requirements.txt
pip3 install -r requirements.txt
```

The `requirements.txt` includes:
- numpy, pandas (data processing)
- scikit-learn (machine learning utilities)
- tensorflow (neural networks)
- matplotlib, seaborn (visualization)
- jupyter, ipython, pyyaml (notebook support)

### Installing Quarto

Visit [quarto.org](https://quarto.org/docs/get-started/) to install Quarto for your operating system.

### Data Setup

Place the following raw data files in `data_org/`:
- `PRCAQualtricsExport_FileC.csv`
- `PRCAProlificExport_FileA.csv`
- `PRCAProlificExport_FileB.csv`

**Important:** Do not commit data files to git. The `data_org/` directory should contain only the original raw data files.

## Reproducing this project

### Step 1: Clean the data

```bash
# Run the data cleaning script to generate cleaned datasets
quarto render scripts/eda/clean_data.qmd
```

This creates cleaned data files in `data_clean/`:
- `survey_clean.csv`
- `survey_transportation.csv`
- `survey_apprehension.csv`

### Step 2: Render the manuscript

```bash
# Set the Python environment for Quarto
export RETICULATE_PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

# Render the main manuscript
quarto render index.qmd
```

This will:
- Run the neural network models
- Generate figures and tables
- Create `index.html` in the project root

### Alternative: Use system Python default

If your system Python is at `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`, you can add the reticulate setup directly to `index.qmd` (already included) and simply run:

```bash
quarto render index.qmd
```

To preview the manuscript while editing, run:

```bash
quarto preview index.qmd
```

## Notes

- The neural network models use TensorFlow/Keras and may take several minutes to train during rendering
- Model outputs are cached in `results/` to speed up subsequent renders
- The `freeze: auto` setting in `index.qmd` caches code execution results when source code hasn't changed

DAGE is life\
DAGE stands for Data Analysis is Great and Excellent\
Or Dasey, Andrew, Grant, Emily.
