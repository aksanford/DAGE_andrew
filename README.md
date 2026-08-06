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
| `index.qmd` | Primary project manuscript. Start here. |
| `contributions.md` | Description of each group member's individual contributions. |
| `memos/` | Individual research memos. |
| `scripts/` | Data-cleaning, modeling, evaluation, and supporting analysis scripts. |
| `results/` | Saved model results, tables, figures, and other generated outputs. |
| `data_org/` | Original or raw project data. |
| `data_clean/` | Cleaned and processed data used in the analyses. |
| `references.bib` | Shared BibTeX bibliography for the manuscript and memos. |
| `requirements.txt` | Python package requirements. |
| `.env.example` | Example environment-variable file. |
| `_quarto.yml` | Project-level Quarto configuration. |
| `DAGE.Rproj` | RStudio project file. |


## Reproducing the Project

### 1. Clone the repository

```bash
git clone https://github.com/grantmooslin/DAGE.git
cd DAGE
```

### 2. Install the required Python packages

```bash
python3 -m pip install -r requirements.txt
```

### 3. Confirm that the required data files are available

The analysis expects the necessary files to be available in `data_org/` and `data_clean/`. Some data files may need to be added locally before the manuscript can be rendered.

### 4. Render the primary manuscript

From the root of the repository, run:

```bash
quarto render index.qmd
```

To preview the manuscript while editing, run:

```bash
quarto preview index.qmd
```

## Notes

DAGE is life\
DAGE stands for Data Analysis is Great and Excellent\
Or Dasey, Andrew, Grant, Emily.
