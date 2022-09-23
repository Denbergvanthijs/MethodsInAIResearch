# MAIR-G8

This project consists of the following baseline systems:

- A baseline system that, regardless of the content of the utterance, always assigns the majority class of in the data. (baseline_majority.py)
- A baseline rule-based system based on keyword matching. (baseline2.py)

As well as three different machine learning classifiers:

- kNN (knn.py)
- Logistic regression (logreg.py)
- Decision trees (lsvc.py)

The table below shows the accuracy value of each of these models.
![Accuracy](https://user-images.githubusercontent.com/113440512/190705271-9dd0c622-f359-4fe5-a8f8-ff54bd6ae8c9.png)

## Installation

Place the required data files in the `./data/` folder.

Install the following packages on Python 3.8+:

```bash
pip install pandas sklearn spacy python-levenshtein nltk
python -m spacy download en_core_web_sm
```

## Development

Install the following for the correct linter and formatter:

```bash
pip install autopep8 flake8 flake8-import-order flake8-blind-except flake8-builtins flake8-docstrings flake8-rst-docstrings flake8-logging-format
```

## Limitations

- Dontcares only work if said after states 2, 3, or 4. Preferably, the system automatically figures out to what slot dontcares belong to
