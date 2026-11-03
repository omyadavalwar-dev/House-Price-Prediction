# House Price Prediction

A beginner-level machine learning project exploring a house-price dataset with Python, Pandas, NumPy, and scikit-learn.

> **Project status: Work in Progress**
>
> This repository is being developed step by step alongside a beginner machine-learning course. The current notebook contains data inspection, basic preprocessing, feature engineering, encoding, scaling, train-test splitting, and an initial regression model. The current model code uses `RandomForestRegressor`; it will be revised as the project is aligned with the Linear Regression topics covered in the course.

## Project files

- `House_Price.ipynb` — Jupyter Notebook containing the project workflow.
- `HousePricePrediction.csv` — Dataset used by the notebook (include it only if its source license permits redistribution).
- `requirements.txt` — Python packages required to run the notebook.
- `.gitignore` — Files and folders that should not be committed.

## Tools and libraries

- Python
- NumPy
- Pandas
- Jupyter Notebook
- scikit-learn

## How to run

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Keep `HousePricePrediction.csv` in the same folder as `House_Price.ipynb`.
4. Open the notebook in Jupyter and run the cells from top to bottom.

## Current workflow

1. Load the dataset.
2. Inspect the dataset using Pandas.
3. Check missing values.
4. Create additional features from year columns.
5. Encode selected categorical columns.
6. Scale selected numerical columns.
7. Split the data into training and testing sets.
8. Train an initial regression model.
9. Evaluate model predictions.

## Important notes

- The notebook is a learning project and is not yet a finalized or production-ready model.
- Review the missing-value handling and train-test preprocessing steps before treating the reported evaluation metrics as final.
- The current notebook's model implementation is `RandomForestRegressor`, even though the learning goal is to practice Linear Regression. This will be updated in a later project revision.
- If the dataset is obtained from a third-party source, check its license and attribution requirements before publishing the CSV publicly.

## Future improvements

- Align the model section with Linear Regression.
- Improve missing-value handling.
- Avoid data leakage by fitting preprocessing steps only on training data.
- Add clear evaluation results and conclusions.
