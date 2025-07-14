# TabNet Customer Churn Prediction

## Project Overview

This project focuses on building and optimizing a customer churn prediction model using TabNet, a deep learning architecture specifically designed for tabular data. Customer churn is a critical problem for many businesses, as retaining existing customers is often more cost-effective than acquiring new ones. By accurately predicting which customers are likely to churn, businesses can implement targeted retention strategies.

This repository contains the code for data preprocessing, feature engineering, TabNet model training, evaluation, and hyperparameter tuning, with a specific focus on handling class imbalance common in churn datasets.

## Key Features

* **Data Preprocessing:** Handles missing values, performs feature engineering (e.g., `AvgChargesPerMonth`, `TotalServices`), and encodes categorical features.
* **Class Imbalance Handling:** Implements **SMOTE (Synthetic Minority Over-sampling Technique)** to address the imbalanced nature of churn data, improving the model's ability to identify the minority class (churners).
* **TabNet Model:** Utilizes the TabNet architecture from `pytorch-tabnet` for its ability to learn interpretable feature importance and achieve strong performance on tabular data.
* **Hyperparameter Tuning:** Includes an optimized set of hyperparameters for TabNet tailored to the dataset.
* **Comprehensive Evaluation:** Provides standard classification metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) and visualizations (Confusion Matrix, ROC Curve, Feature Importances).
* **VS Code Integration:** Designed for a smooth workflow within Visual Studio Code.

## Project Structure

.
├── tele_data.csv             # Raw dataset (or path to it)
├── TabNet_Churn_Prediction.ipynb  # Jupyter Notebook with the full code
├── README.md                 # This README file
└── .gitignore                # Specifies intentionally untracked files


## Installation

To run this project locally, you'll need Python 3.x and the following libraries. It's recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Anandu231/telecomchurn-prediction.git](https://github.com/Anandu231/telecomchurn-prediction.git)
    cd telecomchurn-prediction
    ```
    (Replace `Anandu231` with your actual GitHub username if different.)

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn pytorch-tabnet imbalanced-learn torch
    ```

## Usage

1.  **Place your dataset:** Ensure your `tele_data.csv` file is in the root directory of the cloned repository, or update the path in the notebook/script accordingly.
2.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook TabNet_Churn_Prediction.ipynb
    ```
    This will open the notebook in your web browser.
3.  **Run all cells:** Execute all cells in the notebook sequentially to perform data loading, preprocessing, model training, and evaluation.

## Model Performance

The model's performance metrics after improvements (including SMOTE and hyperparameter tuning) are reported directly in the notebook output. The primary goal was to improve the Recall and F1-Score for the churn class (Class 1) without significantly compromising overall accuracy.

## Results Highlights

* Improved ability to identify actual churners (increased Recall for class 1).
* Balanced F1-Score, considering both precision and recall.
* Robust ROC AUC score indicating good separability of classes.
* Visualizations (Confusion Matrix, ROC Curve, Feature Importances) provide insights into model behavior and key drivers of churn.

## Future Work

* **Advanced Feature Engineering:** Explore more complex interaction features or external data sources.
* **Advanced Imbalance Techniques:** Experiment with different `imbalanced-learn` strategies (e.g., SMOTE-ENN, ADASYN) or custom loss functions.
* **Hyperparameter Optimization Frameworks:** Utilize tools like Optuna or Hyperopt for more exhaustive and efficient hyperparameter tuning.
* **Model Explainability (LIME/SHAP):** Implement techniques to further interpret individual predictions from the TabNet model.
* **Deployment:** Explore deploying the model using frameworks like Flask/FastAPI for real-time predictions.