# ShopSmart: Predicting E-Commerce Purchase Intent 🛍️

## 📌 Problem Statement
**ShopSmart**, an e-commerce platform, struggles to identify which website visitors are likely to make a purchase ("window shoppers" vs. "buyers"). This leads to inefficient marketing spend and lost revenue opportunities.

The goal of this project is to build a machine learning classification model to **predict user purchase intent** (Revenue: True/False) based on session behavior, such as time spent on product pages, bounce rates, and special day proximity.

## 📊 Key Metrics
* **Target Variable:** `Revenue` (Did the user buy? True/False)
* **Challenge:** The dataset is **highly imbalanced** (approx. 85% of users do not buy).
* **Success Metric:** **F1 Score** (Benchmarked > 0.55).
  * *Why F1?* Accuracy is misleading in imbalanced datasets. We need to balance Precision (minimizing false positives) and Recall (catching all actual buyers).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
* **Model:** Decision Tree Classifier
* **Techniques:** Grid Search CV, Feature Selection, Handling Imbalanced Data

## 📂 Dataset Overview
The dataset consists of **12,330 user sessions** with 18 features, including:
* **Behavioral:** `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`
* **Contextual:** `SpecialDay`, `Month`, `Weekend`
* **User Attributes:** `VisitorType` (New vs. Returning), `Region`, `TrafficType`


## ⚙️ Methodology
1.  **Data Preprocessing:**
    * Dropped noisy columns (`OperatingSystems`, `Browser`) that do not correlate with buying behavior.
    * Encoded categorical variables (`Month`, `VisitorType`) using Label Encoding and One-Hot Encoding.
    * Handled class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) and class weighting.

2.  **Exploratory Data Analysis (EDA):**
    * Identified that `PageValues` (average page value) is the strongest predictor of a purchase.
    * Visualized the "Weekend Effect" on shopping habits.

3.  **Model Training:**
    * Algorithm: **Decision Tree Classifier**
    * Hyperparameter Tuning: Used `GridSearchCV` to optimize `max_depth` (10) and `min_samples_split`.

## 🏆 Results
The final model significantly outperformed the project benchmark.

| Metric | Project Benchmark | **My Model Score** |
| :--- | :--- | :--- |
| **F1 Score** | 0.55 | **0.62** 🚀 |
| **Accuracy** | N/A | **89.1%** |

**Key Insight:**
The model successfully identifies potential buyers without overfitting to the majority class ("Non-Buyers"). The most critical feature driving the decision was `PageValues`.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/ShopSmart-Ecommerce-Prediction.git](https://github.com/YourUsername/ShopSmart-Ecommerce-Prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook ShopSmart_Purchase_Prediction.ipynb
    ```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
