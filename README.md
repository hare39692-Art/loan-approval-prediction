# Loan Approval Prediction

A machine learning model that predicts whether a loan application will be **approved** or **rejected**, based on the applicant's income, credit score, loan amount, employment history, and internal evaluation points.

## 📊 Dataset

`loan_approval.csv` contains **2,000 applicant records** with the following columns:

| Column | Description |
|---|---|
| `name` | Applicant name *(dropped before training — not predictive)* |
| `city` | Applicant city *(dropped before training — not predictive)* |
| `income` | Annual income (~30,000–150,000) |
| `credit_score` | Credit score (300–850) |
| `loan_amount` | Requested loan amount (1,000–50,000) |
| `years_employed` | Years in current employment (0–40) |
| `points` | Internal evaluation score |
| `loan_approved` | **Target** — 0 = Rejected, 1 = Approved |

The dataset is fairly balanced (1,121 rejected vs. 879 approved) with no missing values or duplicate rows.

## 🧠 Model

**Algorithm:** Logistic Regression (`scikit-learn`)

**Pipeline:**
1. Load and inspect the dataset (shape, types, missing values, duplicates)
2. Exploratory Data Analysis — distribution plots, boxplots, correlation heatmap
3. Drop non-predictive columns (`name`, `city`)
4. Train/test split — 80% train, 20% test (`random_state=42`)
5. Feature scaling with `StandardScaler`
6. Train the Logistic Regression model
7. Generate predictions and approval probabilities
8. Evaluate with Accuracy, Confusion Matrix, Classification Report, and ROC curve

## 📈 Key Findings

Correlation with `loan_approved`:

- **Points** (0.82) — strongest predictor
- **Credit Score** (0.72) — strong predictor
- **Income** (0.24) — weak positive influence
- **Years Employed** (0.10) — very weak influence
- **Loan Amount** (−0.16) — weak negative influence (larger loans slightly reduce approval odds)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the notebook
1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```
2. Open `Loan_approval_model.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab
3. Make sure `loan_approval.csv` is in the same directory (or update the file path in the notebook — it currently reads from `/content/loan_approval.csv`, which is Colab's default upload path)
4. Run all cells top to bottom

## 📁 Repository Structure

```
loan-approval-prediction/
├── Loan_approval_model.ipynb   # Full training & evaluation pipeline
├── loan_approval.csv           # Dataset
└── README.md
```

## ✅ Results

The model is evaluated on the held-out 20% test set using accuracy, a confusion matrix, a full classification report, and ROC AUC. See the notebook's **Task 8** section for the exact scores from the latest run.

## 🔮 Example Usage

The notebook includes a sample inference block that scores new applicants directly:

```python
# Order: [Income, Credit_score, Loan_amount, Years_employed, Points]
example_inputs = np.array([
    [113810, 389, 39698, 27, 50],
    [44592, 729, 15446, 28, 55],
    [33278, 584, 11189, 13, 45],
])

example_scaled = scaler.transform(example_inputs)
predictions = model.predict(example_scaled)
probabilities = model.predict_proba(example_scaled)[:, 1]
```

## 📄 Conclusion

This project demonstrates that Logistic Regression can effectively predict loan approval outcomes from applicant financial data, with `points` and `credit_score` emerging as the strongest predictors. Such a model could help financial institutions automate initial loan screening and reduce manual review effort.

