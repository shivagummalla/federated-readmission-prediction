# Privacy-Preserving Personalized Federated Learning for 30-Day Hospital Readmission Prediction

## 📌 Project Overview

This project focuses on predicting 30-day hospital readmission using a privacy-preserving approach called Federated Learning. Instead of sharing sensitive patient data, multiple hospitals train models locally and share only model parameters.

We implement three approaches:
- **Local Model** (trained on individual hospital data)
- **Global Federated Model** (using FedAvg)
- **Personalized Federated Model** (global + fine-tuning)

The goal is to compare performance while maintaining data privacy in a non-IID healthcare setting.

---

## 📂 Repository Structure

```
federated-readmission-prediction/
│
├── federated_ml.ipynb             ← Main Jupyter notebook (run this)
├── diabetic_data.csv              ← Dataset (UCI Diabetes 130-US Hospitals)
├── README.md                      ← This file
│
├── results/
│   ├── results_summary.csv
│   ├── feature_importance.csv
│   ├── fig1_performance_comparison.png
│   ├── fig2_roc_curves.png
│   └── fig3_feature_importance.png
```

---

## 📊 Dataset

**Diabetes 130-US Hospitals Dataset**

| Detail | Info |
|--------|------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| Records | 8,000 patient encounters |
| Hospitals | 130 US hospitals (1999–2008) |
| Features | 15 clinical features |
| Target | 30-day readmission (1 = readmitted, 0 = not) |
| Readmission Rate | ~21.6% |

> This dataset is publicly available, fully de-identified, and approved for research use under UCI ML Repository terms. No patient identifiers are present.

---

## ⚙️ Setup Instructions

### 1. Requirements

Make sure you have Python installed (Python 3.8+ recommended).

Install required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
```

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/federated-readmission-prediction.git
cd federated-readmission-prediction
```

### 3. Download the Dataset

**Option A** — Use the CSV already in this repository (`diabetic_data.csv`)

**Option B** — Download the original dataset directly from UCI:
```
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
```
Extract and place `diabetic_data.csv` in the root folder of this repository.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook federated_ml.ipynb
```

> **Using Google Colab instead?**  
> Go to https://colab.research.google.com → File → Upload Notebook → select `federated_ml.ipynb`  
> Then upload `diabetic_data.csv` using the Files panel on the left sidebar.

---

## ▶️ How to Run the Code

### Option A — Jupyter Notebook (Recommended)

Open `federated_ml.ipynb` and run cells in order from top to bottom.

| Step | Cell Description | Output |
|------|-----------------|--------|
| Step 1 | Import libraries | Confirms all packages loaded |
| Step 2 | Load dataset | Shape, column names, class distribution |
| Step 3 | Preprocess data | Encoded features, standardised values |
| Step 4 | Split into 3 hospital silos | Non-IID partition + SMOTE applied locally |
| Step 5 | Train Local-Only models | Random Forest per hospital, metrics printed |
| Step 6 | Train Global Federated model | FedAvg MLP ensemble, metrics printed |
| Step 7 | Train Personalized Federated model | Fine-tuned MLP per hospital, metrics printed |
| Step 8 | Results summary table | All metrics across all configurations |
| Step 9 | Generate figures | 3 plots saved as PNG files |
| Step 10 | Save outputs | CSV files written to disk |

**In Jupyter:** Click `Kernel → Restart & Run All` to run everything at once.  
**In Google Colab:** Click `Runtime → Run all`.

---

### Option B — Python Script

```bash
python federated_ml_code.py
```

All outputs (metrics, figures, CSVs) will be saved to the `data/` folder automatically.

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║         LAYER 1 – DATA INGESTION & PREPARATION              ║
║  Patient Records (CSV / FHIR) → Feature Extraction          ║
║  → SMOTE Applied Locally (no data shared)                    ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    [Hospital A]  [Hospital B]  [Hospital C]
    Local Model   Local Model   Local Model
    (MLP)         (MLP)         (MLP)
          │            │            │
          └────────────┼────────────┘
                  Weights Only
                  (No Raw Data)
                       ▼
╔══════════════════════════════════════════════════════════════╗
║         LAYER 2 – FEDAVG AGGREGATION SERVER                 ║
║  Weighted Average of Model Weights → Global Model           ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║ Global Model Sent Back
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    [Hospital A]  [Hospital B]  [Hospital C]
    Fine-tune     Fine-tune     Fine-tune
    on local data on local data on local data
          │            │            │
          ▼            ▼            ▼
╔══════════════════════════════════════════════════════════════╗
║         LAYER 3 – PERSONALIZED PREDICTION MODEL             ║
║         30-Day Readmission: Yes (1) / No (0)                ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📈 Results Summary

### Per-Hospital Performance

| Hospital | Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|--------------|----------|-----------|--------|----------|---------|
| Hospital A | Local-Only | 0.7234 | 0.2697 | 0.1765 | 0.2133 | 0.5728 |
| Hospital A | Global Federated | 0.7453 | 0.3099 | 0.1618 | 0.2126 | 0.5318 |
| Hospital A | **Personalized Federated** | 0.6719 | 0.2597 | **0.2941** | **0.2759** | 0.5396 |
| Hospital B | Local-Only | 0.6406 | 0.1892 | 0.1826 | 0.1858 | 0.4872 |
| Hospital B | Global Federated | 0.7109 | 0.2381 | 0.1304 | 0.1685 | 0.4912 |
| Hospital B | **Personalized Federated** | 0.6133 | 0.2098 | **0.2609** | **0.2326** | 0.4693 |
| Hospital C | Local-Only | 0.7121 | 0.2203 | 0.1354 | 0.1677 | 0.5576 |
| Hospital C | Global Federated | 0.7254 | 0.1707 | 0.0729 | 0.1022 | 0.5054 |
| Hospital C | **Personalized Federated** | 0.6741 | 0.2768 | **0.3229** | **0.2981** | **0.5415** |

### Average Across All Hospitals

| Configuration | Avg Accuracy | Avg F1-Score | Avg ROC-AUC |
|--------------|-------------|-------------|------------|
| Local-Only | 0.6920 | 0.1889 | 0.5392 |
| Global Federated | 0.7272 | 0.1611 | 0.5095 |
| **Personalized Federated** | 0.6531 | **0.2689** | 0.5168 |

> **Key finding:** Personalized Federated achieves the best F1-Score — a **42.4% improvement** over Global Federated — by recovering recall through local fine-tuning, without any raw patient data leaving each hospital.

---

## 🔑 Key Concepts

| Term | Meaning |
|------|---------|
| **FedAvg** | Federated Averaging — aggregates local model weights proportionally by hospital sample size |
| **Non-IID** | Each hospital has a different patient population — makes federated learning harder and more realistic |
| **SMOTE** | Synthetic Minority Oversampling — fixes class imbalance by generating synthetic readmission cases locally |
| **Fine-tuning** | Taking the global model and training it further on each hospital's own local data |
| **ROC-AUC** | Measures how well the model separates readmitted vs. not-readmitted patients (1.0 = perfect) |
| **F1-Score** | Best metric for imbalanced data — harmonic mean of Precision and Recall |
| **L2 Regularisation** | Penalty on large weights during fine-tuning to prevent overfitting |
| **Early Stopping** | Halts training when validation performance stops improving |

---

## 🔒 Privacy Guarantee

> At **no point** in this framework are raw patient records transmitted between hospitals or to the aggregation server. Only floating-point **model weights** are shared. This is the core privacy guarantee of Federated Learning.

Additional protections in this framework:
- SMOTE is applied **locally** — no data sharing needed to fix class imbalance
- Fine-tuning is done **locally** — global model adapts without exposing local data
- L2 regularisation + early stopping prevent overfitting on small local datasets

---

## 📚 References

1. McMahan et al. (2017). FedAvg — https://arxiv.org/abs/1602.05629
2. Rajkomar et al. (2018). Deep Learning on EHRs — https://www.nature.com/articles/s41746-018-0029-1
3. Rieke et al. (2020). FL in Digital Health — https://www.nature.com/articles/s41746-020-00323-1
4. Fallah et al. (2020). PerFedAvg — https://arxiv.org/abs/2002.07948
5. Sinaci et al. (2024). FL on FAIR FHIR Data — https://www.sciencedirect.com/science/article/pii/S0010482524002634
6. Chawla et al. (2002). SMOTE — https://arxiv.org/abs/1106.1813
7. Lundberg & Lee (2017). SHAP — https://arxiv.org/abs/1705.07874
8. Wilkinson et al. (2016). FAIR Principles — https://www.nature.com/articles/sdata201618
9. Strack et al. (2014). Dataset Paper — https://pubmed.ncbi.nlm.nih.gov/24804245/
10. Dwork & Roth (2014). Differential Privacy — https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf

---

*Submitted for 22AIE213 – Machine Learning | Amrita School of Engineering*
