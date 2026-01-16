# 🏦 Regulatory Credit Risk Scorecard

##  Project Overview
This project implements a comprehensive **Application Scorecard (A-Score)** using the Lending Club dataset (2007-2015). Unlike typical "black box" machine learning projects that simply output a binary prediction, this engine is built for **Regulatory Compliance** (Basel II/III standards).

It utilizes **Logistic Regression** with **Weight of Evidence (WoE)** binning to generate a transparent, interpretable, and FICO-scaled credit score (300-850). The focus was not just on predictive accuracy, but on **Explainability**, **Business Stability**, and **Leakage Prevention**.

---

##  Key Features

### 1. Transparent Scoring Engine
* **No Black Boxes:** Every input variable is transformed into a **Weight of Evidence (WoE)** value, ensuring the model's decisions are linearly interpretable.
* **FICO Calibration:** The raw probability of default is mathematically scaled to a standard 300-850 range using Industry Standard Logic (PDO = 50, Base Odds = 5:1).

### 2. Robust Feature Engineering
* **Monotonic Constraints:** Implemented "Coarse Classing" (Quintile Binning) to ensure that as risk factors increase (e.g., higher Debt-to-Income), the model's penalty strictly increases. This prevents logical violations.
* **Information Value (IV) Selection:** Features were selected based on their predictive power (IV > 0.02) while filtering out weak or redundant variables.

### 3. Production-Ready Deployment
* **Streamlit App:** A client-facing dashboard that simulates a Bank Loan Officer's interface.
* **Dynamic Binning:** The app includes a "Translator" layer that maps raw user inputs (e.g., "$65,000 Income") to the correct model bins in real-time using saved bin edges.

---

## Website Interface
The following dashboard demonstrates the final product, featuring dynamic input handling and the FICO-style gauge chart for risk visualization.

![Website Screenshot](https://github.com/HarshVishwakarma20/Regulatory-Credit-Scorecard/blob/main/chart4.png?raw=true)

## Visualizations & Model Performance

### 1. Target Imbalance
The dataset reflects real-world credit portfolios where defaults are the minority class (~15-20%). We maintained this natural prior during training to ensure calibrated probabilities.

![Target Distribution](https://github.com/HarshVishwakarma20/Regulatory-Credit-Scorecard/blob/main/chart1.png?raw=true)

### 2. Monotonic Risk Trends
A critical requirement for regulatory models is **Monotonicity**. As shown below, our binning logic ensures that the Weight of Evidence (risk signal) moves linearly across bins. This proves the model is learning the "True" relationship, not just noise.

![WoE Trend for feature : DTI](https://github.com/HarshVishwakarma20/Regulatory-Credit-Scorecard/blob/main/chart2a.png?raw=true)
![WoE Trend for feature : Term](https://github.com/HarshVishwakarma20/Regulatory-Credit-Scorecard/blob/main/chart2.png?raw=true)

### 3. Score Separation (The Result)
The final scorecard successfully discriminates between "Good" (Green) and "Bad" (Red) applicants. The clear separation between the two density peaks allows risk managers to set effective cut-off thresholds for loan approval.

![Score Distribution](https://github.com/HarshVishwakarma20/Regulatory-Credit-Scorecard/blob/main/chart3.png?raw=true)

---

## Challenges & Limitations

### 1. The "FICO" Leakage Trap
One of the biggest challenges was **Data Leakage**. The dataset contained a variable `last_fico_range_high` (the borrower's credit score *after* the loan closed).
* **The Trap:** Including this variable yielded a Gini score of ~0.90 (near perfect).
* **The Reality:** This is leakage; you don't know the future credit score when approving a new loan.
* **The Solution:** We rigorously removed all post-origination variables, accepting a lower but **honest** Gini of 0.40, which is the industry standard for Application Scorecards.

### 2. Binning Noise & Monotonicity
Initially, using Decile Binning (`q=10`) caused non-monotonic spikes in variables like `dti` (Debt-to-Income), where risk would randomly dip in the middle ranges.
* **The Fix:** We applied **Coarse Classing (q=5)**. By merging noisy bins, we stabilized the trend, ensuring the model adhered to economic logic.

### 3. Scorecard Calibration
Standard Logistic Regression outputs probabilities (0 to 1). Converting this to a score (300-850) required manual calibration.
* **Issue:** Initial settings caused the average score to be too low (~520).
* **Fix:** We adjusted the **Base Odds** assumption from 15:1 to 5:1 to match the actual portfolio risk, successfully centering the "Good" customers around a score of ~600.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Modeling:** Scikit-Learn (Logistic Regression, Class Weights)
* **Data Processing:** Pandas, NumPy (qcut, log-odds calculation)
* **Visualization:** Plotly (Gauge Charts), Seaborn (KDE Plots), Matplotlib
* **Deployment:** Streamlit (Web App Interface)

---

##  How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HarshVishwakarma20/Regulatory-Credit-Scorecard.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    streamlit run ui.py
    ```

---

##  Author
**Harsh Vishwakarma**
* 2nd Year Engineering Student & Data Science/ML Enthusiast
* https://www.linkedin.com/in/harsh-vishwakarma-407125395/
* https://github.com/HarshVishwakarma20
