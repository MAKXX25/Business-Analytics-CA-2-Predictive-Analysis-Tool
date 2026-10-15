# Decision Intelligence Studio — Predictive Analysis Tool

An enterprise-grade, production-ready predictive modeling dashboard built with Python and Streamlit. Engineered with clean corporate typography, minimal design, and zero jargon.

---

## 🚀 Features

- **Data Ingestion & Profiling:** Upload datasets (.csv, .xlsx, .xls) up to 200MB or load realistic commercial sample data instantly.
- **Exploratory Data Analysis (EDA):** Interactive distributions, correlation heatmaps, scatter & line analysis, and multi-dimensional profiling.
- **Automated Machine Learning Engine:**
  - Automated multivariate regression and classification pipelines.
  - Model benchmarking (Random Forest, Gradient Boosting, Linear/Logistic models).
  - Preprocessing with missing value imputation, scaling, and one-hot encoding.
- **Real-Time Sensitivity & Scenario Simulation:** Interactive what-if scenario testing and parameter tuning with live prediction updates.
- **Executive Reporting & Exporting:** Download clean executive summaries, feature importances, and model metrics.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Spreadsheet Parsing:** [openpyxl](https://openpyxl.readthedocs.io/), [xlrd](https://xlrd.readthedocs.io/)

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/MAKXX25/Business-Analytics-CA-2-Predictive-Analysis-Tool.git
cd Business-Analytics-CA-2-Predictive-Analysis-Tool
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.
