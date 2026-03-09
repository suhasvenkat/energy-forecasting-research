<h1>⚡ Building Energy Forecasting Research — LSTM · Transformer · GNN · SSL</h1>

<p>
  <img src="https://img.shields.io/badge/Models-LSTM%20%7C%20Transformer%20%7C%20GNN%20%7C%20SSL-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-ASHRAE%20Energy-blue?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Research-MSc%20Data%20Science-purple?style=for-the-badge&logo=google-scholar&logoColor=white"/>
  <img src="https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</p>

Deep learning research project comparing LSTM, Transformer, Graph Neural Network (GNN), and
Self-Supervised Learning (SSL) architectures for building-level energy consumption forecasting.

Built as part of MSc Data Science research at the University of Europe for Applied Sciences, Berlin.

---

## 📌 What this project covers

5 research questions answered through systematic experiments on the ASHRAE Energy Consumption dataset:

| # | Research Question | Models Used |
|---|------------------|-------------|
| RQ1 | How do Transformer models compare with LSTM for multi-resolution energy forecasting under seasonal and occupancy variability? | LSTM, Transformer |
| RQ2 | Can GNN architectures modelling hierarchical building structures enhance node-level and aggregate forecasting accuracy? | GCN, LSTM, Transformer |
| RQ3 | What impact does self-supervised pretraining have on model robustness under data sparsity and sensor failure? | SSL + LSTM, SSL + Transformer |
| RQ4 | How do LSTM and Transformer models differ in temporal pattern learning for building-level energy consumption? | LSTM, Transformer |
| RQ5 | Do pretrained models exhibit more stable error behaviour under extreme sensor failure scenarios? | SSL models |

---

## 📊 Key Findings

**RQ1 — LSTM outperforms Transformer on short-horizon forecasting**
LSTM achieved lower MAE and produced smoother forecasts. Transformer showed higher variance during abrupt load changes and a heavier-tailed error distribution — suggesting it needs larger datasets or stronger regularisation for this task.

**RQ2 — GNNs improve aggregate forecasting but not node-level**
Hierarchical GCN outperformed temporal models at site-level aggregate forecasting by leveraging spatial building relationships. However, LSTM remained superior at node-level prediction where fine-grained temporal dynamics dominate.

**RQ3 — SSL pretraining significantly improves robustness**
Pretrained models outperformed from-scratch baselines across all conditions. The largest gains appeared under sensor failure — pretrained models maintained stable predictions where non-pretrained models degraded sharply. Transformer benefited more from pretraining than LSTM due to its higher representational capacity.

**RQ5 — Pretrained models show tighter error distributions under failure**
SSL-pretrained models produced markedly tighter error distributions with shorter high-error tails during extreme corruption scenarios, confirming improved prediction stability.

---

## 🗄️ Dataset

**ASHRAE Great Energy Predictor III** — Kaggle

| File | Description |
|------|-------------|
| `train.csv` | Hourly meter readings per building |
| `building_metadata.csv` | Building type, size, year built, site |
| `weather_train.csv` | Hourly weather data per site |

- **Scale:** Multiple buildings across 16 sites
- **Frequency:** Hourly timestamps
- **Features:** Air temperature, dew temperature, wind speed, cloud coverage, meter type

---

## 🧠 Model Architectures

### LSTM
- Sequence length: 24 hours
- Hidden units: 64
- Loss: MAE · Optimiser: Adam
- Baseline temporal model

### Transformer
- Multi-head self-attention
- Feed-forward layers with ReLU
- Layer normalisation
- Captures long-range temporal dependencies

### Hierarchical GCN (Graph Neural Network)
- Adjacency matrix built from site-level building relationships
- Graph convolution layers propagate information across related buildings
- Node-level + aggregate-level evaluation

### Self-Supervised (SSL) Pretraining
- Models pretrained using self-supervised reconstruction objectives
- Fine-tuned on labelled energy consumption data
- Evaluated under 3 conditions: clean · sparse · sensor failure

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

---

## 📁 Project structure
```
energy-forecasting-research/
├── deassignment.ipynb        # Full research notebook (all 5 RQs)
├── README.md
└── requirements.txt
```

---

## ▶️ How to run

**Option 1 — Kaggle (recommended, dataset already connected)**

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/suhasvenkat/deassignment)

**Option 2 — Local**
```bash
git clone https://github.com/suhasvenkat/energy-forecasting-research.git
cd energy-forecasting-research
pip install -r requirements.txt
jupyter notebook deassignment.ipynb
```

Download the dataset from [ASHRAE on Kaggle](https://www.kaggle.com/competitions/ashrae-energy-prediction/data) and update the file paths in Cell 2.

---

## 🔬 Experimental Design

- 10% stratified sample of buildings used for model training (reproducible with `random_seed=42`)
- Log-transformed meter readings (`log1p`) to handle skewed energy distributions
- 80/20 train/test split
- Sequence length of 24 timesteps (24-hour window)
- Evaluation metric: Mean Absolute Error (MAE) on log-scale predictions

---

## 🔮 What I'd add next

- [ ] MLflow experiment tracking across all RQs
- [ ] Hyperparameter tuning with Optuna
- [ ] Full dataset training (not just 10% sample)
- [ ] Physics-informed neural network (PINN) extension
- [ ] Streamlit dashboard for interactive forecast visualisation

---

## 🔗 Related

This notebook is supported by a separate data engineering pipeline that automates ingestion and preprocessing:

👉 [airflow-weather-pipeline](https://github.com/suhasvenkat/airflow-weather-pipeline) — Airflow DAG that loads and prepares the building + weather datasets used in this research.

---

## 👤 Author

**Suhas Venkat**
MSc Data Science — University of Europe for Applied Sciences, Berlin

[![LinkedIn](https://img.shields.io/badge/LinkedIn-suhas--venkat-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/suhas-venkat)
[![Kaggle](https://img.shields.io/badge/Kaggle-suhasvenkat-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/suhasvenkat)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white)](https://suhas-venkat-data-analys-xtfad0x.gamma.site/)
```

---

## Steps to create the repo

1. Go to github.com → **New repository**
2. Name it: `energy-forecasting-research`
3. Description: `LSTM · Transformer · GNN · SSL — deep learning research on building energy forecasting`
4. Set to **Public**
5. Upload `deassignment.ipynb` (rename it to something cleaner like `energy_forecasting_research.ipynb`)
6. Paste the README above
7. Create a `requirements.txt` with:
```
tensorflow
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyter
