# Project 2 — Applying an MLP to Adult Income (Team 14)

> **TL;DR**: We train and tune a compact Multilayer Perceptron (MLP) to predict whether an individual’s income exceeds **$50K** on the UCI Adult/Census dataset. 
> Our best model (two hidden layers) achieves about **0.855** mean CV accuracy and **0.881** ROC AUC on the dev split. 
> On the validation split, accuracy improves from **0.821** at the default 0.50 threshold to **0.831** at a tuned threshold near **0.78**. 
> All preprocessing is leakage-safe using a single scikit-learn **Pipeline**.

---

## 1) Assignment context

- **Course / Project**: DASC 41103 — Machine Learning, *Project 2: Applying MLP*  
- **Deliverables**: 15‑minute presentation, predictions CSV for the provided validation inputs, and a public GitHub repository with runnable code and a README.  
- **Validation output spec**: Name the file `Group_#_MLP_PredictedOutputs.csv` and map labels as **`1 if x=='>50K' else -1`**.

---

## 2) Data

- **Provided files**: 
  - `project_adult.csv` (training) 
  - `project_validation_inputs.csv` (unlabeled validation inputs)
- **Target**: Binary — `>50K` vs `≤50K` (class imbalance: negatives >> positives)

---

## 3) Method summary (what we did)

### Preprocessing (leakage-safe)
- Numeric: **median imputation → standardization**
- Categorical: **mode imputation → one‑hot encoding**
- Implemented with a single **`ColumnTransformer` + `Pipeline`** so transforms are fit **only** on training folds/splits.
- Final input dimensionality: **108 features** (7 numeric + 101 one‑hot).

### Model family & search
- Model: **MLPClassifier** (scikit‑learn)
- Tuning via **5‑fold cross‑validation** over hidden sizes, activation functions, L2 `alpha`, and learning rates.
- **Selected configuration**: hidden layers **(128, 64)**, activation **tanh**, **alpha = 0.001**, **batch_size = 64**, **learning_rate_init = 0.0005**, optimizer **Adam**, **early stopping** (`patience=10`).  
- Rationale: strongest CV performance, smooth optimization on standardized inputs, compact capacity for regularization.

### Overfitting & evaluation discipline
- Controls: **early stopping**, **L2** regularization, modest network size.
- Procedure: **cross‑validation** for model selection, **clean validation split** for final checks.
- Any class‑imbalance handling or resampling (if used) is **confined to training folds only**; validation remains untouched.
- We **tune the decision threshold** on validation probabilities rather than relying on the default 0.50.

### Fairness & responsible‑use plan
- Adult/Census contains sensitive or proxy‑sensitive attributes (e.g., sex, race, marital/relationship, occupation). 
- For responsible use we would: (1) audit **subgroup metrics** (precision/recall/FPR/FNR and AUC/PR), (2) check **calibration by subgroup**, (3) document limits in a brief **model card**, 
  (4) consider **reweighting/constrained training** or **group‑aware thresholds** if gaps persist, and (5) keep a **human‑in‑the‑loop** for high‑stakes decisions.

---

## 4) Results (dev/validation)

- **Cross‑validation** (model selection): mean CV accuracy ≈ **0.8546** with the (128, 64) tanh MLP.
- **Validation @ 0.50 threshold**: accuracy **0.8211**, ROC AUC **0.8806**.
- **Threshold tuning**: best near **t\* ≈ 0.78**, accuracy **0.8311**.
- **Supporting diagnostics** (dev): PR **Average Precision ≈ 0.707**, **KS ≈ 0.602**, and **reasonable calibration**.

**Interpretation.** Scores are well‑separated (bimodal histograms) and reasonably calibrated, so choosing a non‑default cutoff (~0.78) improves accuracy without brittle behavior. Depending on stakeholder goals, the threshold can be adjusted to trade precision vs. recall.

---

## 5) How to reproduce

> **Prereqs**: Python 3.10+  
> **Install**: `pip install -r requirements.txt`

**A. Run the notebook**
1. Launch Jupyter: `jupyter lab` *(or)* `jupyter notebook`.  
2. Open **`Project2_Team14_Notebook.ipynb`** and run all cells top‑to‑bottom.  
   - The notebook performs leakage‑safe preprocessing, runs CV over candidate MLPs, fits the selected model, evaluates on dev/validation, and writes predictions for the instructor’s validation inputs using the required label mapping (1 / −1).

**B. Produce the required CSV**
- The notebook writes **`Group_14_MLP_PredictedOutputs.csv`** with labels mapped as **`1 if x=='>50K' else -1`**.

---

## 6) Repository guide

- `Project2_Team14_Notebook.ipynb` — end‑to‑end pipeline, CV search, final training, evaluation, and prediction generation.  
- `Project2_Team14_Presentation.pptx` — 15‑minute slide deck (≤2 images/slide; speaker‑notes friendly).  

---

## 7) Key findings (one screen)

- Leakage‑safe preprocessing in a single Pipeline; **108 features** after encoding.  
- Best MLP: **(128, 64)**, **tanh**, **alpha=0.001**, **batch=64**, **lr=0.0005**, **Adam**, **early stopping**.  
- **CV ≈ 0.8546**, **Val AUC ≈ 0.8806**, **Val Acc 0.8211 → 0.8311** at **t\* ≈ 0.78**.  
- Supporting diagnostics: **AP ≈ 0.707**, **KS ≈ 0.602**, and good **calibration**.

---

## 8) Responsible‑use note

This dataset contains sensitive or proxy‑sensitive attributes. Any real‑world use should include subgroup error analysis, calibration by group, documentation of intended use and limits, and human review for high‑stakes actions. Our next step would be to run a subgroup audit on the trained model.

---

## 9) Team

**Team 14 — Billy Buckey & Jackson Green**
