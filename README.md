<img width="1316" height="793" alt="Multi- class particle identification " src="https://github.com/user-attachments/assets/7a64ec76-ff34-4832-b41f-3b2437a64ddc" />





# Multi-Class Particle Classifier

**A reproducible notebook project** that builds and compares multiple machine-learning models to classify subatomic particles (Pion, Proton, Kaon, Electron) using detector features such as momentum and velocity.

---

## Overview 🧭

This project demonstrates end-to-end model development for a physics classification problem using a real dataset (original CSV referenced in the notebook: `/content/drive/My Drive/pid-5M.csv`). The notebook:

Data Set

This Large dataset is from Kaggle (https://www.kaggle.com/datasets/naharrison/particle-identification-from-detector-responses) ,a simulation of electron-proton inelastic scattering measured by a particle detector system.

* Loads and samples the data (50,000 rows sampled from the full dataset)
* Performs exploratory data analysis (correlations, scatter plots)
* Trains five classifiers and a neural network
* Evaluates models on a held-out test set (10,000 particles)
* Analyses per-class performance and highlights the effect of class imbalance

---

## Quick Results Snapshot 🏆

**Final Test Accuracy (10,000-particle test set):**

| Model               |  Accuracy  |
| :------------------ | :--------: |
| **Random Forest**   | **96.85%** |
| **Neural Network**  | **96.70%** |
| K-Nearest Neighbors |   95.62%   |
| Decision Tree       |   95.59%   |
| Logistic Regression |   93.20%   |

> **Winner (by overall accuracy):** Random Forest (n_estimators=200, criterion='entropy') — but see the detailed analysis below. ⚖️

---

## Dataset & Features 🧾

**Source (as referenced in the notebook):** `pid-5M.csv` (mounted from Google Drive in the Colab notebook).

**Sample used:** `data.sample(n=50000, random_state=42)`

**Columns / Features used** (as present in the notebook):

* `id` — particle id (target): `211`=Pion, `2212`=Proton, `321`=Kaon, `-11`=Electron
* `p` — momentum
* `theta` — polar angle
* `beta` — velocity / speed as fraction of c
* `nphe` — number of photoelectrons
* `ein`, `eout` — energy deposit measurements

**Label mapping used in the notebook:**

```python
labels_map = {211: 'Pion (π)', 2212: 'Proton (p)', 321: 'Kaon (K)', -11: 'Electron (e-)'}
```

**Train/Test split:** 80% train (40,000 samples), 20% test (10,000 samples) via `train_test_split(..., test_size=0.2, random_state=42)`.

**Scaling:** StandardScaler applied to models that benefit from scaling (Logistic Regression, KNN, Neural Network). Tree-based models (Decision Tree, Random Forest) trained on raw features.

---

## Models & Key Hyperparameters 🧩

All models were trained on the same 80/20 split for fair comparison.

* **Random Forest (Multi-Class)**

  * `RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)`
  * Trained on unscaled features.

* **Neural Network (Keras / TensorFlow)**

  * Architecture: `Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu) -> Dropout(0.3) -> Dense(4, softmax)`
  * Loss: `categorical_crossentropy` — Optimizer: `adam`
  * Training: `epochs=30`, `batch_size=64`, `validation_split=0.2`
  * Labels one-hot encoded for training.

* **K-Nearest Neighbors**

  * `KNeighborsClassifier(n_neighbors=4)` — trained on scaled features.

* **Decision Tree**

  * `DecisionTreeClassifier(criterion='entropy', random_state=0)`

* **Logistic Regression**

  * `LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)` — trained on scaled features.

---

## Important Findings — Detailed 🔎

> These are the **most critical** takeaways from the evaluation beyond the overall accuracy numbers.

### 1) The Bake-Off — raw accuracy table (see Quick Results Snapshot above)

Random Forest slightly edges out the Neural Network in *overall accuracy* (96.85% vs 96.70%).

### 2) The Imbalanced Data Problem — the real issue 🚨

**The per-class performance for the Random Forest** (winning model by accuracy) shows a strong imbalance-driven effect. The classification report for the Random Forest on the test set is summarized below (this is the exact report produced in the notebook):

| Particle     | Precision |  Recall  | F1-Score | Support (test count) |
| :----------- | :-------: | :------: | :------: | :------------------: |
| **Electron** |    0.89   | **0.28** |   0.42   |          29          |
| **Kaon**     |    0.75   | **0.72** |   0.73   |          425         |
| **Pion**     |    0.98   |   0.97   |   0.98   |         5669         |
| **Proton**   |    0.98   |   0.99   |   0.98   |         3877         |

**Interpretation:**

* The model is excellent at identifying **Pions** and **Protons** (the dominant classes), but it fails to recover the rare **Electrons** (only 28% recall) and misses a significant fraction of **Kaons**.
* This demonstrates the classic *accuracy paradox* where a high overall accuracy masks poor performance on rare but scientifically important classes.

### 3) Feature importance (why the models work) 📈

* Feature-importance plots (from `RFC.feature_importances_`) in the notebook strongly indicate that `beta` (velocity) and `p` (momentum) are dominant predictors — consistent with physics, since mass (and hence particle type) is related to momentum and velocity.

---

## Final Conclusion & Recommendations 🏁

**Bottom line:** ML can classify these subatomic particles with high overall accuracy (≈96.85%). However, for research goals where finding *rare* particles is the objective, raw accuracy is **not** the right metric.

**Key conclusion:**

* The Random Forest produces the best overall accuracy by focusing on majority classes. The Neural Network (96.70% accuracy) is likely more suited to discovery because it appears to be comparatively better at identifying the rarer classes (Electrons and Kaons) — and with hyperparameter tuning it may surpass the Random Forest on both overall and per-class metrics.

**Recommendations for future work:**

1. **Feature engineering** — compute a physics-inspired mass proxy (e.g., functions of `p` and `beta`) and add it as a feature. This could make class separation much easier.
2. **Handle class imbalance** — use oversampling (e.g., SMOTE), class-weighted loss functions (for the NN), or focal loss to give rare classes more influence during training.
3. **Model tuning & ensembling** — perform hyperparameter search for the Neural Network and Random Forest; consider ensembling models to boost rare-class recall.
4. **Evaluation metrics** — prioritize per-class recall / F1 for rare classes, use macro-averaged metrics, and track confusion matrices and ROC/PR curves per class.

---

## How to Reproduce ✅

Clone or open the Jupyter notebook: `Multi_Class_Particle_Classifier_with_Machine_Learning.ipynb` and ensure the dataset `pid-5M.csv` is available.

### Dependencies (tested):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Run in Google Colab (recommended):

1. Upload `pid-5M.csv` to your Google Drive.
2. Open the notebook in Colab (it already contains `drive.mount('/content/drive')` and expects `file_path = '/content/drive/My Drive/pid-5M.csv'`).
3. Execute cells sequentially. The notebook will sample 50k rows and produce plots, training logs, and final evaluation tables.

### Quick commands (local):

* Run the notebook with `jupyter lab` or `jupyter notebook`.
* If running locally and memory is constrained, change `data.sample(n=50000, ...)` to a smaller sample.

---

## File Structure (as expected)

```
Multi_Class_Particle_Classifier_with_Machine_Learning.ipynb
pid-5M.csv
README.md   
```

---

## Notes & Caveats ⚠️

* The notebook uses `data.sample(n=50000, random_state=42)` so results will be reproducible with the same random seed but could vary if different sampling is used.
* Some code cells use trees without scaling (correct approach), and models sensitive to scale use `StandardScaler` prior to fitting.
* The notebook trains the NN for 30 epochs — results may improve with more epochs, learning-rate scheduling, and architecture search.

---

## Contact / Credits ✉️

Project author: Nitesh Kumar ( https://github.com/niteshg97 )

