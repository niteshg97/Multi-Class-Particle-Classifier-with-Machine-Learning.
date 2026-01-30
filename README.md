<img width="1316" height="793" alt="Multi-Class Particle Identification" src="https://github.com/user-attachments/assets/7a64ec76-ff34-4832-b41f-3b2437a64ddc" />

# Multi-Class Particle Classifier

**A reproducible notebook project** that builds and compares multiple machine-learning models to classify subatomic particles (Pion, Proton, Kaon, Electron) using detector features such as momentum and velocity.

---

## Overview 

This project demonstrates end-to-end model development for a physics classification problem using a real dataset (original CSV referenced in the notebook: `/content/drive/My Drive/pid-5M.csv`). The notebook:

Dataset Source:  
This large dataset is from Kaggle — [Particle Identification from Detector Responses](https://www.kaggle.com/datasets/naharrison/particle-identification-from-detector-responses), a simulation of electron-proton inelastic scattering measured by a particle detector system.

* Loads and samples the data (50,000 rows from the full dataset)
* Performs exploratory data analysis (correlations, scatter plots)
* Trains five classical ML models and one neural network
* Evaluates models on a held-out test set (10,000 particles)
* Analyzes per-class performance, highlighting class imbalance and rare-particle detection

---

##  Results  

**Final Test Accuracy (10,000-particle test set):**

| Model               |  Accuracy  |
| :------------------ | :--------: |
| **Random Forest**   | **96.85%** |
| **Neural Network**  | **96.65%** |
| K-Nearest Neighbors |   95.62%   |
| Decision Tree       |   95.59%   |
| Logistic Regression |   93.20%   |

> **Winner (by overall accuracy):** Random Forest — but this metric is misleading. See the detailed findings below. 

---

## Dataset & Features 

**Source:** `pid-5M.csv`  
**Sample used:** `data.sample(n=50000, random_state=42)`

**Columns / Features used:**
* `id` — particle id (target): `211`=Pion, `2212`=Proton, `321`=Kaon, `-11`=Electron  
* `p` — momentum  
* `theta` — polar angle  
* `beta` — velocity (fraction of *c*)  
* `nphe` — number of photoelectrons  
* `ein`, `eout` — energy deposits

**Train/Test split:** 80% train (40,000 samples) / 20% test (10,000 samples).  
**Scaling:** Applied to models sensitive to feature scale (LR, KNN, NN).  

---

## Models & Key Hyperparameters 

All models were trained on the same data split for consistency.

* **Random Forest**  
  `RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)`  

* **Neural Network (Keras/TensorFlow)**  
  Architecture:  
  `Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(4, softmax)`  
  Loss: `categorical_crossentropy`  
  Optimizer: `adam`  
  Training: `epochs=30`, `batch_size=64`, `validation_split=0.2`  

* **KNN** — `KNeighborsClassifier(n_neighbors=4)`  
* **Decision Tree** — `DecisionTreeClassifier(criterion='entropy', random_state=0)`  
* **Logistic Regression** — `LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)`

---

## Important Findings , Detailed 

### Accuracy Comparison 

Random Forest slightly outperformed the Neural Network on overall accuracy (96.85% vs. 96.65%).  
However, deeper metrics reveal a critical imbalance-driven flaw.

### The Class Imbalance Problem 

| Particle | Precision | Recall | F1 | Support |
| :-------- | :--------: | :------: | :------: | :------: |
| **Electron** | 0.89 | **0.28** | 0.42 | 29 |
| **Kaon** | 0.75 | **0.72** | 0.73 | 425 |
| **Pion** | 0.98 | 0.97 | 0.98 | 5669 |
| **Proton** | 0.98 | 0.99 | 0.98 | 3877 |

The Random Forest achieved high accuracy primarily due to strong performance on **Pions** and **Protons**, which together account for **95% of the dataset**.  
It struggled with rare particles — **Electrons** (28% recall) and **Kaons** (72% recall).

###  Feature Importance (Physics Alignment)

Feature importances show `beta` (velocity) and `p` (momentum) dominate — aligning with the physics principle that mass differentiates particles with the same momentum.

---


<img width="2390" height="719" alt="Model Comparison " src="https://github.com/user-attachments/assets/599a0dd8-5c1b-4794-932a-07d6accd4ef1" />


---

## 🏁 Final Conclusion & Key Findings

This project successfully demonstrated the application of multiple machine learning models for multi-class particle identification.

### Key Finding 1: Overall Accuracy is High, but Misleading

While the **Random Forest** appears to be the best model by accuracy, this metric hides the model’s failure on rare classes (Electrons and Kaons).

### Key Finding 2: The Neural Network Excels in Rare Particle Detection

| Particle (Support) | Random Forest (Recall) | Neural Network (Recall) | Winner |
| :--- | :--- | :--- | :--- |
| **Electron** (29) | 0.28 | **0.38** |  **Neural Network** |
| **Kaon** (425) | 0.72 | **0.77** |  **Neural Network** |
| **Pion** (5669) | 0.97 | 0.97 |  Tie |
| **Proton** (3877) | 0.99 | 0.99 |  Tie |

 The Neural Network, despite slightly lower overall accuracy, delivers **superior recall** for rare particles — making it more scientifically valuable.

---

###  Final Recommendation

While **Random Forest** wins on paper, the **Neural Network** is **the more promising model** for this physics problem.  
Future research should focus on improving **Recall** for rare classes, not overall accuracy.

**Recommended next steps:**

1. **Feature Engineering** — create physics-inspired features (e.g., mass proxy from `p` and `β`).  
2. **Data Resampling** — apply **SMOTE** or other resampling methods to balance class representation.  
3. **Weighted Loss Optimization** — penalize misclassification of rare classes using class weights or focal loss in the NN.

---

## How to Reproduce 

Clone or open the Jupyter notebook:  
`Multi_Class_Particle_Classifier_with_Machine_Learning.ipynb`  
Ensure `pid-5M.csv` is accessible.

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow


## Contact / Credits
 ✉️ Project author: Nitesh Kumar ( https://github.com/niteshg97 )

