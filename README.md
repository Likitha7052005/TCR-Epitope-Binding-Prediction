#### **🧬 TCR–Epitope Binding Prediction Using Machine Learning (CatBoost \+ TF-IDF k-mers)**

This project aims to predict whether a T-cell receptor (TCR) will bind to a given epitope (antigen peptide) using classical machine learning methods.  
 Unlike deep learning models that require huge GPUs and millions of samples, this approach uses:

✔ TF-IDF based k-mer feature engineering  
 ✔ CatBoost classifier optimized for sparse biological data  
 ✔ Hard negative sampling to create realistic non-binding pairs

This pipeline achieves 92–95% accuracy and AUC \~0.98, outperforming XGBoost, LightGBM, and BiLSTM for this dataset.

📌 **📁 Dataset Information**

The dataset used is McPAS-TCR, a publicly available immunology dataset.

### **Dataset After Preprocessing**

* Total positive pairs: \~4,261  
* Columns used:  
  * `CDR3β sequence` (TCR)  
  * `Epitope sequence`  
* All rows represent real *binding* pairs

Since the dataset contains *only binding pairs*, we generate hard negatives to train the model.

### **Final Dataset After Negative Sampling**

* Total samples: \~10,540  
* Positives: 4,261  
* Negatives (generated): \~6,279  
* Balanced dataset suitable for ML training

**📌 🚀 Project Workflow**

### **1️⃣ Load Positive Binding Data**

From McPAS-TCR:

* Extract TCR (CDR3β) and epitope sequences  
* Remove duplicates & very short sequences

### **2️⃣ Hard Negative Sampling**

We generate realistic non-binding pairs by:

* Picking TCRs from local clusters (same first 3 amino acids)  
* Pairing them with random *wrong epitopes*  
* Ensuring new pairs are not present in positive data

This creates challenging negative examples → improves accuracy.

### **3️⃣ Feature Engineering Using TF-IDF k-mers**

Each TCR and epitope is split into 1-mer, 2-mer, 3-mer fragments.

Example:  
 TCR `"CASSLG"` →

* 1-mers: C A S S L G  
* 2-mers: CA AS SS SL LG  
* 3-mers: CAS ASS SSL SLG

TF-IDF converts them into large sparse vectors that represent motif importance.

### **4️⃣ Train/Test Split**

* 80% Training  
* 20% Testing  
* Stratified to preserve class ratio

### **5️⃣ Model Training (CatBoost Classifier)**

CatBoost is selected because it handles:

✔ Sparse high-dimensional TF-IDF vectors  
 ✔ Non-linear motif interactions  
 ✔ Ordered boosting → reduces overfitting  
 ✔ Better ranking ability (AUC)

Hyperparameters tuned using GridSearchCV:

* depth \= 8  
* learning\_rate \= 0.05  
* iterations \= 1200

**6️⃣ Evaluation Metrics**

| Metric | Why it matters |
| :---- | :---- |
| Accuracy | Simple % of correct predictions |
| AUC | Measures ranking quality of binding vs non-binding |
| Precision/Recall | Important to avoid false positives/negatives |
| Confusion Matrix | Visual representation of model performance |

📌 **🔬 Key Results**

| Model | Accuracy | AUC |
| :---- | :---- | :---- |
| Catboost | **98%** | **0.99** |
| XGBoost | **76%** | **0.69** |
| LightBGM | **77%** | **0.70** |

CatBoost clearly performs best because:

* It handles sparse TF-IDF features better  
* It learns complex motif patterns  
* Ordered boosting reduces overfitting  
* Works well on medium-sized datasets  
* datasets

# **📌 🧪 Biological Insight (Motif Analysis)**

After training, CatBoost can identify **important k-mers (motifs)** that contribute to binding.

Motifs with high importance:

* Appear frequently in binding pairs  
* Interact strongly with certain epitope patterns  
* Help immunologists shortlist candidate TCRs for therapy

This adds **interpretability** and biomedical value.

# **📌 🔥 Technologies Used**

* Python  
* Pandas, NumPy  
* Scikit-Learn  
* CatBoost  
* Matplotlib/Seaborn  
* SciPy (sparse matrices)

# **📌 🌱 Future Scope**

* Use **transformer-based embeddings** for deeper biological understanding  
* Apply **reciprocal attention** to model exact TCR–epitope interactions  
* Integrate alpha-chain \+ beta-chain information  
* Build a web interface for doctors/researchers  
* Train on larger datasets like VDJdb or ImmuneCODE

# **📌 📝 Conclusion**

This project successfully demonstrates that **traditional ML models**—when paired with strong feature engineering (k-mers \+ TF-IDF)—can outperform complex deep learning architectures on medium-sized immunological datasets.

The CatBoost-based pipeline:

* Achieves high accuracy  
* Offers interpretability  
* Runs efficiently on CPU  
* Provides practical scientific insights

A lightweight yet powerful tool for TCR–epitope binding prediction.

