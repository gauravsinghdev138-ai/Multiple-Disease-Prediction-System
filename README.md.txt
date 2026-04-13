# Multiple Disease Prediction System

A professional Machine Learning-based web application that predicts three major health conditions: **Diabetes**, **Heart Disease**, and **Chronic Kidney Disease (CKD)**. Built with Python and Streamlit.

## 🌟 Features
- **Diabetes Prediction:** Predicts diabetes status using a Random Forest Classifier.
- **Heart Disease Prediction:** Predicts heart health using a Support Vector Machine (SVM) with data scaling.
- **Kidney Disease Prediction:** Predicts Chronic Kidney Disease using an Extra Trees Classifier with data scaling and categorical mapping.
- **Interactive UI:** Built with Streamlit for a smooth user experience.
- **Responsive Results:** Clear, color-coded result boxes (Red for Disease, Green for Normal).

## 🛠️ Tech Stack
- **Language:** Python 3
- **Frontend:** Streamlit
- **ML Libraries:** Scikit-learn, Pandas, Numpy
- **Model Storage:** Pickle (Saved as .sav files)

## 📁 Project Structure
```text
├── app.py	       # final main application code (after dimentionality reduction and advanced UI/UX)
├── final_main_app.py               # Main application code (without dimentionality reduction)
└── Dataset/                  # Folder containing Dataset for project
    ├── diabetes.csv		https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset
    ├── heart_disease_uci.csv	https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
    └── kidney_disease.csv	https://www.kaggle.com/datasets/mansoordaku/ckdisease
├── README.md                 # Project documentation (this folder)
├── requirements.txt          # List of dependencies
└── saved_models/             # Folder containing trained models and scalers
    ├── diabetes_modelRF.sav		#final diabetes prediction model and scaler
    ├── diabetes_scalerRF.sav
    ├── heart_disease_modelSVM.sav	#heart disease prediction model and scaler (without dimentionality reduction)
    ├── heart_scaler_final.sav
    ├── heart_disease_model_final.sav	#heart disease prediction model and scaler (after dimentionality reduction)
    ├── heart_scalerSVM.sav
    ├── kidney_modelET.sav		#kidney disease prediction model and scaler (without dimentionality reduction)
    └── kidney_scalerET.sav
    ├── kidney_model_final.sav		#kidney disease prediction model and scaler (after dimentionality reduction)
    └── kidney_scaler_final.sav		
