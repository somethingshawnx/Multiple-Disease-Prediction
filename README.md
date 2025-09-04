# 🏥 Multiple Disease Prediction System
A comprehensive machine learning-based healthcare prediction system that can predict three major diseases: **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using advanced ML algorithms and real medical datasets.

## 🎯 Project Overview

This project represents my first deep dive into healthcare AI, combining machine learning with medical diagnostics to create a user-friendly web application. The system analyzes various medical parameters to provide accurate disease predictions, potentially assisting in early detection and preventive healthcare.

## ✨ Features

- **🩺 Multi-Disease Prediction**: Supports prediction for 3 critical diseases
- **🖥️ Interactive Web Interface**: Built with Streamlit for seamless user experience
- **📊 Real-time Predictions**: Instant results with confidence indicators
- **🔒 Data Privacy**: All processing done locally, no data storage
- **📱 Responsive Design**: Works on desktop and mobile devices
- **⚡ Fast Processing**: Optimized models for quick predictions
- **📈 Model Performance**: High accuracy rates across all disease predictions

## 🔬 Supported Disease Predictions

### 1. 🍯 Diabetes Prediction
- **Parameters**: 8 medical indicators
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Model**: Logistic Regression
- **Accuracy**: ~85%
- **Input Example**: Pregnancies=6, Glucose=148, BloodPressure=72, SkinThickness=35, Insulin=0, BMI=33.6, DiabetesPedigreeFunction=0.627, Age=50

### 2. ❤️ Heart Disease Prediction  
- **Parameters**: 13 cardiac indicators
- **Features**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Rest ECG, Max Heart Rate, Exercise Induced Angina, ST Depression, ST Slope, Number of Vessels, Thalassemia
- **Model**: Logistic Regression
- **Accuracy**: ~85%
- **Input Example**: Age=63, Sex=1, ChestPainType=3, RestingBP=145, Cholesterol=233, FastingBS=1, RestingECG=0, MaxHR=150, ExerciseAngina=0, Oldpeak=2.3, ST_Slope=0, CA=0, Thal=1

### 3. 🧠 Parkinson's Disease Prediction
- **Parameters**: 22 voice biomarkers
- **Features**: 
  - **Fundamental Frequency**: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
  - **Jitter Measures**: MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP
  - **Shimmer Measures**: MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA
  - **Harmonic Measures**: NHR, HNR
  - **Nonlinear Measures**: RPDE, DFA, spread1, spread2, D2, PPE
- **Model**: SVM with Linear Kernel + StandardScaler
- **Accuracy**: ~90%+
- **Data Processing**: Features are standardized using StandardScaler for optimal SVM performance

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Libraries**: 
  - scikit-learn (SVM, Logistic Regression)
  - pandas (Data manipulation)
  - numpy (Numerical operations)
- **Model Serialization**: pickle
- **Data Processing**: StandardScaler (for Parkinson's prediction)
- **Visualization**: matplotlib, seaborn

## 📋 Prerequisites

```bash
Python 3.8 or higher
pip package manager
```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multiple-disease-prediction.git
cd multiple-disease-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv disease_prediction_env
source disease_prediction_env/bin/activate  # On Windows: disease_prediction_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train and save the models** (if not already done)
```bash
# Run the training notebooks to generate model files
jupyter notebook notebooks/diabetes_training.ipynb
jupyter notebook notebooks/heart_disease_training.ipynb
jupyter notebook notebooks/parkinsons_training.ipynb
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Access the application**
```
Open your browser and navigate to: http://localhost:8501
```

## 📁 Project Structure

```
multiple-disease-prediction/
│
├── app.py                          # Main Streamlit application
├── models/
│   ├── diabetes_model.pkl         # Trained diabetes prediction model
│   ├── heart_disease_model.pkl    # Trained heart disease model
│   ├── parkinsons_model.pkl       # Trained Parkinson's model
│   └── parkinsons_scaler.pkl      # StandardScaler for Parkinson's features
├── notebooks/
│   ├── diabetes_training.ipynb    # Diabetes model training notebook
│   ├── heart_disease_training.ipynb # Heart disease model training
│   └── parkinsonsdisease.ipynb    # Parkinson's model training notebook
├── data/
│   ├── diabetes.csv              # Diabetes dataset
│   ├── heart.csv                 # Heart disease dataset
│   └── parkinsons.csv            # Parkinson's dataset
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```

## 💻 Usage

1. **Select Disease Type**: Choose from the sidebar which disease you want to predict
2. **Input Parameters**: Enter the required medical parameters in the form
3. **Get Prediction**: Click the "Predict" button to get instant results
4. **View Results**: See the prediction result with confidence indicators

### Example Usage:

#### Diabetes Prediction:
```python
# Sample input values
pregnancies = 6
glucose = 148
blood_pressure = 72
skin_thickness = 35
insulin = 0
bmi = 33.6
diabetes_pedigree_function = 0.627
age = 50
```

#### Heart Disease Prediction:
```python
# Sample input values
age = 63
sex = 1  # 1 = Male, 0 = Female
cp = 3   # Chest pain type
trestbps = 145  # Resting blood pressure
chol = 233      # Cholesterol
fbs = 1         # Fasting blood sugar
restecg = 0     # Resting ECG results
thalach = 150   # Maximum heart rate
exang = 0       # Exercise induced angina
oldpeak = 2.3   # ST depression
slope = 0       # ST slope
ca = 0          # Number of major vessels
thal = 1        # Thalassemia
```

#### Parkinson's Disease Prediction:
```python
# Sample input values (22 voice biomarkers)
input_data = (
    197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 
    0.00498, 0.01098, 0.097, 0.00563, 0.00680, 0.00802, 0.01689, 
    0.00339, 26.775, 0.422229, 0.741367, -7.3483, 0.177551, 
    1.743867, 0.085569
)
```

## 🧠 Model Information

### Training Process:
1. **Data Preprocessing**: 
   - Handling missing values
   - Feature scaling (StandardScaler for Parkinson's)
   - Data normalization
2. **Model Selection**: Comparison of multiple algorithms
3. **Training**: Using train-test split with stratification
4. **Evaluation**: Accuracy, precision, recall, F1-score metrics
5. **Model Persistence**: Saving trained models using pickle

### Performance Metrics:
- **Diabetes Model**: ~85% accuracy (Logistic Regression)
- **Heart Disease Model**: ~85% accuracy (Logistic Regression)
- **Parkinson's Model**: ~90% accuracy (SVM with Linear Kernel)
  - Training Data Accuracy: 88.46%
  - Test Data Accuracy: 87.18%

### Model Details:
- **Diabetes & Heart Disease**: Simple Logistic Regression models with good interpretability
- **Parkinson's Disease**: 
  - SVM with linear kernel for better performance on voice biomarker data
  - StandardScaler preprocessing for feature normalization
  - 22 voice-related features including jitter, shimmer, and harmonic measures

## 📊 Datasets

- **Diabetes**: Pima Indians Diabetes Database (768 samples, 8 features)
- **Heart Disease**: UCI Heart Disease Dataset (303 samples, 13 features)
- **Parkinson's**: UCI Parkinson's Disease Dataset (195 samples, 22 voice biomarkers)

*All datasets are publicly available and used for educational purposes.*

## 🔬 Key Features by Disease

### Diabetes Features:
1. Pregnancies - Number of pregnancies
2. Glucose - Glucose level in blood
3. BloodPressure - Blood pressure measurement
4. SkinThickness - Thickness of the skin
5. Insulin - Insulin level in blood
6. BMI - Body mass index
7. DiabetesPedigreeFunction - Diabetes percentage
8. Age - Age of the person

### Heart Disease Features:
1. Age - Age in years
2. Sex - Gender (1 = male, 0 = female)
3. CP - Chest pain type (0-3)
4. Trestbps - Resting blood pressure
5. Chol - Serum cholesterol
6. FBS - Fasting blood sugar
7. Restecg - Resting electrocardiographic results
8. Thalach - Maximum heart rate achieved
9. Exang - Exercise induced angina
10. Oldpeak - ST depression induced by exercise
11. Slope - Slope of peak exercise ST segment
12. CA - Number of major vessels colored by fluoroscopy
13. Thal - Thalassemia

### Parkinson's Disease Features:
Voice biomarkers including fundamental frequency variations, jitter, shimmer, noise-to-harmonic ratios, and nonlinear dynamical complexity measures that can detect vocal impairments associated with Parkinson's disease.

## 🚀 Future Enhancements

- [ ] **Cloud Deployment** (Heroku/AWS/Google Cloud)
- [ ] **Model Performance Dashboard**
- [ ] **Advanced Ensemble Methods**
- [ ] **Real-time Model Monitoring**
- [ ] **Mobile App Development**
- [ ] **API Development** for third-party integration
- [ ] **Enhanced Visualizations** and feature importance plots
- [ ] **Multi-language Support**
- [ ] **Integration with medical devices** for automatic data input
- [ ] **Confidence intervals** for predictions
- [ ] **Model explainability** features (SHAP, LIME)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

**This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.**

## 🙏 Acknowledgments

- **Datasets**: UCI Machine Learning Repository
- **Libraries**: scikit-learn, Streamlit, pandas, numpy
- **Inspiration**: Healthcare AI research community
- **Special Thanks**: To the open-source community for providing excellent tools and datasets

## 📈 Model Training Results

### Parkinson's Disease Model Details:
- **Algorithm**: Support Vector Machine (SVM) with Linear Kernel
- **Preprocessing**: StandardScaler for feature normalization
- **Dataset Size**: 195 samples with 22 features
- **Training Accuracy**: 88.46%
- **Test Accuracy**: 87.18%
- **Key Insight**: Voice biomarkers provide excellent discrimination between Parkinson's and healthy individuals

### Performance Summary:
| Disease | Model | Accuracy | Features | Dataset Size |
|---------|-------|----------|----------|--------------|
| Diabetes | Logistic Regression | ~85% | 8 | 768 |
| Heart Disease | Logistic Regression | ~85% | 13 | 303 |
| Parkinson's | SVM (Linear) | ~90% | 22 | 195 |

---

⭐ **If you found this project helpful, please give it a star!** ⭐
