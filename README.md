# Mobile Phone Price Prediction 📱

This project builds a **machine learning model** to predict the **price range of mobile phones** based on their specifications such as RAM, battery power, storage, connectivity features, etc.  
The dataset is sourced from Kaggle: [Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).

---

## 📂 Project Structure
├── mobile_price_prediction.ipynb # Jupyter Notebook with full code
├── export/ # Saved trained models (.joblib)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Requirements
Install dependencies with:

bash
pip install -r requirements.txt
requirements.txt

nginx
Copy code
pandas
numpy
scikit-learn
matplotlib
joblib
🧑‍💻 Skills Gained
Data preprocessing & feature analysis

Exploratory Data Analysis (EDA) & visualization

Handling categorical and numerical features

Building classification models (Logistic Regression, Decision Tree, Random Forest, KNN, SVM)

Model evaluation with accuracy, precision, recall, F1-score

Hyperparameter tuning with GridSearchCV

Model persistence using joblib

Making predictions on new unseen data

 How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/mobile-price-prediction.git
cd mobile-price-prediction
Open Jupyter Notebook:

bash
Copy code
jupyter notebook
and run mobile_price_prediction.ipynb

Or run saved model for prediction:

python
Copy code
import joblib, pandas as pd

bundle = joblib.load("./export/mobile_price_best_model_rf_xxxxx.joblib")
model = bundle["model"]
features = bundle["features"]

# Example phone specs
sample_phone = {f: 0 for f in features}
sample_phone.update({
    "battery_power": 1500,
    "ram": 3000,
    "blue": 1,
    "wifi": 1
})

sample_df = pd.DataFrame([sample_phone])
pred = model.predict(sample_df)[0]

price_map = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
print("Predicted Price Range:", price_map[pred])
📊 Dataset Info
Target: Mobile price range (0 = Low, 1 = Medium, 2 = High, 3 = Very High)

Features: RAM, battery power, internal memory, Bluetooth, Wi-Fi, screen size, etc.

 Results
Best performing model: Random Forest

Accuracy: ~95% on validation data

Saved model available in ./export/

 Future Improvements
Try deep learning models (ANN, CNN with feature engineering)

Build a simple Flask/Django web app for interactive predictions

Deploy on cloud (Heroku / Streamlit / AWS)

Author
Madhu Lavan Kumar
B.Tech CSE, Amrita Vishwa Vidyapeetham (2027)
Aspiring Data Scientist | Machine Learning Enthusiast
