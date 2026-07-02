import pandas as pd
import joblib
# load model
loaded_model = joblib.load(r'D:\Pycharm\Fraude-Detector.Project\models\random_forest_model.pkl')
loaded_scaler = joblib.load(r'D:\Pycharm\Fraude-Detector.Project\models\scaler.pkl')

print("✅ Model loaded successfully")
print("✅ Scaler loaded successfully")

#model type
print("model type: ",loaded_model.__class__.__name__)
print("scaler type: ",loaded_scaler.__class__.__name__)