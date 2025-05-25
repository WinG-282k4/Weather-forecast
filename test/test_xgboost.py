import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from settings import BASE_DIR, TARGET, FEATURES
from train.train_XGBoostClassifier import X_test, y_test

test_data_path = os.path.join(BASE_DIR, 'clean_data', 'clean_data.csv')
df = pd.read_csv(test_data_path)

# Đường dẫn đến thư mục lưu model
save_dir = os.path.join(BASE_DIR, 'model', 'xgboost', 'best_model')

# Load lại các đối tượng đã lưu
model = joblib.load(os.path.join(save_dir, 'model.pkl'))
scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

# Load lại dữ liệu
df['weather_encoded'] = label_encoder.transform(df[TARGET])

X = df[FEATURES].values
y = df['weather_encoded'].values

# Chuẩn hóa
X_scaled = scaler.transform(X)

y_pred = model.predict(X_test)

# Báo cáo
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

with open(os.path.join(save_dir, 'test_report.txt'), 'w') as f:
    f.write(report)
