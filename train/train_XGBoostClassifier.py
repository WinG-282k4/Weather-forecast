import os.path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from settings import BASE_DIR, TARGET, FEATURES
from utils import save_result

# Đường dẫn dữ liệu
data_path = os.path.join(BASE_DIR, 'clean_data', 'clean_data.csv')
df = pd.read_csv(data_path)

# Encode nhãn weather
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df[TARGET])

# Đặc trưng và nhãn
X = df[FEATURES].values
y = df['weather_encoded'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Khởi tạo model
xgb = XGBClassifier(eval_metric='mlogloss')

# Lưới siêu tham số
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Grid search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Huấn luyện
grid_search.fit(X_train, y_train)

# Mô hình tốt nhất
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Dự đoán
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Tính report
train_report = classification_report(y_train, y_train_pred, target_names=le.classes_)
val_report = classification_report(y_test, y_test_pred, target_names=le.classes_)

# Lưu kết quả
save_dir = os.path.join(BASE_DIR, 'model', 'xgboost', 'best_model')
save_result(save_dir, train_report, val_report, scaler, best_model, grid_search.best_params_, le)

# In ra màn hình nếu muốn
print("Train Report:\n", train_report)
print("Validation (Test) Report:\n", val_report)
