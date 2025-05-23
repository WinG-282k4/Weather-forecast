import json
import os

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from settings import BASE_DIR


def save_model(model, model_name, fold, directory="saved_models"):
    model_dir = os.path.join(directory, model_name, f"fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")

    joblib.dump(model, model_path)


def load_model(model_name, fold, directory="best_models"):
    model_path = os.path.join(directory, model_name, f"fold_{fold}", "model.pkl")
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    return joblib.load(model_path)

def processing_data(df_train: pd.DataFrame, df_val: pd.DataFrame, fold: int, features, target, model_name):
    save_folder = os.path.join(BASE_DIR, "best_models", f"{model_name}", f"fold_{fold+1}")
    os.makedirs(save_folder, exist_ok=True)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train[features].values)
    X_val = scaler.transform(df_val[features].values)

    # Lưu scaler
    scaler_path = os.path.join(save_folder, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")

    y_train = df_train[target].values
    y_val = df_val[target].values

    # Biểu đồ phân phối target
    plt.figure(figsize=(20, 10))
    for i, (df, label, color) in enumerate(zip([df_train, df_val], ['y_train', 'y_val'], ['blue', 'orange'])):
        plt.subplot(1, 2, i+1)
        sns.histplot(df["log_gross"], kde=True, color=color, label=label)
        plt.title(f"Phân phối của {label} (thang gốc) - Fold {fold + 1}")
        plt.xlabel('Log Gross')
        plt.ylabel('Tần suất')
        plt.legend()

    return X_train, y_train, X_val, y_val


def save_result(save_dir, train_report, val_report, scaler, best_model, best_params, label_encoder):
    # === LƯU KẾT QUẢ ===
    os.makedirs(save_dir, exist_ok=True)

    # Lưu báo cáo đánh giá
    with open(os.path.join(save_dir, 'train_report.txt'), 'w') as f:
        f.write(train_report)

    with open(os.path.join(save_dir, 'val_report.txt'), 'w') as f:
        f.write(val_report)

    # Lưu scaler
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    # Lưu mô hình
    joblib.dump(best_model, os.path.join(save_dir, 'model.pkl'))

    # Lưu best_params
    with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))


def load_models(input_dim, base_dir="best_models/neural_net"):
    models = []
    for fold in range(1, 6):
        model_dir = os.path.join(base_dir, f"fold_{fold}")
        model_path = os.path.join(model_dir, "model.pt")
        params_path = os.path.join(model_dir, "params.json")

        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise FileNotFoundError(f"Missing model or params for fold {fold}")

        # Load params
        with open(params_path, "r") as f:
             params = json.load(f)
             print(f"Loaded params for fold {fold}: {params}")


        model = Net(input_dim=input_dim,
                    num_hidden_layers=params["num_hidden_layers"],
                    dropout_rate=params["dropout_rate"]).to(DEVICE)

        # Load trọng số
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models.append(model)

    return model