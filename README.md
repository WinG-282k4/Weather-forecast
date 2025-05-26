# Dự Án Xử Lý Dữ Liệu Thời Tiết Đà Nẵng (Phiên Bản V2)

Dự án này bao gồm thu thập, làm sạch, phân tích và xây dựng mô hình dự đoán thời tiết cho Đà Nẵng từ năm 2019 đến 2025.

## Cấu Trúc Dự Án

```
├── Crawl_data.ipynb                    # Script thu thập dữ liệu từ web
├── Clean_data_V2.ipynb                 # Script xử lý và làm sạch dữ liệu V2
├── Show_data.ipynb                     # Script hiển thị và phân tích dữ liệu
├── analyze_data_weather_v2.ipynb       # Phân tích chi tiết dữ liệu V2
├── requirements.txt                    # Danh sách thư viện Python cần thiết
├── Raw_data/                          # Thư mục chứa dữ liệu thô
│   ├── raw_data_train.csv             # Dữ liệu thô tập huấn luyện
│   └── raw_data_test.csv              # Dữ liệu thô tập kiểm tra
├── Clean_data_v2/                     # Thư mục chứa dữ liệu đã xử lý V2
│   ├── clean_data_train.csv           # Dữ liệu sạch tập huấn luyện
│   ├── clean_data_test.csv            # Dữ liệu sạch tập kiểm tra
│   └── weather_visibility.json        # Tham số visibility theo loại thời tiết
├── train/                             # Thư mục chứa script huấn luyện mô hình
│   ├── train_XGBoostClassifier.py     # Huấn luyện mô hình XGBoost
│   └── train_neural_net.py            # Huấn luyện mô hình Neural Network
├── test/                              # Thư mục chứa script kiểm tra mô hình
│   ├── test_xgboost.py               # Kiểm tra mô hình XGBoost
│   └── test_neural_net.py            # Kiểm tra mô hình Neural Network
├── model/                             # Thư mục chứa mô hình đã huấn luyện
└── clustering_model/                  # Thư mục chứa mô hình phân cụm
```

## Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Jupyter Notebook hoặc JupyterLab
- Trình duyệt Chrome (cho Selenium web crawling)

## Cài Đặt

1. Clone repository này hoặc tải xuống tất cả các file vào máy tính của bạn
2. Cài đặt các thư viện Python cần thiết:
   ```powershell
   pip install -r requirements.txt
   ```

## Hướng Dẫn Chạy Dự Án

### Bước 1: Thu Thập Dữ Liệu

Chạy notebook `Crawl_data.ipynb` để thu thập dữ liệu thời tiết từ web:

1. Mở `Crawl_data.ipynb` trong Jupyter Notebook/Lab
2. Thực thi tất cả các cell theo thứ tự
3. Dữ liệu thu thập sẽ được lưu vào `Raw_data/raw_data_train.csv` và `Raw_data/raw_data_test.csv`

**Lưu ý:** Quá trình này có thể mất thời gian tùy thuộc vào khoảng thời gian thu thập.

### Bước 2: Xử Lý và Làm Sạch Dữ Liệu (V2)

Chạy notebook `Clean_data_V2.ipynb` để xử lý và làm sạch dữ liệu:

1. Mở `Clean_data_V2.ipynb` trong Jupyter Notebook/Lab
2. Thực thi tất cả các cell theo thứ tự
3. Dữ liệu đã được xử lý sẽ được lưu vào thư mục `Clean_data_v2/`

**Quy trình xử lý bao gồm:**

- Xử lý giá trị trống (missing values)
- Đơn giản hóa mô tả thời tiết thành 5 loại chính: Clear, Cloudy, Rainy, Foggy, Stormy
- Phát hiện và xử lý outliers theo phương pháp IQR
- Chuyển đổi biến thời gian thành dạng cyclical (sin/cos)
- Chuyển đổi thông tin gió thành vector components
- Xử lý tập train và test với cùng tham số

**Xử lý outliers:**

- `wind_speed`: 2,518 outliers (3.07%) - Áp dụng Winsorization
- `visibility`: 5,795 outliers (6.8%) - Áp dụng Winsorization
- Các biến khác có ít outliers nên không cần xử lý

### Bước 3: Hiển Thị và Phân Tích Dữ Liệu

Chạy các notebook sau để phân tích và hiển thị dữ liệu:

#### Show_data.ipynb

1. Mở `Show_data.ipynb` trong Jupyter Notebook/Lab
2. Thực thi tất cả các cell để xem:
   - Phân phối của các biến số
   - Tần suất xuất hiện của các loại thời tiết
   - Các thống kê cơ bản

#### analyze_data_weather_v2.ipynb

1. Mở `analyze_data_weather_v2.ipynb` để phân tích chi tiết hơn
2. Xem các biểu đồ correlation, heatmap và phân tích sâu hơn

### Bước 4: Huấn Luyện Mô Hình (Tùy Chọn)

Nếu muốn huấn luyện mô hình dự đoán:

#### XGBoost Classifier

```powershell
cd train
python train_XGBoostClassifier.py
```

#### Neural Network

```powershell
cd train
python train_neural_net.py
```

### Bước 5: Kiểm Tra Mô Hình (Tùy Chọn)

Để kiểm tra hiệu suất mô hình:

```powershell
cd test
python test_xgboost.py
python test_neural_net.py
```

## Đặc Điểm Phiên Bản V2

- **Xử lý dữ liệu nâng cao:** Tách riêng tập train/test với cùng tham số xử lý
- **Feature Engineering:** Chuyển đổi biến thời gian thành dạng cyclical và gió thành vector
- **Xử lý outliers thông minh:** Sử dụng phương pháp Winsorization
- **Đơn giản hóa weather:** Gom nhóm thành 5 loại thời tiết chính
- **Lưu tham số:** Lưu `weather_visibility.json` để áp dụng cho tập test

## Lưu Ý

- Quá trình crawling sử dụng Selenium với Chrome WebDriver, cần cài đặt Chrome browser
- WebDriver được quản lý tự động bởi thư viện webdriver-manager
- Dữ liệu được xử lý theo chuẩn machine learning với tách train/test riêng biệt
- Tất cả tham số xử lý từ tập train được áp dụng cho tập test để tránh data leakage
