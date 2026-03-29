# Dự án Phát hiện Gian lận Giao dịch Tài chính

## 1. Tổng quan dự án
Dự án này xây dựng mô hình machine learning nhằm phát hiện các giao dịch gian lận trong bộ dữ liệu giao dịch tài chính. Đây là một bài toán classification, trong đó số lượng giao dịch Fraud rất nhỏ so với Non-Fraud.

Dự án tập trung vào 2 nội dung chính:
- EDA (Exploratory Data Analysis)
- Train, so sánh và đánh giá các mô hình

---

## 2. Mục tiêu
Mục tiêu chính của dự án là xây dựng một mô hình có khả năng:

- phát hiện được càng nhiều giao dịch gian lận càng tốt
- đồng thời kiểm soát số lượng cảnh báo nhầm ở mức chấp nhận được

Cụ thể hơn, dự án hướng đến:
- hiểu đặc điểm của các giao dịch Fraud và Non-Fraud (thông qua EDA)
- lựa chọn phương pháp phù hợp để xử lý class imbalance
- so sánh hiệu quả giữa các mô hình phân loại
- sử dụng các metric phù hợp với dữ liệu mất cân bằng

---

## 3. Bộ dữ liệu
Mô hình sử dụng bộ dữ liệu mô phỏng PaySim trên Kaggle gồm khoảng **6 triệu quan sát** với biến mục tiêu là `isFraud`:

- `0`: giao dịch hợp lệ (Non-Fraud)
- `1`: giao dịch gian lận (Fraud)

Đây là một bài toán **imbalanced classification** điển hình vì số lượng giao dịch Fraud chiếm tỷ lệ rất nhỏ trong toàn bộ dữ liệu.

---

## 4. Thách thức của bài toán
Dự án có một số thách thức chính:

- dữ liệu bị class imbalance mạnh
- Chất lượng một số features còn kém, có thể do bộ dữ liệu gốc đã mask dữ liệu
- kích thước dữ liệu lớn, dẫn đến thời gian huấn luyện cao
- cần đánh đổi giữa khả năng phát hiện fraud và số lượng cảnh báo nhầm

---

## 5. Phân tích dữ liệu khám phá (EDA)
Quá trình EDA cho thấy một số điểm quan trọng:

- biến mục tiêu `isFraud` bị mất cân bằng nghiêm trọng
- giao dịch fraud có xu hướng có giá trị lớn hơn
- fraud chỉ xuất hiện đối với loại giao dịch **Transfer** và **Cash Out**
- các biến liên quan đến khối lượng giao dịch (oldbalance, newbalance,...) đều có dấu hiệu bị can thiệp, do hầu hết đều không tương thích với biến amount
- Số lần giao dịch của mỗi tài khoản không thể hiện mối quan hệ với fraud
- số lượng fraud biến động nhẹ theo khung giờ -> **có thể tạo thêm feature Hour** 
- mối quan hệ giữa các biến đầu vào và khả năng gian lận không mang tính tuyến tính rõ rệt -> **các mô hình tree-based sẽ phù hợp hơn** so với các mô hình tuyến tính như Logistic Regression.

---

## 6. Phương pháp thực hiện

### 6.1. Xử lý mất cân bằng lớp
Để xử lý class imbalance, tôi cân nhắc hai hướng tiếp cận:

- **Random UnderSampling**
- **SMOTE**

Do bộ dữ liệu có quy mô rất lớn (~6tr dòng), tôi lựa chọn **Random UnderSampling** nhằm giảm thời gian huấn luyện và tăng tốc độ thử nghiệm mô hình.  
Để tránh data leakage, undersampling được tích hợp trực tiếp vào pipeline và chỉ được áp dụng trên **train fold** trong từng vòng cross-validation.

### 6.2. Mô hình sử dụng
Hai mô hình được huấn luyện và so sánh trong dự án là:

- **XGBoost**
- **Random Forest**

Các mô hình này được lựa chọn vì:
- phù hợp với quan hệ phi tuyến
- có khả năng xử lý interaction giữa các biến tốt hơn
- thường hoạt động hiệu quả trong các bài toán phân loại mất cân bằng
- không cần scale


Sau đó, **GridSearchCV** được sử dụng để tìm bộ hyperparameters tốt nhất cho từng mô hình.

Trong quá trình lựa chọn mô hình qua **cross validation**, tôi sử dụng **Average Precision** làm metric chính, vì chỉ số này phù hợp  trong bối cảnh dữ liệu mất cân bằng và phản ánh tốt hơn chất lượng của mô hình trên đường Precision-Recall.

-> Sau khi đánh giá, **XGBoost được chọn với AP ~94%**
---

## 7. Đánh giá mô hình
Sau khi chọn được mô hình tốt nhất, tôi đánh giá trên tập test gốc.

Các công cụ đánh giá được sử dụng gồm:
- **Classification Report**
- **Confusion Matrix**
- **Precision-Recall Curve**
- **Average Precision**

Việc đánh giá trên tập test gốc giúp phản ánh đúng hơn hiệu quả thực tế của mô hình trong bối cảnh fraud detection.

---

## 8. Kết quả chính
Kết quả cho thấy mô hình có khả năng phát hiện gian lận khá tốt trên tập test:

- mô hình chỉ bỏ sót một số lượng nhỏ giao dịch Fraud (58 trên ~2500)
- tuy nhiên vẫn còn khá nhiều giao dịch Non-Fraud bị phân loại nhầm thành Fraud

Điều này cho thấy mô hình có **recall cao đối với Fraud**, nhưng vẫn phải đánh đổi bằng việc tăng số lượng **false positive**.

Một nguyên nhân có thể đến từ việc Random UnderSampling làm thay đổi phân phối lớp trong tập huấn luyện, khiến mô hình nhạy hơn với lớp Fraud. Đây là một đánh đổi hợp lý trong bối cảnh bộ dữ liệu rất lớn và mong muốn rút ngắn thời gian huấn luyện của tôi.

---


## 9. Hướng phát triển thêm
Trong tương lai, dự án có thể được mở rộng theo các hướng sau:

- thử nghiệm thêm **SMOTE** hoặc các kỹ thuật resampling khác
- tối ưu thêm threshold thay vì chỉ sử dụng ngưỡng mặc định
- bổ sung các metric như ROC-AUC để tham khảo thêm
- phân tích sâu hơn các trường hợp false positive và false negative
- triển khai mô hình thành một API và chạy test với dữ liệu mới/ dữ liệu simulated.

---

## 10. Công cụ:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
- Path
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Jupyter Notebook