# MNIST Dataset — Interactive High-Dimensional Data Explorer

> **Mục tiêu:** sử dụng MNIST để nghiên cứu giảm chiều dữ liệu từ không gian 784 chiều xuống không gian 3D, phân cụm và khám phá cấu trúc dữ liệu bằng trực quan hóa tương tác.

## 1. Tổng quan

MNIST (Modified National Institute of Standards and Technology database) là bộ dữ liệu ảnh chữ số viết tay thường dùng trong nghiên cứu thị giác máy tính và học máy.

Mỗi ảnh là ảnh xám có kích thước **28 × 28 pixel**. Khi làm phẳng ảnh thành vector, mỗi mẫu có **784 đặc trưng pixel**. Bộ dữ liệu có tổng cộng **70.000 ảnh**, gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra theo cách chia chuẩn.

| Thuộc tính | Giá trị |
|---|---|
| Loại dữ liệu | Ảnh chữ số viết tay, grayscale |
| Tổng số mẫu | 70.000 |
| Số chiều mỗi mẫu | 784 |
| Kích thước ảnh | 28 × 28 pixel |
| Nhãn | 10 lớp: chữ số 0–9 |
| Tập train chuẩn | 60.000 ảnh |
| Tập test chuẩn | 10.000 ảnh |
| Giá trị pixel gốc | 0–255 |
| Giá trị pixel sau chuẩn hóa | 0–1 |

**Nguồn dữ liệu:**
- TensorFlow/Keras: https://keras.io/api/datasets/mnist/
- Trang MNIST của Yann LeCun: http://yann.lecun.com/exdb/mnist/

## 2. Vì sao chọn MNIST cho đề tài?

Đề tài của nhóm: **Xây dựng hệ thống khám phá cấu trúc dữ liệu đa chiều bằng giảm chiều và phân cụm trong không gian 3D.**

MNIST phù hợp để xây dựng một trường hợp nghiên cứu có thể tái lập vì:

1. **Có số chiều cao hơn 3D:** 784 đặc trưng pixel cho phép khảo sát việc chuyển từ không gian gốc xuống biểu diễn 3D.
2. **Có nhãn tham chiếu:** nhãn chữ số 0–9 giúp so sánh kết quả phân cụm với lớp đã biết bằng ARI và NMI. Nhãn chỉ dùng để đánh giá/hiển thị, không đưa vào thuật toán phân cụm không giám sát.
3. **Quy mô vừa phải:** 70.000 ảnh đủ để thử nghiệm nhiều thuật toán, nhưng có thể lấy mẫu phân tầng để phát triển nhanh trên máy cá nhân.
4. **Dễ giải thích:** người xem có thể nhìn ảnh mẫu của từng điểm và so sánh nhãn thật với cụm mà thuật toán phát hiện.
5. **Có thể so sánh nhiều phương pháp:** PCA, UMAP, t-SNE; K-Means, DBSCAN hoặc HDBSCAN; đồng thời đo chất lượng biểu diễn và chất lượng cụm.
6. **Dễ tái lập:** bộ dữ liệu phổ biến, có nguồn tải chính thức và được hỗ trợ trực tiếp bởi TensorFlow/Keras.

## 3. Ưu điểm và giới hạn

### Ưu điểm

- Cấu trúc dữ liệu đơn giản, dễ tải và dễ kiểm tra.
- Có nhãn tham chiếu đầy đủ cho các chữ số.
- Phù hợp minh họa trực quan hóa 3D tương tác.
- Có thể thiết kế thí nghiệm so sánh không gian gốc, PCA-50 và không gian 3D.
- Thích hợp làm baseline trước khi thử dữ liệu phức tạp hơn như Fashion-MNIST hoặc dữ liệu bảng nhiều chiều.

### Giới hạn cần nêu trong báo cáo

- MNIST được xây dựng cho nhận dạng chữ số, không phải bảo đảm có đúng 10 cụm tự nhiên tách biệt.
- Các chữ số có hình dạng gần nhau có thể chồng lấn; nhãn lớp không nhất thiết tương đương với cấu trúc cụm.
- Pixel thô nhạy với độ nghiêng, độ dày nét và vị trí chữ số.
- PCA tối ưu phương sai tuyến tính, không bảo đảm giữ cấu trúc cụm.
- UMAP/t-SNE có thể làm thay đổi khoảng cách toàn cục; các “đảo” tách biệt trên biểu đồ không tự động chứng minh phân cụm tốt.
- K-Means giả định cụm tương đối lồi và có dạng gần hình cầu; DBSCAN/HDBSCAN có thể gặp khó khi mật độ thay đổi hoặc tham số chưa phù hợp.

## 4. Cấu trúc dữ liệu sau khi tải

Nếu tải bằng `tensorflow.keras.datasets.mnist`, TensorFlow/Keras thường lưu file cache trong thư mục cache của Keras. Nếu dự án yêu cầu quản lý file dữ liệu trực tiếp tại repository, có thể tải file `mnist.npz` vào `data/raw/` bằng một script riêng.

Cấu trúc dự án gợi ý:

```text
Interactive-High-Dimensional-Data-Explorer/
├── data/
│   ├── raw/
│   │   └── mnist.npz          # dữ liệu gốc; không commit lên Git
│   ├── processed/             # dữ liệu đã tiền xử lý; thường không commit
│   └── samples/               # mẫu nhỏ phục vụ demo (nếu cần)
├── notebooks/
│   └── mnist_exploration.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dimensionality_reduction.py
│   ├── clustering.py
│   └── visualization_3d.py
├── outputs/                   # biểu đồ và kết quả chạy
├── requirements.txt
├── .gitignore
└── README.md
```

> `mnist.npz` là file nén chứa các mảng train/test và nhãn tương ứng. Không cần giải nén thành 70.000 file ảnh riêng lẻ để phân tích bằng Python.

## 5. Cách tải và kiểm tra dữ liệu

### Cách A — Dùng TensorFlow/Keras

Cài thư viện:

```bash
python -m pip install tensorflow numpy
```

Tạo `src/data_loader.py`:

```python
import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # 28 × 28 -> 784 chiều; chuẩn hóa pixel về [0, 1]
    X = X.reshape(-1, 784).astype("float32") / 255.0

    print("X shape:", X.shape)       # (70000, 784)
    print("y shape:", y.shape)       # (70000,)
    print("Labels:", np.unique(y))   # 0, 1, ..., 9
    print("Pixel range:", X.min(), X.max())

    return X, y

if __name__ == "__main__":
    X, y = load_mnist()
```

Chạy từ thư mục gốc dự án:

```bash
python src/data_loader.py
```

Lần chạy đầu cần Internet để tải dữ liệu. Phương pháp này sử dụng cache của TensorFlow/Keras; nó **không tự động đảm bảo** file `mnist.npz` nằm trong `data/raw/`.

### Cách B — Lưu file `mnist.npz` trực tiếp vào `data/raw/`

Nếu nhóm cần quản lý file tại một vị trí cụ thể, dùng script tải file vào thư mục dự án:

```python
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

data_file = data_dir / "mnist.npz"
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

if not data_file.exists():
    print(f"Downloading MNIST to {data_file}")
    urlretrieve(url, data_file)

with np.load(data_file) as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)
```

Chỉ sử dụng một trong hai cách tải trong cùng pipeline để tránh tải hoặc lưu trữ trùng lặp không cần thiết.

## 6. Những hướng khai thác trong đề tài

### 6.1. Khám phá dữ liệu (EDA)

- Xem số lượng mẫu theo từng chữ số.
- Hiển thị lưới ảnh mẫu 0–9.
- Kiểm tra giá trị pixel, hình dạng mảng, kiểu dữ liệu và phân bố nhãn.
- So sánh độ sáng, độ dày nét và biến thiên hình dạng giữa các chữ số.

### 6.2. Giảm chiều từ 784D xuống 3D

**PCA**
- Tạo baseline tuyến tính.
- Báo cáo tổng explained variance ratio của ba thành phần đầu.
- Có thể thử PCA 50D làm bước trung gian trước UMAP hoặc phân cụm.

**UMAP**
- Khám phá cấu trúc láng giềng phi tuyến.
- Thử các giá trị `n_neighbors` và `min_dist` khác nhau.
- Đánh giá độ ổn định khi thay đổi tham số và random seed.

**t-SNE (tùy chọn)**
- So sánh trực quan hóa cục bộ với PCA/UMAP.
- Cần lưu ý chi phí tính toán và không diễn giải quá mức khoảng cách toàn cục giữa các cụm.

### 6.3. Phân cụm

Các thí nghiệm có thể gồm:

1. K-Means với `k=10` làm baseline.
2. DBSCAN hoặc HDBSCAN để kiểm tra khả năng phát hiện cấu trúc không cần chỉ định số cụm trước.
3. Phân cụm trên không gian pixel 784D, PCA-50 và tọa độ 3D.
4. So sánh các kết quả trên cùng tập mẫu và cùng quy tắc tiền xử lý.

**Quan trọng:** nhãn chữ số 0–9 không được dùng để huấn luyện thuật toán phân cụm không giám sát. Chỉ dùng nhãn sau khi phân cụm để tô màu, tính chỉ số tham chiếu và phân tích lỗi.

### 6.4. Đánh giá định lượng

| Chỉ số | Câu hỏi trả lời |
|---|---|
| ARI | Cụm dự đoán đồng thuận với nhãn chữ số đến mức nào? |
| NMI | Thông tin giữa nhãn cụm và nhãn chữ số liên hệ ra sao? |
| Silhouette | Các điểm có kết dính trong cụm và tách biệt với cụm khác không? |
| Trustworthiness | Biểu diễn 3D có giữ các láng giềng gần trong không gian gốc không? |
| Explained variance ratio | PCA 3D giữ được bao nhiêu phương sai tổng? |
| Thời gian và bộ nhớ | Pipeline có đáp ứng tài nguyên máy của nhóm không? |
| Độ ổn định | Kết quả thay đổi thế nào khi đổi seed hoặc tham số? |

Không nên so sánh trực tiếp Silhouette giữa các không gian khác nhau mà không ghi rõ metric, khoảng cách và không gian tính toán. ARI/NMI cao cũng không phải điều kiện duy nhất để kết luận biểu diễn 3D hữu ích.

### 6.5. Tính năng có thể phát triển trong hệ thống

- Biểu đồ 3D có thể xoay, zoom và chọn điểm.
- Chuyển màu giữa nhãn thật và nhãn cụm dự đoán.
- Chọn phương pháp giảm chiều: PCA / UMAP / t-SNE.
- Chọn thuật toán phân cụm và điều chỉnh tham số.
- Chọn một điểm để xem ảnh chữ số gốc 28 × 28.
- Hiển thị số điểm mỗi cụm, điểm nhiễu và chỉ số đánh giá.
- So sánh kết quả giữa các phương pháp trên cùng một tập mẫu.
- Xuất ảnh biểu đồ, bảng metrics và cấu hình thí nghiệm.

## 7. Thiết kế thí nghiệm đề xuất

Để kết quả có ý nghĩa khoa học, nên so sánh ít nhất ba pipeline:

| Pipeline | Quy trình | Mục đích |
|---|---|---|
| A — Baseline | Pixel 784D → K-Means | Đánh giá phân cụm trong không gian đầu vào |
| B — Giảm chiều trước | Pixel 784D → PCA-50 → K-Means | Đo tác động của giảm chiều trung gian |
| C — Trực quan hóa 3D | Pixel 784D → PCA-3 hoặc UMAP-3 → phân cụm | Đánh giá khi phân cụm trực tiếp trên biểu diễn 3D |

Ngoài ra, có thể dùng PCA-50 → UMAP-3 chỉ để hiển thị kết quả phân cụm được học ở PCA-50. Khi đó phải ghi rõ tọa độ 3D chỉ phục vụ hiển thị, không phải không gian mà thuật toán phân cụm đã sử dụng.

Để giảm thời gian phát triển, bắt đầu với mẫu phân tầng 10.000 ảnh, cố định random seed và ghi lại tham số. Khi pipeline ổn định, tăng kích thước mẫu để kiểm tra khả năng mở rộng.

## 8. Quản lý dataset trên GitHub

Không nên commit toàn bộ file `mnist.npz`, dữ liệu đã xử lý lớn hoặc môi trường `.venv` vào Git.

Gợi ý `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/
.keras/
outputs/
.ipynb_checkpoints/
```

Nếu cần chia sẻ kết quả, hãy commit code tải dữ liệu, hướng dẫn tái lập, cấu hình thí nghiệm, bảng metrics và một số hình demo nhỏ. Mỗi thành viên có thể tải dữ liệu từ cùng nguồn chính thức.

## 9. Kết luận

MNIST là lựa chọn phù hợp để xây dựng **baseline nghiên cứu giảm chiều và phân cụm trong không gian 3D** vì có 784 đặc trưng, 70.000 mẫu, nhãn tham chiếu và khả năng tái lập tốt.

Giá trị nghiên cứu không nằm ở việc tạo được một biểu đồ 3D đẹp, mà ở khả năng trả lời các câu hỏi:

- Giảm từ 784 chiều xuống 3 chiều làm mất thông tin gì?
- PCA và UMAP bảo toàn cấu trúc khác nhau như thế nào?
- Phân cụm trong không gian gốc khác gì so với phân cụm trong không gian 3D?
- Các cụm được tìm thấy có đồng thuận với nhãn chữ số không?
- Kết quả có ổn định khi thay đổi tham số và số lượng mẫu không?

Sau khi hoàn thiện pipeline trên MNIST, nhóm nên thử thêm Fashion-MNIST hoặc một bộ dữ liệu bảng nhiều chiều để kiểm tra khả năng tổng quát hóa.

---

## Tài liệu tham khảo

1. TensorFlow/Keras, *MNIST dataset*: https://keras.io/api/datasets/mnist/
2. Yann LeCun, *The MNIST Database of Handwritten Digits*: http://yann.lecun.com/exdb/mnist/
3. McInnes, Healy & Melville, *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*: https://arxiv.org/abs/1802.03426
4. scikit-learn, *Clustering performance evaluation*: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
