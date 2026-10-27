# HƯỚNG DẪN XÂY DỰNG HỆ THỐNG KHÁM PHÁ CẤU TRÚC DỮ LIỆU ĐA CHIỀU
## BẰNG GIẢM CHIỀU VÀ PHÂN CỤM TRONG KHÔNG GIAN 3D (100% TYPESCRIPT / JAVASCRIPT)

---

## 1. Tiêu chuẩn bộ dữ liệu có cấu trúc (Structured / Tabular Data)

Đối với đề tài xây dựng hệ thống trên nền tảng **TypeScript / JavaScript** (chạy trên Web Client hoặc Fullstack Node.js), bộ dữ liệu cần đáp ứng các tiêu chí sau để vừa đảm bảo tính khoa học, vừa tối ưu hiệu năng tính toán và hiển thị:

### 1.1. Số lượng mẫu (Samples / Records)
- **Khoảng lý tưởng:** **500 – 5.000 mẫu**.
  - *Lý do:* 
    - Hiệu năng hiển thị: Three.js (WebGL) render cực kỳ mượt mà ở 60 FPS với số lượng hạt dưới 10.000 điểm mà không cần kỹ thuật nén quá phức tạp.
    - Hiệu năng thuật toán: Các giải thuật giảm chiều phi tuyến như t-SNE hoặc UMAP có độ phức tạp tính toán xấp xỉ $O(N^2)$ hoặc $O(N \log N)$. Nếu chạy trực tiếp bằng JavaScript trên trình duyệt, 500 – 3.000 mẫu sẽ cho thời gian tính toán chỉ mất **2 – 8 giây**, không gây đơ/lag trình duyệt.
- **Ngưỡng tối đa cho Web Client:** ~10.000 mẫu (nếu dùng Web Workers hoặc kỹ thuật lấy mẫu phân tầng - Stratified Sampling).

### 1.2. Số lượng trường và đặc trưng (Fields & Features)
- **Đặc trưng số (Numerical Features) để giảm chiều:** **10 – 50 đặc trưng**.
  - *Tại sao không nên dưới 4 đặc trưng?* Nếu dữ liệu gốc chỉ có 2-3 đặc trưng thì đã có thể trực quan trực tiếp lên không gian 2D/3D mà không cần kỹ thuật giảm chiều (PCA/t-SNE/UMAP). Mục tiêu của đề tài là chứng minh giá trị của việc "nén thông tin từ không gian nhiều chiều ($D \ge 10$) về 3 chiều ($d = 3$)".
  - *Tại sao không nên quá 100 đặc trưng trên JS thuần?* Với dữ liệu bảng (Tabular), 10 – 50 đặc trưng là tiêu chuẩn của các bài toán phân loại/phân cụm thực tế (y tế, tài chính, cảm biến), giúp ma trận hiệp phương sai tính toán nhanh chóng.
- **Trường định danh & Nhãn (Metadata / Ground-truth Labels):** Cần **1 – 2 trường**.
  - `ID`: Mã định danh bản ghi (để hiển thị tooltip, tra cứu chi tiết).
  - `Label / Category`: Nhãn phân loại thực tế (Ground truth) để:
    - Tô màu điểm dữ liệu theo nhãn gốc nhằm so sánh trực quan với nhãn cụm (Clusters) do thuật toán tìm ra.
    - Tính toán độ đo đánh giá ngoài (External Metrics): **ARI** (Adjusted Rand Index), **NMI** (Normalized Mutual Information).

### 1.3. Đánh giá bộ dữ liệu sẵn có trong dự án: `data.csv` (Breast Cancer Wisconsin)
Tệp `data.csv` trong thư mục của bạn là một lựa chọn **kinh điển và cực kỳ xuất sắc** cho đề tài này:
- **Số mẫu:** 569 dòng (rất vừa vặn, tính toán cực nhanh trong trình duyệt).
- **Số đặc trưng:** 30 đặc trưng số liên tục mô tả kích thước, kết cấu, chu vi tế bào ung thư.
- **Nhãn phân loại:** Cột `diagnosis` (M = Malignant / Ác tính, B = Benign / Lành tính).
- **Ý nghĩa thực tế cao:** Trực quan hóa giúp bác sĩ/chuyên gia quan sát xem các ca lành tính và ác tính tách cụm như thế nào trong không gian 3 chiều sau khi giảm từ 30 chiều.

---

## 2. Kiến trúc tổng thể hệ thống (System Architecture)

Hệ thống được thiết kế theo mô hình **Client-Side SPA (Single Page Application)** hoặc **Fullstack TypeScript**:

```
+-----------------------------------------------------------------------------------+
|                                GIAO DIỆN NGƯỜI DÙNG (UI/UX)                        |
|  - Bảng điều khiển tham số (Control Panel: k, perplexity, neighbors, metric...)   |
|  - Khung hiển thị đồ họa 3D (Three.js WebGL Canvas + OrbitControls)               |
|  - Biểu đồ phụ trợ (Loss curve, Silhouette plot, Scree plot)                      |
|  - Bảng thông tin chi tiết điểm (Tooltip / Inspector Panel khi click vào điểm)    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        TẦNG TIỀN XỬ LÝ (PREPROCESSING)                            |
|  - Parse CSV / JSON (PapaParse)                                                   |
|  - Tách Features (Ma trận X) và Labels (Vector y)                                 |
|  - Chuẩn hóa dữ liệu: Z-Score (StandardScaler) hoặc Min-Max Normalization         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                TẦNG TÍNH TOÁN & THUẬT TOÁN (WEB WORKERS / WORKER THREADS)         |
|  + Giảm chiều (Dimensionality Reduction):                                         |
|    - PCA (Principal Component Analysis): ml-pca                                   |
|    - t-SNE (t-Distributed Stochastic Neighbor Embedding): tsne-js                |
|    - UMAP (Uniform Manifold Approximation and Projection): umap-js                |
|  + Phân cụm (Clustering):                                                         |
|    - K-Means: ml-kmeans                                                           |
|    - DBSCAN: density-clustering                                                   |
|  + Đánh giá chất lượng (Evaluation Metrics):                                      |
|    - Internal: Silhouette Coefficient, Davies-Bouldin Index                       |
|    - External: ARI (Adjusted Rand Index), Accuracy/Purity (so với nhãn gốc)       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      TẦNG RENDER ĐỒ HỌA 3D (THREE.JS / WEBGL)                     |
|  - THREE.Points / THREE.InstancedMesh (Tọa độ X, Y, Z lấy từ kết quả giảm chiều)  |
|  - Color Mapping: Tô màu theo Nhãn gốc HOẶC theo Cụm K-Means/DBSCAN                |
|  - Tương tác: Xoay, Zoom, Pan, Raycasting (Hover/Click điểm để xem chi tiết)     |
|  - Hiển thị tâm cụm (Cluster Centroids) & bounding hull 3D                        |
+-----------------------------------------------------------------------------------+
```

---

## 3. Các thư viện TypeScript / JavaScript đề xuất sử dụng

| Nhóm chức năng | Thư viện khuyến nghị | Mục đích sử dụng |
|---|---|---|
| **Bundler & Framework** | Vite + React + TypeScript | Khởi tạo dự án nhanh, type-safe, hot reload |
| **Giao diện & UI** | Tailwind CSS / Lucide React / shadcn-ui | Giao diện hiện đại, tối ưu, hỗ trợ Dark Mode |
| **Xử lý dữ liệu & CSV** | `papaparse` | Đọc và phân tích file CSV cực nhanh |
| **Toán học & Tiền xử lý** | `mathjs` hoặc `ml-matrix` | Các phép toán vector, ma trận, chuẩn hóa |
| **Giảm chiều dữ liệu** | `ml-pca` (PCA)<br>`tsne-js` (t-SNE)<br>`umap-js` (UMAP) | Giảm từ $D$ chiều xuống 3 chiều ($X, Y, Z$) |
| **Phân cụm dữ liệu** | `ml-kmeans` (K-Means)<br>`density-clustering` (DBSCAN) | Phân nhóm dữ liệu trong không gian gốc hoặc 3D |
| **Hiển thị 3D tương tác** | `three` (@types/three)<br>`@react-three/fiber` (nếu dùng React) | Render không gian 3D, camera, raycasting |
| **Biểu đồ bổ trợ 2D** | `chart.js` hoặc `plotly.js-dist-min` | Biểu đồ Silhouette, Scree plot (Explained Variance) |

---

## 4. Hướng dẫn chi tiết từng bước xây dựng hệ thống từ đầu

### Bước 1: Khởi tạo dự án với Vite & TypeScript
Chạy các lệnh trong terminal để tạo khung dự án:
```bash
# Khởi tạo dự án React + TypeScript với Vite
npm create vite@latest 3d-data-explorer -- --template react-ts

# Chuyển vào thư mục dự án
cd 3d-data-explorer

# Cài đặt các thư viện cốt lõi
npm install three @types/three papaparse @types/papaparse ml-matrix ml-pca ml-kmeans tsne-js umap-js lucide-react clsx
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Bước 2: Xây dựng Module nạp và Tiền xử lý dữ liệu (Preprocessing)
1. **Đọc dữ liệu CSV:** Cho phép người dùng tải file CSV lên hoặc chọn sẵn dataset mẫu (`data.csv`).
2. **Lọc trường:** Cho phép người dùng chọn:
   - Các cột làm đặc trưng tính toán (Numerical Features).
   - Cột nhãn phân loại (Ground-truth Label - ví dụ `diagnosis`).
   - Cột định danh (ID).
3. **Chuẩn hóa (Normalization / Standardization):**
   - **Bắt buộc** phải chuẩn hóa vì các biến có đơn vị khác nhau (ví dụ: `area_mean` ~ 1000, trong khi `smoothness_mean` ~ 0.1).
   - Công thức chuẩn hóa Z-score:
     $$z = \frac{x - \mu}{\sigma}$$

### Bước 3: Xây dựng Module Giảm chiều (Dimensionality Reduction)
Hệ thống cần cung cấp ít nhất 2 đến 3 thuật toán để người dùng so sánh:
1. **PCA (Principal Component Analysis):** Tuyến tính, bảo toàn phương sai lớn nhất, tính toán tức thì. Lấy 3 thành phần chính đầu tiên ($PC_1, PC_2, PC_3$) làm tọa độ $(x, y, z)$.
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Phi tuyến, bảo toàn tốt cấu trúc lân cận địa phương, tách cụm trực quan rất rõ ràng. Cho phép tinh chỉnh: `perplexity`, `learningRate`, `iterations`.
3. **UMAP (Uniform Manifold Approximation and Projection):** Cân bằng giữa cấu trúc cục bộ và toàn cục, chạy nhanh hơn t-SNE. Cho phép tinh chỉnh: `nNeighbors`, `minDist`.

> **Mẹo kỹ thuật quan trọng:** Các thuật toán t-SNE và UMAP nên được chạy trong **Web Worker** để không làm đóng băng giao diện người dùng (UI) khi đang tính toán vòng lặp.

### Bước 4: Xây dựng Module Phân cụm (Clustering)
Người dùng có thể chọn:
1. **K-Means:** Cho phép chọn số cụm $K$ (từ 2 đến 10). Tìm tâm cụm (Centroids) và gán nhãn cụm cho từng điểm.
2. **DBSCAN:** Phân cụm dựa trên mật độ, có khả năng phát hiện điểm nhiễu (Outliers / Noise).
3. **Lựa chọn không gian phân cụm:**
   - Phân cụm trên **không gian gốc** (30 chiều) rồi chiếu nhãn lên 3D.
   - Hoặc phân cụm trực tiếp trên **không gian 3D** sau giảm chiều.

### Bước 5: Xây dựng Canvas 3D tương tác (Three.js WebGL Engine)
1. **Dựng Scene, Perspective Camera, Ambient/Directional Light**.
2. **Sử dụng `THREE.Points` hoặc `THREE.InstancedMesh`:**
   - Đặt tọa độ mỗi hạt tại $(x, y, z)$ thu được từ bước giảm chiều.
   - Scale tọa độ về khoảng $[-50, 50]$ để camera quan sát vừa vặn.
3. **Tô màu linh hoạt (Dynamic Color Mapping):**
   - Chế độ 1: Tô màu theo **Nhãn thực tế (Ground-truth)**.
   - Chế độ 2: Tô màu theo **Cụm thuật toán tìm ra (Predicted Clusters)**.
4. **Tương tác trực quan:**
   - `OrbitControls`: Xoay 360 độ, phóng to, thu nhỏ, trượt khung nhìn.
   - `Raycasting`: Khi người dùng di chuột (hover) hoặc bấm vào một điểm hạt:
     - Làm nổi bật điểm đó.
     - Hiển thị popover/tooltip chi tiết thông số gốc của mẫu (ID, diagnosis, radius, texture, v.v.).

### Bước 6: Đánh giá chất lượng và Thống kê (Evaluation Metrics)
Hệ thống tính toán và hiển thị các chỉ số:
1. **Chất lượng phân cụm nội tại (Internal):**
   - **Silhouette Score:** Đánh giá mức độ tách biệt giữa các cụm (từ -1 đến 1).
2. **Độ tương đồng với nhãn thật (External):**
   - **Adjusted Rand Index (ARI):** So sánh độ khớp giữa nhãn K-Means và nhãn ung thư gốc (`M` và `B`).

---

## 5. Kế hoạch phân công công việc gợi ý cho nhóm

| Giai đoạn | Nội dung công việc | Người phụ trách / Vai trò |
|---|---|---|
| **Tuần 1** | Khảo sát dữ liệu `data.csv`, thiết lập khung dự án Vite + TypeScript, dựng giao diện cơ bản | Thành viên 1 + 2 |
| **Tuần 2** | Viết module tiền xử lý (CSV Parse, Z-score) và tích hợp Three.js 3D Scatter Plot cơ bản | Thành viên 1 |
| **Tuần 3** | Cài đặt thuật toán PCA, t-SNE, UMAP (chạy trên Web Worker) để sinh tọa độ 3D | Thành viên 2 |
| **Tuần 4** | Cài đặt thuật toán K-Means, DBSCAN và tính toán Silhouette Score, ARI | Thành viên 3 |
| **Tuần 5** | Hoàn thiện tương tác Raycasting (Hover/Click), bộ lọc (Filters), tinh chỉnh UI/UX | Cả nhóm |
| **Tuần 6** | Viết báo cáo đề tài, đối sánh kết quả các thuật toán và chuẩn bị slide demo | Cả nhóm |

---

## 6. Các tính năng chuyên sâu & Nâng cấp đẳng cấp đồ án điểm 10

Hệ thống đã được trang bị đầy đủ các tính năng hiện đại nhất theo tiêu chuẩn phần mềm khoa học dữ liệu quốc tế:

1. **PCA 3D Biplot Feature Vectors (Véc-tơ đặc trưng không gian 3D):**
   - Trích xuất 3 eigenvector từ ma trận hiệp phương sai.
   - Tính độ lớn véc-tơ đặc trưng $L_j = \sqrt{V_{1,j}^2 + V_{2,j}^2 + V_{3,j}^2}$.
   - Vẽ mũi tên 3D phát sáng và nhãn văn bản billboard (Top 6 đặc trưng có ảnh hưởng mạnh nhất như `radius_mean`, `concavity_mean`...) định hình sự phân bố của cụm tế bào ung thư.

2. **Dò tìm số cụm K tối ưu (Elbow Method & Silhouette Curve):**
   - Tự động chạy $K = 2 \to 8$, tính toán đồng thời tổng bình phương độ lệch trong cụm (Inertia/WCSS) và hệ số Silhouette Score.
   - Vẽ biểu đồ đường 2 trục tương tác SVG, tự động phát hiện điểm khuỷu tay (Elbow Point tại $K = 2$) và cung cấp nút "Áp dụng K này ngay".

3. **Màn hình đôi 3D so sánh đồng bộ (Dual Synchronized Viewports):**
   - Cho phép đặt 2 thuật toán cạnh nhau (ví dụ: Màn hình A chạy PCA 3D, Màn hình B chạy t-SNE 3D).
   - Đồng bộ góc xoay camera qua `requestAnimationFrame` và `OrbitControls` mượt mà ở 60 FPS.

4. **Tụ cụm động thời gian thực (Live t-SNE / UMAP Optimization):**
   - Xem trực tiếp các điểm hạt di chuyển, co cụm và tách nhóm qua từng vòng lặp Gradient Descent trong không gian 3 chiều.

5. **Trung tâm Bảo mật dữ liệu & Chia sẻ Mạng LAN Wi-Fi:**
   - Bảo mật 4 lớp: Mã hóa TLS 1.3 / SSL, xác thực SCRAM-SHA-256, ẩn biến môi trường `.env`, cơ chế tự hành In-Memory Fallback khi mất kết nối cloud.
   - Trình chiếu đa thiết bị: Chạy lệnh `npm run dev:host` để giảng viên và sinh viên truy cập qua IP Wi-Fi `http://192.168.x.x:3000`.

6. **Chụp ảnh 3D 4K & Xuất dữ liệu CSV đã gắn nhãn cụm:**
   - Nút "Chụp 3D" lưu ảnh PNG sắc nét phục vụ báo cáo.
   - Nút "Xuất file CSV" lưu toàn bộ dữ liệu kèm tọa độ $X, Y, Z$ và nhãn cụm dự đoán `predicted_cluster`.

