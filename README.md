# Hệ thống gợi ý xuyên dịch vụ

Project này xây một pipeline recommendation chạy bằng script Python thuần, không phụ thuộc notebook. Ý tưởng chính là chuẩn hoá mọi nguồn dữ liệu về cùng một format:

```text
user_id | item_id | category | timestamp | interaction_strength
```

Nhờ hợp đồng dữ liệu này, phần EDA, chia train/test theo thời gian, train ALS, train Two-Tower, đánh giá cross-category và FastAPI serving đều dùng chung code. Khi chuyển từ Giai đoạn A sang Giai đoạn B, phần cần đổi chủ yếu nằm ở `src/data_loader.py`.

Xem thêm [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) để biết mục tiêu bài toán và cách đánh giá.

## Mục tiêu

Bài toán không chỉ là gợi ý item cùng loại, mà là gợi ý item/dịch vụ tiếp theo cho user dựa trên lịch sử tương tác xuyên nhiều category. Ví dụ: từ hành vi ở nhóm mobility/taxi có thể gợi ý hospitality/spa/hotel, hoặc từ lịch sử shopping có thể gợi ý một dịch vụ khác phù hợp.

Điểm cần chứng minh trong project:

- Pipeline chạy end-to-end và đo lường được.
- Có metric riêng cho gợi ý cross-category.
- Có baseline ALS để so sánh.
- Có model Two-Tower dùng embedding và feature category.
- Có API trả gợi ý kèm lý do đơn giản.
- Có thể đổi nguồn dữ liệu mà không phải viết lại toàn bộ hệ thống.

## Kiến trúc

```text
Amazon Reviews 2023
        |
CSV dịch vụ thật hoặc mô phỏng
        |
        v
Canonical interactions
        |
        v
Temporal train/test split
        |
        +--> ALS baseline
        |
        +--> Two-Tower + FAISS
        |
        v
Recall@K / NDCG@K
overall + cross-category
        |
        v
FastAPI serving
```

Model artifact lưu user/item factors, mapping metadata, lịch sử category của user và thống kê category co-occurrence. Serving vì vậy không cần import trực tiếp class PyTorch hoặc thư viện `implicit`. Với Two-Tower, FAISS index được tạo nếu môi trường hỗ trợ; nếu không, hệ thống vẫn có fallback bằng NumPy cho catalog nhỏ.

## Cấu trúc thư mục

```text
cross-service-rec/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── features.py
│   ├── train_baseline.py
│   ├── train_embedding.py
│   ├── evaluate.py
│   ├── serve.py
│   └── api/
│       └── main.py
├── reports/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

## Cài đặt nhanh

Khuyến nghị dùng Python 3.10 hoặc 3.11.

```bash
python -m venv .venv

# Linux / Colab:
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Chạy thử nhanh bằng dữ liệu demo:

```bash
python -m src.data_loader --source demo-amazon
python -m src.eda
python -m src.train_baseline --backend als
python -m src.train_embedding --epochs 80 --device auto
python -m src.serve amazon_user_00000 --model models/two_tower_v2.pkl -k 5
pytest -q
```

Dữ liệu demo chỉ dùng để kiểm tra pipeline trong vài phút. Không nên trình bày metric demo như kết quả thật trên Amazon Reviews.

## Giai đoạn A: Amazon Reviews 2023

Dataset chính thức: [McAuley-Lab/Amazon-Reviews-2023 trên Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

Dataset này rất lớn, nên loader đọc theo streaming từng config `raw_review_<category>` và giới hạn số dòng mỗi category. Nên bắt đầu nhỏ, chạy end-to-end trước, rồi tăng dần `--max-rows-per-category`.

```bash
python -m src.data_loader \
  --source amazon-hf \
  --categories All_Beauty Electronics Home_and_Kitchen Grocery_and_Gourmet_Food \
  --max-rows-per-category 250000 \
  --min-user-interactions 5 \
  --min-item-interactions 3 \
  --min-user-categories 2 \
  --output data/processed/amazon_interactions.parquet

python -m src.eda \
  --data data/processed/amazon_interactions.parquet \
  --output-dir reports/eda

python -m src.train_baseline \
  --data data/processed/amazon_interactions.parquet \
  --model-output models/als_v1.pkl \
  --metrics-output reports/baseline_metrics.json

python -m src.train_embedding \
  --data data/processed/amazon_interactions.parquet \
  --model-output models/two_tower_v2.pkl \
  --epochs 20 \
  --batch-size 4096 \
  --device auto
```

Nếu sau khi lọc không còn user đa category, nguyên nhân thường là dữ liệu lấy mẫu còn quá ít hoặc ngưỡng lọc quá chặt. Hãy tăng `--max-rows-per-category`, chọn category nhỏ hơn, hoặc tạm giảm `--min-user-interactions` và `--min-item-interactions` khi khám phá.

## Giai đoạn B: Ghép nhiều dataset public để giả lập đa dịch vụ (khuyến nghị)

Hiện không có dataset public nào mô tả đầy đủ hành vi của cùng một người dùng trên nhiều dịch vụ giống hệ sinh thái Vingroup. Vì vậy, hướng triển khai phù hợp là ghép 2-3 dataset thuộc các domain khác nhau, xem mỗi dataset như một dịch vụ, sau đó tạo `user_id` giả để mô phỏng hành vi xuyên dịch vụ.

| Dịch vụ giả lập | Dataset gợi ý |
|---|---|
| Di chuyển (tương tự Xanh SM) | **NYC Taxi Trip Data** hoặc **Chicago Taxi Trips** (NYC Open Data/Kaggle) |
| Lưu trú, nghỉ dưỡng (tương tự Vinpearl) | **Hotel Booking Demand** (Kaggle) |
| Mua sắm, ăn uống | **Yelp Open Dataset** hoặc **Instacart Market Basket Analysis** |
| Y tế (tương tự Vinmec) | **Medical Appointment No Shows** (Kaggle); chỉ sử dụng dữ liệu lịch hẹn, không sử dụng thông tin y tế nhạy cảm |

Mỗi nguồn dữ liệu được chuẩn hóa độc lập về cùng schema:

```text
user_id | item_id | category | timestamp | interaction_strength
```

Do các dataset không có khóa người dùng chung, bước ghép dữ liệu sẽ sinh `user_id` giả và gán các bản ghi từ từng dịch vụ theo xác suất có trọng số. Việc gán cần có các ràng buộc hợp lý để tạo pattern có thể học được:

- **Thời gian:** các sự kiện của cùng một user nằm trong chuỗi thời gian hợp lý, ví dụ đặt taxi rồi nhận phòng khách sạn sau 1-3 ngày.
- **Địa điểm:** khi nguồn dữ liệu có thông tin vị trí, ưu tiên ghép chuyến đi và dịch vụ ở cùng khu vực.
- **Persona:** tạo các nhóm như khách du lịch cuối tuần, cư dân địa phương hoặc người dùng thường xuyên mua sắm; mỗi nhóm có xác suất sử dụng dịch vụ khác nhau.
- **Tính tái lập:** cố định random seed và lưu cấu hình trọng số để có thể sinh lại cùng một phiên bản dữ liệu.

Đây là **dữ liệu mô phỏng có kiểm soát**, không phải dữ liệu khách hàng thật và không được dùng để khẳng định insight kinh doanh thực tế. Mục tiêu của Giai đoạn B là chứng minh pipeline có thể hoạt động trên nhiều nguồn dữ liệu (ghép nhiều dataset public) và kiểm thử ý tưởng recommendation xuyên dịch vụ — **giai đoạn này không đụng tới và không mô phỏng dữ liệu Vingroup thật**, các tên dịch vụ (`xanh_sm`, `vinpearl`, `vinmec`, `shopping`) chỉ là nhãn category minh hoạ cho kịch bản đa dịch vụ.

Repo hiện cung cấp `normalize_service_csv` để ánh xạ từng file CSV về schema chung và `generate_multi_service_simulation` để tạo tập dữ liệu persona mẫu. Có thể chạy nhanh bộ sinh dữ liệu mẫu bằng lệnh:

```bash
python -m src.data_loader \
  --source multi-service-sim \
  --n-users 5000 \
  --output data/processed/multi_service_sim.parquet

python -m src.eda --data data/processed/multi_service_sim.parquet
python -m src.train_baseline --data data/processed/multi_service_sim.parquet
python -m src.train_embedding --data data/processed/multi_service_sim.parquet --device auto
```

`data_loader` ghi kèm 1 file `<output>.provenance.json` bên cạnh dữ liệu, đánh dấu nguồn là `synthetic_controlled` hoặc `public_dataset`. `train_baseline.py`/`train_embedding.py` đọc lại file này và nhúng `data_provenance` vào artifact, `reports/*_metrics.json` và `reports/model_comparison.md` — nếu dữ liệu là mô phỏng, report và cả response API `/recommend` sẽ tự in kèm cảnh báo "không dùng số liệu này làm bằng chứng kinh doanh".

Trong bản mô phỏng mẫu, nhóm khách du lịch cuối tuần thường có chuỗi taxi → khách sạn trong vài ngày; nhóm cư dân địa phương có thể có lịch hẹn y tế và mua sắm. Tất cả ID mô phỏng bắt đầu bằng `sim:`. Khi đã tải và chuẩn hóa các dataset public, chỉ cần thay đầu vào ở `src/data_loader.py`; phần EDA, train, evaluate và API được giữ nguyên.

## Đánh giá

Project dùng temporal leave-one-out: với mỗi user, giữ item mới nhất làm test, các tương tác trước đó làm train. Recommendation sẽ loại các item user đã thấy trong train.

Metric chính:

- `Recall@K`: item test có xuất hiện trong top K gợi ý không.
- `NDCG@K`: giống Recall nhưng thưởng thêm nếu item đúng nằm ở vị trí cao.
- `Cross-category Recall@K`: chỉ tính các case mà category của item test khác category chiếm ưu thế trong lịch sử train của user.

Không dùng Accuracy/RMSE làm metric chính vì đây là bài toán ranking, không phải bài toán dự đoán chính xác rating.

Kết quả được lưu tại:

- `reports/baseline_metrics.json`
- `reports/two_tower_metrics.json`
- `reports/evaluation_metrics.json`
- `reports/model_comparison.md`

Kết quả demo (dữ liệu mô phỏng, `data_provenance: synthetic_controlled` — chỉ để kiểm tra pipeline, không phải bằng chứng kinh doanh):

| Model | Recall@10 overall | NDCG@10 overall | Recall@10 cross-category | NDCG@10 cross-category |
|---|---:|---:|---:|---:|
| ALS | 0.5680 | 0.2819 | 0.5361 | 0.2696 |
| Two-Tower | 0.6880 | 0.3198 | 0.6084 | 0.2799 |

Kết quả thật trên Amazon Reviews 2023 (`data_provenance: public_dataset`, categories `All_Beauty`/`Electronics`/`Home_and_Kitchen`/`Grocery_and_Gourmet_Food`, `--max-rows-per-category 250000`, sau lọc còn 16,261 user đa category, 298,387 interaction):

| Model | Recall@10 overall | NDCG@10 overall | Recall@10 cross-category | NDCG@10 cross-category |
|---|---:|---:|---:|---:|
| ALS | 0.0251 | 0.0156 | 0.0249 | 0.0157 |
| Two-Tower | 0.0289 | 0.0165 | 0.0210 | 0.0122 |

**Đọc kết quả thật này một cách trung thực:** giả thuyết chính của project — category embedding giúp Two-Tower gợi ý cross-category tốt hơn — **không được xác nhận trên dữ liệu thật**. Two-Tower cross-category (0.0210) thấp hơn cả ALS cross-category (0.0249) và thấp hơn chính Two-Tower same-category (0.0352), ngược lại hoàn toàn so với kết quả trên dữ liệu mô phỏng. Lý do nhiều khả năng: category của Amazon (nhóm sản phẩm) không mang tính "hành trình dịch vụ nối tiếp nhau" như kịch bản mô phỏng (taxi → khách sạn) — không có cơ sở tự nhiên để việc mua đồ điện tử dự đoán được việc mua mỹ phẩm, nên feature category-embedding gây nhiễu nhiều hơn là giúp ích cho đúng phần cross-category. Recall tuyệt đối thấp (~0.02-0.03) cũng phản ánh catalog 48k item và cấu hình mặc định (`epochs=15`, `embedding_dim=64`, `category_weight=0.2`) chưa được tune cho quy mô này — chưa thử tăng epoch, tune `category_weight`, hay lọc catalog theo phổ biến trước khi kết luận đây là giới hạn cứng của kiến trúc.

## API serving

Chạy API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Gọi thử:

```bash
curl "http://127.0.0.1:8000/recommend/sim_user_000000?context=xanh_sm_trip_to_vinpearl&top_k=5"
```

Có thể chọn model bằng biến môi trường:

```bash
MODEL_PATH=models/als_v1.pkl uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Query parameter hữu ích:

- `top_k`: số lượng item cần gợi ý.
- `target_category`: chỉ gợi ý trong category cụ thể.
- `cross_category_only`: ưu tiên loại category user đã tương tác nhiều nhất.
- `context`: mô phỏng ngữ cảnh, ví dụ `xanh_sm_trip_to_vinpearl`.

User chưa có trong model (cold-start) không trả lỗi 404 — API trả HTTP 200 với `personalized: false` và danh sách item phổ biến nhất (ưu tiên trong `target_category` nếu suy ra được từ `context`, ngược lại lấy phổ biến toàn catalog). Đây là fallback tối thiểu; production thật nên nâng cấp thêm rule theo context chi tiết hơn hoặc session-based recommendation.

Mỗi response còn kèm `data_provenance` và `caution` (nếu model được train trên dữ liệu mô phỏng) để phân biệt rõ gợi ý nào có bằng chứng dữ liệu thật đứng sau, gợi ý nào chỉ đang chạy trên dữ liệu kiểm thử.

### Bảo vệ vận hành tối thiểu

- **Logging**: mỗi request tới `/recommend` ghi 1 dòng log có `request_id`, `user_id`, `model_type`, `personalized`, `data_provenance`, `latency_ms`.
- **API key**: đặt biến môi trường `API_KEY` để bắt buộc header `X-API-Key` cho `/reload-model` (nếu không đặt, endpoint mở — chỉ phù hợp môi trường dev). `/recommend` hiện không yêu cầu key để giữ đơn giản cho demo; khi triển khai thật nên bọc toàn bộ API sau API gateway/reverse proxy có auth.
- **Rate limit**: giới hạn mặc định 60 request/phút mỗi API key (hoặc mỗi IP nếu không dùng key), cấu hình qua `RATE_LIMIT_PER_MINUTE` (đặt `0` để tắt). Đây là rate limit trong-process, không chia sẻ giữa nhiều worker/instance — chỉ đủ cho 1 instance đơn, không thay thế API gateway thật khi scale ngang.
- **Hot reload model**: gọi `POST /reload-model` (kèm header `X-API-Key` nếu đã cấu hình) để nạp lại artifact tại `MODEL_PATH` mà không cần restart process. Lưu ý: endpoint này chỉ đổi file model đang phục vụ, **không** tự xoá dữ liệu 1 user khỏi artifact hiện tại — xem [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) mục 3.

```bash
API_KEY=secret123 RATE_LIMIT_PER_MINUTE=120 uvicorn src.api.main:app --host 0.0.0.0 --port 8000

curl -X POST -H "X-API-Key: secret123" http://127.0.0.1:8000/reload-model
```

### Container hoá (serving only)

`Dockerfile` chỉ đóng gói phần serving (`requirements-serving.txt`: numpy, pandas, joblib, fastapi, uvicorn, pydantic) — không cài `torch`/`implicit`/`faiss-cpu`. Train vẫn chạy ngoài container (máy local/Colab) rồi mount artifact vào.

```bash
docker build -t cross-service-rec .
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/models/als_v1.pkl \
  -v "$(pwd)/models:/app/models" \
  cross-service-rec
```

Đã build + chạy thử thành công (`/health` → 200, `/recommend/<user>` trả gợi ý kèm `data_provenance`). Lưu ý nếu chạy lệnh trên bằng **Git Bash trên Windows**: Git Bash tự dịch các đường dẫn kiểu Unix (`/app/...`) trong biến môi trường sang đường dẫn Windows trước khi truyền vào container, làm sai `MODEL_PATH`. Nếu gặp lỗi "No trained artifact at C:/Program Files/...", thêm `MSYS_NO_PATHCONV=1` trước lệnh `docker run` (PowerShell/CMD/Linux/macOS không bị ảnh hưởng).

## Chạy trên Colab Pro

Nên lưu code trong GitHub hoặc Google Drive. Notebook chỉ dùng như terminal chạy lệnh, không đặt logic train trong cell.

```bash
git clone <your-repository-url>
cd cross-service-rec
pip install -r requirements.txt

python -m src.data_loader --source demo-amazon
python -m src.train_baseline
python -m src.train_embedding --device cuda --batch-size 8192
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Trước khi chạy lâu, hãy chạy một vòng nhỏ end-to-end. Sau đó mới tăng kích thước dữ liệu. Nên lưu `data/processed`, `models` và `reports` vào Drive hoặc tải xuống trước khi runtime Colab bị ngắt.

Ghi chú: ALS từ thư viện `implicit` chủ yếu chạy CPU. Phần Two-Tower bằng PyTorch mới hưởng lợi rõ từ GPU.

## Mapping checkpoint

- CP0: problem statement và setup project.
- CP1: loader chuẩn hoá dữ liệu, lọc sparsity, giữ user đa category.
- CP2: EDA category distribution và category co-occurrence.
- CP3: ALS baseline, split theo thời gian, Recall@10/NDCG@10.
- CP4: đánh giá riêng cross-category.
- CP5: Two-Tower có category embedding và FAISS index.
- CP6: FastAPI serving có context và lý do gợi ý.
- CP7-9: chuẩn hoá/mô phỏng dữ liệu đa dịch vụ và chạy lại pipeline.
- CP10: README, report, test và phần giới hạn.

## Giới hạn và hướng mở rộng

Phần giải thích trong API dựa trên category co-occurrence mô tả, không phải quan hệ nhân quả và không phải xác suất đã hiệu chỉnh. Dữ liệu mô phỏng chỉ kiểm tra phần mềm và pattern được cài vào dữ liệu, không chứng minh insight kinh doanh thật.

Nếu có dữ liệu thật trong hệ sinh thái, bước tiếp theo nên gồm consented shared identity, kiểm soát privacy/governance, real-time context, feature store dùng chung, cold-start strategy, drift monitoring, online A/B test và ràng buộc an toàn theo business.
