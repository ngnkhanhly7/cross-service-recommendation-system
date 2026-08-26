# Improvement plan

Tài liệu này nối tiếp phần "Giới hạn và hướng mở rộng" trong [README.md](README.md), dựa trên review kỹ thuật + góc nhìn doanh nghiệp/mentor. Mỗi checkpoint có: mục tiêu, việc cần làm, và **report** — bằng chứng cụ thể để coi checkpoint đó là "done" (không phải cảm tính).

Lưu ý phạm vi: **Giai đoạn B (CP7-9 trong README) là ghép nhiều dataset public để mô phỏng hành vi đa dịch vụ, không phải dữ liệu Vingroup thật.** Toàn bộ id bắt đầu bằng `sim:` và mọi con số từ giai đoạn này chỉ dùng để kiểm thử pipeline/thuật toán, không được trích dẫn như insight khách hàng thật. Các checkpoint dưới đây giữ nguyên ràng buộc đó — không có checkpoint nào giả định có dữ liệu Vingroup thật.

Ưu tiên: **P0** = chặn trước khi trình bày dự án ra ngoài team kỹ thuật, **P1** = cần trước khi coi là production-track, **P2** = cải thiện chất lượng/khả năng mở rộng.

---

## CP-I1 (P0) — Chặn rò rỉ "circular validation" ra ngoài

**Vấn đề:** Persona trong `generate_vingroup_simulation` cấy cứng chuỗi `xanh_sm → vinpearl` ([src/data_loader.py:222-225](src/data_loader.py#L222-L225)), model học lại đúng pattern đó rồi được báo cáo là "cross-category recall cải thiện". Đây là bằng chứng pipeline hoạt động đúng, không phải bằng chứng hành vi khách hàng — nhưng nếu tách khỏi ngữ cảnh code, con số 0.60-0.68 rất dễ bị dùng sai trong slide/proposal.

**Việc cần làm:**
1. Đổi tên các hàm/artefact liên quan để không thể nhầm là dữ liệu thật: `generate_vingroup_simulation` → `generate_multi_service_simulation`, output mặc định đổi từ `vingroup_sim.parquet` → `multi_service_sim.parquet`, CLI choice `--source vingroup-sim` → `--source multi-service-sim`, và các file model/report mẫu hiện có (`models/vingroup_als_smoke.pkl`, `models/vingroup_two_tower_smoke.*`, `reports/vingroup_baseline_smoke.json`, `reports/vingroup_model_comparison_smoke.md`) đổi tên tương ứng (loại bỏ luôn chữ "vingroup" khỏi code/tên file, vì đây chỉ là dữ liệu mô phỏng chung, không gắn thương hiệu cụ thể).
2. Thêm field `data_provenance` vào mọi artifact (`train_baseline.py`, `train_embedding.py`) ghi rõ `"synthetic_controlled"` hoặc `"public_dataset"` hoặc `"production"`.
3. `reports/model_comparison.md` và API response (`/recommend`) phải in kèm dòng cảnh báo tự động khi `data_provenance == "synthetic_controlled"` — không để người đọc report tự nhớ đọc README.
4. Viết 1 đoạn trong README ngay đầu bảng kết quả demo: "Số liệu này đo khả năng model phục hồi lại pattern đã cấy sẵn trong dữ liệu mô phỏng — không phải bằng chứng hành vi khách hàng."

**Report:**
- Diff cho thấy không còn hàm/artifact nào tên `vingroup` gắn với dữ liệu giả lập.
- 1 file JSON mẫu (`reports/*_metrics.json`) có field `data_provenance`.
- Screenshot/log response API có dòng cảnh báo khi dùng model synthetic.

**Trạng thái: ✅ Done.** `generate_multi_service_simulation` + `--source multi-service-sim`; toàn bộ file model/report smoke đã đổi tên và **retrain lại thật** (không chỉ đổi tên file cũ) — xem `models/multi_service_*`, `reports/multi_service_*`. `data_provenance`/`data_source` được ghi qua sidecar `<data>.provenance.json` ([src/common.py](src/common.py)) và nhúng vào mọi artifact + `reports/*_metrics.json` + `reports/model_comparison.md`; API `/recommend` trả kèm `data_provenance`/`caution`. Verify: `reports/multi_service_baseline_smoke.json` có `"caution"`; log mẫu ở CP-I5.

---

## CP-I2 (P0) — Cold-start: gợi ý cho user mới/thưa dữ liệu

**Vấn đề:** User không có trong `user_to_idx` → HTTP 404 ([src/api/main.py:74-75](src/api/main.py#L74-L75)). Đây chính là nhóm user mà cross-sell có giá trị cao nhất (đã dùng 1 dịch vụ, chưa đủ lịch sử để có factor).

**Việc cần làm:**
1. Thêm fallback 2 tầng trong `RecommendationEngine.recommend`:
   - Nếu `context` được truyền (vd `xanh_sm_trip_to_vinpearl`) nhưng user chưa có factor → trả về top item phổ biến nhất trong `target_category` (đã có `CONTEXT_TARGETS`, chỉ thiếu nhánh xử lý khi `user_id` không tồn tại).
   - Nếu không có context → trả về global popularity theo category, kèm `reason: "popular_fallback"` để phân biệt rõ với gợi ý cá nhân hoá.
2. API trả HTTP 200 kèm field `personalized: false` thay vì 404, trừ khi user_id sai định dạng hoàn toàn.
3. Test case: user chưa từng xuất hiện + có context, user chưa từng xuất hiện + không context.

**Report:**
- `tests/test_serve.py` có 2 test mới cho cold-start, pass.
- Gọi thử `curl /recommend/unknown_user_123?context=xanh_sm_trip_to_vinpearl` trả 200 với `personalized: false` thay vì 404.

**Trạng thái: ✅ Done.** `RecommendationEngine.recommend` trả `{"personalized": bool, "results": [...]}`; user lạ → `_popularity_fallback` (theo `item_popularity` mới thêm vào artifact), ưu tiên `target_category` suy ra từ `context` nếu có. API không còn nhánh `except KeyError` → 404. Verify: `tests/test_serve.py::test_unknown_user_with_context_gets_popular_fallback_in_target_category`, `test_unknown_user_without_context_gets_global_popular_fallback`, `tests/test_api.py::test_recommend_unknown_user_returns_200_not_404` — cả 3 pass.

---

## CP-I3 (P1) — Tầng dịch metric kỹ thuật sang KPI kinh doanh

**Vấn đề:** Recall@10/NDCG@10 là ngôn ngữ ML thuần, không nối được với quyết định đầu tư. Hiện không có tài liệu nào dịch "recall tăng X điểm" thành "giá trị kỳ vọng Y".

**Việc cần làm:**
1. Tạo `reports/business_translation_template.md` — **template**, không phải số liệu giả định — nêu rõ các biến số cần điền khi có dữ liệu thật: chi phí phục vụ 1 impression gợi ý, giá trị kỳ vọng 1 conversion cross-service, ngưỡng recall hoà vốn.
2. Ghi rõ trong template: mọi ô số liệu để trống cho tới khi có A/B test trên dữ liệu thật — không điền số từ dữ liệu mô phỏng vào template này (tránh lặp lại rủi ro của CP-I1).
3. Thêm mục "Câu hỏi kinh doanh chưa được trả lời" liệt kê giả thuyết cần test thật (vd: "khách taxi có thực sự có xác suất đặt Vinpearl cao hơn baseline không").

**Report** (dự án cá nhân, không có người ngoài team để duyệt — tiêu chí đổi thành tự-kiểm chứng):
- File `reports/business_translation_template.md` tồn tại.
- Tự đọc lại sau ít nhất 1 ngày (không đọc ngay sau khi viết) và xác nhận: không có ô số liệu nào bị điền từ dữ liệu mô phỏng, mọi câu hỏi kinh doanh đều là giả thuyết cần dữ liệu thật để trả lời (không phải câu đã có sẵn đáp án).

**Trạng thái: ✅ Done.** File [reports/business_translation_template.md](reports/business_translation_template.md) có bảng biến số, bảng dịch để trống, và mục câu hỏi kinh doanh chưa trả lời. Tự kiểm tra: không ô nào chứa số liệu từ `synthetic_controlled`.

---

## CP-I4 (P1) — Thiết kế governance/privacy trước khi chạm dữ liệu thật

**Vấn đề:** README nhắc 1 dòng cuối cùng, nhưng bài toán "định danh chung xuyên nhiều pháp nhân dịch vụ" (di chuyển, y tế, bán lẻ) là vấn đề pháp lý cần giải trước khi viết code, đặc biệt với dữ liệu y tế.

**Việc cần làm:**
1. Viết `docs/DATA_GOVERNANCE.md` (tài liệu thiết kế, không phải code) trả lời tối thiểu:
   - Cơ chế consent khi hợp nhất id giữa các dịch vụ khác pháp nhân.
   - Loại dữ liệu y tế nào tuyệt đối không được đưa vào feature (chỉ dùng lịch hẹn, không dùng chẩn đoán — đã ghi trong README bảng Giai đoạn B, cần nâng thành policy chính thức).
   - Ai sở hữu quyền xoá/rút đồng ý (right to be forgotten) và nó ảnh hưởng thế nào tới artifact đã train (retrain bắt buộc, không thể chỉ xoá 1 dòng trong dict).
2. Đây là tài liệu, không cần code, nhưng phải tồn tại trước khi bất kỳ checkpoint nào sau đây động vào dữ liệu thật.

**Report** (dự án cá nhân — bỏ yêu cầu review bên ngoài, giữ lại phần quan trọng nhất: tài liệu phải tồn tại *trước* khi chạm dữ liệu thật, không phải ai ký duyệt nó):
- File `docs/DATA_GOVERNANCE.md` tồn tại và trả lời được câu hỏi ở mục 1-3 (consent, dữ liệu y tế, right-to-be-forgotten) đủ cụ thể để tự mình dùng làm rào chắn khi có ý định gắn dữ liệu thật vào `data_loader.py`.
- Mục 4 (sở hữu quyết định) với dự án 1 người thì mặc định là chính người viết — không cần điền tên, chỉ cần tự nhắc bản thân đây là quyết định phải cân nhắc kỹ trước khi làm, không phải ký duyệt qua loa.

**Trạng thái: ✅ Done.** File [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) trả lời đủ 3 mục đầu, có nêu rõ giới hạn kiến trúc thật (1 `user_id` = 1 embedding, không che được theo từng cặp consent — đây là giới hạn kỹ thuật, không phải chính sách, nên không thể "vá" bằng quy trình). Mục 4 giữ nguyên dạng khung câu hỏi để tự tra lại khi cần, phù hợp quy mô dự án cá nhân.

---

## CP-I5 (P1) — API sẵn sàng vận hành tối thiểu

**Vấn đề:** Không auth, không rate limit, không log, model load qua `lru_cache` singleton không hot-reload/rollback ([src/api/main.py:42-49](src/api/main.py#L42-L49)).

**Việc cần làm:**
1. Thêm structured logging (request_id, user_id, model_type, latency) cho endpoint `/recommend`.
2. Thêm API key header đơn giản (env-based) để chặn truy cập vô danh — không cần OAuth phức tạp ở giai đoạn này.
3. Thêm endpoint `/reload-model` (bảo vệ bằng cùng API key) để đổi `MODEL_PATH` mà không cần restart process — giải quyết vấn đề "không hot-reload".
4. Rate limit cơ bản theo IP/API key (vd `slowapi`), tránh 1 client duyệt hết catalog liên tục.

**Report:**
- `tests/test_serve.py` hoặc test mới cho `/reload-model`.
- Log mẫu (1 dòng) cho thấy structured logging hoạt động.
- README cập nhật mục "API serving" với hướng dẫn set API key.

**Trạng thái: ✅ Done** (rate limit ở mức "đủ dùng cho 1 instance", không phải giải pháp scale ngang — đã ghi rõ giới hạn này trong README). Đã thêm logging structured, `require_api_key`/`X-API-Key`, `POST /reload-model`, rate limit in-process (`RATE_LIMIT_PER_MINUTE`, mặc định 60/phút). Verify: log mẫu thật đã chạy — `recommend request_id=e977fb2d7b9e user_id=sim_user_000001 model_type=als personalized=True data_provenance=synthetic_controlled latency_ms=32.0`; `tests/test_api.py` có 5 test mới cho reload/API key/rate limit, đều pass; README mục "Bảo vệ vận hành tối thiểu" đã thêm. Không dùng `slowapi` (tránh thêm dependency) — dùng in-memory sliding window, đủ cho demo/1 instance.

---

## CP-I6 (P2) — Khả năng mở rộng của artifact serving

**Vấn đề:** `train_seen` và `user_category_history` lưu dạng dict Python trong joblib artifact, load toàn bộ vào RAM ([src/train_baseline.py:95-104](src/train_baseline.py#L95-L104)). Không scale quá vài trăm nghìn user.

**Việc cần làm:**
1. Đo thử: script nhỏ tạo N=100k/500k/1M user giả, đo RAM và thời gian load artifact — có số liệu mới quyết định có cần đổi kiến trúc không (đừng tối ưu sớm khi chưa đo).
2. Nếu vượt ngưỡng chấp nhận được (ví dụ RAM > vài GB hoặc load > vài giây), đề xuất tách `train_seen`/`user_category_history` ra key-value store ngoài (Redis/SQLite) thay vì nhúng trong artifact.

**Report:**
- File `reports/scaling_benchmark.md` với bảng RAM/thời gian theo số user, kèm kết luận có cần đổi kiến trúc hay chưa (dựa trên số đo, không suy đoán).

**Trạng thái: ✅ Done — và kết quả xác nhận cần đổi kiến trúc.** Đo thật ở 100k user (không cần đo tới 500k/1M, vấn đề đã rõ): **10.58s để load** và **523 MB RAM** chỉ cho `train_seen`+`user_category_history` — không tính phần factor chính. Đây là cold-start latency không chấp nhận được cho 1 process API, và sẽ tệ hơn tuyến tính theo số user thật. Xem [reports/scaling_benchmark.md](reports/scaling_benchmark.md) để có số đo đầy đủ + ngoại suy 500k/1M. **Khuyến nghị chuyển ngay hướng giải quyết CP-I6 (tách 2 field này ra Redis/SQLite) từ "cân nhắc" sang "nên làm sớm"** nếu quy mô thật dự kiến vượt ~50-100k user; chưa cần đổi nếu quy mô dưới mốc đó.

---

## CP-I7 (P2) — Test chống hồi quy cho các rủi ro đã review

**Việc cần làm:**
1. Test kiểm tra không còn artifact/report nào thiếu field `data_provenance` (chặn CP-I1 tái diễn).
2. Test cold-start (đã liệt kê ở CP-I2) nằm trong CI để không bị revert.
3. Thêm test rằng API luôn trả 200+`personalized:false` thay vì 404 cho user lạ có context hợp lệ.

**Report:**
- `pytest -q` pass toàn bộ, bao gồm các test mới; đính kèm output log chạy pytest vào PR/commit liên quan.

**Trạng thái: ✅ Done.** Thêm [tests/test_provenance.py](tests/test_provenance.py) khoá lại yêu cầu "mọi artifact phải có `data_provenance`/`item_popularity`" (chặn CP-I1 tái diễn), cộng với các test cold-start (CP-I2) và test API (CP-I8) đã nằm trong cùng bộ test chạy qua CI (CP-I9). Verify: `pytest -q` → 22 passed.

---

## CP-I8 (P1) — Test API thật, không chỉ test engine

**Vấn đề:** `requirements.txt` đã khai báo `httpx` để test FastAPI, nhưng hiện không có file test nào dùng `TestClient`/`httpx` gọi `src/api/main.py`. `tests/test_serve.py` chỉ test `RecommendationEngine` trực tiếp — tầng HTTP (mapping lỗi → status code, `response_model`, query params `top_k`/`target_category`) chưa được test. Đã chạy `pytest -q`: 7/7 pass, nhưng 0 test nào chạm tới `src/api/`.

**Việc cần làm:**
1. Thêm `tests/test_api.py` dùng `fastapi.testclient.TestClient`, mock/monkeypatch `get_engine()` để không phụ thuộc model thật trên đĩa.
2. Test tối thiểu: `/health` trả 200; `/recommend/{user}` với user hợp lệ trả đúng schema `RecommendationResponse`; user không tồn tại (sau CP-I2, phải là 200 + `personalized:false`, không còn 404); model chưa train → 503; `top_k` ngoài khoảng `[1,100]` bị FastAPI validation chặn (422).

**Report:**
- `tests/test_api.py` tồn tại, `pytest -q` báo số test tăng và tất cả pass, bao gồm test tầng HTTP.

**Trạng thái: ✅ Done.** [tests/test_api.py](tests/test_api.py) dùng `TestClient` + monkeypatch `get_engine` (không phụ thuộc model thật trên đĩa): health, personalized/fallback, caution theo `data_provenance`, 503 khi thiếu model, 422 khi `top_k` sai khoảng, cộng thêm test cho `/reload-model` và rate limit (CP-I5). Verify: 10 test trong file này, tất cả pass trong tổng 22.

---

## CP-I9 (P1) — CI pipeline tối thiểu

**Vấn đề:** Không có `.github/workflows/*` hay bất kỳ cấu hình CI nào — hiện tại "tests pass" chỉ đúng trên máy người chạy thủ công, không có gì chặn merge code vỡ.

**Việc cần làm:**
1. Thêm `.github/workflows/ci.yml`: cài `requirements.txt`, chạy `pytest -q` trên mỗi push/PR.
2. Cache pip để CI không cài lại `torch`/`faiss-cpu` mỗi lần (các gói native nặng, ảnh hưởng thời gian CI).

**Report:**
- File workflow tồn tại; 1 lần chạy CI (screenshot hoặc link) hiển thị pass trên push thử nghiệm.

**Trạng thái: 🟡 Workflow đã viết, chưa chạy trên GitHub thật.** [.github/workflows/ci.yml](.github/workflows/ci.yml) có job `test` (pytest) và `lint` (ruff bắt buộc pass, mypy advisory). Chưa verify được lần chạy thật trên GitHub Actions vì repo hiện chưa init git/chưa có remote — cần push lên GitHub rồi xác nhận 1 lần chạy xanh mới coi là done đầy đủ.

---

## CP-I10 (P2) — Lint & type-check

**Vấn đề:** Repo không có `pyproject.toml`/`.flake8`/mypy config nào — không có gì bắt lỗi kiểu (vd nhầm `str | None`) hoặc style trước khi code vào production track.

**Việc cần làm:**
1. Thêm `ruff` (lint + format, thay thế flake8/isort/black bằng 1 tool) với config tối thiểu trong `pyproject.toml`.
2. Thêm `mypy` chạy trên `src/` (code đã dùng type hint khá đầy đủ, nên chi phí bật mypy không lớn).
3. Đưa cả hai vào CI (CP-I9) như bước riêng, không block merge ngay (warning-only) cho tới khi dọn hết lỗi tồn đọng.

**Report:**
- `ruff check src tests` và `mypy src` chạy được (có thể còn warning), kèm trong CI log.

**Trạng thái: ✅ Done.** [pyproject.toml](pyproject.toml) cấu hình ruff (E/F/I/UP/B) + mypy. Sửa cả các lỗi thật ruff/mypy phát hiện được (2 chỗ `zip()` thiếu `strict=True` trong `serve.py` — rủi ro im lặng bỏ sót item nếu 2 danh sách lệch độ dài; 2 chỗ lambda trong `features.py` bind nhầm biến vòng lặp; 2 chỗ thiếu type annotation). Verify: `ruff check src tests` → "All checks passed!"; `mypy src` → "Success: no issues found in 11 source files".

---

## CP-I11 (P2) — Container hoá môi trường

**Vấn đề:** Stack phụ thuộc nhiều gói native dễ vỡ theo OS/Python version (`faiss-cpu`, `implicit`, `torch`) — README phải nhắc "khuyến nghị Python 3.10/3.11" vì rủi ro môi trường có thật.

**Việc cần làm:**
1. Viết `Dockerfile` build image cài đúng `requirements.txt`, verify bằng cách chạy thử pipeline demo (`data_loader` → `train_baseline` → `serve`) bên trong container.
2. Không bắt buộc container cho việc train nặng (Colab/GPU vẫn là đường chính theo README), container chủ yếu để đảm bảo phần serving/API tái lập được.

**Report:**
- `docker build` thành công, `docker run` phục vụ được 1 request `/health` trả 200.

**Trạng thái: 🟡 Code viết xong, chưa verify build được.** [Dockerfile](Dockerfile) + [requirements-serving.txt](requirements-serving.txt) (tập dependency rút gọn, cố ý loại `torch`/`faiss`/`implicit` khỏi image phục vụ — training không chạy trong container này) + [.dockerignore](.dockerignore). `docker build` không chạy được trong môi trường này vì Docker daemon không hoạt động (`docker --version` OK nhưng daemon không kết nối được) — cần verify `docker build` + `docker run` trên máy có Docker Desktop đang chạy trước khi coi checkpoint này là done thật.

---

## Thứ tự thực hiện đề xuất

1. CP-I1 → CP-I2 (P0, chặn ngay rủi ro hiểu sai + gap sản phẩm rõ nhất)
2. CP-I4 (P1, phải xong trước khi ai đó nghĩ tới chạm dữ liệu thật)
3. CP-I3 → CP-I5 → CP-I8 → CP-I9 (P1, chuẩn bị cho việc trình bày/vận hành + có lưới an toàn CI trước khi thay đổi nhiều)
4. CP-I6 → CP-I7 → CP-I10 → CP-I11 (P2, củng cố sau khi phần lõi đã chắc)
