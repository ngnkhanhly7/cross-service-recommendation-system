# End-to-end checklist

Chạy tuần tự để xác nhận toàn bộ pipeline còn hoạt động đúng sau khi sửa code. Mỗi bước có lệnh cụ thể + kết quả mong đợi — nếu khác, dừng lại và sửa trước khi qua bước sau.

## 0. Môi trường

```bash
python -m pip install -r requirements.txt
```

- [ ] Cài xong không lỗi (Python 3.10/3.11 khuyến nghị).

## 1. Test + lint + type-check

```bash
python -m pytest -q
python -m pip install ruff mypy --quiet
python -m ruff check src tests
python -m mypy src
```

- [ ] `pytest`: **22 passed**, 0 failed.
- [ ] `ruff check`: "All checks passed!"
- [ ] `mypy src`: "Success: no issues found in 11 source files"

## 2. Data pipeline (dữ liệu mô phỏng — nhanh nhất để kiểm tra)

```bash
python -m src.data_loader --source demo-amazon --output data/processed/e2e_check.csv
python -m src.eda --data data/processed/e2e_check.csv --output-dir reports/eda_e2e_check
```

- [ ] `data_loader` in ra `Before filtering` / `After filtering` với `multi_category_user_pct > 0`.
- [ ] File `data/processed/e2e_check.csv` và `data/processed/e2e_check.csv.provenance.json` tồn tại, provenance = `synthetic_controlled`.
- [ ] `reports/eda_e2e_check/category_interactions.png` và `category_cooccurrence.png` được tạo.

## 3. Train baseline (ALS) + evaluate

```bash
python -m src.train_baseline --data data/processed/e2e_check.csv \
  --model-output models/e2e_check_als.pkl \
  --metrics-output reports/e2e_check_baseline.json
```

- [ ] Output JSON có đủ field: `overall`, `cross_category`, `same_category`, `data_provenance: "synthetic_controlled"`, `caution` (không rỗng).
- [ ] `models/e2e_check_als.pkl` tồn tại.

## 4. Serve từ CLI — user có lịch sử (personalized) và user lạ (cold-start fallback)

```bash
python -m src.serve amazon_user_00000 --model models/e2e_check_als.pkl -k 3
python -m src.serve user_khong_ton_tai --model models/e2e_check_als.pkl -k 3
```

- [ ] Lệnh 1: `"personalized": true`, `results` không rỗng.
- [ ] Lệnh 2: `"personalized": false`, `results` vẫn không rỗng (fallback theo popularity), **không có traceback lỗi**.

## 5. API (FastAPI + TestClient, không cần model thật)

```bash
python -m pytest tests/test_api.py -q
```

- [ ] 10 test pass (health, personalized, cold-start 200, caution theo provenance, 503 khi thiếu model, 422 khi `top_k` sai khoảng, reload-model + API key, rate limit).

## 6. API thật qua uvicorn

```bash
MODEL_PATH=models/e2e_check_als.pkl uvicorn src.api.main:app --port 8000 &
sleep 2
curl -s http://127.0.0.1:8000/health
curl -s "http://127.0.0.1:8000/recommend/amazon_user_00000?top_k=3"
curl -s "http://127.0.0.1:8000/recommend/user_khong_ton_tai?top_k=3"
kill %1
```

- [ ] `/health` → `{"status":"ok"}`.
- [ ] `/recommend/amazon_user_00000` → `"personalized": true`, có `data_provenance` + `caution`.
- [ ] `/recommend/user_khong_ton_tai` → HTTP 200 (không phải 404), `"personalized": false`.

## 7. Docker (serving image)

```bash
docker build -t cross-service-rec:e2e .
MSYS_NO_PATHCONV=1 docker run -d --name e2e-check -p 8001:8000 \
  -e MODEL_PATH=/app/models/e2e_check_als.pkl \
  -v "$(pwd)/models:/app/models" cross-service-rec:e2e
sleep 2
curl -s http://127.0.0.1:8001/health
docker stop e2e-check && docker rm e2e-check
```

*(`MSYS_NO_PATHCONV=1` chỉ cần khi chạy trên Git Bash/Windows — xem README mục "Container hoá".)*

- [ ] `docker build` thành công (~20-30s, không cần build tool nặng vì dùng `requirements-serving.txt`).
- [ ] `/health` qua container trả `{"status":"ok"}`.

## 8. CI

- [ ] Push lên GitHub, mở tab **Actions**, xác nhận cả job `test` và `lint` đều ✓ xanh cho commit mới nhất.

## 9. Dọn dẹp file tạm của lần check này

```bash
rm -f data/processed/e2e_check.csv data/processed/e2e_check.csv.provenance.json
rm -f models/e2e_check_als.pkl reports/e2e_check_baseline.json
rm -rf reports/eda_e2e_check
```

- [ ] Không còn file `e2e_check*` sót lại trong `git status`.

---

**Kết quả tổng:** nếu tất cả các mục trên đều tick, pipeline end-to-end (data → train → evaluate → serve CLI → API → Docker → CI) đang hoạt động đúng như thiết kế trong [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).
