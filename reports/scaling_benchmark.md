# Scaling benchmark: dict-in-artifact design (CP-I6)

Đo bằng `scripts/benchmark_scaling.py`, dựng `train_seen` + `user_category_history` đúng hình dạng artifact thật (15 item đã xem/user, 2 category/user). `*_peak_mb` là bộ nhớ Python-attributable qua `tracemalloc` (cận dưới, RSS thật của tiến trình sẽ cao hơn).

## Số đo thật (chạy trên máy người dùng, Windows)

| Users | Build time (s) | Build peak (MB) | Dump time (s) | File size (MB) | Load time (s) | Load peak (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 12.27 | 158.5 | 45.75 | 37.9 | 10.58 | 523.4 |

## Ngoại suy tuyến tính (chưa đo, chỉ ước lượng — cấu trúc dict lồng nhau tăng gần tuyến tính theo số user)

| Users | Load time ước lượng (s) | Load peak ước lượng (MB) |
|---:|---:|---:|
| 500,000 | ~53 | ~2,600 |
| 1,000,000 | ~106 | ~5,200 |

## Kết luận

**Đã vượt ngưỡng chấp nhận được ngay ở 100k user — không cần chờ tới 500k/1M mới thấy vấn đề.**

- **10.58 giây chỉ để load 2 dict** (chưa tính `user_factors`/`item_factors`, chưa tính thời gian `joblib.load` toàn bộ artifact) là không chấp nhận được cho cold-start của 1 process API: mỗi lần restart/`--reload`/deploy mới, service sẽ "đơ" hơn 10 giây trước khi phục vụ được request đầu tiên. Ở quy mô 500k-1M user (ngoại suy), con số này lên tới 1-2 phút — vượt xa mọi SLA khởi động hợp lý.
- **523 MB RAM chỉ cho 100k user** ở 2 field phụ trợ (không phải model chính) có nghĩa là kiến trúc "nhúng hết vào 1 file joblib, load hết vào RAM 1 process" không scale quá vài trăm nghìn user trên một instance có RAM tiêu chuẩn (thường 1-4 GB cho 1 container nhỏ).
- **Dump time (45.75s) cũng đáng chú ý**: nghĩa là mỗi lần train xong và ghi artifact ra đĩa đã mất gần 1 phút chỉ cho phần dict này — ảnh hưởng trực tiếp tới thời gian của pipeline train, không chỉ lúc serve.

**Quyết định:** kích hoạt hướng giải quyết đã đề xuất trong [IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md) CP-I6 — tách `train_seen` và `user_category_history` ra khỏi artifact joblib, chuyển sang key-value store ngoài (Redis cho production, SQLite đủ cho single-instance/demo lớn hơn). `user_factors`/`item_factors` (mảng NumPy thuần) không nằm trong phạm vi benchmark này và không có cùng vấn đề — đó là dữ liệu dày đặc, load nhanh hơn nhiều so với dict Python lồng nhau chứa hàng triệu string.

**Không nằm trong phạm vi khuyến nghị này:** không cần đổi kiến trúc ngay lập tức nếu quy mô thật đang dưới ~50k user (build/load time dưới vài giây, RAM dưới 300MB theo ngoại suy ngược) — chỉ nên ưu tiên khi số user thật tiến gần mốc 100k trở lên.
