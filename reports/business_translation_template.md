# Business translation template (điền khi có dữ liệu thật + A/B test)

Mục đích: dịch metric ML (`Recall@K`, `NDCG@K`) sang ngôn ngữ quyết định đầu tư. **Không điền số liệu từ dữ liệu mô phỏng (`data_provenance: synthetic_controlled`) vào bảng dưới đây** — làm vậy lặp lại đúng rủi ro "circular validation" mà [IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md) (CP-I1) đã cảnh báo: số liệu đó chỉ đo việc model phục hồi lại pattern do chính generator cấy vào, không phải hành vi khách hàng. Mọi ô số bên dưới để trống cho tới khi có kết quả A/B test trên dữ liệu thật (`data_provenance: production`).

## 1. Biến số cần có trước khi điền bảng

| Biến số | Nguồn lấy số | Trạng thái |
|---|---|---|
| Chi phí phục vụ 1 impression gợi ý (compute + vị trí UI chiếm chỗ) | Team hạ tầng / product | _chưa có_ |
| Giá trị kỳ vọng 1 conversion cross-service (vd 1 lượt đặt Vinpearl từ gợi ý sau chuyến taxi) | Team business/finance của từng dịch vụ | _chưa có_ |
| Baseline conversion rate hiện tại khi KHÔNG có gợi ý cá nhân hoá (popularity/rule-based) | Đo trực tiếp trên production hiện hành | _chưa có_ |
| Ngưỡng recall/NDCG tương ứng ROI hoà vốn | Suy ra từ 3 dòng trên, không đoán | _chưa có_ |

## 2. Bảng dịch (để trống, chỉ điền sau A/B test thật)

| Model | Recall@10 (offline) | Conversion rate thật (A/B) | Chi phí/1000 impression | Giá trị/1000 impression | ROI |
|---|---:|---:|---:|---:|---:|
| Baseline (popularity/rule) | — | — | — | — | — |
| Model đề xuất | — | — | — | — | — |

Ghi chú bắt buộc khi điền: `data_provenance` của dòng dữ liệu offline dùng để tính Recall@10 tương ứng phải là `production` hoặc `public_dataset` đã qua A/B thật — nếu là `synthetic_controlled`, không được đưa vào bảng này.

## 3. Câu hỏi kinh doanh chưa được trả lời

Đây là danh sách giả thuyết cần kiểm chứng bằng dữ liệu thật, không phải bằng dữ liệu mô phỏng:

- Khách dùng dịch vụ di chuyển có thực sự có xác suất chuyển đổi sang dịch vụ lưu trú cao hơn baseline không, và cao hơn bao nhiêu?
- Cửa sổ thời gian hợp lý để gợi ý cross-service là bao lâu (giờ, ngày)? Dữ liệu mô phỏng hiện đặt cứng "1-3 ngày" — đây là giả định cần kiểm chứng, không phải kết luận.
- Gợi ý cross-service có làm giảm trải nghiệm trong chính dịch vụ gốc không (vd gợi ý khách sạn ngay giữa luồng đặt taxi có gây phiền không)?
- Nhóm khách hàng nào (persona) thực sự phản hồi tích cực với gợi ý xuyên dịch vụ, nhóm nào nên tắt tính năng này?

## 4. Điều kiện để coi bảng mục 2 là "đã điền hợp lệ"

1. Có ít nhất 1 vòng A/B test trên dữ liệu thật (`data_provenance: production`).
2. Cỡ mẫu đủ lớn để chênh lệch conversion rate có ý nghĩa thống kê (không chỉ nhìn con số tuyệt đối).
3. Người điền không phải là người trực tiếp làm model (tránh thiên vị xác nhận — nên do team business/analytics điền dựa trên log thật).
