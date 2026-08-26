# Data governance (design doc, pre-real-data)

Tài liệu này phải được điền/duyệt **trước** khi bất kỳ checkpoint nào của dự án chạm vào dữ liệu khách hàng thật của một doanh nghiệp cụ thể. Ở trạng thái hiện tại, toàn bộ pipeline chỉ chạy trên Amazon Reviews (public) và dữ liệu mô phỏng tự sinh (`sim:` prefix) — tài liệu này chưa được kích hoạt, nó tồn tại để chuẩn bị sẵn khung trả lời trước khi câu hỏi trở nên khẩn cấp.

## 1. Consent khi hợp nhất định danh xuyên dịch vụ

Câu hỏi cần trả lời trước khi code, không phải sau:

- Cơ chế nào chứng minh người dùng đã đồng ý cho phép **dịch vụ A** chia sẻ tín hiệu hành vi với **dịch vụ B** để tạo gợi ý? (opt-in rõ ràng theo từng cặp dịch vụ, hay một consent tổng cho "hệ sinh thái"?)
- Nếu người dùng đồng ý chia sẻ với dịch vụ A→B nhưng không đồng ý B→C, hệ thống hợp nhất định danh (`user_id` chung) có vô tình làm rò rỉ tín hiệu sang C không? (Rủi ro kỹ thuật thật: một `user_id` chung trong `user_factors` là một embedding duy nhất — không thể "che" một phần lịch sử theo từng cặp consent bằng kiến trúc hiện tại. Đây là giới hạn kiến trúc cần biết trước, không phải chi tiết vận hành có thể vá sau.)
- Consent hết hạn/rút lại xử lý thế nào — xoá interaction khỏi dữ liệu train tiếp theo là đủ, hay phải retrain ngay lập tức artifact đang phục vụ?

**Trạng thái:** chưa có câu trả lời — không nằm trong phạm vi pipeline hiện tại. Bất kỳ ai định gắn dữ liệu thật của công ty cụ thể vào `src/data_loader.py` phải điền mục này trước.

## 2. Dữ liệu y tế — loại trừ nghiêm ngặt

Bảng dataset gợi ý trong README (Giai đoạn B) đã tự giới hạn: chỉ dùng dữ liệu lịch hẹn (appointment no-show), không dùng thông tin y tế nhạy cảm. Chính sách chính thức cần mở rộng thêm:

- Danh sách trắng field được phép đưa vào `category`/`item_id`: loại dịch vụ (khám tổng quát, nha khoa...), không bao gồm chẩn đoán, mã bệnh, kết quả xét nghiệm, tên thuốc.
- `interaction_strength` cho category y tế không được suy ra từ mức độ nghiêm trọng bệnh lý — chỉ dựa trên tần suất/có đến hẹn hay không, để tránh feature vô tình mã hoá tình trạng sức khoẻ.
- Bất kỳ field nào không chắc có vi phạm hay không → mặc định loại trừ, không đưa vào rồi tính sau.

## 3. Quyền xoá / rút đồng ý (right to be forgotten) và artifact đã train

Đây là điểm kiến trúc hiện tại **chưa hỗ trợ**: artifact (`models/*.pkl`) là snapshot bất biến chứa `user_factors`, `train_seen`, `user_category_history` cho toàn bộ user tại thời điểm train. Khi một user yêu cầu xoá dữ liệu:

- Xoá khỏi `data/processed/*.csv` cho lần train tiếp theo là cần nhưng không đủ — artifact đang phục vụ (`models/*.pkl` đang load trong `get_engine()`) vẫn còn factor của user đó cho tới lần retrain + reload kế tiếp.
- Cần định nghĩa SLA: tối đa bao lâu sau yêu cầu xoá, artifact đang phục vụ phải ngừng chứa dữ liệu của user đó? SLA này quyết định retrain có cần chạy theo lịch cố định (vd hàng ngày) hay theo trigger xoá dữ liệu.
- `/reload-model` (xem [IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md) CP-I5) chỉ đổi artifact đang phục vụ sang 1 file đã có sẵn trên đĩa — nó không tự xoá user khỏi artifact hiện tại; retrain vẫn là bước bắt buộc.

## 4. Sở hữu quyết định

Điền trước khi có dữ liệu thật — ai là người:
- Duyệt việc thêm 1 nguồn dữ liệu mới (dịch vụ mới) vào `data_loader.py`?
- Duyệt SLA xoá dữ liệu ở mục 3?
- Là đầu mối khi 1 audit/pháp lý hỏi "tại sao user X được gợi ý item Y"? (liên quan tới `reason` field trong response — hiện tại chỉ diễn giải co-occurrence, không phải căn cứ có thể defend trước audit).

---

*Dự án cá nhân, không có team pháp lý/compliance riêng để duyệt — tài liệu này đóng vai trò rào chắn tự đặt ra: nếu một ngày có ý định gắn dữ liệu thật của một doanh nghiệp cụ thể vào `data_loader.py`, hãy quay lại đọc kỹ mục 1-3 trước khi viết dòng code đầu tiên, và nếu có thể, tìm người thật (bạn bè làm pháp lý, mentor) để hỏi ý kiến về mục 1 và 2 trước khi triển khai thật.*
