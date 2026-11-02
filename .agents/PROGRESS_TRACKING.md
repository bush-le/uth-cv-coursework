# PROGRESS TRACKING - THEO DÕI TIẾN ĐỘ & TRẠNG THÁI CÔNG VIỆC

Mọi tác vụ phải được gắn trạng thái rõ ràng theo chuẩn Kanban để AI không lặp lại công việc đã làm hoặc bỏ sót bước quan trọng.

---

## 5 Trạng thái Chuẩn của Công việc (Kanban States)

```text
[ To Do ] ──────► [ In Progress ] ──────► [ Done ]
                        │
                        ├──────► [ On Hold ]
                        │
                        └──────► [ Cancelled ]
```

| Trạng thái | Định nghĩa | Ý nghĩa đối với AI Agent |
| :--- | :--- | :--- |
| **`To Do`** | Task đã được lên kế hoạch và có Detailed Plan, nhưng chưa bắt đầu code. | AI chỉ đọc để nắm bức tranh tổng thể, KHÔNG tự ý code trước khi được chỉ định. |
| **`In Progress`** | Task đang được AI và Kỹ sư trực tiếp triển khai tại thời điểm hiện tại. | Trọng tâm duy nhất của Agent trong phiên làm việc hiện thời. |
| **`Done`** | Task đã vượt qua toàn bộ Codebase Audit, Smoke Test và Normal Test. | Mã nguồn đã đóng băng, AI KHÔNG được sửa đổi trừ khi có yêu cầu refactor. |
| **`On Hold`** | Task bị tạm dừng do thiếu dữ liệu, thiếu phần cứng hoặc chờ kết quả từ task khác. | AI ghi nhận nguyên nhân nghẽn (Blocker) và chuyển sang task khả thi khác. |
| **`Cancelled`** | Task bị hủy bỏ do thay đổi kiến trúc hoặc không còn phù hợp. | AI bỏ qua hoàn toàn, không tạo code liên quan. |

---

## Bảng Theo dõi Tiến độ Dự án (Progress Tracking Board)

Duy trì một bảng Markdown ngắn gọn trong quá trình làm việc để đồng bộ giữa Kỹ sư và Agent:

```markdown
# Current Sprint / Lab Progress

| Task ID | Tên công việc | Phụ trách | Trạng thái | Blockers / Ghi chú |
| :---: | :--- | :--- | :---: | :--- |
| **T-01** | Tạo cấu trúc dataset & checksum | Engineer | `Done` | Hoàn thành |
| **T-02** | Xây dựng pipeline transforms | AI Agent | `Done` | Đã pass unit test |
| **T-03** | Triển khai DataModule nạp batch | AI Agent | `In Progress` | Đang viết test cho batch shape |
| **T-04** | Xây dựng Model Architecture | AI Agent | `To Do` | Chờ xong T-03 |
| **T-05** | Training pipeline & Evaluation | AI Agent | `To Do` | Chờ xong T-04 |
```

---

## Nguyên tắc Chống Lệch Tiến độ của AI
1. **Chỉ làm 1 task `In Progress` tại một thời điểm**: Tránh tình trạng Agent sinh code dàn trải cho nhiều file cùng lúc dẫn đến lỗi khó cô lập.
2. **Không tự ý đánh dấu `Done`**: Chỉ chuyển sang `Done` sau khi đã chạy lệnh kiểm thử thực tế trên terminal và nhận được kết quả Pass.
3. **Báo cáo trạng thái sau mỗi lượt trao đổi**: Sau mỗi bước, Agent tóm tắt ngắn gọn: *"Task X đã hoàn thành, tiếp theo chuyển sang Task Y."*
