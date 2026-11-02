# WORKFLOW OVERVIEW - TỔNG QUAN TOÀN BỘ QUY TRÌNH HỆ THỐNG

Toàn cảnh luồng thực thi khép kín từ Requirement đến Complete. Kết hợp Quản trị LLM và Kỹ thuật Phần mềm chuẩn công nghiệp.

---

## Sơ đồ Luồng Tổng thể (End-to-End Execution Flowchart)

```text
Requirement (Đề bài Lab / Yêu cầu Kỹ thuật)
        │
        ▼
LLM Context Management (CONTEXT_MANAGEMENT.md)
[Context Relevance │ Faithfulness │ Hallucination Control │ Knowledge Store]
        │
        ▼
Behavior Layer - Prompting (BEHAVIORAL_LAYER.md)
[Markdown Schema: Name → Description → Purpose → Input → Output → Workflow]
        │
        ▼
Task Description (TASK_DESCRIPTION.md)
[Plan → Pipeline → Detailed Execution Plan]
        │
        ▼
Progress Tracking (PROGRESS_TRACKING.md)
[Kanban Statuses: To Do → In Progress → Done → On Hold → Cancelled]
        │
        ▼
Pipeline Separation (PIPELINE_SEPARATION.md)
[Data Preparation │ Model Architecture │ Training │ Evaluation]
        │
        ▼
AI Reference Standards (AI_REFERENCE.md)
[Naming Conventions │ Folder Structure Hierarchy]
        │
        ▼
AI Generate Code (AGENT_INTERACTION_LOOP.md)
[Code Generation strictly within scoped contract]
        │
        ▼
Codebase Audit (CODEBASE_AUDIT.md)
[Audit Architecture → Audit Detailed Plan → Audit Conventions]
        │
        ▼
Unit Test Authoring (TESTING_STRATEGY.md)
[Viết test cases đối chiếu với Detail Plan & Invariants]
        │
        ▼
Smoke Test (TESTING_STRATEGY.md)
[Kiểm tra nhanh: cú pháp, import, instantiate, zero-crash]
        │
        ▼
Normal Test (TESTING_STRATEGY.md)
[Kiểm tra đầy đủ: logic, shapes, numerical invariants, performance]
        │
        ▼
Complete & Commit (AI_FIRST_LIFECYCLE.md & AI_REFERENCE.md)
[Conventional Commits & Technical Documentation]
```

---

## Mối quan hệ giữa 2 Cấp độ Vòng lặp

### 1. Macro Loop (Vòng lặp Kỹ thuật Cấp cao)
Quy trình phát triển toàn diện của kỹ sư:
`Define → Design → Breakdown → Build MVP → Implement → Test → Refactor → Commit → Document`

### 2. Micro Loop (Chu trình Tương tác Cấp thấp với Agent)
Áp dụng cho từng đầu việc nhỏ (Task) trong giai đoạn `Implement`:
`Describe → Generate → Review → Run → Test → Refactor → Commit`

---

## Cổng Kiểm soát Chất lượng (Quality Gates)
Trước khi chuyển từ giai đoạn này sang giai đoạn tiếp theo, hệ thống bắt buộc phải vượt qua các cổng kiểm soát:
1. **Gate 1 (Context Gate)**: AI đã nạp đủ tài liệu tham chiếu từ `.agents/AI_REFERENCE.md` chưa?
2. **Gate 2 (Design Gate)**: Đã có Detailed Plan và Definition of Done chưa?
3. **Gate 3 (Audit Gate)**: Code sinh ra có bám sát kiến trúc và không có code thừa/tự ý bịa đặt không?
4. **Gate 4 (Test Gate)**: Đã pass cả Smoke Test và Normal Test chưa?
5. **Gate 5 (Document & Delivery Gate)**:
   - **Báo cáo chuyên sâu (`docs/`)**: Đã có đủ cặp tệp Markdown báo cáo và HTML trực quan tương tác, tích hợp đầy đủ Sơ đồ Luồng toàn diện và Mục Tài liệu Tham khảo (References) 4 nhóm chuẩn chưa?
   - **Jupyter Notebook (`notebooks/**/*.ipynb`)**: Đã tinh gọn theo chuẩn (Cell 0: Mục tiêu, Cell 1: Bảng Roadmap 4 cột kèm Quick Links tới các tài liệu liên quan), tuyệt đối KHÔNG chứa sơ đồ workflow diagram hay references học thuật chưa?
   - **Bảo mật & Điều hướng (`Ctrl + Click`)**: 100% liên kết nội bộ là đường dẫn tương đối (`../../src/...`), tuyệt đối KHÔNG chứa đường dẫn tuyệt đối chứa thông tin người dùng (`file:///home/...`) chưa?
