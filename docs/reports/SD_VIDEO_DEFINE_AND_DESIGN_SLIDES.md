# SD VIDEO — DEFINE & DESIGN SLIDES

- **Motivation/Background**: Cần bộ slide đủ chi tiết, theo phong cách HTML cũ của nhánh baseline.
- **Purpose**: Trình bày bài toán, thiết kế và cách kiểm chứng dự án SD video.
- **Overview Pipeline**: Inputs → pretrained benchmark → selection → dataset → conditional detector.
- **Detailed Plan**: 8 slide trực quan, 300 giây, với lời dẫn và cổng kiểm chứng.
- **References**: [HTML slides](SD_VIDEO_DEFINE_AND_DESIGN_SLIDES.html), [master report](SD_VIDEO_DEFINE_AND_DESIGN.md), [protocol](../experiments/SD_VIDEO_BASELINE_PROTOCOL.md).
- **Created**: 2026-11-07T14:32:21+07:00
- **Last Updated**: 2026-11-07T15:11:37+07:00



---

**8 slide / 5 phút.** Mũi tên chuyển trang, N mở lời dẫn, R mở nguồn, Esc đóng, Ctrl+P xuất PDF. SVG chân dung được kế thừa từ bản HTML mới ở baseline commit `1be72ef`, các sơ đồ được thiết kế lại cho video. Không có ảnh người thật, output model hoặc số đo benchmark được trình bày như kết quả thực. Không tải font hay thư viện ngoài khi trình chiếu.

## 01. Ý tưởng — 25 giây

Đề tài của nhóm là xây dựng và đánh giá pipeline sinh video deepfake bằng các phương pháp dựa trên Stable Diffusion. Ảnh A cung cấp danh tính, video B cung cấp chuyển động. Đầu ra mong muốn là video giữ diễn xuất của B nhưng mang danh tính A. Nhóm sẽ benchmark pretrained trước rồi mới chọn cấu hình để sinh dữ liệu.

## 02. Define bài toán — 35 giây

Bài toán có ba yêu cầu: đổi đúng danh tính, giữ đúng diễn xuất và ổn định giữa các frame. Các hình ở đây chỉ minh họa mục tiêu, không phải output thực nghiệm. Chuyển động nhanh và vật che mặt dễ gây lỗi nên cần đánh giá ở cấp clip. Câu hỏi chính là backend nào cân bằng chất lượng với chi phí trên Kaggle.

## 03. Các ứng viên — 35 giây

Nhóm khảo sát bốn ứng viên. Face-Adapter và REFace xử lý ảnh nên cần wrapper để chạy từng frame. VFace và VividFace có thành phần thời gian. VFace mở rộng trên REFace nên không xem là backbone độc lập. Đây là registry để thử khả thi, chưa kết luận model nào tốt nhất. Mỗi backend cần đúng preprocessing, weights và giấy phép.

## 04. Thiết kế pipeline — 50 giây

Đầu tiên kiểm tra ảnh và video, lưu nguồn rồi tạo manifest. Chia dữ liệu theo danh tính và video cha để tránh rò rỉ. Cùng cặp đầu vào được chạy qua bốn backend với preprocessing riêng. Các clip được đánh giá chất lượng, lỗi và chi phí. Chọn cấu hình bằng development, khóa rồi kiểm tra holdout. Sau đó mới sinh batch có QA. Mọi job lưu revision, config, seed, hash và trạng thái để tái lập hoặc tiếp tục khi phiên Kaggle bị ngắt.

## 05. Pilot benchmark — 35 giây

Pilot đề xuất ba cặp cho bốn backend, tối đa mười hai clip. Mỗi clip năm giây ở mười FPS, tương ứng năm mươi frame. Crop 512 chỉ dùng khi backend hỗ trợ. Pilot giúp biết có chạy được và lỗi ở đâu, chưa đủ để xếp hạng tổng quát. Đo thời gian thực tế rồi mở rộng. Fine-tune chỉ xét sau phân tích lỗi nếu có dữ liệu và upstream hỗ trợ.

## 06. Đánh giá và success criteria — 50 giây

Đánh giá theo ba trục. Danh tính dùng cosine đầu ra với A, đồng thời kiểm tra có còn giống người B hay không. Diễn xuất so pose, biểu cảm và chuyển động với video B. Thời gian dùng warping có bù chuyển động và mask vùng che, kết hợp xem clip để chấm flicker, seam và biến dạng. Báo cả tỷ lệ qua QA và chi phí. Chọn ngưỡng trên development rồi khóa trước holdout. Nhóm chưa đặt số điểm giả hoặc kết luận từ một frame đẹp.

## 07. Dataset và utility — 35 giây

Mỗi clip có manifest truy vết nguồn, backend, cấu hình và kết quả QA. Danh tính cùng video cha được khóa nhóm trước khi tạo biến thể. Nếu còn đủ dữ liệu và thời gian, nhóm kiểm chứng lợi ích cho detector bằng ba điều kiện: dữ liệu gốc, augmentation thông thường và thêm dữ liệu SD. Giữ detector, ngân sách và test độc lập như nhau, đo AUC, AP và TPR tại FPR đã chọn.

## 08. Milestones và kết thúc — 35 giây

Hiện nhóm đã có hồ sơ define/design. Tiếp theo là đầu vào và khả thi từng backend, rồi benchmark để chọn cấu hình, sinh batch và xuất dataset card. Detector là phần mở rộng có điều kiện. Sản phẩm cuối gồm pipeline, video, manifest, benchmark và tài liệu tái lập. Nguyên tắc của nhóm là dùng pretrained trước, đo chất lượng trước và sinh dữ liệu sau.

## Chi tiết khi được hỏi

- Pilot là feasibility, không đủ để kết luận thống kê tổng quát. 50 frame/clip không đồng nghĩa 50 forward.
- Mỗi adapter pin đúng SD base, checkpoint, revision và preprocessing của upstream. Public weights không đồng nghĩa license không hạn chế.
- Tách theo connected components của danh tính / video cha. Biến thể của cùng cặp ở chung nhóm.
- Warping chỉ đo vùng có correspondence hợp lệ, mask occlusion và kết hợp motion fidelity. Không dùng PSNR/SSIM toàn mặt so với target làm metric chính.
- Đếm clip, danh tính, cặp và video cha riêng. Batch hỗ trợ resume bằng hash; output tạm phải validate trước khi hoàn tất.
- Detector dùng trung bình điểm frame thì chỉ là frame detector có tổng hợp điểm, chưa phải mô hình temporal. Test khác generator/checkpoint nếu đủ dữ liệu. Ngưỡng chọn trên validation, CI theo nhóm.

## Nhóm tài liệu tham khảo

- Official implementations / API: nguồn model và weights trong [protocol](../experiments/SD_VIDEO_BASELINE_PROTOCOL.md).
- Standards / data contracts: shape, RGB, bbox và timeline trong [system design](../phases/SD_VIDEO_SYSTEM_DESIGN.md).
- CV evaluation conventions: identity, motion, temporal, QA và detector utility trong [protocol](../experiments/SD_VIDEO_BASELINE_PROTOCOL.md).
- Internal architecture: [DEFINE](../PURPOSE.md), [master report](SD_VIDEO_DEFINE_AND_DESIGN.md), [roadmap](../progress/SD_VIDEO_ROADMAP.md).
