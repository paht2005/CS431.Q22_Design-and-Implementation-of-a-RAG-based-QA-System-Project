# Bộ 20 Câu Hỏi - Trả Lời Bảo Vệ Đồ Án

Mục tiêu: dùng cho phần phản biện 10-15 phút.

Quy ước:
- Trả lời ngắn: 20-30 giây, khi cần chốt nhanh.
- Trả lời chi tiết: 60-90 giây, khi hội đồng hỏi sâu.
- Khi cần show code, dùng đúng file và hàm được gợi ý trong từng câu.

Nguồn số liệu chính:
- [docs/RESULTS.md](docs/RESULTS.md)
- [docs/slide/main.tex](docs/slide/main.tex)

## Nhóm A - Phương pháp thực hiện

### Câu 1. Bài toán của nhóm là gì và vì sao chọn hướng zero-shot object counting?
Trả lời ngắn:
Nhóm giải bài toán đếm đối tượng khi số lớp rất đa dạng và không thể gán nhãn đầy đủ cho mọi lớp mới. Zero-shot giúp dùng prompt ngôn ngữ + exemplar để đếm lớp chưa thấy trong train, phù hợp bối cảnh thực tế.

Trả lời chi tiết:
Bài toán của nhóm là đếm số lượng đối tượng theo prompt lớp trên ảnh, trong bối cảnh không thể xây một bộ nhãn đầy đủ cho mọi lớp. Thay vì huấn luyện detector đóng trên một số lớp cố định, nhóm dùng pipeline zero-shot dựa trên VA-Count: tạo exemplar dương/âm từ detector + prompt, sau đó đưa vào mô hình density map để suy ra count. Hướng này phù hợp vì có thể mở rộng lớp mới bằng ngôn ngữ tự nhiên, giảm phụ thuộc annotate thủ công.

Bằng chứng:
- [docs/slide/main.tex](docs/slide/main.tex)
- [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L593)

Show code nếu bị hỏi:
- Mở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L593), hàm full_counting_pipeline.

### Câu 2. Kiến trúc tổng thể pipeline của nhóm gồm những bước nào?
Trả lời ngắn:
Pipeline có 4 bước chính: prompt enhancement, detection, lọc exemplar đơn đối tượng, rồi counting bằng density map.

Trả lời chi tiết:
Pipeline bắt đầu từ ảnh và class prompt. Nếu bật Rich Prompt, hệ thống gọi Gemini 2.5 Flash để tạo mô tả chi tiết hơn, sau đó dùng CLIP ViT-B/32 re-rank candidate theo semantic similarity với class name gốc. Sau đó dùng detector (GroundingDINO hoặc YOLO-World) lấy candidate boxes. Các box được lọc bằng binary classifier để giữ exemplar khả năng là một instance đơn. Tiếp theo xếp hạng và lấy top-3 exemplar (hoặc top-5 luồng dương khi bật Rich Prompt) để feed vào mô hình đếm. Mô hình xuất density map, rồi tổng density để ra count cuối.

Bằng chứng:
- [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L593)
- [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L519)

Show code nếu bị hỏi:
- Lần lượt mở detect_with_grounding_dino, detect_with_yolo, filter_single_objects, select_exemplars, run_counting_inference trong [code/source-code/demo_inference.py](code/source-code/demo_inference.py).

### Câu 3. Rich Prompt của nhóm hoạt động thế nào và đóng vai trò gì?
Trả lời ngắn:
Rich Prompt mở rộng mô tả ngữ nghĩa của đối tượng, giúp detector và bước chọn exemplar bớt mơ hồ, đặc biệt hữu ích cho YOLO-World.

Trả lời chi tiết:
Nhóm tích hợp Gemini 2.5 Flash để sinh mô tả chi tiết từ ảnh và class name, đồng thời có logic chuẩn hóa class về dạng số ít để prompt ổn định. Tiếp theo, CLIP ViT-B/32 re-rank các candidate exemplar theo semantic similarity với tên lớp gốc — đây là bộ lọc ngữ nghĩa cấp hai đảm bảo patch khớp với class name. Số lượng exemplar dương cũng mở rộng từ top-3 (mặc định) lên top-5 khi bật Rich Prompt. Rich Prompt giúp mô tả rõ màu sắc, hình dạng, ngữ cảnh xuất hiện của object. Nếu API lỗi hoặc response kém, pipeline có fallback để không làm chết luồng inference.

Bằng chứng:
- [code/source-code/prompt_enhancer.py](code/source-code/prompt_enhancer.py#L49)
- [docs/slide/main.tex](docs/slide/main.tex)

Show code nếu bị hỏi:
- Mở enhance_prompt_with_gemini trong [code/source-code/prompt_enhancer.py](code/source-code/prompt_enhancer.py#L49).

### Câu 4. Vì sao nhóm dùng cả exemplar dương và exemplar âm?
Trả lời ngắn:
Exemplar dương giúp mô hình biết cần đếm gì, exemplar âm giúp mô hình biết không được đếm nhầm gì.

Trả lời chi tiết:
Nếu chỉ có exemplar dương, mô hình dễ over-count trong bối cảnh nền phức tạp. Exemplar âm đóng vai trò ràng buộc phủ định: các pattern giống nhưng không phải lớp mục tiêu cần bị triệt tiêu tín hiệu. Trong training, loss cho nhánh âm giúp mô hình học phân biệt tốt hơn giữa đối tượng mục tiêu và distractor.

Bằng chứng:
- [code/source-code/FSC_train.py](code/source-code/FSC_train.py)
- [docs/slide/main.tex](docs/slide/main.tex)

Show code nếu bị hỏi:
- Mở phần tính loss trong [code/source-code/FSC_train.py](code/source-code/FSC_train.py).

## Nhóm B - Thiết kế thực nghiệm

### Câu 5. Nhóm đánh giá trên dữ liệu nào và dùng metric gì?
Trả lời ngắn:
Nhóm đánh giá trên FSC-147 test split với MAE và RMSE, kèm latency demo để phản ánh khả năng triển khai.

Trả lời chi tiết:
FSC-147 là benchmark phổ biến cho few-shot và zero-shot counting nên phù hợp để so sánh với hướng tiếp cận exemplar-based. Nhóm dùng MAE để đo sai số tuyệt đối trung bình, RMSE để nhấn mạnh các lỗi lớn. Ngoài chất lượng đếm, nhóm đo thêm thời gian inference và chi phí sinh exemplar nhằm đánh giá trade-off accuracy-speed thực tế.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)
- [code/source-code/FSC_test.py](code/source-code/FSC_test.py#L164)

Show code nếu bị hỏi:
- Mở batched_rmse trong [code/source-code/FSC_test.py](code/source-code/FSC_test.py#L164).

### Câu 6. 4 cấu hình thực nghiệm chính của nhóm là gì?
Trả lời ngắn:
Có 4 cấu hình: baseline VA-Count, baseline + Rich Prompt, YOLO-World, và YOLO-World + Rich Prompt.

Trả lời chi tiết:
Nhóm thiết kế 4 nhánh để tách rõ tác động của 2 biến: detector backbone và Rich Prompt. Baseline dùng GroundingDINO theo VA-Count gốc. Sau đó thêm Rich Prompt để xem gain ngữ nghĩa trên cùng backbone. Tiếp theo thay detector bằng YOLO-World để kiểm tra hướng tối ưu tốc độ. Cuối cùng kết hợp YOLO-World + Rich Prompt để xem có thể kéo lại accuracy không.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)
- [docs/slide/main.tex](docs/slide/main.tex)

### Câu 7. Kết quả MAE/RMSE tổng quan của 4 cấu hình là gì?
Trả lời ngắn:
Baseline 17.99 MAE, +Rich Prompt tốt nhất 17.80, YOLO đơn 19.03, YOLO+Rich Prompt 17.91 gần baseline.

Trả lời chi tiết:
Trên test split FSC-147, kết quả là: VA-Count baseline MAE 17.99, RMSE 129.39. VA-Count + Rich Prompt đạt MAE 17.80 (tốt nhất), RMSE 129.69. VA-Count + YOLO-World đạt MAE 19.03, RMSE 131.55. VA-Count + YOLO-World + Rich Prompt kéo về MAE 17.91, RMSE 130.98. Điều này cho thấy Rich Prompt giúp mạnh ở nhánh YOLO, còn nhánh GroundingDINO cải thiện nhẹ.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 8. Nhóm đo hiệu năng runtime thế nào?
Trả lời ngắn:
Nhóm đo cả latency inference và thời gian sinh exemplar toàn bộ FSC-147, vì đây là nút thắt triển khai thực tế.

Trả lời chi tiết:
Ngoài MAE/RMSE, nhóm đo thời gian demo per image và tổng wall-clock để tạo annotation exemplar cho 6,146 ảnh. Đây là phần quan trọng vì nếu chỉ nhìn accuracy thì khó đánh giá tính khả thi chạy nhiều ablation trên 1 GPU phổ thông. Kết quả cho thấy detector choice ảnh hưởng cực lớn tới thời gian chuẩn bị dữ liệu và thời gian phản hồi demo.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)

## Nhóm C - Kết quả thực nghiệm và phản biện sâu

### Câu 9. Vì sao kết quả cải thiện MAE không nhiều dù thêm nhiều kỹ thuật?
Trả lời ngắn:
Vì pipeline bị giới hạn bởi bottleneck ở khối counter/density decoding và chất lượng exemplar chưa luôn ổn định trong các ảnh khó.

Trả lời chi tiết:
Có 2 lý do chính. Một là giới hạn kiến trúc downstream: dù cải thiện EEM, phần đếm vẫn dùng cùng một backbone/counter nên biên độ gain bị trần. Hai là sai số do dữ liệu khó vẫn tồn tại: dense scenes và fragmented objects gây under-count hoặc over-count lớn. Khi error mode mang tính cấu trúc chưa được xử lý triệt để, cải thiện ở đầu vào thường chỉ tăng vừa phải trên MAE tổng.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)
- [docs/slide/main.tex](docs/slide/main.tex)

### Câu 10. Nhóm giải thích thế nào về hiện tượng “GroundingDINO cải thiện ít, YOLO cải thiện nhiều khi thêm Rich Prompt”?
Trả lời ngắn:
GroundingDINO vốn mạnh ngữ nghĩa nên gain thêm nhỏ, còn YOLO-World hưởng lợi nhiều hơn từ ngữ cảnh prompt.

Trả lời chi tiết:
Trong kết quả, delta MAE khi thêm Rich Prompt cho GroundingDINO chỉ là **Δ = 0.19** (17.99 → 17.80), còn YOLO-World cải thiện rõ hơn với **Δ = 1.12** (19.03 → 17.91). Diễn giải kỹ thuật là GroundingDINO đã có năng lực language grounding tốt từ đầu nên thêm mô tả giàu ngữ nghĩa chỉ còn dư địa nhỏ — đạt gần "semantic saturation". Với YOLO-World, Rich Prompt (Gemini 2.5 Flash + CLIP ViT-B/32 re-ranking) giúp chọn exemplar tốt hơn và giảm ambiguity nên hiệu quả thấy rõ hơn. Đây là luận điểm "semantic saturation" mà nhóm nhấn mạnh.

Bằng chứng:
- [docs/slide/main.tex](docs/slide/main.tex)
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 11. Tại sao YOLO-World đơn lẻ kém MAE hơn GroundingDINO?
Trả lời ngắn:
YOLO-World nhanh hơn nhưng chất lượng exemplar thô thường thấp hơn ở các cảnh phức tạp, làm tăng sai số đầu vào cho nhánh đếm.

Trả lời chi tiết:
YOLO-World có lợi thế tốc độ, nhưng ở chế độ không tăng cường prompt và re-ranking mạnh, candidate box dễ lẫn nhiễu hơn với cảnh đông, chồng lấp, hoặc đối tượng mảnh. Sai lệch ở bước exemplar truyền thẳng vào density estimation, kéo MAE xấu hơn baseline GroundingDINO. Tuy nhiên khi thêm Rich Prompt, khoảng cách này giảm đáng kể.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)
- [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L258)
- [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L173)

### Câu 12. YOLO-World hơn GroundingDINO ở những điểm nào?
Trả lời ngắn:
YOLO-World hơn rõ ở tốc độ inference và chi phí sinh exemplar, phù hợp triển khai demo và chạy thử nghiệm nhanh.

Trả lời chi tiết:
Theo số liệu, YOLO-World có latency thấp hơn đáng kể và giảm mạnh tổng thời gian tạo exemplar toàn dataset. Đây là lợi thế vận hành quan trọng: giúp thử nhiều cấu hình hơn trong cùng tài nguyên phần cứng. Khi cộng Rich Prompt, YOLO đạt MAE gần baseline GroundingDINO nhưng giữ lợi thế vận hành, nên phù hợp làm cấu hình triển khai.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 13. Cụ thể tốc độ hơn bao nhiêu và ý nghĩa thực tế là gì?
Trả lời ngắn:
YOLO nhanh hơn rõ ở cả demo time và exemplar generation: demo 0.60 s/img vs 1.47 s/img baseline, sinh exemplar ≈ 4 giờ vs ≈ 25 giờ khi có Rich Prompt.

Trả lời chi tiết:
Về demo time trên RTX 4060: YOLO-World đơn đạt **0.60 s/img** (nhanh gấp 2.5× so với baseline 1.47 s/img); YOLO+Rich Prompt **2.41 s/img** — vẫn nhanh hơn GDino+Rich Prompt ở 5.76 s/img. Về sinh exemplar toàn FSC-147 (6,146 ảnh): YOLO+Rich Prompt chỉ mất **≈ 4 giờ**, so với **≈ 25 giờ** của GroundingDINO+Rich Prompt. Ý nghĩa là nhóm có thể chạy thêm ablation và tinh chỉnh nhiều vòng trên một máy RTX 4060.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 14. Những failure case lớn nhất của hệ thống là gì?
Trả lời ngắn:
Ba nhóm lỗi chính: cảnh dày đặc bị under-count, đối tượng phân mảnh bị over-count, và đôi lúc đếm “hợp lý giả” dù exemplar chưa tốt.

Trả lời chi tiết:
Trong phân tích lỗi, dense scenes làm các peak density dính nhau nên đếm thiếu. Fragmented objects như các cấu trúc có nhiều thành phần làm mô hình đếm thừa. Ngoài ra có trường hợp exemplar không tối ưu nhưng output count vẫn có vẻ hợp lý do ảnh hưởng priors, gây rủi ro sai lệch có hệ thống. Đây là lý do nhóm không tuyên bố vượt trội tuyệt đối mà nhấn mạnh trade-off có kiểm soát.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 15. Nếu được làm tiếp, nhóm ưu tiên cải tiến kỹ thuật nào để tăng MAE rõ hơn?
Trả lời ngắn:
Ưu tiên 1 là nâng cấp counter/decoder và loss cho cảnh dense; ưu tiên 2 là tăng chất lượng exemplar bằng re-ranking/validation chặt hơn.

Trả lời chi tiết:
Kết quả cho thấy chỉ tối ưu EEM thì gain hữu hạn. Hướng tiếp theo nên đi vào phần đếm: tăng khả năng tách instance gần nhau, điều chỉnh loss cho dense scenes, và bổ sung ràng buộc hình học. Đồng thời cần cải thiện khâu exemplar với schema quality score rõ ràng để giảm nhiễu đầu vào. Khi 2 nhánh này đi cùng nhau, kỳ vọng MAE cải thiện ổn định hơn thay vì chỉ dao động nhỏ.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)
- [code/source-code/models_mae_cross.py](code/source-code/models_mae_cross.py)

## Nhóm D - Đóng góp chính của đồ án

### Câu 16. Đóng góp khoa học/kỹ thuật chính của đồ án là gì?
Trả lời ngắn:
Đóng góp chính là tích hợp Rich Prompt và YOLO-World vào VA-Count, rồi chứng minh thực nghiệm rõ trade-off accuracy-speed, cùng quan sát semantic saturation.

Trả lời chi tiết:
Đồ án có 3 đóng góp lớn. Thứ nhất, xây pipeline hoàn chỉnh từ prompt enhancement đến counting inference có thể chạy demo. Thứ hai, thực nghiệm 4 cấu hình với số liệu đầy đủ MAE/RMSE/latency/extraction cost, giúp so sánh công bằng. Thứ ba, chỉ ra hiện tượng saturation: backbone ngữ nghĩa mạnh như GroundingDINO nhận gain nhỏ từ Rich Prompt, còn backbone nhanh hơn như YOLO hưởng lợi mạnh hơn. Đây là insight hữu ích cho quyết định triển khai.

Bằng chứng:
- [docs/slide/main.tex](docs/slide/main.tex)
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 17. Vì sao nhóm chọn cấu hình triển khai là YOLO-World + Rich Prompt?
Trả lời ngắn:
Vì cấu hình này cân bằng tốt nhất giữa độ chính xác gần baseline và chi phí vận hành thấp.

Trả lời chi tiết:
Nếu chọn tuyệt đối theo MAE, GroundingDINO+Rich Prompt đứng đầu nhẹ. Nhưng xét sản phẩm/demo thực tế, chi phí inference và tạo exemplar cũng quan trọng. YOLO+Rich Prompt đạt MAE gần baseline GroundingDINO, trong khi giảm đáng kể chi phí thời gian vận hành. Do đó nhóm chọn cấu hình này làm điểm cân bằng giữa chất lượng và khả năng triển khai.

Bằng chứng:
- [docs/RESULTS.md](docs/RESULTS.md)

### Câu 18. Điểm mới so với chỉ dùng VA-Count gốc là gì?
Trả lời ngắn:
Điểm mới là thêm lớp ngữ nghĩa giàu hơn bằng Rich Prompt và mở rộng detector sang YOLO-World để đạt cấu hình thực dụng hơn.

Trả lời chi tiết:
VA-Count gốc là nền tảng tốt, nhưng đồ án mở rộng theo hướng hệ thống: prompt enhancement tự động, nhiều detector backend, cơ chế đánh giá theo cả accuracy và runtime, và demo hóa pipeline. Đổi mới không chỉ là thay 1 model mà là thiết kế một khung thực nghiệm có thể đưa ra quyết định triển khai dựa trên dữ liệu định lượng.

Bằng chứng:
- [docs/slide/main.tex](docs/slide/main.tex)
- [code/source-code/demo_app_advanced.py](code/source-code/demo_app_advanced.py)

## Nhóm E - Câu hỏi show code (trọng tâm)

### Câu 19. Nếu thầy yêu cầu show luồng end-to-end, mở code nào theo thứ tự?
Trả lời ngắn:
Mở demo_inference theo thứ tự: full pipeline -> detect -> filter -> exemplar select -> counting inference.

Trả lời chi tiết:
Thứ tự show code đề xuất:
1. full_counting_pipeline ở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L593)
2. detect_with_grounding_dino ở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L173)
3. detect_with_yolo ở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L258)
4. filter_single_objects ở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L328)
5. select_exemplars ở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L389)
6. run_counting_inference ở [code/source-code/demo_inference.py](code/source-code/demo_inference.py#L519)

Thông điệp khi show:
- Detector quyết định chất lượng candidate ban đầu.
- Classifier + exemplar selection quyết định chất lượng support signal.
- Counter quyết định sai số cuối cùng.

### Câu 20. Nếu thầy hỏi sâu về “học gì trong training” và “tính metric ra sao”, mở code nào?
Trả lời ngắn:
Mở FSC_train để giải thích loss, models_mae_cross để giải thích kiến trúc, FSC_test để giải thích RMSE.

Trả lời chi tiết:
Playbook mở code:
1. Loss và huấn luyện: [code/source-code/FSC_train.py](code/source-code/FSC_train.py)
2. Kiến trúc encoder-decoder và cross-attention: [code/source-code/models_mae_cross.py](code/source-code/models_mae_cross.py#L18), [code/source-code/models_mae_cross.py](code/source-code/models_mae_cross.py#L136), [code/source-code/models_mae_cross.py](code/source-code/models_mae_cross.py#L150)
3. Metric: batched_rmse ở [code/source-code/FSC_test.py](code/source-code/FSC_test.py#L164)

Thông điệp khi show:
- Training học cả phân biệt dương/âm và tối ưu density count.
- Kiến trúc cross-attention dùng exemplar để điều kiện hóa dự đoán density.
- Metric được tính tách bạch và nhất quán trên test split.

---

## Gợi ý mini-script 60 giây khi bị hỏi bất ngờ
- Chốt mục tiêu: đếm zero-shot theo prompt với exemplar tự sinh.
- Chốt kết quả: MAE tốt nhất ở GroundingDINO+RP, nhưng YOLO+RP gần tương đương và vận hành rẻ hơn.
- Chốt hạn chế: dense/fragmented scenes vẫn là điểm nghẽn chính.
- Chốt đóng góp: insight semantic saturation + khung so sánh thực nghiệm đầy đủ accuracy-speed.

## Checklist trước lúc lên bảo vệ
- Thuộc 4 con số MAE chính của 4 cấu hình trong [docs/RESULTS.md](docs/RESULTS.md).
- Thuộc 3 failure modes chính trong [docs/RESULTS.md](docs/RESULTS.md).
- Luyện mở 6 hàm trọng tâm trong [code/source-code/demo_inference.py](code/source-code/demo_inference.py) trong dưới 10 giây.
- Chuẩn bị một câu kết luận trade-off: accuracy gần baseline, tốc độ/chi phí tốt hơn.
- Nhớ Rich Prompt = **Gemini 2.5 Flash** (sinh mô tả) + **CLIP ViT-B/32** (re-rank exemplar), top-3 → top-5 luồng dương.
- Nhớ 2 số Δ MAE: GroundingDINO Δ = 0.19, YOLO-World Δ = 1.12 — luận điểm semantic saturation.
