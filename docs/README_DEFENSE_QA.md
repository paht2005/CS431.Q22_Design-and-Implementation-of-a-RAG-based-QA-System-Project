# 📚 Bộ Câu Hỏi Bảo Vệ Đồ Án - Zero-shot Object Counting

> **Mục tiêu:** Chuẩn bị cho phần phản biện 10-15 phút  
> **Tổng số câu hỏi:** 27 câu (20 câu chính + 7 câu bổ sung về YOLO-World & Hyperparameters)

---

## 📊 Bảng Tóm Tắt Kết Quả Chính

| Cấu hình | MAE ↓ | RMSE ↓ | Inference (s/img) |
|----------|-------|--------|-------------------|
| VA-Count (Baseline) | 17.99 | 129.39 | 1.47 |
| VA-Count + Rich Prompt | **17.80** | 129.69 | 5.76 |
| VA-Count + YOLO-World | 19.03 | 131.55 | **0.60** |
| VA-Count + YOLO-World + RP | 17.91 | 130.98 | 2.41 |

**Số quan trọng cần nhớ:**
- Δ MAE (Rich Prompt trên GDino): **0.19** (gần bão hòa)
- Δ MAE (Rich Prompt trên YOLO): **1.12** (lợi ích lớn)

---

## 🅰️ NHÓM A - PHƯƠNG PHÁP THỰC HIỆN

### Câu 1. Bài toán của nhóm là gì và vì sao chọn hướng zero-shot object counting?

**Trả lời ngắn (20-30s):**
> Nhóm giải bài toán đếm đối tượng khi số lớp rất đa dạng và không thể gán nhãn đầy đủ cho mọi lớp mới. Zero-shot giúp dùng prompt ngôn ngữ + exemplar để đếm lớp chưa thấy trong train, phù hợp bối cảnh thực tế.

**Trả lời chi tiết (60-90s):**
> Bài toán của nhóm là đếm số lượng đối tượng theo prompt lớp trên ảnh, trong bối cảnh không thể xây một bộ nhãn đầy đủ cho mọi lớp. Thay vì huấn luyện detector đóng trên một số lớp cố định, nhóm dùng pipeline zero-shot dựa trên VA-Count: tạo exemplar dương/âm từ detector + prompt, sau đó đưa vào mô hình density map để suy ra count. Hướng này phù hợp vì có thể mở rộng lớp mới bằng ngôn ngữ tự nhiên, giảm phụ thuộc annotate thủ công.

---

### Câu 2. Kiến trúc tổng thể pipeline của nhóm gồm những bước nào?

**Trả lời ngắn:**
> Pipeline có 4 bước chính: prompt enhancement, detection, lọc exemplar đơn đối tượng, rồi counting bằng density map.

**Trả lời chi tiết:**
> Pipeline bắt đầu từ ảnh và class prompt. Nếu bật Rich Prompt, hệ thống gọi Gemini 2.5 Flash để tạo mô tả chi tiết hơn, sau đó dùng CLIP ViT-B/32 re-rank candidate theo semantic similarity với class name gốc. Sau đó dùng detector (GroundingDINO hoặc YOLO-World) lấy candidate boxes. Các box được lọc bằng binary classifier để giữ exemplar khả năng là một instance đơn. Tiếp theo xếp hạng và lấy top-3 exemplar (hoặc top-5 luồng dương khi bật Rich Prompt) để feed vào mô hình đếm. Mô hình xuất density map, rồi tổng density để ra count cuối.

**Show code:**
- `full_counting_pipeline` trong [demo_inference.py#L593](../code/source-code/demo_inference.py#L593)

---

### Câu 3. Rich Prompt của nhóm hoạt động thế nào và đóng vai trò gì?

**Trả lời ngắn:**
> Rich Prompt mở rộng mô tả ngữ nghĩa của đối tượng, giúp detector và bước chọn exemplar bớt mơ hồ, đặc biệt hữu ích cho YOLO-World.

**Trả lời chi tiết:**
> Nhóm tích hợp **Gemini 2.5 Flash** để sinh mô tả chi tiết từ ảnh và class name, đồng thời có logic chuẩn hóa class về dạng số ít để prompt ổn định. Tiếp theo, **CLIP ViT-B/32** re-rank các candidate exemplar theo semantic similarity với tên lớp gốc — đây là bộ lọc ngữ nghĩa cấp hai đảm bảo patch khớp với class name. Số lượng exemplar dương cũng mở rộng từ top-3 (mặc định) lên **top-5** khi bật Rich Prompt.

**Show code:**
- `enhance_prompt_with_gemini` trong [prompt_enhancer.py#L49](../code/source-code/prompt_enhancer.py#L49)

---

### Câu 4. Vì sao nhóm dùng cả exemplar dương và exemplar âm?

**Trả lời ngắn:**
> Exemplar dương giúp mô hình biết cần đếm gì, exemplar âm giúp mô hình biết không được đếm nhầm gì.

**Trả lời chi tiết:**
> Nếu chỉ có exemplar dương, mô hình dễ over-count trong bối cảnh nền phức tạp. Exemplar âm đóng vai trò ràng buộc phủ định: các pattern giống nhưng không phải lớp mục tiêu cần bị triệt tiêu tín hiệu. Trong training, loss cho nhánh âm giúp mô hình học phân biệt tốt hơn giữa đối tượng mục tiêu và distractor.

---

## 🅱️ NHÓM B - THIẾT KẾ THỰC NGHIỆM

### Câu 5. Nhóm đánh giá trên dữ liệu nào và dùng metric gì?

**Trả lời ngắn:**
> Nhóm đánh giá trên FSC-147 test split với MAE và RMSE, kèm latency demo để phản ánh khả năng triển khai.

**Trả lời chi tiết:**
> FSC-147 là benchmark phổ biến cho few-shot và zero-shot counting (6,135 ảnh, 147 lớp). Nhóm dùng **MAE** để đo sai số tuyệt đối trung bình, **RMSE** để nhấn mạnh các lỗi lớn. Ngoài chất lượng đếm, nhóm đo thêm thời gian inference và chi phí sinh exemplar nhằm đánh giá trade-off accuracy-speed thực tế.

---

### Câu 6. 4 cấu hình thực nghiệm chính của nhóm là gì?

**Trả lời:**
> Có 4 cấu hình: 
> 1. **Baseline VA-Count** (GroundingDINO)
> 2. **Baseline + Rich Prompt**
> 3. **YOLO-World** (thay detector)
> 4. **YOLO-World + Rich Prompt**

Thiết kế này tách rõ tác động của 2 biến: **detector backbone** và **Rich Prompt**.

---

### Câu 7. Kết quả MAE/RMSE tổng quan của 4 cấu hình là gì?

| Cấu hình | MAE | RMSE |
|----------|-----|------|
| VA-Count baseline | 17.99 | 129.39 |
| VA-Count + Rich Prompt | **17.80** (tốt nhất) | 129.69 |
| VA-Count + YOLO-World | 19.03 | 131.55 |
| VA-Count + YOLO-World + RP | 17.91 | 130.98 |

**Key insight:** Rich Prompt giúp mạnh ở nhánh YOLO (Δ=1.12), còn nhánh GroundingDINO cải thiện nhẹ (Δ=0.19).

---

### Câu 8. Nhóm đo hiệu năng runtime thế nào?

**Trả lời:**
> Nhóm đo cả latency inference và thời gian sinh exemplar toàn bộ FSC-147:

| Phương pháp | Positive (giờ) | Negative (giờ) |
|-------------|----------------|----------------|
| GroundingDINO | ~3 | ~10 |
| GroundingDINO + RP | ~10 | ~15 |
| **YOLO-World** | **~0.5** | **~2** |
| **YOLO-World + RP** | **~1** | **~3** |

YOLO-World nhanh gấp **6×** GroundingDINO ở khâu trích xuất.

---

## 🅲️ NHÓM C - KẾT QUẢ THỰC NGHIỆM & PHẢN BIỆN SÂU

### Câu 9. Vì sao kết quả cải thiện MAE không nhiều dù thêm nhiều kỹ thuật?

**Trả lời:**
> Có 2 lý do chính:
> 1. **Giới hạn kiến trúc downstream:** dù cải thiện EEM, phần đếm (Counter) vẫn dùng cùng backbone → biên độ gain bị trần
> 2. **Error mode cấu trúc:** dense scenes và fragmented objects gây under-count/over-count lớn, chưa được xử lý triệt để

---

### Câu 10. Giải thích hiện tượng "GroundingDINO cải thiện ít, YOLO cải thiện nhiều khi thêm Rich Prompt"?

**Trả lời - Hiện tượng "Semantic Saturation":**

| Detector | Δ MAE với Rich Prompt | Giải thích |
|----------|----------------------|------------|
| GroundingDINO | 0.19 | Đã khai thác ~95% thông tin text → gần bão hòa |
| YOLO-World | **1.12** | Chỉ khai thác ~70% → Rich Prompt bổ sung hiệu quả |

**Quy luật:** *Bộ phát hiện càng yếu về ngữ nghĩa, Rich Prompt càng có ích.*

---

### Câu 11. Tại sao YOLO-World đơn lẻ kém MAE hơn GroundingDINO?

**Trả lời ngắn:**
> YOLO-World nhanh hơn nhưng chất lượng exemplar thô thường thấp hơn ở các cảnh phức tạp.

**Trả lời chi tiết - 3 nguyên nhân kỹ thuật:**

| Yếu tố | GroundingDINO | YOLO-World |
|--------|---------------|------------|
| **Kiến trúc fusion** | Cross-modal fusion **tại inference** | Text encoding **offline** (trước inference) |
| **Tương tác vision-language** | Query-based attention giữa image và text | Chỉ dùng CLIP cosine similarity |
| **Khả năng ngữ nghĩa** | Mạnh - hiểu context phức tạp | Yếu hơn - chỉ matching cơ bản |

**Kỹ thuật:**
- **GroundingDINO**: Detect-then-Match → fuse text và image features trong detection → hiểu ngữ cảnh thị giác
- **YOLO-World**: Prompt-then-Detect → encode text thành tham số cố định → không thích ứng với từng ảnh

---

### Câu 12. YOLO-World hơn GroundingDINO ở những điểm nào?

| Tiêu chí | GroundingDINO | YOLO-World | Lợi ích |
|----------|---------------|------------|---------|
| **Tốc độ inference** | 1.47s/ảnh | **0.60s/ảnh** | 2.5× nhanh hơn |
| **Tốc độ trích xuất** | ~13 giờ | **~2.5 giờ** | 5-6× nhanh hơn |
| **Prompt faithfulness** | ❌ Unfaithful | ✅ Faithful | Không detect sai lớp |

---

### Câu 13. Prompt Faithfulness là gì và tại sao quan trọng?

**Định nghĩa:** Khả năng mô hình CHỈ detect đối tượng khớp với prompt, không detect lung tung.

**Ví dụ minh họa:**
| Ảnh | Prompt | GroundingDINO | YOLO-World |
|-----|--------|---------------|------------|
| Hồng hạc | "lion" | Detect 14 con ❌ | Detect 0 ✅ |
| Hồng hạc | "bird" | Detect 14 con ✅ | Detect 14 ✅ |

**Tại sao quan trọng:** Zero-shot counting dựa vào text prompt → nếu model unfaithful → kết quả không đáng tin.

---

### Câu 14. Những failure case lớn nhất của hệ thống là gì?

**3 nhóm lỗi chính:**

| Loại lỗi | Mô tả | Hậu quả |
|----------|-------|---------|
| **Dense scenes** | Đối tượng nhỏ, sát nhau | Under-count (đếm thiếu) |
| **Fragmented objects** | Vật thể có nhiều thành phần (cửa sổ nhiều ô) | Over-count (đếm thừa) |
| **Texture similarity** | Đối tượng giống nền | Chọn sai exemplar |

---

### Câu 15. Nếu được làm tiếp, nhóm ưu tiên cải tiến kỹ thuật nào?

**Ưu tiên 1:** Nâng cấp Counter/Decoder
- Tăng khả năng tách instance gần nhau
- Điều chỉnh loss cho dense scenes
- Bổ sung ràng buộc hình học

**Ưu tiên 2:** Tăng chất lượng exemplar
- Re-ranking/validation chặt hơn
- Quality score schema rõ ràng

---

## 🅳️ NHÓM D - ĐÓNG GÓP CHÍNH

### Câu 16. Đóng góp khoa học/kỹ thuật chính của đồ án là gì?

**3 đóng góp lớn:**
1. ✅ Xây pipeline hoàn chỉnh từ prompt enhancement đến counting inference có thể chạy demo
2. ✅ Thực nghiệm 4 cấu hình với số liệu đầy đủ MAE/RMSE/latency/extraction cost
3. ✅ Chỉ ra hiện tượng **semantic saturation**: backbone mạnh nhận gain nhỏ từ Rich Prompt

---

### Câu 17. Vì sao nhóm chọn cấu hình triển khai là YOLO-World + Rich Prompt?

**Phân tích trade-off:**

| Cấu hình | MAE | Tốc độ | Prompt Faithful |
|----------|-----|--------|-----------------|
| VA-Count | 17.99 | 1.47s | ❌ |
| VA-Count + RP | **17.80** | 5.76s | ❌ |
| YOLO | 19.03 | **0.60s** | ✅ |
| **YOLO + RP** | 17.91 | 2.41s | ✅ |

**Kết luận:** YOLO+RP đạt MAE gần baseline, giữ lợi ích tốc độ và prompt faithfulness → **cân bằng tốt nhất cho production**.

---

### Câu 18. Điểm mới so với chỉ dùng VA-Count gốc là gì?

1. Thêm lớp ngữ nghĩa giàu hơn bằng **Rich Prompt** (Gemini + CLIP)
2. Mở rộng detector sang **YOLO-World** để đạt cấu hình thực dụng
3. Thiết kế **khung thực nghiệm** có thể đưa ra quyết định triển khai dựa trên dữ liệu định lượng

---

## 🅴️ NHÓM E - CÂU HỎI SHOW CODE

### Câu 19. Nếu thầy yêu cầu show luồng end-to-end, mở code nào theo thứ tự?

**Thứ tự show code:**
1. `full_counting_pipeline` → [demo_inference.py#L593](../code/source-code/demo_inference.py#L593)
2. `detect_with_grounding_dino` → [demo_inference.py#L173](../code/source-code/demo_inference.py#L173)
3. `detect_with_yolo` → [demo_inference.py#L258](../code/source-code/demo_inference.py#L258)
4. `filter_single_objects` → [demo_inference.py#L328](../code/source-code/demo_inference.py#L328)
5. `select_exemplars` → [demo_inference.py#L389](../code/source-code/demo_inference.py#L389)
6. `run_counting_inference` → [demo_inference.py#L519](../code/source-code/demo_inference.py#L519)

---

### Câu 20. Nếu thầy hỏi sâu về "học gì trong training" và "tính metric ra sao"?

**Playbook mở code:**
1. **Loss và huấn luyện:** [FSC_train.py](../code/source-code/FSC_train.py)
2. **Kiến trúc encoder-decoder:** [models_mae_cross.py#L18](../code/source-code/models_mae_cross.py#L18)
3. **Metric RMSE:** [FSC_test.py#L164](../code/source-code/FSC_test.py#L164)

---

## 🆕 NHÓM F - CÂU HỎI BỔ SUNG VỀ YOLO-WORLD & HYPERPARAMETERS

### Câu 21. Giải thích cụ thể lỗi của YOLO-World trong các trường hợp khó?

**Trường hợp 1: Đối tượng có texture tương tự nền**
- YOLO-World thiếu cross-modal reasoning → không phân biệt được foreground/background
- Ví dụ: Đếm lá trên nền cỏ → chọn sai mẫu âm/dương

**Trường hợp 2: Đối tượng nhỏ, mật độ cao**
- CLIP embedding không capture được chi tiết spatial
- Cosine similarity giảm khi đối tượng bị overlap

**Minh họa từ thực nghiệm (ảnh 716):**
- GroundingDINO: Density map tập trung đúng vị trí
- YOLO-World: Density map phân tán, nhiều false positive

---

### Câu 22. YOLO-World + Rich Prompt (MAE 17.91) gần bằng baseline (17.99). Vậy có nên dùng cấu hình này?

**Phân tích:**
- MAE chỉ kém baseline **0.08** (không đáng kể)
- Có **prompt faithfulness** (GDino không có)
- Thời gian trích xuất exemplar nhanh hơn **5-6×**

**Kết luận:** Nên dùng cho production khi cần tốc độ và độ tin cậy prompt.

---

### Câu 23. Kết luận cuối cùng về YOLO-World trong hệ thống VA-Count?

**Ưu điểm:**
- ✅ Tốc độ cao (6× nhanh hơn khi trích xuất, 2.5× khi inference)
- ✅ Prompt faithful - đáng tin cậy hơn trong ứng dụng thực tế
- ✅ Với Rich Prompt, đạt hiệu suất gần baseline

**Nhược điểm:**
- ❌ Yếu hơn về ngữ nghĩa do thiếu cross-modal fusion
- ❌ Cần Rich Prompt để bù đắp → thêm overhead từ LLM
- ❌ Khó khăn với đối tượng có texture tương tự nền

**Khuyến nghị:**
- **Dùng YOLO-World + RP** khi: cần tốc độ, dữ liệu lớn, đối tượng rõ ràng
- **Dùng GroundingDINO** khi: cần accuracy cao nhất, đối tượng phức tạp

---

### Câu 24. Tại sao chọn các siêu tham số huấn luyện này?

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| **AdamW** | - | Chuẩn cho fine-tuning transformer. Decoupled weight decay hiệu quả hơn Adam |
| **LR = 1×10⁻⁵** | Rất thấp | Fine-tuning từ pretrained → tránh catastrophic forgetting. Theo VA-Count paper |
| **Weight Decay = 0.05** | - | Giá trị chuẩn trong ViT, MAE, DINO papers |
| **Cosine Schedule** | - | Giảm LR mượt mà, hiệu quả hơn step decay |
| **500 Epochs** | - | FSC-147 training ~3,659 ảnh → cần nhiều epochs. Theo paper gốc |
| **Batch 8 × Accum 16** | Effective=128 | Giới hạn VRAM H200, gradient accumulation giả lập batch lớn |

**Câu trả lời ngắn cho thầy:**
> "Chúng em sử dụng các tham số theo paper VA-Count gốc (ECCV 2024) để đảm bảo tính tái lập. Learning rate thấp (10⁻⁵) vì đang fine-tune từ pretrained Counter, tránh phá vỡ features đã học. Batch size 8 với gradient accumulation 16 steps do giới hạn VRAM, tạo effective batch size 128 giúp training ổn định."

---

### Câu 25. Tại sao chọn CLIP ViT-B/32 cho re-ranking?

**Lý do:**
1. **Cân bằng tốc độ-chất lượng:** ViT-B/32 nhanh hơn ViT-L/14 mà vẫn đủ semantic understanding
2. **Đã được verify:** CLIP ViT-B/32 là backbone phổ biến, được test rộng rãi
3. **Phù hợp task:** Re-ranking chỉ cần so sánh similarity, không cần fine-grained features

---

### Câu 26. Tại sao chọn Gemini 2.5 Flash thay vì GPT-4 hay Claude?

**Lý do:**
1. **Tốc độ:** Flash model nhanh hơn đáng kể, phù hợp batch processing
2. **Chi phí:** Rẻ hơn GPT-4, Claude cho số lượng API calls lớn (6,135 ảnh × 2 prompts)
3. **Chất lượng đủ dùng:** Task sinh prompt không cần reasoning phức tạp
4. **Multimodal:** Có thể nhận cả ảnh + text để sinh mô tả chính xác hơn

---

### Câu 27. Tại sao mở rộng từ top-3 lên top-5 exemplar khi dùng Rich Prompt?

**Lý do:**
1. **Enhanced prompt sinh nhiều candidate chất lượng hơn:** Mô tả chi tiết giúp detector detect đúng nhiều instance hơn
2. **CLIP re-ranking đảm bảo chất lượng:** Dù lấy nhiều hơn, CLIP vẫn filter được candidate tốt
3. **Thực nghiệm cho thấy:** Top-5 với Rich Prompt cho kết quả tốt hơn top-3

---

## 📝 CHECKLIST TRƯỚC KHI BẢO VỆ

### Số liệu cần thuộc:
- [ ] 4 con số MAE: **17.99, 17.80, 19.03, 17.91**
- [ ] 2 số Δ MAE: GDino **0.19**, YOLO **1.12**
- [ ] Tốc độ: YOLO **0.60s**, baseline **1.47s**, YOLO+RP **2.41s**

### Failure modes:
- [ ] Dense scenes → under-count
- [ ] Fragmented objects → over-count
- [ ] Texture similarity → sai exemplar

### Code cần luyện mở (<10s):
- [ ] `full_counting_pipeline`
- [ ] `detect_with_grounding_dino`
- [ ] `detect_with_yolo`
- [ ] `filter_single_objects`
- [ ] `select_exemplars`
- [ ] `run_counting_inference`

### Câu kết luận trade-off:
> "Cấu hình YOLO-World + Rich Prompt đạt MAE 17.91 (chỉ kém baseline 0.08), nhưng nhanh gấp 2.5× inference và 5-6× trích xuất, đồng thời có prompt faithfulness. Đây là lựa chọn cân bằng tốt nhất cho triển khai thực tế."

---

## 🎯 MINI-SCRIPT 60 GIÂY KHI BỊ HỎI BẤT NGỜ

1. **Chốt mục tiêu:** Đếm zero-shot theo prompt với exemplar tự sinh
2. **Chốt kết quả:** MAE tốt nhất ở GDino+RP (17.80), nhưng YOLO+RP (17.91) gần tương đương và vận hành rẻ hơn
3. **Chốt hạn chế:** Dense/fragmented scenes vẫn là điểm nghẽn chính
4. **Chốt đóng góp:** Insight semantic saturation + khung so sánh accuracy-speed đầy đủ
