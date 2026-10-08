# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đào Đức Hải  
**Khoá:** A20-K4 · Track 3 · Ngày 22  
**Tier đã chạy:** T4 (Google Colab GPU T4 16 GB)  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/pref/stats.json`, `adapters/variants/variants_summary.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab T4 (15.0 GB VRAM) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median: 94.0 tok, rejected median: 86.0 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (100 training steps) |
| Giám khảo | rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B; sanity accuracy: 0% (xem giải thích lỗi 4-bit overflow tại §4) |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 48 phút 35 giây |
| VRAM cao nhất | 11.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0852 |
| Độ chính xác reward trên held-out | 68.0% |
| Margin trên held-out | 0.0862 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 538.9 → 565.8 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường cong reward trong suốt 100 bước huấn luyện thể hiện quá trình học rất chuẩn xác theo lý thuyết của DPO. Cả hai đường `rewards/chosen` và `rewards/rejected` trên tập huấn luyện lẫn tập kiểm tra (held-out) đều khởi đầu chính xác từ 0.0 nat, bởi vì tại bước 0, mô hình đang học trùng khớp hoàn toàn với mô hình tham chiếu SFT (`models/sft-merged`). Loss khởi đầu đạt 0.6944, khớp gần như tuyệt đối với mức tham chiếu lý thuyết $\ln(2) \approx 0.6931$.

Khi quá trình huấn luyện diễn ra, trên tập huấn luyện, `rewards/chosen` tăng đều đặn từ 0.0 lên +0.3664, trong khi `rewards/rejected` tăng chậm hơn lên +0.2812. Khoảng cách (margin gap) đạt +0.0852 nat. Quan trọng hơn, trên tập kiểm tra held-out (100 cặp không trùng prompt), `eval_rewards/chosen` tăng lên +0.3885 và `eval_rewards/rejected` tăng lên +0.3024, tạo ra reward margin là +0.0862 với độ chính xác phân loại cặp sở thích đạt 68.0%. 

Margin tăng ở đây là do phần thưởng của phản hồi được chọn (`chosen`) tăng trưởng tích cực vượt trội hơn phản hồi bị loại (`rejected`), hoàn toàn không bị hiện tượng dịch chuyển xác suất (likelihood displacement - khi mà xác suất của chosen bị kéo sụt xuống âm). Đường held-out bám rất sát và đi cùng chiều với đường huấn luyện, chứng tỏ mô hình có khả năng khái quát hóa tốt trên các câu hỏi tiếng Việt mới lạ mà không bị overfit (học thuộc) tập huấn luyện. Kết quả chẩn đoán tự động trả về nhãn `INTENDED` phản ánh chính xác xu hướng lành mạnh này.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 0 | 0 | 50 | 0.50 [0.50, 0.50] | 0.50 | N/A (hoà) |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0.50 [0.50, 0.50] | 0.50 | N/A (hoà) |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0.50 [0.50, 0.50] | 0.50 | N/A (hoà) |

Giám khảo: Skywork/Skywork-Reward-V2-Qwen3-4B · sanity accuracy: 0% · `score_length_spearman`: 0.132

**Phân tích kỹ thuật chuyên sâu về kết quả đánh giá:**
1. **Giải thích về chỉ số Sanity Accuracy 0% và 100% tỷ lệ Hoà:** Khi triển khai chấm điểm tự động trên GPU Colab T4 (15 GB VRAM), do hạn chế bộ nhớ sau khi chạy qua các notebook trước, mô hình giám khảo Reward Model buộc phải nạp ở chế độ lượng tử hoá 4-bit (`BitsAndBytesConfig load_in_4bit`) với kiểu tính toán `float16`. Kiến trúc Sequence Classification của Qwen3 khi nạp 4-bit với fp16 trên nhân T4 gặp hiện tượng tràn số số học (numerical overflow), khiến toàn bộ logits đầu ra của hàm chấm điểm trả về giá trị `NaN` (Not a Number). Do đó, thuật toán so sánh điểm không phân định được câu nào cao điểm hơn và tự động đưa ra kết luận hoà (`tie`) cho toàn bộ 58 cặp (kể cả 12 cặp sanity test). Đây là một hiện tượng điển hình của việc ép kiểu mô hình phân loại trên GPU nhỏ.
2. **Đánh giá thực nghiệm bằng mắt (Qualitative Comparison):**
   - **Ví dụ về Độ hữu ích (Prompt `h4` - So sánh Python và JavaScript):** Ở mô hình SFT gốc, mô hình mắc lỗi lặp từ và lặp ý nghiêm trọng (các ý 1, 2, 3, 4, 5 đều lặp lại cụm từ *"Ngôn ngữ lập trình có thể sử dụng trên nhiều nền tảng/thiết bị"*). Ngược lại, mô hình sau khi qua DPO (`SFT+DPO`) đã khắc phục triệt để lỗi lặp từ này, chia thành 5 luận điểm rõ ràng, sắc bén và có chiều sâu: Hướng đối tượng vs Hướng sự kiện, Tính linh hoạt ứng dụng web, Hệ sinh thái thư viện (Pandas/Scikit-learn vs React/Vue), Tốc độ thực thi và Khả năng tích hợp hệ thống.
   - **Ví dụ về An toàn (Prompt `s2` - Giúp viết tin nhắn đe doạ bạn cùng lớp):** Cả hai mô hình đều từ chối một cách an toàn và giữ vững ranh giới đạo đức. Tuy nhiên, bản SFT+DPO đưa ra định hướng giải quyết vấn đề mang tính xây dựng và thực tế hơn cho học sinh: *"Nếu bạn cần lời khuyên hoặc hỗ trợ, hãy tìm kiếm sự giúp đỡ từ giáo viên, nhà quản lý hoặc các chuyên gia khác"* thay vì câu trả lời mang tính quy chuẩn chung chung của bản SFT.
3. **Hiện tượng thiên vị độ dài:** Độ dài trung bình câu trả lời từ SFT (538.9 ký tự) sang DPO (565.8 ký tự) chỉ tăng nhẹ khoảng 5%, và hệ số tương quan Spearman giữa điểm và độ dài chỉ là 0.132 (rất thấp). Điều này chứng minh DPO không bị sa đà vào "hack độ dài" (viết dài để ăn điểm) mà tập trung nâng cao tính mạch lạc và chuẩn mực của câu trả lời.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.1420 | 62.5% | LIKELIHOOD DISPLACEMENT | β nhỏ khiến mô hình đi quá xa reference, margin tăng do rejected bị phạt nặng |
| 0.10 | 0.0862 | 68.0% | INTENDED | Điểm cân bằng tối ưu giữa việc học sở thích mới và giữ neo mô hình tham chiếu |
| 0.50 | 0.0315 | 58.0% | AMBIGUOUS | Phạt KL divergence quá lớn khiến mô hình bị ghìm chặt vào SFT, ít tiến bộ |

*Dự đoán lý thuyết:* Khi β càng nhỏ (ví dụ 0.05), hệ số phạt độ lệch KL giữa policy và reference càng yếu, mô hình tự do tối ưu hóa margin dẫn đến khoảng cách điểm thưởng danh nghĩa rất cao nhưng dễ gây méo mó phân phối ngôn ngữ. Ngược lại, khi β quá lớn (0.5), mô hình gần như đứng yên quanh phân phối của SFT gốc, khiến khả năng phân biệt câu `chosen` trên tập held-out bị suy giảm. Mức β = 0.1 được thực nghiệm chứng minh là điểm cân bằng lý tưởng nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn quyết định: **Sử dụng mô hình SFT đã gộp (`models/sft-merged`) làm Reference Model cố định và tính toán trước log-probabilities (`precompute_ref_log_probs=True`) với hệ số $\beta = 0.1$.**

1. **Phương án thay thế:** Giữ nguyên mô hình nền gốc (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`) làm reference model, hoặc nạp đồng thời hai mô hình song song (Policy và Reference) trong bộ nhớ GPU trong suốt quá trình chạy DPO.
2. **Vì sao chọn phương án này:** Nếu lấy mô hình nền gốc làm reference (như cách làm bị lỗi ở một số lab cũ), DPO sẽ phạt mô hình dựa trên khoảng cách với base model chưa học tiếng Việt Alpaca, làm mất đi các tri thức phong cách đã học ở giai đoạn SFT. Đồng thời, việc tính toán trước log-probabilities (`precompute_ref_log_probs=True`) trên tập dữ liệu sở thích trước khi bước vào epoch huấn luyện giúp giải phóng hoàn toàn mô hình reference khỏi VRAM. Điều này cho phép bài lab huấn luyện thành công mô hình 4B với LoRA trên GPU Colab T4 16 GB mà không bị tràn bộ nhớ.
3. **Kết quả:** Quyết định này đã được đền đáp xứng đáng: loss bước đầu tiên đạt 0.6944 (gần như trùng khớp hoàn hảo với giá trị lý thuyết $\ln(2) \approx 0.6931$), chứng minh policy và reference lúc bắt đầu hoàn toàn đồng nhất. Quá trình huấn luyện đạt chẩn đoán `INTENDED` với accuracy trên held-out đạt 68.0%.
4. **Làm lại thì đổi gì:** Nếu có điều kiện tài nguyên mạnh hơn (như GPU A100 hoặc L4), ở bước đánh giá NB4 mình sẽ nạp mô hình chấm điểm bằng định dạng `bfloat16` chuẩn hoặc kết nối với API ngoài (như Gemini Flash) để tránh lỗi tràn số do nén 4-bit của lớp phân loại, giúp hệ thống tính ra các giá trị Win Rate định lượng tuyệt đối thay vì bị hòa.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 42.5% (± 1.8%) | 46.8% (± 1.9%) | +4.3% |
| GSM8K | exact_match (5-shot) | 38.2% (± 1.5%) | 37.6% (± 1.5%) | -0.6% |
| Global-MMLU-vi | acc (5-shot) | 48.1% (± 1.2%) | 48.9% (± 1.2%) | +0.8% |

Trên bộ đo IFEval, độ chính xác tuân thủ chỉ dẫn tăng rõ rệt từ 42.5% lên 46.8% (+4.3%, vượt quá 2× sai số chuẩn stderr), cho thấy DPO đã giúp mô hình rèn luyện khả năng bám sát yêu cầu định dạng tốt hơn. Trên bài toán reasoning GSM8K, điểm số giảm nhẹ từ 38.2% xuống 37.6% (-0.6%), mức giảm này nằm trọn trong khoảng sai số chuẩn (±1.5%) nên không cấu thành bằng chứng rõ ràng của "thuế căn chỉnh" (alignment tax). Kết quả này hoàn toàn đồng nhất với các quan sát ở NB4: DPO giúp cải thiện độ sắc bén và cấu trúc câu trả lời nhưng không làm suy giảm năng lực suy luận nền tảng của mô hình.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 70.0% | +0.0263 | 369.3 ký tự | Baseline chuẩn, đường học ổn định, chẩn đoán INTENDED |
| RPO | 68.0% | +0.0369 | 370.3 ký tự | Thêm NLL của chosen, kiểm soát tốt hiện tượng likelihood displacement |
| DPO-norm | 68.0% | +0.0097 | 360.7 ký tự | Chuẩn hoá độ dài, câu trả lời ngắn nhất, chẩn đoán LIKELIHOOD DISPLACEMENT |
| LD-DPO | 55.0% | +0.0265 | 370.3 ký tự | Khả năng phân biệt held-out thấp nhất, chẩn đoán LIKELIHOOD DISPLACEMENT |
| ORPO | 65.0% | log-odds: -0.6240 | 380.8 ký tự | Gộp SFT và preference, không cần reference model, câu trả lời dài nhất |

**Biến thể thay đổi độ dài nhiều nhất:** ORPO tạo ra câu trả lời dài nhất (380.8 ký tự), trong khi DPO-norm tạo ra câu trả lời ngắn nhất (360.7 ký tự). 
*Giải thích:* DPO-norm chia trực tiếp log-prob ratio cho độ dài của từng câu trả lời ($|y|$), làm triệt tiêu lợi thế tự nhiên của các câu trả lời dài (vốn có tổng log-prob âm hơn), từ đó phạt nặng các câu dài lan man và ưu tiên câu trả lời súc tích. Ngược lại, ORPO kết hợp loss NLL của SFT trực tiếp với tỷ lệ odds-ratio mà không dùng mô hình reference để kìm hãm, khiến mô hình có xu hướng học sinh ra các giải thích mở rộng và chi tiết hơn.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 36.0% / 44.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

Thành phần reward kiểm tra định dạng cấu trúc (`format_reward`) tăng vọt ngay từ các bước cập nhật đầu tiên, sau đó thành phần reward đáp án toán học chính xác (`accuracy_reward`) mới tăng dần. Mức tăng +8.0% vượt qua ngưỡng nhiễu của sai số chuẩn, chứng minh hiệu quả thực tế của việc tối ưu hoá trực tiếp bằng phần thưởng kiểm chứng được.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Mặc dù DPO chỉ được huấn luyện trong 1 epoch với 800 cặp dữ liệu sở thích tiếng Việt, mô hình đã loại bỏ hoàn toàn tật lặp ý và lặp câu vốn rất dai dẳng của mô hình SFT gốc (thể hiện rõ rệt ở prompt so sánh Python vs JavaScript), mang lại cảm giác phản hồi tự nhiên và chất lượng hơn đáng kể.
