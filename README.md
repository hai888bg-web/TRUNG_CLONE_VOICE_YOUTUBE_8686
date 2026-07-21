# TRUNG_CLONE_VOICE_YOUTUBE_8686

Giao diện tiếng Việt clone giọng nói (VoxCPM2), chạy hàng loạt từ file .txt ra MP3 đánh số, hỗ trợ 6 ngôn ngữ (VN/EN/ES/KR/JP/PT).

## Cài đặt & chạy

**Yêu cầu:** máy đã cài [Git](https://git-scm.com/downloads) (Windows) — Mac thường có sẵn.

1. Tải repo này về (Code → Download ZIP, hoặc `git clone`).
2. Chạy đúng file theo máy:
   - **Windows**: double-click `chay_win.bat` — máy khỏe (RAM ≥ 28GB) tự tải thêm model chạy local cho nhanh (~vài phút, chỉ lần đầu).
   - **macOS**: double-click `chay_mac.command` — chạy qua cloud, nhẹ, không cần RAM mạnh.
3. Lần đầu chờ cài môi trường (vài phút). Trình duyệt tự mở `http://localhost:7860`.

## Ghi chú

- Giọng đã lưu (`voices/`) và lịch sử (`history/`) nằm trên máy, không đồng bộ qua GitHub.
- Windows tự tải thêm code model gốc ([OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)) vào thư mục `engine/` khi cần chạy local — không cần thao tác thủ công.
