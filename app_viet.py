# Giao dien tieng Viet - tu nhan dien may de chon local hoac cloud
# Local: nhanh (3-5s), can RAM >=28GB hoac GPU NVIDIA
# Cloud: cham (10-30s), chay duoc moi may
import gradio as gr
import numpy as np
import soundfile as sf
import psutil

# --- Tu nhan dien phan cung ---
ram_gb = psutil.virtual_memory().total / (1024**3)

if ram_gb >= 28:
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    if has_cuda:
        MODE = "local"
        DEVICE = "cuda"
        print(f"[GPU NVIDIA] Chay VoxCPM2 local tren CUDA — nhanh nhat!")
    else:
        MODE = "local"
        DEVICE = "cpu"
        print(f"[RAM {ram_gb:.0f}GB] Chay VoxCPM2 local tren CPU — nhanh 3-5s/cau")
else:
    MODE = "cloud"
    DEVICE = None
    print(f"[RAM {ram_gb:.0f}GB] Chay qua cloud API — cham hon nhung khong ton RAM")

# --- Khoi tao theo mode ---
MODEL = None
API = None

if MODE == "local":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from voxcpm import VoxCPM
    model_id = "openbmb/VoxCPM2"
    print(f"Dang nap model {model_id} (lan dau tai ve, xin cho)...")
    MODEL = VoxCPM.from_pretrained(model_id, load_denoiser=False, device=DEVICE)
    print("Model da san sang!")
    toc_do_text = "Chạy trực tiếp trên máy bạn"
    if has_cuda:
        toc_do_text += " (GPU NVIDIA) — nhanh ~2-3 giây/câu"
    else:
        toc_do_text += f" (CPU, {ram_gb:.0f}GB RAM) — mỗi câu mất ~3-10 giây"
else:
    from gradio_client import Client, handle_file
    SPACE = "openbmb/VoxCPM-Demo"
    print(f"Dang ket noi den {SPACE}...")
    API = Client(SPACE)
    print("Da ket noi thanh cong!")
    toc_do_text = "Chạy qua server cloud GPU miễn phí — mỗi câu mất ~10-30 giây"

CAU_MAU = {
    "🇻🇳 Tiếng Việt": "Xin chào, đây là giọng nói được nhân bản bằng trí tuệ nhân tạo.",
    "🇬🇧 Tiếng Anh": "Hello, this is a voice cloned by artificial intelligence.",
    "🇪🇸 Tiếng Tây Ban Nha": "Hola, esta es una voz clonada por inteligencia artificial.",
    "🇰🇷 Tiếng Hàn": "안녕하세요, 이것은 인공지능으로 복제된 목소리입니다.",
    "🇯🇵 Tiếng Nhật": "こんにちは、これは人工知能によってクローンされた声です。",
    "🇵🇹 Tiếng Bồ Đào Nha": "Olá, esta é uma voz clonada por inteligência artificial.",
}


def tao_giong_local(audio_mau, transcript_mau, van_ban, cfg, progress):
    transcript_mau = (transcript_mau or "").strip()
    params = dict(
        text=van_ban.strip(),
        cfg_value=float(cfg),
        inference_timesteps=10,
    )
    if transcript_mau:
        params.update(
            prompt_wav_path=audio_mau, prompt_text=transcript_mau,
            reference_wav_path=audio_mau,
        )
        che_do = "Clone chất lượng cao (có lời thoại mẫu)"
    else:
        params.update(reference_wav_path=audio_mau)
        che_do = "Clone nhanh (chỉ dùng âm thanh mẫu)"

    progress(0.2, desc="Đang tạo giọng nói trên máy bạn...")
    wav = MODEL.generate(**params)
    sr = MODEL.tts_model.sample_rate
    do_dai = len(wav) / sr
    return (sr, np.asarray(wav, dtype=np.float32)), f"✅ Xong! {che_do} • Độ dài: {do_dai:.1f} giây"


def tao_giong_cloud(audio_mau, transcript_mau, van_ban, cfg, progress):
    transcript_mau = (transcript_mau or "").strip()
    use_prompt = bool(transcript_mau)

    progress(0.1, desc="Đang upload giọng mẫu lên server cloud...")
    progress(0.3, desc="Server đang xử lý (chờ 10-30 giây)...")

    result_path = API.predict(
        text_input=van_ban.strip(),
        control_instruction="",
        reference_wav_path_input=handle_file(audio_mau),
        use_prompt_text=use_prompt,
        prompt_text_input=transcript_mau,
        cfg_value_input=float(cfg),
        do_normalize=False,
        denoise=False,
        api_name="/generate",
    )

    wav, sr = sf.read(result_path)
    do_dai = len(wav) / sr
    che_do = "Clone chất lượng cao (có lời thoại mẫu)" if use_prompt else "Clone nhanh (chỉ dùng âm thanh mẫu)"
    return (sr, np.asarray(wav, dtype=np.float32)), f"✅ Xong! {che_do} • Độ dài: {do_dai:.1f} giây"


def tao_giong(audio_mau, transcript_mau, van_ban, cfg,
              progress=gr.Progress(track_tqdm=True)):
    if not audio_mau:
        raise gr.Error("Chưa có giọng mẫu! Hãy bấm thu âm hoặc tải file giọng của bạn lên (Bước 1).")
    if not van_ban or not van_ban.strip():
        raise gr.Error("Chưa nhập văn bản! Hãy gõ câu bạn muốn giọng đó đọc (Bước 2).")

    if MODE == "local":
        return tao_giong_local(audio_mau, transcript_mau, van_ban, cfg, progress)
    else:
        return tao_giong_cloud(audio_mau, transcript_mau, van_ban, cfg, progress)


with gr.Blocks(title="Clone Giọng Nói - VoxCPM2") as app:
    gr.Markdown(
        f"""
        # 🎙️ Nhân Bản Giọng Nói (VoxCPM2)

        **"Nhân bản giọng" nghĩa là:** Bạn đưa vào 1 đoạn ghi âm giọng ai đó (ví dụ giọng bạn),
        rồi máy sẽ tạo ra giọng nói **giống hệt người đó** nhưng đọc bất kỳ câu nào bạn muốn, bằng bất kỳ ngôn ngữ nào.

        ⚡ {toc_do_text}
        """
    )

    gr.Markdown("---")

    gr.Markdown(
        """
        ## BƯỚC 1: Đưa giọng mẫu (giọng bạn muốn nhân bản)

        **Bạn có 2 cách:**
        - 🎤 **Thu âm trực tiếp:** Bấm nút micro, đọc 1 đoạn bất kỳ 5-15 giây, rồi bấm dừng.
        - 📁 **Tải file lên:** Nếu đã có sẵn file ghi âm (mp3, wav...), bấm nút upload.

        ⚠️ **Lưu ý:** Giọng mẫu càng rõ, ít ồn, thì kết quả clone càng giống!
        """
    )
    audio_mau = gr.Audio(
        sources=["upload", "microphone"], type="filepath",
        label="Giọng mẫu (thu âm hoặc tải file lên)",
    )
    with gr.Accordion("Nâng cao: Gõ lời thoại trong đoạn mẫu (clone giống hơn, KHÔNG bắt buộc)", open=False):
        transcript_mau = gr.Textbox(
            label="Nội dung người trong đoạn ghi âm đã nói",
            placeholder="Ví dụ: Xin chào mọi người, hôm nay tôi sẽ giới thiệu về...",
            lines=2,
        )

    gr.Markdown("---")

    gr.Markdown(
        """
        ## BƯỚC 2: Nhập câu muốn giọng đó đọc

        Gõ bất kỳ câu nào — giọng nhân bản sẽ đọc câu này.
        Có thể gõ tiếng Việt, Anh, Hàn, Nhật, Tây Ban Nha, Bồ Đào Nha — máy tự nhận diện ngôn ngữ.
        """
    )
    van_ban = gr.Textbox(
        label="Văn bản cần đọc",
        placeholder="Ví dụ: Xin chào các bạn, đây là giọng nói nhân bản của tôi.",
        lines=3,
    )
    gr.Markdown("**Thử nhanh** — bấm nút để điền câu mẫu:")
    with gr.Row():
        for ten, cau in list(CAU_MAU.items())[:3]:
            gr.Button(ten, size="sm").click(lambda c=cau: c, outputs=van_ban)
    with gr.Row():
        for ten, cau in list(CAU_MAU.items())[3:]:
            gr.Button(ten, size="sm").click(lambda c=cau: c, outputs=van_ban)

    gr.Markdown("---")

    gr.Markdown("## BƯỚC 3: Bấm tạo giọng nói và chờ kết quả")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Tùy chỉnh (không bắt buộc)", open=False):
                cfg = gr.Slider(1.0, 3.0, value=2.0, step=0.1,
                                label="Độ bám giọng mẫu (cao = giống mẫu hơn)")
        with gr.Column(scale=2):
            nut_tao = gr.Button("🎤 TẠO GIỌNG NÓI", variant="primary", size="lg")
            trang_thai = gr.Markdown("*Sẵn sàng. Bấm nút ở trên để bắt đầu.*")
            ket_qua = gr.Audio(label="🔊 Kết quả — nghe thử và tải về", type="numpy")

    nut_tao.click(
        tao_giong,
        inputs=[audio_mau, transcript_mau, van_ban, cfg],
        outputs=[ket_qua, trang_thai],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", inbrowser=True)
