"""
TRUNG_CLONE_VOICE_YOUTUBE_8686 (v4)
Batch clone giong 6 ngon ngu - hybrid local+API - loc van ban triet de
Chay: uv run python app_cloud.py
"""
import os, sys, tempfile, json, time, threading, webbrowser, shutil, re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
from gradio_client import Client, handle_file
from num2words import num2words
import numpy as np
import soundfile as sf
import lameenc
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(SCRIPT_DIR, "voices")
HISTORY_DIR = os.path.join(SCRIPT_DIR, "history")
DICT_FILE = os.path.join(SCRIPT_DIR, "dictionary.json")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
PORT = 7860

# --- Phan luong: may khoe (RAM>=28GB) = 5 API + 1 local; may yeu = 3 API ---
RAM_GB = psutil.virtual_memory().total / (1024**3)
LOCAL_ENV = os.environ.get("VOX_LOCAL", "")  # "1"=ep bat local, "mock"=gia lap de test
USE_LOCAL = RAM_GB >= 28 or LOCAL_ENV in ("1", "mock")
WORKERS_API = 5 if USE_LOCAL else 3
LOCAL = {"model": None, "ready": False, "mock": LOCAL_ENV == "mock"}
print(f"RAM {RAM_GB:.0f}GB -> {WORKERS_API} luong API"
      + (" + 1 luong local (dang nap model nen...)" if USE_LOCAL else " (API-only)"),
      flush=True)


def load_local_model():
    if LOCAL["mock"]:
        LOCAL["ready"] = True
        print("[local] MOCK model san sang (chi de test phan luong)", flush=True)
        return
    try:
        # "engine/src" = ban dong goi (repo GitHub, launcher tu tai OpenBMB/VoxCPM ve engine/)
        # "src"        = ban dev tai cho (thu muc nay von la git clone cua OpenBMB/VoxCPM)
        candidates = [os.path.join(SCRIPT_DIR, "engine", "src"),
                      os.path.join(SCRIPT_DIR, "src")]
        engine_src = next((p for p in candidates if os.path.isdir(p)), None)
        if not engine_src:
            raise RuntimeError(
                "Chua co code model VoxCPM2 (engine/src hoac src). Launcher se tu tai truoc khi vao day.")
        sys.path.insert(0, engine_src)
        from voxcpm import VoxCPM
        print("[local] Dang nap VoxCPM2 tren may (lan dau tai model ~vai GB, xin cho)...", flush=True)
        LOCAL["model"] = VoxCPM.from_pretrained("openbmb/VoxCPM2",
                                                load_denoiser=False, device="cpu")
        LOCAL["ready"] = True
        print("[local] Model local san sang -> tong 6 luong!", flush=True)
    except Exception as e:
        print(f"[local] Khong nap duoc model local ({e}) -> chay API-only", flush=True)


if USE_LOCAL:
    threading.Thread(target=load_local_model, daemon=True).start()

SPACE = "openbmb/VoxCPM-Demo"
print(f"Dang ket noi den {SPACE}...", flush=True)
API = Client(SPACE)
print("Da ket noi thanh cong!", flush=True)

IDX_LOCK = threading.Lock()

# ---------------- Tu dien phat am ----------------
DICT_LOCK = threading.Lock()
DICT = {}
if os.path.exists(DICT_FILE):
    try:
        DICT = json.load(open(DICT_FILE, encoding="utf-8"))
    except Exception:
        DICT = {}


def save_dict():
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(DICT, f, ensure_ascii=False, indent=2)


def _idx(d):
    return os.path.join(d, "index.json")


def load_idx(d):
    p = _idx(d)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def save_idx(d, data):
    with open(_idx(d), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pcm16_to_mp3(data, sr, mp3_path):
    enc = lameenc.Encoder()
    enc.set_bit_rate(192)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(data.tobytes()) + enc.flush()
    with open(mp3_path, "wb") as f:
        f.write(bytes(mp3))


def read_mono16(wav_path):
    data, sr = sf.read(wav_path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0].copy()
    return data, sr


# ============ PIPELINE LOC VAN BAN (6 ngon ngu) ============
# Nguyen tac: van ban den model CHI con chu + 4 dau . , ! ?
# Thu tu: tu dien -> the [nghi x] -> dau ban xu -> ky hieu thanh chu
#         -> so thanh chu -> ha cap dau -> xoa cam -> quet cuoi

# Dau ban xu (Nhat/Han/Trung/fullwidth) -> ASCII chuan
FULLWIDTH_MAP = {
    "。": ". ", "、": ", ", "，": ", ", "！": "! ", "？": "? ",
    "；": ". ", "：": ", ", "‥": ". ", "…": ". ",
    "「": "", "」": "", "『": "", "』": "", "【": "", "】": "",
    "（": ", ", "）": ", ", "・": " ", "～": "", "―": ", ",
}

# Ky hieu -> chu theo ngon ngu
SYMBOL_WORDS = {
    "VN": {"%": " phần trăm", "$": " đô la", "€": " ơ rô", "₫": " đồng",
           "¥": " yên", "£": " bảng", "&": " và ", "+": " cộng ",
           "=": " bằng ", "°": " độ ", "@": " a còng ", "№": " số "},
    "EN": {"%": " percent", "$": " dollars", "€": " euros", "₫": " dong",
           "¥": " yen", "£": " pounds", "&": " and ", "+": " plus ",
           "=": " equals ", "°": " degrees ", "@": " at ", "№": " number "},
    "ES": {"%": " por ciento", "$": " dólares", "€": " euros", "₫": " dong",
           "¥": " yenes", "£": " libras", "&": " y ", "+": " más ",
           "=": " igual a ", "°": " grados ", "@": " arroba ", "№": " número "},
    "KR": {"%": " 퍼센트", "$": " 달러", "€": " 유로", "₫": " 동",
           "¥": " 엔", "£": " 파운드", "&": " 그리고 ", "+": " 더하기 ",
           "=": " 는 ", "°": " 도 ", "@": " 골뱅이 ", "№": " 번호 "},
    "JP": {"%": " パーセント", "$": " ドル", "€": " ユーロ", "₫": " ドン",
           "¥": " 円", "£": " ポンド", "&": " と ", "+": " プラス ",
           "=": " は ", "°": " 度 ", "@": " アットマーク ", "№": " 番号 "},
    "PT": {"%": " por cento", "$": " dólares", "€": " euros", "₫": " dong",
           "¥": " ienes", "£": " libras", "&": " e ", "+": " mais ",
           "=": " igual a ", "°": " graus ", "@": " arroba ", "№": " número "},
}

# Dau doc nham thanh tieng -> ha cap ve dau chuan
DOWNGRADE_MAP = {
    "—": ", ", "–": ", ", ";": ". ", ":": ", ", "|": ", ",
    "/": " ", "\\": " ", "*": " ", "#": " ", "_": " ", "^": " ",
    "•": " ", "·": " ", "¿": " ", "¡": " ", "§": " ", "~": " ",
    "“": "", "”": "", "„": "", "‘": "", "’": "", '"': "", "'": "",
    "«": "", "»": "",
    "(": ", ", ")": ", ", "[": ", ", "]": ", ", "{": ", ", "}": ", ",
}

NUM_LANG = {"VN": "vi", "EN": "en", "ES": "es", "KR": "ko", "JP": "ja", "PT": "pt"}


def apply_dictionary(t, lang):
    with DICT_LOCK:
        entries = list(DICT.get(lang, []))
    for e in entries:
        tu = (e.get("tu") or "").strip()
        doc = (e.get("doc") or "").strip()
        if tu:
            t = re.sub(rf"(?i)(?<!\w){re.escape(tu)}(?!\w)", doc, t)
    return t


def num_to_words_all(t, lang):
    code = NUM_LANG.get(lang, "en")

    def conv(m):
        s = m.group(0)
        digits_only = re.sub(r"[.,]", "", s)
        try:
            # so dai (dien thoai, ma so) -> doc tung chu so
            if len(digits_only) > 12 or ("." not in s and "," not in s and len(s) > 6):
                return " " + " ".join(num2words(int(d), lang=code) for d in digits_only) + " "
            # nhom nghin: 1.000.000 / 1,000,000
            if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):
                return " " + num2words(int(digits_only), lang=code) + " "
            # thap phan: 3.5 / 3,5
            if re.fullmatch(r"\d+[.,]\d{1,2}", s):
                return " " + num2words(float(s.replace(",", ".")), lang=code) + " "
            return " " + num2words(int(s), lang=code) + " "
        except Exception:
            return " " + " ".join(num2words(int(d), lang=code)
                                  for d in digits_only if d.isdigit()) + " "

    return re.sub(r"\d+(?:[.,]\d+)*", conv, t)


def sanitize_text(t, lang="VN"):
    t = apply_dictionary(t, lang)
    for k, v in FULLWIDTH_MAP.items():
        t = t.replace(k, v)
    # tien te dung truoc so ($100) -> dao ra sau (100 $) roi moi thay chu
    t = re.sub(r"([$€£¥₫])\s*(\d[\d.,]*)", r"\2 \1", t)
    for k, v in SYMBOL_WORDS.get(lang, SYMBOL_WORDS["EN"]).items():
        t = t.replace(k, v)
    t = num_to_words_all(t, lang)
    for k, v in DOWNGRADE_MAP.items():
        t = t.replace(k, v)
    t = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]{2,}", " ", t)          # kkk/hhh tieng Han (cuoi)
    t = re.sub(r"\b[A-Z]{4,}\b", lambda m: m.group(0).lower(), t)  # CAPS gao
    removed = set(re.findall(r"[^\w\s.,!?]", t))
    t = re.sub(r"[^\w\s.,!?]", " ", t)               # quet cuoi: chi giu chu + . , ! ?
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.,!?])", r"\1", t)
    t = re.sub(r"([.,!?]){2,}", r"\1", t)
    t = re.sub(r"^[.,!? ]+", "", t).strip()
    if t and t[-1] not in ".!?":
        t += "."                                      # luon ket cau bang dau cham
    if removed:
        print(f"[loc] Da bo ky tu la: {' '.join(sorted(removed))}", flush=True)
    return t


# The nghi: [nghỉ 1.5] / [nghi 1.5] / [pause 1.5] -> chen lang dung giay (cap 3s)
PAUSE_RE = re.compile(r"\[\s*(?:ngh[iỉ]|pause)\s*([\d.,]+)?\s*\]", re.IGNORECASE)


def build_segments(text, lang):
    """Tra ve [(cau_da_loc, giay_nghi_sau_cau)] — xuong dong = nghi 0.5s"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", " [nghi 0.5] ", text)
    pieces, last = [], 0
    for m in PAUSE_RE.finditer(text):
        dur = float((m.group(1) or "0.5").replace(",", "."))
        pieces.append((text[last:m.start()], min(dur, 3.0)))
        last = m.end()
    pieces.append((text[last:], 0.0))

    segs = []
    for raw, pause in pieces:
        s = sanitize_text(raw, lang)
        if not s:
            if segs:  # the nghi dung canh nhau -> don nghi vao cau truoc
                segs[-1] = (segs[-1][0], min(segs[-1][1] + pause, 3.0))
            continue
        chunks = split_sentences(s)
        for i, c in enumerate(chunks):
            segs.append((c, 0.25 if i < len(chunks) - 1 else pause))
    return segs


def split_sentences(t, max_len=120):
    """Tach doan dai thanh tung cau ngan — model doc cau ngan chinh xac hon nhieu"""
    parts = re.split(r"(?<=[.!?;:])\s+", t)
    chunks, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_len:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(p) > max_len:
                cut = p.rfind(",", 0, max_len)
                if cut <= 0:
                    cut = max_len
                chunks.append(p[:cut + 1].strip(" ,"))
                p = p[cut + 1:].strip()
            cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def history_file(hid):
    for ext in ("mp3", "wav"):
        fp = os.path.join(HISTORY_DIR, f"{hid}.{ext}")
        if os.path.exists(fp):
            return fp, ext
    return None, None


def add_history(src_path, ext, text, voice_name, number=None):
    hid = f"{int(time.time()*1000)}_{number if number is not None else 0}"
    shutil.copy2(src_path, os.path.join(HISTORY_DIR, f"{hid}.{ext}"))
    with IDX_LOCK:
        hist = load_idx(HISTORY_DIR)
        hist.insert(0, {
            "id": hid, "text": text[:100], "voice": voice_name,
            "time": time.strftime("%H:%M %d/%m"),
        })
        if len(hist) > 50:
            for old in hist[50:]:
                fp, _ = history_file(old["id"])
                if fp:
                    os.unlink(fp)
            hist = hist[:50]
        save_idx(HISTORY_DIR, hist)
    return hid


# ---------------- Batch ----------------
BATCH = {
    "running": False, "stop": False,
    "items": [], "folder": "", "voice_name": "", "language": "VN",
    "voice_path": None, "transcript": "", "cfg": 2.0, "denoise": False,
    "control": "", "tmp_voice": None,
}


def num_name(n):
    return f"{n:02d}.mp3"


def scan_folder(folder):
    """Tra ve (so_file, so_lon_nhat) cua cac file dang NN.mp3"""
    if not os.path.isdir(folder):
        return 0, 0
    nums = []
    for f in os.listdir(folder):
        m = re.fullmatch(r"(\d+)\.mp3", f)
        if m:
            nums.append(int(m.group(1)))
    return len(nums), max(nums) if nums else 0


def gen_chunk(chunk, kind):
    """Tao am thanh 1 cau — kind: 'api' (cloud) hoac 'local' (model tren may)"""
    transcript = BATCH["transcript"]
    control = "" if transcript else BATCH["control"]

    if kind == "local":
        if LOCAL["mock"]:
            return np.zeros(8000, dtype=np.int16), 16000  # 0.5s lang de test
        m = LOCAL["model"]
        kw = dict(text=chunk, cfg_value=BATCH["cfg"], inference_timesteps=10)
        if transcript:
            kw.update(prompt_wav_path=BATCH["voice_path"], prompt_text=transcript,
                      reference_wav_path=BATCH["voice_path"])
        else:
            kw["reference_wav_path"] = BATCH["voice_path"]
            if control:
                kw["text"] = f"({control}){chunk}"
        wav = m.generate(**kw)
        sr = m.tts_model.sample_rate
        data = (np.clip(np.asarray(wav, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
        return data, sr

    result = API.predict(
        text_input=chunk,
        control_instruction=control,
        reference_wav_path_input=handle_file(BATCH["voice_path"]),
        use_prompt_text=bool(transcript),
        prompt_text_input=transcript,
        cfg_value_input=BATCH["cfg"],
        do_normalize=False,
        denoise=BATCH["denoise"],
        api_name="/generate",
    )
    return read_mono16(result)


def gen_one(it, kind="api"):
    if BATCH["stop"]:
        return
    it["status"] = "dang_chay"
    try:
        segs = build_segments(it["text"], BATCH["language"])
        if not segs:
            raise ValueError("Văn bản rỗng sau khi lọc ký tự")

        parts, sr = [], None
        for chunk, pause in segs:
            if BATCH["stop"]:
                it["status"] = "da_dung"
                return
            data, s = gen_chunk(chunk, kind)
            if sr is None:
                sr = s
            elif s != sr:
                raise ValueError(f"Sample rate lệch: {s} vs {sr}")
            parts.append(data)
            if pause > 0:
                parts.append(np.zeros(int(sr * min(pause, 3.0)), dtype=np.int16))

        if BATCH["stop"]:
            it["status"] = "da_dung"
            return
        full = np.concatenate(parts)

        mp3_path = os.path.join(BATCH["folder"], num_name(it["number"]))
        pcm16_to_mp3(full, sr, mp3_path)
        it["hid"] = add_history(mp3_path, "mp3", it["text"],
                                BATCH["voice_name"], it["number"])
        it["status"] = "xong"
    except Exception as e:
        it["status"] = "loi"
        it["err"] = str(e)[:200]
        it["retries"] = it.get("retries", 0) + 1


def run_batch():
    try:
        while not BATCH["stop"]:
            pend = [it for it in BATCH["items"] if it["status"] != "xong"]
            if not pend:
                break
            queue = list(pend)
            qlock = threading.Lock()

            def take():
                with qlock:
                    return queue.pop(0) if queue else None

            def worker(kind):
                while not BATCH["stop"]:
                    item = take()
                    if item is None:
                        return
                    gen_one(item, kind)

            threads = [threading.Thread(target=worker, args=("api",))
                       for _ in range(WORKERS_API)]
            if USE_LOCAL and LOCAL["ready"]:
                threads.append(threading.Thread(target=worker, args=("local",)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if BATCH["stop"]:
                break
            if any(it["status"] != "xong" for it in BATCH["items"]):
                time.sleep(3)  # nghi truoc vong chay lai file thieu
    finally:
        if BATCH["stop"]:
            for it in BATCH["items"]:
                if it["status"] not in ("xong",):
                    it["status"] = "da_dung"
        BATCH["running"] = False
        tmp = BATCH.get("tmp_voice")
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        BATCH["tmp_voice"] = None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRUNG_CLONE_VOICE_YOUTUBE_8686</title>
<style>
:root{--bg:#0f0f0f;--panel:#1a1a1a;--card:#242424;--accent:#c8ff00;--accent2:#b3e000;
--text:#e0e0e0;--muted:#777;--border:#333;--red:#ff4455;--green:#4ecca3;--blue:#5599ff}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
background:var(--bg);color:var(--text);min-height:100vh}
.app{max-width:1100px;margin:0 auto;padding:16px;display:flex;flex-direction:column;min-height:100vh}

.header{display:flex;justify-content:space-between;align-items:center;padding:10px 0;
border-bottom:1px solid var(--border);margin-bottom:14px}
.logo{font-size:1.05em;font-weight:700;letter-spacing:.5px}.logo em{color:var(--accent);font-style:normal}

.main{display:flex;gap:14px;flex:1;min-height:0}
.editor-panel{flex:3;display:flex;flex-direction:column;gap:8px}
.control-panel{flex:2;background:var(--panel);border-radius:12px;padding:14px;
display:flex;flex-direction:column;gap:13px;overflow-y:auto;max-height:calc(100vh - 170px)}

.edit-top{display:flex;justify-content:space-between;align-items:center}
.chip{display:none;align-items:center;gap:8px;background:var(--card);border:1px solid var(--accent);
border-radius:16px;padding:4px 12px;font-size:12px;width:fit-content}
.chip b{color:var(--accent)}
.chip button{background:none;border:none;color:var(--red);cursor:pointer;font-size:13px}

.editor-panel textarea{flex:1;min-height:170px;width:100%;padding:14px;background:var(--panel);
border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:15px;
resize:none;outline:none;line-height:1.6}
.editor-panel textarea:focus{border-color:var(--accent)}
.editor-panel textarea[readonly]{opacity:.75;border-style:dashed}
.editor-footer{display:flex;justify-content:space-between;align-items:center;padding:4px}
.quick-langs{display:flex;gap:5px;flex-wrap:wrap;align-items:center}
.qb{padding:4px 12px;background:var(--card);border:1px solid var(--border);border-radius:14px;
color:var(--text);cursor:pointer;font-size:12px;font-weight:600}
.qb:hover{border-color:var(--accent)}
.qb.active{background:var(--accent);color:#000;border-color:var(--accent)}

.slbl{display:block;color:var(--muted);font-size:12px;font-weight:600;
text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.csec{padding-bottom:12px;border-bottom:1px solid var(--border)}
.csec:last-child{border-bottom:none;padding-bottom:0}
.shdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}

.voice-row{display:flex;gap:6px;align-items:center}
.voice-row select{flex:1;padding:8px 10px;background:var(--card);border:1px solid var(--border);
border-radius:8px;color:var(--text);font-size:13px;outline:none;min-width:0}
.voice-row select:focus{border-color:var(--accent)}
.bi{width:34px;height:34px;background:var(--card);border:1px solid var(--border);
border-radius:8px;color:var(--accent);cursor:pointer;display:flex;align-items:center;
justify-content:center;font-size:14px;flex-shrink:0}
.bi:hover{background:var(--accent);color:#000}
.basm{padding:3px 10px;background:var(--accent);color:#000;border:none;border-radius:14px;
font-size:12px;font-weight:600;cursor:pointer}
.basm:hover{background:var(--accent2)}

.ordiv{text-align:center;color:var(--muted);font-size:11px;margin:6px 0;position:relative}
.ordiv::before,.ordiv::after{content:'';position:absolute;top:50%;width:28%;height:1px;background:var(--border)}
.ordiv::before{left:0}.ordiv::after{right:0}

.fi{width:100%;padding:8px;background:var(--card);border:1px dashed var(--border);
border-radius:8px;color:var(--text);cursor:pointer;font-size:12px}
.fi:hover{border-color:var(--accent)}

.srow{display:flex;align-items:center;gap:6px}
.srow input[type=range]{flex:1;accent-color:var(--accent)}
.mt{color:var(--muted);font-size:11px;white-space:nowrap}
.ac{color:var(--accent);font-weight:bold}

.tags{display:flex;flex-wrap:wrap;gap:6px}
.tag{padding:5px 12px;background:var(--card);border:1px solid var(--border);border-radius:16px;
color:var(--text);cursor:pointer;font-size:12px;transition:.15s}
.tag:hover{border-color:var(--accent)}
.tag.active{background:var(--accent);color:#000;border-color:var(--accent)}

.note{color:var(--muted);font-size:11px;padding:3px 0;line-height:1.4}

.tin{width:100%;padding:8px 10px;background:var(--card);border:1px solid var(--border);
border-radius:8px;color:var(--text);font-size:13px;outline:none}
.tin:focus{border-color:var(--accent)}

.bbar{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:14px 0;
margin-top:12px;border-top:1px solid var(--border)}
.status{color:var(--muted);font-size:13px;flex:1}
.bgen{padding:12px 32px;background:var(--accent);color:#000;border:none;border-radius:40px;
font-size:15px;font-weight:700;cursor:pointer;transition:.15s;flex-shrink:0}
.bgen:hover:not(:disabled){background:var(--accent2);transform:scale(1.02)}
.bgen:disabled{background:#444;color:#777;cursor:not-allowed;transform:none}
.bstop{padding:12px 28px;background:transparent;color:var(--red);border:2px solid var(--red);
border-radius:40px;font-size:15px;font-weight:700;cursor:pointer;flex-shrink:0}
.bstop:hover:not(:disabled){background:var(--red);color:#fff}
.bstop:disabled{border-color:#444;color:#555;cursor:not-allowed}

.pbox{margin-top:6px;display:none}
.pbox h3{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.prow{display:grid;grid-template-columns:56px 1fr 130px 60px;gap:8px;align-items:center;
padding:7px 10px;background:var(--panel);border-radius:8px;margin-bottom:4px;font-size:13px}
.pnum{color:var(--accent);font-weight:700}
.ptext{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#bbb}
.pst{font-size:12px}
.st-cho{color:var(--muted)}.st-chay{color:var(--blue)}.st-xong{color:var(--green)}.st-loi{color:var(--red)}
.pplay{background:none;border:1px solid var(--border);border-radius:12px;color:var(--accent);
cursor:pointer;font-size:11px;padding:2px 8px;visibility:hidden}
.pplay.on{visibility:visible}
.pplay:hover{background:var(--accent);color:#000}

.rsec audio{width:100%;margin-top:6px}

.bosm{padding:3px 10px;background:transparent;border:1px solid var(--border);border-radius:14px;
color:var(--text);cursor:pointer;font-size:12px;text-decoration:none;display:inline-block}
.bosm:hover{border-color:var(--accent);color:var(--accent)}

.ov{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100}
.hpan{position:fixed;top:0;right:0;width:360px;height:100vh;background:var(--panel);
z-index:101;display:flex;flex-direction:column;box-shadow:-4px 0 20px rgba(0,0,0,.5);
transform:translateX(100%);transition:transform .25s}
.hpan.open{transform:translateX(0)}
.hhdr{display:flex;justify-content:space-between;align-items:center;padding:14px;
border-bottom:1px solid var(--border)}
.hhdr h3{font-size:1em}
.bcl{width:30px;height:30px;background:var(--card);border:none;border-radius:50%;
color:var(--text);cursor:pointer;font-size:14px}
.bcl:hover{background:var(--red);color:#fff}
.hlist{flex:1;overflow-y:auto;padding:8px}
.hitem{padding:10px;background:var(--card);border-radius:8px;margin-bottom:6px;
border:1px solid transparent;transition:.15s}
.hitem:hover{border-color:var(--accent)}
.htext{font-size:13px;margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;
-webkit-box-orient:vertical;overflow:hidden}
.hmeta{font-size:11px;color:var(--muted);display:flex;justify-content:space-between}
.hacts{display:flex;gap:4px;margin-top:6px;flex-wrap:wrap}
.empty{text-align:center;color:var(--muted);padding:40px 16px;font-size:13px}

.modal{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:200;display:flex;
align-items:center;justify-content:center}
.mcont{background:var(--panel);border-radius:14px;width:92%;max-width:460px;max-height:92vh;overflow-y:auto}
.mhdr{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;
border-bottom:1px solid var(--border)}
.mhdr h3{font-size:1em}
.mbod{padding:16px 18px;display:flex;flex-direction:column;gap:8px}
.mbod label{color:var(--muted);font-size:12px;font-weight:600}
.mbod input[type=text]{padding:8px 10px;background:var(--card);border:1px solid var(--border);
border-radius:8px;color:var(--text);font-size:13px;outline:none}
.mbod input[type=text]:focus{border-color:var(--accent)}
.mbod textarea{width:100%;padding:8px;background:var(--card);border:1px solid var(--border);
border-radius:8px;color:var(--text);font-size:13px;resize:vertical;outline:none}
.mbod textarea:focus{border-color:var(--accent)}
.mbod input[type=range]{width:100%;accent-color:var(--accent)}
.langs{display:flex;gap:6px;flex-wrap:wrap}
.lg{padding:6px 14px;background:var(--card);border:1px solid var(--border);border-radius:14px;
color:var(--text);cursor:pointer;font-size:13px;font-weight:600}
.lg.active{background:var(--accent);color:#000;border-color:var(--accent)}
.chk{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text);cursor:pointer}
.chk input{accent-color:var(--accent);width:16px;height:16px}
.drow{display:grid;grid-template-columns:1fr 20px 1fr 30px;gap:6px;align-items:center;margin-bottom:6px}
.drow input{padding:7px 9px;background:var(--card);border:1px solid var(--border);
border-radius:8px;color:var(--text);font-size:13px;outline:none;min-width:0}
.drow input:focus{border-color:var(--accent)}
.drow .arr{color:var(--accent);text-align:center}
.drow .del{background:none;border:none;color:var(--red);cursor:pointer;font-size:15px}
.mftr{display:flex;justify-content:flex-end;gap:8px;padding:14px 18px;border-top:1px solid var(--border)}
.bout{padding:7px 18px;background:transparent;border:1px solid var(--border);border-radius:8px;
color:var(--text);cursor:pointer;font-size:13px}
.bout:hover{border-color:var(--accent);color:var(--accent)}
.bacc{padding:7px 18px;background:var(--accent);color:#000;border:none;border-radius:8px;
font-size:13px;font-weight:600;cursor:pointer}
.bacc:hover:not(:disabled){background:var(--accent2)}
.bacc:disabled{background:#444;color:#777;cursor:wait}

@media(max-width:768px){.main{flex-direction:column}.control-panel{max-height:none}.hpan{width:100%}
.prow{grid-template-columns:44px 1fr 100px 50px}}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="logo"><em>&#127908;</em> TRUNG_CLONE_VOICE_YOUTUBE_8686</div>
    <div style="display:flex;gap:8px;align-items:center">
      <span class="mt">Cloud GPU</span>
      <button class="bout" onclick="toggleHistory()">&#9203; Lịch sử</button>
    </div>
  </header>

  <div class="main">
    <div class="editor-panel">
      <div class="edit-top">
        <label class="slbl" style="margin:0">Nhập văn bản cần đọc</label>
        <button class="bout" onclick="document.getElementById('txt-file').click()">&#128196; Nạp file .txt</button>
        <input type="file" id="txt-file" accept=".txt" style="display:none">
      </div>
      <div class="chip" id="txt-chip">
        <span id="chip-label"></span>
        <button onclick="clearTxt()" title="Bỏ file">&#10005;</button>
      </div>
      <textarea id="text-input" placeholder="Gõ hoặc dán văn bản = 1 file âm thanh (kể cả nhiều dòng).&#10;Muốn tách nhiều file: dùng nút Nạp file .txt — mỗi dòng trong file = 1 file âm thanh riêng."></textarea>
      <div class="editor-footer">
        <div class="quick-langs">
          <span class="mt">Ngôn ngữ giọng:</span>
          <button class="qb" data-lang="VN" onclick="selectLang('VN')">VN</button>
          <button class="qb" data-lang="EN" onclick="selectLang('EN')">EN</button>
          <button class="qb" data-lang="ES" onclick="selectLang('ES')">ES</button>
          <button class="qb" data-lang="KR" onclick="selectLang('KR')">KR</button>
          <button class="qb" data-lang="JP" onclick="selectLang('JP')">JP</button>
          <button class="qb" data-lang="PT" onclick="selectLang('PT')">PT</button>
          <button class="qb" onclick="openDictModal()">&#128214; Từ điển</button>
        </div>
        <span class="mt" id="char-count">0 ký tự</span>
      </div>
      <div class="note">&#128161; Mẹo: gõ <b>[nghỉ 1.5]</b> để lặng đúng 1.5 giây •
        xuống dòng khi dán trực tiếp = nghỉ 0.5 giây •
        số và ký hiệu (%, $...) tự đọc thành chữ theo ngôn ngữ đang chọn</div>
    </div>

    <div class="control-panel">
      <div class="csec">
        <div class="shdr">
          <label class="slbl" style="margin:0">Giọng nhân bản</label>
          <button class="basm" onclick="openVoiceModal()">+ Thêm</button>
        </div>
        <div class="voice-row">
          <select id="voice-select"><option value="">-- Chọn giọng đã lưu --</option></select>
          <button class="bi" onclick="previewVoice()" title="Nghe thử">&#9654;</button>
          <button class="bi" onclick="deleteSelectedVoice()" title="Xóa giọng" style="color:var(--red)">&#128465;</button>
        </div>
        <div class="ordiv">hoặc upload giọng tạm</div>
        <input type="file" id="temp-audio" accept="audio/*" class="fi">
      </div>

      <div class="csec">
        <label class="slbl">Tốc độ: <span class="ac" id="speed-val">1.0x</span></label>
        <div class="srow">
          <span class="mt">Chậm hơn</span>
          <input type="range" id="speed" min="0.5" max="1.5" step="0.1" value="1.0">
          <span class="mt">Nhanh hơn</span>
        </div>
      </div>

      <div class="csec">
        <label class="slbl">Phong cách</label>
        <div class="tags">
          <button class="tag" data-ctrl="with clear pauses between sentences" onclick="toggleTag(this)">&#9199; Ngắt nghỉ</button>
          <button class="tag" data-ctrl="fluent continuous speech" onclick="toggleTag(this)">&#128279; Lưu loát</button>
          <button class="tag" data-ctrl="emotional and expressive voice" onclick="toggleTag(this)">&#10024; Truyền cảm</button>
          <button class="tag" data-ctrl="energetic and enthusiastic" onclick="toggleTag(this)">&#9889; Năng lượng</button>
        </div>
        <div class="note" id="style-note" style="display:none">
          &#9888; Tốc độ và phong cách không có tác dụng khi giọng có lời thoại mẫu (chế độ clone chuẩn nhất)
        </div>
      </div>

      <div class="csec">
        <label class="slbl">Thư mục tải về *</label>
        <input type="text" id="folder" class="tin"
               placeholder="VD: /Users/ban/Desktop/audio hoặc C:\Users\ban\audio">
        <div class="note">File tự lưu vào đây, đặt tên 01.mp3, 02.mp3... theo thứ tự câu</div>
      </div>

      <div class="csec rsec" id="result-sec" style="display:none">
        <label class="slbl">&#128266; Kết quả mới nhất</label>
        <audio id="result-audio" controls></audio>
      </div>
    </div>
  </div>

  <div class="bbar">
    <div class="status" id="status">Sẵn sàng. Chọn giọng, nhập văn bản và thư mục tải về.</div>
    <button class="bstop" id="stop-btn" onclick="stopBatch()" disabled>&#9209; KẾT THÚC</button>
    <button class="bgen" id="start-btn" onclick="startBatch()">&#9654; BẮT ĐẦU</button>
  </div>

  <div class="pbox" id="prog-box">
    <h3>Tiến trình</h3>
    <div id="prog-list"></div>
  </div>
</div>

<div class="ov" id="hist-ov" onclick="toggleHistory()" style="display:none"></div>
<div class="hpan" id="hist-pan">
  <div class="hhdr">
    <h3>Lịch sử</h3>
    <button class="bcl" onclick="toggleHistory()">&#10005;</button>
  </div>
  <div class="hlist" id="hist-list"><div class="empty">Chưa có lịch sử</div></div>
</div>

<div class="modal" id="voice-modal" style="display:none">
  <div class="mcont">
    <div class="mhdr">
      <h3>Thêm giọng nhân bản mới</h3>
      <button class="bcl" onclick="closeVM()">&#10005;</button>
    </div>
    <div class="mbod">
      <label>Tên giọng *</label>
      <input type="text" id="v-name" placeholder="VD: ana">
      <label>Giọng này là của nước nào? *</label>
      <div class="langs" id="v-langs">
        <button class="lg" data-lang="VN" onclick="pickVLang(this)">&#127483;&#127475; VN</button>
        <button class="lg" data-lang="EN" onclick="pickVLang(this)">&#127468;&#127463; EN</button>
        <button class="lg" data-lang="ES" onclick="pickVLang(this)">&#127466;&#127480; ES</button>
        <button class="lg" data-lang="KR" onclick="pickVLang(this)">&#127472;&#127479; KR</button>
        <button class="lg" data-lang="JP" onclick="pickVLang(this)">&#127471;&#127477; JP</button>
        <button class="lg" data-lang="PT" onclick="pickVLang(this)">&#127477;&#127481; PT</button>
      </div>
      <label>File giọng mẫu * (5-15 giây, rõ tiếng)</label>
      <input type="file" id="v-file" accept="audio/*" class="fi">
      <audio id="v-preview" controls style="display:none;width:100%;margin-top:4px"></audio>
      <label>Lời thoại trong đoạn mẫu (nên có — clone chuẩn nhất)</label>
      <textarea id="v-trans" rows="2" placeholder="Gõ chính xác nội dung đoạn mẫu, hoặc bấm nút bên dưới..."></textarea>
      <button class="bout" id="asr-btn" onclick="autoASR()">&#127908; Tự nhận diện lời thoại</button>
      <div class="note" id="asr-warn" style="display:none;color:#ffa500">
        &#9888; Máy nhận diện kém chính xác với tiếng này — nên nghe lại và sửa tay cho đúng</div>
      <label class="chk"><input type="checkbox" id="v-denoise"> Khử tạp âm (bật nếu giọng mẫu có tiếng ồn)</label>
      <label>Độ bám giọng: <span class="ac" id="v-cfg-val">2.0</span></label>
      <input type="range" id="v-cfg" min="1" max="3" step="0.1" value="2.0"
             oninput="document.getElementById('v-cfg-val').textContent=this.value">
      <div class="note">&#128161; Muốn clone chuẩn nhất: có lời thoại + để 2.0.
      Cao hơn = bám giọng mẫu chặt hơn nhưng giọng có thể cứng; thấp hơn = tự nhiên hơn nhưng ít giống hơn.</div>
    </div>
    <div class="mftr">
      <button class="bout" onclick="closeVM()">Hủy</button>
      <button class="bacc" id="save-v-btn" onclick="saveVoice()">Lưu giọng</button>
    </div>
  </div>
</div>

<div class="modal" id="dict-modal" style="display:none">
  <div class="mcont">
    <div class="mhdr">
      <h3>&#128214; Từ điển phát âm — <span id="dict-lang-label"></span></h3>
      <button class="bcl" onclick="closeDM()">&#10005;</button>
    </div>
    <div class="mbod">
      <div class="note">Sửa cách đọc từ khó: từ mượn, viết tắt, tên riêng.
        VD: <b>server &#8594; sơ vơ</b>, <b>AI &#8594; ây ai</b>, <b>TP.HCM &#8594; thành phố Hồ Chí Minh</b>.
        Áp dụng cho mọi lần tạo với ngôn ngữ này.</div>
      <div id="dict-rows"></div>
      <button class="bout" onclick="addDictRow('','')">+ Thêm dòng</button>
    </div>
    <div class="mftr">
      <button class="bout" onclick="closeDM()">Hủy</button>
      <button class="bacc" onclick="saveDict()">Lưu từ điển</button>
    </div>
  </div>
</div>

<script>
let voices=[], curLang=localStorage.getItem('lang')||'VN', txtSegments=null, pollTimer=null;

document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('folder').value=localStorage.getItem('folder')||'';
  selectLang(curLang);
  loadVoices(); loadHistory();
  document.getElementById('text-input').addEventListener('input',updCC);
  document.getElementById('speed').addEventListener('input',function(){
    document.getElementById('speed-val').textContent=parseFloat(this.value).toFixed(1)+'x';
  });
  document.getElementById('folder').addEventListener('input',function(){
    localStorage.setItem('folder',this.value.trim());
  });
  document.getElementById('v-file').addEventListener('change',function(){
    const p=document.getElementById('v-preview');
    if(this.files.length){p.src=URL.createObjectURL(this.files[0]);p.style.display='block'}
    else p.style.display='none';
  });
  document.getElementById('temp-audio').addEventListener('change',function(){
    if(this.files.length) document.getElementById('voice-select').value='';
    checkStyleNote();
  });
  document.getElementById('voice-select').addEventListener('change',function(){
    if(this.value) document.getElementById('temp-audio').value='';
    checkStyleNote();
  });
  document.getElementById('txt-file').addEventListener('change',async function(){
    if(!this.files.length) return;
    const f=this.files[0];
    const buf=await f.arrayBuffer();
    let txt;
    try{ txt=new TextDecoder('utf-8',{fatal:true}).decode(buf); }
    catch(e){
      const u8=new Uint8Array(buf);
      if(u8[0]===0xFF&&u8[1]===0xFE) txt=new TextDecoder('utf-16le').decode(buf);
      else if(u8[0]===0xFE&&u8[1]===0xFF) txt=new TextDecoder('utf-16be').decode(buf);
      else txt=new TextDecoder('windows-1258').decode(buf);
    }
    const lines=txt.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
    if(!lines.length){alert('File txt rỗng!');this.value='';return}
    txtSegments=lines;
    const ta=document.getElementById('text-input');
    ta.value=lines.join('\n');ta.readOnly=true;
    document.getElementById('chip-label').innerHTML=
      esc(f.name)+' — <b>'+lines.length+' câu = '+lines.length+' file</b>';
    document.getElementById('txt-chip').style.display='flex';
    updCC();
    this.value='';
  });
  poll(true); // neu server dang chay batch do (F5 giua chung) thi noi lai
});

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function pad(n){return String(n).padStart(2,'0')}

function clearTxt(){
  txtSegments=null;
  const ta=document.getElementById('text-input');
  ta.readOnly=false;ta.value='';
  document.getElementById('txt-chip').style.display='none';
  updCC();
}

function updCC(){
  const el=document.getElementById('char-count');
  if(txtSegments) el.textContent=txtSegments.length+' câu = '+txtSegments.length+' file';
  else el.textContent=document.getElementById('text-input').value.length+' ký tự (1 file)';
}

function selectLang(l){
  curLang=l;localStorage.setItem('lang',l);
  document.querySelectorAll('.quick-langs .qb').forEach(b=>
    b.classList.toggle('active',b.dataset.lang===l));
  renderVoiceOptions();
}

function renderVoiceOptions(){
  const s=document.getElementById('voice-select');
  const prev=s.value;
  s.innerHTML='<option value="">-- Chọn giọng đã lưu --</option>';
  voices.filter(v=>!v.language||v.language===curLang).forEach(v=>{
    const o=document.createElement('option');o.value=v.id;o.textContent=v.name;s.appendChild(o);
  });
  if([...s.options].some(o=>o.value===prev)) s.value=prev;
  checkStyleNote();
}

function checkStyleNote(){
  const vid=document.getElementById('voice-select').value;
  const v=voices.find(x=>x.id===vid);
  document.getElementById('style-note').style.display=(v&&v.transcript)?'block':'none';
}

async function loadVoices(){
  const r=await fetch('/api/voices');voices=await r.json();
  renderVoiceOptions();
}

let vLang='';
function pickVLang(el){
  vLang=el.dataset.lang;
  document.querySelectorAll('#v-langs .lg').forEach(b=>
    b.classList.toggle('active',b===el));
  document.getElementById('asr-warn').style.display=vLang&&vLang!=='EN'?'block':'none';
}

function openVoiceModal(){
  document.getElementById('voice-modal').style.display='flex';
  document.getElementById('v-name').value='';
  document.getElementById('v-file').value='';
  document.getElementById('v-trans').value='';
  document.getElementById('v-denoise').checked=false;
  document.getElementById('v-cfg').value=2.0;
  document.getElementById('v-cfg-val').textContent='2.0';
  document.getElementById('v-preview').style.display='none';
  vLang='';
  document.querySelectorAll('#v-langs .lg').forEach(b=>b.classList.remove('active'));
}
function closeVM(){document.getElementById('voice-modal').style.display='none'}

async function autoASR(){
  const fi=document.getElementById('v-file');
  if(!fi.files.length){alert('Chưa chọn file giọng mẫu!');return}
  const btn=document.getElementById('asr-btn');
  btn.disabled=true;btn.textContent='Đang nhận diện... (chờ chút)';
  const fd=new FormData();fd.append('audio',fi.files[0]);
  try{
    const r=await fetch('/api/asr',{method:'POST',body:fd});
    if(!r.ok) throw new Error(await r.text());
    const d=await r.json();
    document.getElementById('v-trans').value=d.text||'';
    if(!d.text) alert('Không nhận diện được — hãy gõ tay lời thoại.');
  }catch(e){alert('Lỗi nhận diện: '+e.message)}
  btn.disabled=false;btn.innerHTML='&#127908; Tự nhận diện lời thoại';
}

async function saveVoice(){
  const name=document.getElementById('v-name').value.trim();
  const fi=document.getElementById('v-file');
  if(!name){alert('Chưa đặt tên giọng!');return}
  if(!vLang){alert('Bắt buộc phải chọn giọng này là của nước nào!');return}
  if(!fi.files.length){alert('Chưa chọn file giọng mẫu!');return}
  const fd=new FormData();
  fd.append('name',name);fd.append('language',vLang);
  fd.append('audio',fi.files[0]);
  fd.append('transcript',document.getElementById('v-trans').value.trim());
  fd.append('denoise',document.getElementById('v-denoise').checked?'1':'');
  fd.append('cfg',document.getElementById('v-cfg').value);
  const btn=document.getElementById('save-v-btn');
  btn.disabled=true;
  try{
    const r=await fetch('/api/voices',{method:'POST',body:fd});
    if(!r.ok) throw new Error(await r.text());
    const d=await r.json();closeVM();
    selectLang(vLang);
    await loadVoices();
    document.getElementById('voice-select').value=d.id;
    checkStyleNote();
  }catch(e){alert('Lỗi: '+e.message)}
  btn.disabled=false;
}

async function deleteSelectedVoice(){
  const vid=document.getElementById('voice-select').value;
  if(!vid){alert('Chưa chọn giọng!');return}
  if(!confirm('Xóa giọng này?')) return;
  await fetch('/api/voices/'+encodeURIComponent(vid),{method:'DELETE'});
  await loadVoices();
}

function previewVoice(){
  const vid=document.getElementById('voice-select').value;
  if(!vid){alert('Chưa chọn giọng!');return}
  new Audio('/api/voices/'+encodeURIComponent(vid)+'/audio').play();
}

function toggleTag(el){el.classList.toggle('active')}

function buildCtrl(){
  const spd=parseFloat(document.getElementById('speed').value);
  const parts=[];
  if(spd<=0.6) parts.push('speak very slowly with deliberate pauses');
  else if(spd<=0.8) parts.push('speak slowly');
  else if(spd>=1.4) parts.push('speak very fast');
  else if(spd>=1.2) parts.push('speak fast');
  document.querySelectorAll('.tags .tag.active').forEach(t=>parts.push(t.dataset.ctrl));
  return parts.join(', ');
}

function setRunning(on){
  document.getElementById('start-btn').disabled=on;
  document.getElementById('stop-btn').disabled=!on;
}

async function startBatch(){
  const st=document.getElementById('status');
  const raw=document.getElementById('text-input').value.trim();
  const segs=txtSegments||(raw?[raw]:[]);
  const vid=document.getElementById('voice-select').value;
  const tmp=document.getElementById('temp-audio');
  const folder=document.getElementById('folder').value.trim();

  if(!segs.length){st.innerHTML='<span style="color:var(--red)">Chưa nhập văn bản!</span>';return}
  if(!vid&&!tmp.files.length){st.innerHTML='<span style="color:var(--red)">Chưa chọn giọng hoặc upload file!</span>';return}
  if(!folder){st.innerHTML='<span style="color:var(--red)">&#10071; Bắt buộc phải nhập thư mục tải về!</span>';return}

  let start=1, delOld=false;
  try{
    const r=await fetch('/api/batch/check',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({folder})});
    if(!r.ok) throw new Error(await r.text());
    const d=await r.json();
    if(d.count>0){
      delOld=confirm('Thư mục đã có '+d.count+' file âm thanh (đến số '+pad(d.max)+').\n\n'+
        'Bạn có muốn xoá các file đã làm không?\n\n'+
        'OK = CÓ, xoá hết, chạy lại từ 01.mp3\n'+
        'Cancel = KHÔNG xoá, chạy nối tiếp');
      if(!delOld){
        const ok=confirm('Lần này bắt đầu từ '+pad(d.max+1)+'.mp3\n\n'+
          'OK = Chạy luôn\nCancel = Dừng lại');
        if(!ok){st.textContent='Đã hủy. Chưa chạy gì.';return}
        start=d.max+1;
      }
    }
  }catch(e){st.innerHTML='<span style="color:var(--red)">Lỗi thư mục: '+esc(e.message)+'</span>';return}

  const fd=new FormData();
  fd.append('texts',JSON.stringify(segs));
  fd.append('folder',folder);
  fd.append('start',start);
  fd.append('delete_old',delOld?'1':'');
  fd.append('control',buildCtrl());
  fd.append('language',curLang);
  if(tmp.files.length) fd.append('audio',tmp.files[0]);
  else fd.append('voice_id',vid);

  setRunning(true);
  st.textContent='Đang khởi động '+segs.length+' câu...';
  try{
    const r=await fetch('/api/batch/start',{method:'POST',body:fd});
    if(!r.ok) throw new Error(await r.text());
    poll();
  }catch(e){
    st.innerHTML='<span style="color:var(--red)">Lỗi: '+esc(e.message)+'</span>';
    setRunning(false);
  }
}

async function stopBatch(){
  await fetch('/api/batch/stop',{method:'POST'});
  document.getElementById('status').textContent='Đang dừng...';
}

function poll(initial){
  if(pollTimer) clearTimeout(pollTimer);
  fetch('/api/batch/status').then(r=>r.json()).then(d=>{
    if(!d.items.length){if(!initial) setRunning(false);return}
    renderProg(d);
    if(d.running){
      setRunning(true);
      pollTimer=setTimeout(poll,1500);
    }else{
      setRunning(false);
      const st=document.getElementById('status');
      if(d.done===d.total)
        st.innerHTML='<span style="color:var(--green)">&#9989; Hoàn tất! Đã lưu '+d.total+' file MP3 vào thư mục.</span>';
      else
        st.innerHTML='Đã dừng. Xong '+d.done+'/'+d.total+' file. Ấn BẮT ĐẦU để chạy tiếp các file thiếu (chọn KHÔNG xoá).';
      loadHistory();
    }
  }).catch(()=>{pollTimer=setTimeout(poll,3000)});
}

function renderProg(d){
  document.getElementById('prog-box').style.display='block';
  const st=document.getElementById('status');
  if(d.running){
    const missing=d.items.filter(i=>i.status==='loi').map(i=>pad(i.number));
    let msg='&#128260; Xong '+d.done+'/'+d.total+' file';
    if(missing.length) msg+=' • Sẽ chạy lại file lỗi: '+missing.join(', ');
    st.innerHTML=msg;
  }
  const stMap={cho:['⏳ Chờ','st-cho'],dang_chay:['🔄 Đang chạy','st-chay'],
               xong:['✅ Xong','st-xong'],loi:['❌ Lỗi—sẽ thử lại','st-loi'],
               da_dung:['⏹ Đã dừng','st-cho']};
  document.getElementById('prog-list').innerHTML=d.items.map(i=>{
    const [txt,cls]=stMap[i.status]||['?',''];
    return '<div class="prow"><span class="pnum">'+pad(i.number)+'</span>'+
      '<span class="ptext">'+esc(i.text)+'</span>'+
      '<span class="pst '+cls+'">'+txt+'</span>'+
      '<button class="pplay'+(i.hid?' on':'')+'" onclick="playHid(\''+(i.hid||'')+'\')">&#9654;</button></div>';
  }).join('');
  const done=d.items.filter(i=>i.hid);
  if(done.length){
    const last=done[done.length-1];
    const a=document.getElementById('result-audio');
    const src='/api/history/'+last.hid+'/audio';
    if(!a.src.endsWith(src)){a.src=src}
    document.getElementById('result-sec').style.display='block';
  }
}

function playHid(hid){if(hid) new Audio('/api/history/'+hid+'/audio').play()}

async function loadHistory(){
  const r=await fetch('/api/history');const h=await r.json();
  const el=document.getElementById('hist-list');
  if(!h.length){el.innerHTML='<div class="empty">Chưa có lịch sử</div>';return}
  el.innerHTML=h.map(i=>`<div class="hitem">
    <div class="htext">${esc(i.text)}</div>
    <div class="hmeta"><span>${esc(i.voice)}</span><span>${i.time}</span></div>
    <div class="hacts">
      <button class="bosm" onclick="playHid('${i.id}')">&#9654; Nghe</button>
      <a class="bosm" href="/api/history/${i.id}/download">&#128190; Tải về</a>
      <button class="bosm" onclick="delHist('${i.id}')" style="color:var(--red)">Xóa</button>
    </div></div>`).join('');
}

async function delHist(id){
  await fetch('/api/history/'+id,{method:'DELETE'});loadHistory();
}

function toggleHistory(){
  const p=document.getElementById('hist-pan');
  const o=document.getElementById('hist-ov');
  const open=p.classList.contains('open');
  if(open){p.classList.remove('open');o.style.display='none'}
  else{p.classList.add('open');o.style.display='block';loadHistory()}
}

// ---- Tu dien phat am ----
async function openDictModal(){
  document.getElementById('dict-lang-label').textContent=curLang;
  document.getElementById('dict-rows').innerHTML='';
  try{
    const r=await fetch('/api/dictionary');const d=await r.json();
    (d[curLang]||[]).forEach(e=>addDictRow(e.tu,e.doc));
  }catch(e){}
  if(!document.querySelectorAll('#dict-rows .drow').length) addDictRow('','');
  document.getElementById('dict-modal').style.display='flex';
}
function closeDM(){document.getElementById('dict-modal').style.display='none'}

function addDictRow(tu,doc){
  const row=document.createElement('div');row.className='drow';
  row.innerHTML='<input placeholder="Từ gốc (VD: server)" class="d-tu">'+
    '<span class="arr">&#8594;</span>'+
    '<input placeholder="Cách đọc (VD: sơ vơ)" class="d-doc">'+
    '<button class="del" onclick="this.parentNode.remove()">&#128465;</button>';
  row.querySelector('.d-tu').value=tu||'';
  row.querySelector('.d-doc').value=doc||'';
  document.getElementById('dict-rows').appendChild(row);
}

async function saveDict(){
  const entries=[...document.querySelectorAll('#dict-rows .drow')].map(r=>({
    tu:r.querySelector('.d-tu').value.trim(),
    doc:r.querySelector('.d-doc').value.trim(),
  })).filter(e=>e.tu);
  try{
    const r=await fetch('/api/dictionary',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lang:curLang,entries})});
    if(!r.ok) throw new Error(await r.text());
    closeDM();
  }catch(e){alert('Lỗi lưu từ điển: '+e.message)}
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ct, download_name=None):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        if download_name:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(data)

    def _send_err(self, msg, status=500):
        body = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_form(self):
        import cgi
        ct = self.headers.get("Content-Type", "")
        return cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ct},
        )

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ---------------- GET ----------------
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/voices":
            self._send_json(load_idx(VOICES_DIR))

        elif path.startswith("/api/voices/") and path.endswith("/audio"):
            vid = unquote(path[len("/api/voices/"):-len("/audio")])
            fp = os.path.join(VOICES_DIR, f"{vid}.wav")
            if os.path.exists(fp):
                self._send_file(fp, "audio/wav")
            else:
                self._send_err("Not found", 404)

        elif path == "/api/history":
            self._send_json(load_idx(HISTORY_DIR))

        elif path.startswith("/api/history/") and path.endswith("/audio"):
            hid = path[len("/api/history/"):-len("/audio")]
            fp, ext = history_file(hid)
            if fp:
                self._send_file(fp, "audio/mpeg" if ext == "mp3" else "audio/wav")
            else:
                self._send_err("Not found", 404)

        elif path.startswith("/api/history/") and path.endswith("/download"):
            hid = path[len("/api/history/"):-len("/download")]
            fp, ext = history_file(hid)
            if fp:
                self._send_file(fp, "application/octet-stream",
                                download_name=f"giong_{hid}.{ext}")
            else:
                self._send_err("Not found", 404)

        elif path == "/api/batch/status":
            self._send_json({
                "running": BATCH["running"],
                "total": len(BATCH["items"]),
                "done": sum(1 for it in BATCH["items"] if it["status"] == "xong"),
                "items": [{"number": it["number"], "text": it["text"][:60],
                           "status": it["status"], "hid": it.get("hid")}
                          for it in BATCH["items"]],
            })

        elif path == "/api/dictionary":
            with DICT_LOCK:
                self._send_json(DICT)

        else:
            self.send_error(404)

    # ---------------- POST ----------------
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/voices":
            self._do_save_voice()
        elif path == "/api/asr":
            self._do_asr()
        elif path == "/api/batch/check":
            self._do_batch_check()
        elif path == "/api/batch/start":
            self._do_batch_start()
        elif path == "/api/batch/stop":
            BATCH["stop"] = True
            self._send_json({"ok": True})
        elif path == "/api/dictionary":
            try:
                data = self._read_json()
                lang = data.get("lang", "")
                entries = data.get("entries", [])
                if lang not in ("VN", "EN", "ES", "KR", "JP", "PT"):
                    self._send_err("Ngôn ngữ không hợp lệ", 400)
                    return
                with DICT_LOCK:
                    DICT[lang] = [e for e in entries
                                  if (e.get("tu") or "").strip()]
                    save_dict()
                self._send_json({"ok": True})
            except Exception as e:
                self._send_err(str(e), 400)
        else:
            self.send_error(404)

    # ---------------- DELETE ----------------
    def do_DELETE(self):
        path = urlparse(self.path).path

        if path.startswith("/api/voices/"):
            vid = unquote(path[len("/api/voices/"):])
            with IDX_LOCK:
                voices = [v for v in load_idx(VOICES_DIR) if v["id"] != vid]
                save_idx(VOICES_DIR, voices)
            fp = os.path.join(VOICES_DIR, f"{vid}.wav")
            if os.path.exists(fp):
                os.unlink(fp)
            self._send_json({"ok": True})

        elif path.startswith("/api/history/"):
            hid = path[len("/api/history/"):]
            with IDX_LOCK:
                hist = [h for h in load_idx(HISTORY_DIR) if h["id"] != hid]
                save_idx(HISTORY_DIR, hist)
            fp, _ = history_file(hid)
            if fp:
                os.unlink(fp)
            self._send_json({"ok": True})

        else:
            self.send_error(404)

    # ---------------- handlers ----------------
    def _do_save_voice(self):
        form = self._parse_form()
        name = form.getvalue("name", "").strip()
        language = form.getvalue("language", "").strip()
        transcript = form.getvalue("transcript", "").strip()
        denoise = bool(form.getvalue("denoise", ""))
        try:
            cfg = float(form.getvalue("cfg", "2.0"))
        except ValueError:
            cfg = 2.0

        if not name:
            self._send_err("Chưa đặt tên giọng", 400)
            return
        if language not in ("VN", "EN", "ES", "KR", "JP", "PT"):
            self._send_err("Bắt buộc phải chọn giọng của nước nào", 400)
            return
        if "audio" not in form:
            self._send_err("Chưa chọn file giọng", 400)
            return

        vid = re.sub(r"[^\w\-]", "_", name) + "_" + language
        fp = os.path.join(VOICES_DIR, f"{vid}.wav")
        with open(fp, "wb") as f:
            f.write(form["audio"].file.read())

        with IDX_LOCK:
            voices = [v for v in load_idx(VOICES_DIR) if v["id"] != vid]
            voices.append({
                "id": vid, "name": name, "language": language,
                "transcript": transcript, "denoise": denoise, "cfg": cfg,
                "created": time.strftime("%Y-%m-%d %H:%M"),
            })
            save_idx(VOICES_DIR, voices)
        self._send_json({"ok": True, "id": vid})

    def _do_asr(self):
        form = self._parse_form()
        if "audio" not in form:
            self._send_err("Thiếu file audio", 400)
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(form["audio"].file.read())
        tmp.close()
        try:
            text = API.predict(
                checked=True,
                audio_path=handle_file(tmp.name),
                api_name="/_run_asr_if_needed",
            )
            if isinstance(text, dict):  # gradio tra ve dang update {'value': ...}
                text = text.get("value", "")
            self._send_json({"text": (text or "").strip()})
        except Exception as e:
            self._send_err(str(e))
        finally:
            os.unlink(tmp.name)

    def _do_batch_check(self):
        try:
            data = self._read_json()
            folder = os.path.expanduser(data.get("folder", "").strip())
            if not folder:
                self._send_err("Thiếu thư mục", 400)
                return
            os.makedirs(folder, exist_ok=True)
            count, mx = scan_folder(folder)
            self._send_json({"count": count, "max": mx})
        except Exception as e:
            self._send_err(f"Không tạo được thư mục: {e}", 400)

    def _do_batch_start(self):
        if BATCH["running"]:
            self._send_err("Đang chạy — ấn KẾT THÚC trước", 409)
            return

        form = self._parse_form()
        try:
            texts = json.loads(form.getvalue("texts", "[]"))
        except json.JSONDecodeError:
            texts = []
        folder = os.path.expanduser(form.getvalue("folder", "").strip())
        start = int(form.getvalue("start", "1"))
        delete_old = bool(form.getvalue("delete_old", ""))
        control = form.getvalue("control", "").strip()
        voice_id = form.getvalue("voice_id", "").strip()
        language = form.getvalue("language", "VN").strip() or "VN"

        if not texts:
            self._send_err("Chưa có văn bản", 400)
            return
        if not folder:
            self._send_err("Bắt buộc phải có thư mục tải về", 400)
            return
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            self._send_err(f"Không tạo được thư mục: {e}", 400)
            return

        # chuan bi giong
        tmp_voice = None
        if voice_id:
            voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
            if not os.path.exists(voice_path):
                self._send_err("Giọng không tìm thấy", 404)
                return
            info = next((v for v in load_idx(VOICES_DIR) if v["id"] == voice_id), {})
            voice_name = info.get("name", voice_id)
            transcript = info.get("transcript", "")
            cfg = float(info.get("cfg", 2.0))
            denoise = bool(info.get("denoise", False))
        elif "audio" in form:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(form["audio"].file.read())
            tmp.close()
            voice_path = tmp_voice = tmp.name
            voice_name = "Upload trực tiếp"
            transcript, cfg, denoise = "", 2.0, False
        else:
            self._send_err("Chưa có giọng mẫu", 400)
            return

        if delete_old:
            for f in os.listdir(folder):
                if re.fullmatch(r"\d+\.mp3", f):
                    os.unlink(os.path.join(folder, f))
            start = 1

        BATCH.update({
            "running": True, "stop": False, "folder": folder,
            "voice_path": voice_path, "voice_name": voice_name,
            "transcript": transcript, "cfg": cfg, "denoise": denoise,
            "control": control, "tmp_voice": tmp_voice, "language": language,
            "items": [{"number": start + i, "text": t, "status": "cho"}
                      for i, t in enumerate(texts)],
        })
        threading.Thread(target=run_batch, daemon=True).start()
        self._send_json({"ok": True, "total": len(texts)})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = None
    for p in range(PORT, PORT + 6):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", p), Handler)
            PORT = p
            break
        except OSError:
            print(f"Port {p} dang ban, thu port tiep theo...", flush=True)
    if server is None:
        sys.exit("Khong tim duoc port trong 7860-7865. Tat cac app cu roi chay lai.")
    print(f"\n  Mo trinh duyet tai: http://localhost:{PORT}\n", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    server.serve_forever()
