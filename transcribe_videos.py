import os
import sys
import re
import subprocess
import zlib
import datetime
import numpy as np
import cv2

from utils import get_user_data_dir, load_json_safe, save_json_atomic

SAMPLE_RATE = 16000

# 選べる音声認識モデル (faster-whisper のモデル名: 画面に出す説明)
WHISPER_MODELS = {
    "large-v3-turbo": "高精度 (推奨)",
    "large-v3": "最高精度・低速",
    "small": "高速・低精度",
}
DEFAULT_WHISPER_MODEL = "large-v3-turbo"

# 発話区間の前後にこの秒数の余裕を持たせて人物の出現を判定する
# (スキャンは0.5秒間隔なので、それより少し広めに取る)
PRESENCE_TOLERANCE_SEC = 0.75

# 単語間にこれ以上の無音があれば別の発話として扱う
UTTERANCE_GAP_SEC = 1.0
# 長い発話は複数人の声が混ざりやすいので、短い間(ま)でも区切る
LONG_UTTERANCE_SEC = 3.0
LONG_UTTERANCE_GAP_SEC = 0.4
SENTENCE_ENDINGS = ("。", "？", "！", "?", "!")

# Whisperが無音や雑音に対して出しがちな、実際には話されていない定型文
HALLUCINATION_PHRASES = (
    "視聴ありがとうございました", "ご覧いただきありがとうございます", "チャンネル登録",
    "最後までご視聴", "字幕", "thank you", "thanks for watching",
)

# 単語の確からしさ(平均)がこれ未満の発話は、雑音から作られた言葉とみなして捨てる
MIN_WORD_PROB = 0.15
# 自信がない区間でランダム性を加えて再挑戦した(温度>0)出力は作り話が多いので、より厳しく判定する
MIN_WORD_PROB_FALLBACK = 0.25
# 「お客様にお客様に…」のような同じ言葉の繰り返しは、圧縮率がこれを超える (Whisper自身の判定基準と同じ値)
MAX_COMPRESSION_RATIO = 2.4
MIN_TEXT_LEN_FOR_COMPRESSION = 10

# --- 口元の動きによる話者判定 ---
# 判定ロジックを変えたら上げる (古い判定結果を持つ動画は判定だけやり直す)
SPEAKER_ANALYSIS_VERSION = 2
MOUTH_SAMPLE_FPS = 8
MIN_MOUTH_SAMPLES = 4
MAX_MOUTH_SAMPLES = 40
MOUTH_CROP_SIZE = 300       # 顔周辺の切り出しをこの大きさ(px)に揃える
MIN_MOUTH_WIDTH_PX = 25     # これより口が小さいと開き具合を正しく測れない
MOUTH_MOVING_THRESHOLD = 0.03  # 口の開き具合(MAR)の標準偏差がこれ以上なら話している
# 食事中など話していなくても口が動く人がいるので、発話の合間(無音)の口の動きを
# その動画での「ふだんの口の動き」として測り、それを十分上回ったときだけ話していると判定する
BASELINE_WINDOW_SEC = 1.0
BASELINE_MAX_WINDOWS = 8
BASELINE_MIN_WINDOWS = 3
BASELINE_RATIO = 1.5
OFFSCREEN_LABEL = "画面外・撮影者"

# --- 会話タイムライン ---
TIMELINE_SPEAKER_OFFSCREEN = "撮影者"
MERGE_SAME_SPEAKER_GAP_SEC = 1.5   # 同じ話者の発話がこの間隔以内に続けば1つの発言にまとめる
DIALOGUE_PAIR_MAX_GAP_SEC = 5.0    # 話者が替わるまでの間隔がこれ以内なら「対話ペア」とみなす


def get_ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"


def load_audio(video_path):
    """ffmpegで動画の音声を16kHzモノラルのfloat32配列として取り出す"""
    cmd = [
        get_ffmpeg_exe(), "-nostdin", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-loglevel", "error", "-"
    ]
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    result = subprocess.run(cmd, capture_output=True, startupinfo=startupinfo)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0


def load_whisper_model(model_size=DEFAULT_WHISPER_MODEL):
    from faster_whisper import WhisperModel
    model_dir = os.path.join(get_user_data_dir(), "models", "whisper")
    os.makedirs(model_dir, exist_ok=True)
    print(f"音声認識モデルを準備中 ({model_size}) ... 初回はダウンロードに時間がかかります")
    if not os.environ.get("HF_TOKEN"):
        print("  (HF_TOKEN 未設定の警告が出ることがありますが、ダウンロードはそのまま続行されます)")
    sys.stdout.flush()
    # ctranslate2 は Apple Silicon の GPU に非対応のため CPU(int8) で実行する
    return WhisperModel(model_size, device="auto", compute_type="int8", download_root=model_dir)


def transcribe_video(model, video_path, language="ja", stop_event=None):
    """1本の動画を文字起こしし、セグメントのリストを返す。音声が無い場合は空リスト"""
    audio = load_audio(video_path)
    if audio is None or len(audio) == 0:
        print(f"  音声トラックがありません: {os.path.basename(video_path)}")
        return []

    # condition_on_previous_text=False: 同じ文の繰り返しや幻聴の連鎖を防ぐ
    segments, _ = model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                                   word_timestamps=True, condition_on_previous_text=False)

    results = []
    for seg in segments:
        if stop_event and stop_event.is_set():
            break
        min_prob = MIN_WORD_PROB_FALLBACK if seg.temperature > 0 else MIN_WORD_PROB
        results.extend(u for u in split_into_utterances(seg.words or [])
                       if u["prob"] >= min_prob and not is_repetitive(u["text"]))
    return results


def split_into_utterances(words):
    """Whisperのセグメントは複数人の発話をまとめがちなので、句点や無音の切れ目で細かく分ける"""
    utterances = []
    current = []

    def flush():
        text = "".join(w.word for w in current).strip()
        if text:
            utterances.append({
                "start": round(float(current[0].start), 2),
                "end": round(float(current[-1].end), 2),
                "text": text,
                "prob": round(float(np.mean([w.probability for w in current])), 3),
            })

    for w in words:
        if current:
            gap = w.start - current[-1].end
            is_long = current[-1].end - current[0].start >= LONG_UTTERANCE_SEC
            if gap >= UTTERANCE_GAP_SEC or (is_long and gap >= LONG_UTTERANCE_GAP_SEC):
                flush()
                current = []
        current.append(w)
        if w.word.strip().endswith(SENTENCE_ENDINGS):
            flush()
            current = []
    if current:
        flush()
    return utterances


def is_repetitive(text):
    data = text.encode("utf-8")
    if len(text) < MIN_TEXT_LEN_FOR_COMPRESSION:
        return False
    return len(data) / len(zlib.compress(data)) > MAX_COMPRESSION_RATIO


def is_hallucination(text):
    normalized = text.strip().strip("。.!！ ").lower()
    return any(p in normalized for p in HALLUCINATION_PHRASES)


def visible_people(scan_results, video_path, start, end):
    """発話区間中に画面に映っていた人物と、その区間付近の検出結果を返す"""
    lo = start - PRESENCE_TOLERANCE_SEC
    hi = end + PRESENCE_TOLERANCE_SEC
    hits = {}
    for name, videos in scan_results.get("people", {}).items():
        dets = [d for d in videos.get(video_path, []) if lo <= d.get("t", -1) <= hi and d.get("face_loc")]
        if dets:
            hits[name] = dets
    return hits


def mouth_aspect_ratio(landmarks):
    """唇の内側の開き / 口の幅 (dlib 68点の 61-67, 62-66, 63-65 と 48-54)"""
    top, bottom = landmarks["top_lip"], landmarks["bottom_lip"]
    pairs = [(top[10], bottom[8]), (top[9], bottom[9]), (top[8], bottom[10])]
    opening = np.mean([np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in pairs])
    width = np.hypot(top[0][0] - top[6][0], top[0][1] - top[6][1])
    if width < MIN_MOUTH_WIDTH_PX:
        return None
    return float(opening / width)


def measure_mouth(frame, face_loc):
    """スキャン時の顔位置の周辺を切り出し、口の開き具合を測る。判定できなければ None"""
    import face_recognition
    top, right, bottom, left = face_loc
    h, w = frame.shape[:2]
    pad = int(max(bottom - top, right - left) * 0.6)
    crop = frame[max(0, top - pad):min(h, bottom + pad), max(0, left - pad):min(w, right + pad)]
    if crop.size == 0:
        return None
    # 顔の大きさを揃えて、小さく写った顔でもランドマークを取れるようにする
    scale = MOUTH_CROP_SIZE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    if not locs:
        return None
    loc = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
    landmarks = face_recognition.face_landmarks(rgb, [loc])
    if not landmarks:
        return None
    return mouth_aspect_ratio(landmarks[0])


def nearest_detection(dets, t):
    best = min(dets, key=lambda d: abs(d["t"] - t))
    return best if abs(best["t"] - t) <= PRESENCE_TOLERANCE_SEC else None


def read_frames(cap, start, end):
    """発話区間のフレームを MOUTH_SAMPLE_FPS 程度の間隔で読み出す"""
    duration = max(end - start, 0.3)
    n = min(MAX_MOUTH_SAMPLES, max(MIN_MOUTH_SAMPLES, int(duration * MOUTH_SAMPLE_FPS)))
    times = np.linspace(start, start + duration, n)
    frames = []
    cap.set(cv2.CAP_PROP_POS_MSEC, times[0] * 1000)
    idx = 0
    while idx < len(times):
        if not cap.grab():
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos + 1e-3 >= times[idx]:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((pos, frame))
            while idx < len(times) and times[idx] <= pos + 1e-3:
                idx += 1
        if pos > times[-1] + 0.5:
            break
    return frames


def mouth_activity(frames, dets):
    """フレーム列から口の開き具合の標準偏差を求める。判定できなければ None"""
    ratios = []
    for t, frame in frames:
        det = nearest_detection(dets, t)
        if det is None:
            continue
        mar = measure_mouth(frame, det["face_loc"])
        if mar is not None:
            ratios.append(mar)
    if len(ratios) < MIN_MOUTH_SAMPLES:
        return None
    return float(np.std(ratios))


def silent_windows(segments, duration):
    """発話と発話の間の無音区間から、口の動きの基準を測る時間帯を選ぶ"""
    edges = [(0.0, 0.0)] + sorted((s["start"], s["end"]) for s in segments) + [(duration, duration)]
    windows = []
    for (_, prev_end), (next_start, _) in zip(edges, edges[1:]):
        gap = next_start - prev_end
        if gap >= BASELINE_WINDOW_SEC + 0.5:
            mid = (prev_end + next_start) / 2
            windows.append((mid - BASELINE_WINDOW_SEC / 2, mid + BASELINE_WINDOW_SEC / 2))
    if len(windows) > BASELINE_MAX_WINDOWS:
        idx = np.linspace(0, len(windows) - 1, BASELINE_MAX_WINDOWS).astype(int)
        windows = [windows[i] for i in idx]
    return windows


def measure_baselines(cap, scan_results, video_path, segments):
    """人物ごとに、話していない時間帯の口の動き(標準偏差の中央値)を測る"""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    samples = {}
    for start, end in silent_windows(segments, duration):
        visible = visible_people(scan_results, video_path, start, end)
        if not visible:
            continue
        frames = read_frames(cap, start, end)
        for name, dets in visible.items():
            a = mouth_activity(frames, dets)
            if a is not None:
                samples.setdefault(name, []).append(a)
    return {name: round(float(np.median(v)), 4) for name, v in samples.items() if len(v) >= BASELINE_MIN_WINDOWS}


def analyze_speakers(cap, scan_results, video_path, seg, baselines=None):
    """口元の動きから発話者を判定し、セグメントに結果を書き込む"""
    visible = visible_people(scan_results, video_path, seg["start"], seg["end"])
    seg["visible"] = sorted(visible.keys(), key=lambda n: -len(visible[n]))
    seg["mouth_activity"] = {}
    seg["speaker_v"] = SPEAKER_ANALYSIS_VERSION

    if not visible:
        seg["speakers"], seg["speaker_method"] = [], "offscreen"
        return

    frames = read_frames(cap, seg["start"], seg["end"])
    unjudged = []
    for name, dets in visible.items():
        a = mouth_activity(frames, dets)
        if a is not None:
            seg["mouth_activity"][name] = round(a, 4)
        else:
            unjudged.append(name)

    baselines = baselines or {}

    def threshold(name):
        return max(MOUTH_MOVING_THRESHOLD, baselines.get(name, 0) * BASELINE_RATIO)

    moving = [n for n, a in seg["mouth_activity"].items() if a >= threshold(n)]
    if moving:
        seg["speakers"] = sorted(moving, key=lambda n: -seg["mouth_activity"][n])
        seg["speaker_method"] = "mouth"
    elif unjudged:
        # 顔が小さい・横向きなどで口元を判定できない場合は、映っている人物の発話と推定する
        seg["speakers"], seg["speaker_method"] = unjudged, "estimated"
    else:
        # 映っている全員の口が動いていない = 撮影者など画面外の人の声
        seg["speakers"], seg["speaker_method"] = [], "offscreen"


def analyze_video_speakers(video_path, segments, scan_results, stop_event=None):
    """動画内の全発話の話者を判定する。成功すれば人物ごとのふだんの口の動きを、失敗すれば None を返す"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        baselines = measure_baselines(cap, scan_results, video_path, segments) if segments else {}
        for seg in segments:
            if stop_event and stop_event.is_set():
                return None
            analyze_speakers(cap, scan_results, video_path, seg, baselines)
    finally:
        cap.release()
    return baselines


def needs_speaker_analysis(info):
    return any(seg.get("speaker_v") != SPEAKER_ANALYSIS_VERSION for seg in info.get("segments", []))


def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "unknown"


def format_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def apply_speech_start_dates(speakers, date, speech_start_dates):
    """「話し始めた日付」より前の動画では、その人物を話者から外す (赤ちゃんの口の動きを発話と誤認しないため)"""
    if not speech_start_dates or not date:
        return speakers
    return [n for n in speakers if not speech_start_dates.get(n) or date[:10] >= speech_start_dates[n]]


def build_people_index(transcripts, scan_results, speech_start_dates=None):
    """動画ごとの文字起こしを話者ごとに振り分ける"""
    people = {name: [] for name in scan_results.get("people", {}).keys()}
    offscreen = []
    metadata = scan_results.get("metadata", {})

    for video_path, info in transcripts.items():
        date = metadata.get(video_path, {}).get("date", "")
        for seg in info.get("segments", []):
            if is_hallucination(seg["text"]) or "speakers" not in seg:
                continue
            speakers = apply_speech_start_dates(seg["speakers"], date, speech_start_dates)
            entry = {"video": video_path, "date": date, "start": seg["start"], "end": seg["end"],
                     "text": seg["text"], "method": seg["speaker_method"],
                     "visible": seg.get("visible", [])}
            if not speakers:
                offscreen.append(entry)
            for name in speakers:
                people.setdefault(name, []).append(dict(entry, with_people=[n for n in speakers if n != name]))

    for entries in list(people.values()) + [offscreen]:
        entries.sort(key=lambda e: (e["date"], e["video"], e["start"]))
    return people, offscreen


def build_conversations(transcripts, scan_results, speech_start_dates=None):
    """動画ごとに全話者の発話を時刻順に並べた会話と、話者が替わる受け答え(対話ペア)を作る"""
    metadata = scan_results.get("metadata", {})
    conversations = []
    pairs = []

    for video_path, info in transcripts.items():
        date = metadata.get(video_path, {}).get("date", "")
        turns = []
        for seg in sorted(info.get("segments", []), key=lambda x: x["start"]):
            if is_hallucination(seg["text"]) or "speakers" not in seg:
                continue
            speakers = apply_speech_start_dates(seg["speakers"], date, speech_start_dates)
            speaker = "/".join(speakers) if speakers else TIMELINE_SPEAKER_OFFSCREEN
            estimated = bool(speakers) and seg["speaker_method"] == "estimated"
            prev = turns[-1] if turns else None
            # 同じ話者の細切れの発話は1つの発言にまとめる
            if (prev and prev["speaker"] == speaker and prev["estimated"] == estimated
                    and seg["start"] - prev["end"] <= MERGE_SAME_SPEAKER_GAP_SEC):
                prev["end"] = seg["end"]
                prev["text"] += seg["text"]
                continue
            turns.append({"start": seg["start"], "end": seg["end"], "speaker": speaker,
                          "estimated": estimated, "text": seg["text"]})
        if not turns:
            continue
        conversations.append({"video": video_path, "date": date, "turns": turns})

        for a, b in zip(turns, turns[1:]):
            if a["speaker"] != b["speaker"] and b["start"] - a["end"] <= DIALOGUE_PAIR_MAX_GAP_SEC:
                pairs.append({"video": video_path, "date": date, "time": a["start"],
                              "speaker": a["speaker"], "text": a["text"],
                              "reply_speaker": b["speaker"], "reply_text": b["text"],
                              "estimated": a["estimated"] or b["estimated"]})

    conversations.sort(key=lambda c: (c["date"], c["video"]))
    pairs.sort(key=lambda p: (p["date"], p["video"], p["time"]))
    return conversations, pairs


def turn_label(turn):
    return f"{turn['speaker']}(推定)" if turn["estimated"] else turn["speaker"]


def write_conversation_files(conversations, pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    written = []
    if conversations:
        n_turns = sum(len(c["turns"]) for c in conversations)
        lines = [f"# 会話タイムライン ({len(conversations)} 本 / {n_turns} 発言)", ""]
        for c in conversations:
            lines.append("")
            lines.append(f"## {os.path.basename(c['video'])}  {c['date']}")
            for t in c["turns"]:
                lines.append(f"[{format_time(t['start'])}] {turn_label(t)}: {t['text']}")
        path = os.path.join(output_dir, "会話タイムライン.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written.append(path)
    if pairs:
        lines = [f"# 対話ペア ({len(pairs)} 組)", "", "話者が替わった直後の受け答えを、問いかけ → 返答の形で並べています。", ""]
        current_video = None
        for p in pairs:
            if p["video"] != current_video:
                current_video = p["video"]
                lines.append("")
                lines.append(f"## {os.path.basename(current_video)}  {p['date']}")
            mark = "  (推定を含む)" if p["estimated"] else ""
            lines.append(f"[{format_time(p['time'])}] {p['speaker']}「{p['text']}」 → {p['reply_speaker']}「{p['reply_text']}」{mark}")
        path = os.path.join(output_dir, "対話ペア.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written.append(path)
    return written


def write_text_file(path, title, entries, note_fn):
    lines = [f"# {title} ({len(entries)} 件)", ""]
    current_video = None
    for e in entries:
        if e["video"] != current_video:
            current_video = e["video"]
            lines.append("")
            lines.append(f"## {os.path.basename(current_video)}  {e['date']}")
        lines.append(f"[{format_time(e['start'])} - {format_time(e['end'])}] {e['text']}{note_fn(e)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def person_note(e):
    notes = []
    if e["method"] == "estimated":
        notes.append("推定")
    if e["with_people"]:
        notes.append(f"同時: {', '.join(e['with_people'])}")
    return f"  ({' / '.join(notes)})" if notes else ""


def offscreen_note(e):
    return f"  (画面: {', '.join(e['visible'])})" if e["visible"] else ""


def write_text_files(people, offscreen, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for name, entries in people.items():
        if not entries:
            continue
        path = os.path.join(output_dir, f"{safe_filename(name)}.txt")
        write_text_file(path, f"{name} の発話テキスト", entries, person_note)
        written.append(path)
    if offscreen:
        path = os.path.join(output_dir, f"{OFFSCREEN_LABEL}.txt")
        write_text_file(path, f"{OFFSCREEN_LABEL}の発話テキスト", offscreen, offscreen_note)
        written.append(path)
    return written


def update_outputs(data, scan_results, output_json, text_output_dir, speech_start_dates=None):
    people, offscreen = build_people_index(data["videos"], scan_results, speech_start_dates)
    conversations, pairs = build_conversations(data["videos"], scan_results, speech_start_dates)
    data["people"] = people
    data["offscreen"] = offscreen
    data["conversations"] = conversations
    data["dialogue_pairs"] = pairs
    data.pop("unassigned", None)
    save_json_atomic(output_json, data)
    written = write_text_files(people, offscreen, text_output_dir)
    written += write_conversation_files(conversations, pairs, text_output_dir)
    return people, offscreen, written


def run_transcription(scan_json=None, output_json=None, text_output_dir=None,
                      model_size=DEFAULT_WHISPER_MODEL, language="ja", force=False, stop_event=None,
                      speech_start_dates=None):
    """speech_start_dates: {人物名: "YYYY-MM-DD"} この日付より前の動画ではその人物を話者にしない"""
    user_data_dir = get_user_data_dir()
    if scan_json is None:
        scan_json = os.path.join(user_data_dir, "scan_results.json")
    if output_json is None:
        output_json = os.path.join(user_data_dir, "transcripts.json")
    if text_output_dir is None:
        text_output_dir = os.path.join(user_data_dir, "Output", "transcripts")

    scan_results = load_json_safe(scan_json, {"people": {}, "metadata": {}})
    data = load_json_safe(output_json, {"videos": {}})
    transcripts = data.setdefault("videos", {})

    # 登録人物が1回でも検出された動画だけを対象にする
    target_videos = sorted({
        v for videos in scan_results.get("people", {}).values()
        for v, dets in videos.items() if dets
    })
    if not target_videos:
        print("人物が検出されたスキャン済み動画がありません。先に動画スキャンを実行してください。")
        return None

    # 文字起こし済みでも話者判定がまだの動画は、判定だけを行う
    # 別のモデルで文字起こしした動画は、選んだモデルでやり直す
    def needs_transcription(v):
        return force or v not in transcripts or transcripts[v].get("model") != model_size

    to_process = [v for v in target_videos
                  if os.path.exists(v) and (needs_transcription(v) or needs_speaker_analysis(transcripts[v]))]
    n_transcribe = sum(1 for v in to_process if needs_transcription(v))
    print(f"処理対象: {len(to_process)} 本 (うち文字起こし {n_transcribe} 本 / 人物検出済み動画 {len(target_videos)} 本中)")

    model = None
    for i, video_path in enumerate(to_process, 1):
        if stop_event and stop_event.is_set():
            print("\nユーザーによる中断がリクエストされました。")
            break
        name = os.path.basename(video_path)
        try:
            if needs_transcription(video_path):
                if model is None:
                    model = load_whisper_model(model_size)
                print(f"  文字起こし中: {name}")
                sys.stdout.flush()
                segments = transcribe_video(model, video_path, language=language, stop_event=stop_event)
                info = {
                    "model": model_size,
                    "language": language,
                    "transcribed_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "segments": segments,
                }
            else:
                info = transcripts[video_path]
                segments = info["segments"]

            if segments:
                print(f"  話者判定中: {name} ({len(segments)} 発話)")
                sys.stdout.flush()
            baselines = analyze_video_speakers(video_path, segments, scan_results, stop_event=stop_event)
        except Exception as e:
            print(f"  エラー ({name}): {e}")
            continue
        if stop_event and stop_event.is_set():
            # 途中までの結果は保存しない (次回やり直す)
            print("\nユーザーによる中断がリクエストされました。")
            break
        if baselines is None:
            print(f"  警告: 動画を開けませんでした: {name}")
            continue
        info["mouth_baseline"] = baselines

        transcripts[video_path] = info
        # 1本ごとに人物別テキストを更新し、途中でも結果を確認できるようにする
        update_outputs(data, scan_results, output_json, text_output_dir, speech_start_dates)

        speakers = [s for seg in segments for s in seg.get("speakers", [])]
        n_off = sum(1 for seg in segments if seg.get("speaker_method") == "offscreen")
        pct = int(i / len(to_process) * 100)
        print(f"進捗: {pct}%| ({i}/{len(to_process)}本完了) - {name} [{len(segments)} 発話 / 画面外 {n_off}]"
              + (f" 話者: {', '.join(sorted(set(speakers)))}" if speakers else ""))
        sys.stdout.flush()

    people, offscreen, written = update_outputs(data, scan_results, output_json, text_output_dir, speech_start_dates)
    print(f"\n人物ごとのテキストを保存しました: {text_output_dir}")
    for name, entries in people.items():
        if entries:
            print(f"- {name}: {len(entries)} 件")
    if offscreen:
        print(f"- {OFFSCREEN_LABEL}: {len(offscreen)} 件")
    return written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="スキャン済み動画から人物ごとの発話テキストを抽出")
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL, help=f"Whisperモデル ({'/'.join(WHISPER_MODELS)})")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--force", action="store_true", help="文字起こし済みの動画もやり直す")
    args = parser.parse_args()

    run_transcription(model_size=args.model, language=args.language, force=args.force)
