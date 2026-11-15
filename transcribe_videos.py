import os
import sys
import re
import subprocess
import datetime
import numpy as np

from utils import get_user_data_dir, load_json_safe, save_json_atomic

SAMPLE_RATE = 16000

# 発話区間の前後にこの秒数の余裕を持たせて人物の出現を判定する
# (スキャンは0.5秒間隔なので、それより少し広めに取る)
PRESENCE_TOLERANCE_SEC = 0.75

# 単語間にこれ以上の無音があれば別の発話として扱う
UTTERANCE_GAP_SEC = 1.0
SENTENCE_ENDINGS = ("。", "？", "！", "?", "!")


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


def load_whisper_model(model_size="small"):
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

    segments, _ = model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                                   word_timestamps=True)

    results = []
    for seg in segments:
        if stop_event and stop_event.is_set():
            break
        results.extend(split_into_utterances(seg.words or []))
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
            })

    for w in words:
        if current and w.start - current[-1].end >= UTTERANCE_GAP_SEC:
            flush()
            current = []
        current.append(w)
        if w.word.strip().endswith(SENTENCE_ENDINGS):
            flush()
            current = []
    if current:
        flush()
    return utterances


def people_on_screen(scan_results, video_path, start, end):
    """発話区間中に画面に映っていた人物を、映っていた割合の高い順に返す"""
    lo = start - PRESENCE_TOLERANCE_SEC
    hi = end + PRESENCE_TOLERANCE_SEC
    hits = []
    for name, videos in scan_results.get("people", {}).items():
        dets = videos.get(video_path, [])
        count = sum(1 for d in dets if lo <= d.get("t", -1) <= hi)
        if count > 0:
            hits.append((name, count))
    hits.sort(key=lambda x: -x[1])
    return [name for name, _ in hits]


def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "unknown"


def format_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_people_index(transcripts, scan_results):
    """動画ごとの文字起こしを人物ごとに振り分ける (スキャン結果が変わっても毎回再計算する)"""
    people = {name: [] for name in scan_results.get("people", {}).keys()}
    unassigned = []
    metadata = scan_results.get("metadata", {})

    for video_path, info in transcripts.items():
        date = metadata.get(video_path, {}).get("date", "")
        for seg in info.get("segments", []):
            names = people_on_screen(scan_results, video_path, seg["start"], seg["end"])
            seg["people"] = names
            entry = {"video": video_path, "date": date, "start": seg["start"], "end": seg["end"], "text": seg["text"]}
            if not names:
                unassigned.append(entry)
            for name in names:
                people.setdefault(name, []).append(dict(entry, with_people=[n for n in names if n != name]))

    for entries in people.values():
        entries.sort(key=lambda e: (e["date"], e["video"], e["start"]))
    return people, unassigned


def write_text_files(people, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for name, entries in people.items():
        if not entries:
            continue
        path = os.path.join(output_dir, f"{safe_filename(name)}.txt")
        lines = [f"# {name} の発話テキスト ({len(entries)} 件)", ""]
        current_video = None
        for e in entries:
            if e["video"] != current_video:
                current_video = e["video"]
                lines.append("")
                lines.append(f"## {os.path.basename(current_video)}  {e['date']}")
            others = f"  (同席: {', '.join(e['with_people'])})" if e["with_people"] else ""
            lines.append(f"[{format_time(e['start'])} - {format_time(e['end'])}] {e['text']}{others}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written.append(path)
    return written


def run_transcription(scan_json=None, output_json=None, text_output_dir=None,
                      model_size="small", language="ja", force=False, stop_event=None):
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

    to_process = [v for v in target_videos
                  if os.path.exists(v) and (force or v not in transcripts)]
    print(f"文字起こし対象: {len(to_process)} 本 (人物検出済み動画 {len(target_videos)} 本中)")

    if to_process and not (stop_event and stop_event.is_set()):
        model = load_whisper_model(model_size)
        for i, video_path in enumerate(to_process, 1):
            if stop_event and stop_event.is_set():
                print("\nユーザーによる中断がリクエストされました。")
                break
            print(f"  文字起こし中: {os.path.basename(video_path)}")
            sys.stdout.flush()
            try:
                segments = transcribe_video(model, video_path, language=language, stop_event=stop_event)
            except Exception as e:
                print(f"  エラー ({os.path.basename(video_path)}): {e}")
                continue
            if stop_event and stop_event.is_set():
                # 途中までの結果は保存しない (次回やり直す)
                print("\nユーザーによる中断がリクエストされました。")
                break

            transcripts[video_path] = {
                "model": model_size,
                "language": language,
                "transcribed_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "segments": segments,
            }
            save_json_atomic(output_json, data)

            pct = int(i / len(to_process) * 100)
            print(f"進捗: {pct}%| ({i}/{len(to_process)}本完了) - {os.path.basename(video_path)} [{len(segments)} 発話]")
            sys.stdout.flush()

    people, unassigned = build_people_index(transcripts, scan_results)
    data["people"] = people
    data["unassigned"] = unassigned
    save_json_atomic(output_json, data)

    written = write_text_files(people, text_output_dir)
    print(f"\n人物ごとのテキストを保存しました: {text_output_dir}")
    for name, entries in people.items():
        if entries:
            print(f"- {name}: {len(entries)} 件")
    if unassigned:
        print(f"(登録人物が映っていない発話: {len(unassigned)} 件 → {output_json} の unassigned に保存)")
    return written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="スキャン済み動画から人物ごとの発話テキストを抽出")
    parser.add_argument("--model", default="small", help="Whisperモデル (tiny/base/small/medium/large-v3)")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--force", action="store_true", help="文字起こし済みの動画もやり直す")
    args = parser.parse_args()

    run_transcription(model_size=args.model, language=args.language, force=args.force)
