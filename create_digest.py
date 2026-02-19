import os
import sys
import json
import pickle
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import random
from utils import resource_path

def load_scan_results(json_path='scan_results.json'):
    if not os.path.exists(json_path):
        print(f"エラー: スキャン結果ファイルが見つかりません: {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


from utils import load_config

def apply_blur(frame, target_encodings, blur_enabled):
    if not blur_enabled:
        return frame
    
    # MoviePy frames are often read-only. Create a copy to modify.
    processed_frame = frame.copy()
    
    # face_recognition expects RGB, which MoviePy already provides.
    face_locations = face_recognition.face_locations(processed_frame)
    if not face_locations:
        return processed_frame
    
    face_encodings = face_recognition.face_encodings(processed_frame, face_locations)
    known_encodings = list(target_encodings.values())
    
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        # ターゲットに一致するか判定
        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.5)
        if not any(matches):
            # ターゲット以外ならぼかし
            face_region = processed_frame[top:bottom, left:right]
            if face_region.size == 0: continue

            # ぼかし強度
            kh = (bottom - top) // 4 * 2 + 1
            kw = (right - left) // 4 * 2 + 1
            ksize = (max(1, kw), max(1, kh))
            
            blurred_face = cv2.GaussianBlur(face_region, ksize, 30)
            processed_frame[top:bottom, left:right] = blurred_face
            
    return processed_frame

def create_digest(scan_results_path, target_person_name=None, config_path='config.json', base_output_dir='output', period="All Time", focus="Balance", blur_enabled=None):
    results = load_scan_results(scan_results_path)
    config = load_config(config_path)
    
    # 引数、環境変数、Configの順で優先
    if blur_enabled is None:
        blur_enabled = str(os.environ.get("DIGEST_BLUR", config.get("blur_enabled", False))).lower() in ("1", "true", "yes")
    
    # ターゲットのエンコーディングをロード（ぼかし判定用）
    target_encodings = {}
    if blur_enabled:
        target_pkl = resource_path('target_faces.pkl')
        if os.path.exists(target_pkl):
            with open(target_pkl, 'rb') as f:
                target_encodings = pickle.load(f)
        else:
            # Fallback to absolute/local path if not found in bundle
            if os.path.exists('target_faces.pkl'):
                with open('target_faces.pkl', 'rb') as f:
                    target_encodings = pickle.load(f)
    
    if not results:
        print("スキャン結果が空です。")
        return

    people_data = results.get("people", {})
    metadata = results.get("metadata", {})

    for person_name, video_map in people_data.items():
        if target_person_name and person_name != target_person_name:
            continue
        if not video_map:
            continue
            
        print(f"\n>>>> Starting Digest for: {person_name} (Period: {period}, Focus: {focus}) <<<<")
        
        # 月ごとにグループ化し、期間でフィルタリング
        monthly_groups = {}
        for video_path, ts in video_map.items():
            month = metadata.get(video_path, {}).get('month', 'unknown')
            
            # フィルタリング
            if period != "All Time":
                if period.count("-") == 1: # YYYY-MM
                    if month != period: continue
                else: # YYYY
                    if month.split("-")[0] != period: continue

            if month not in monthly_groups:
                monthly_groups[month] = []
            monthly_groups[month].append((video_path, ts))
            
        if not monthly_groups:
            print(f"  指定された期間 ({period}) の素材が見つかりませんでした。")
            continue

        total_groups = len(monthly_groups)
        processed_groups = 0

        for month_str, video_list in monthly_groups.items():
            print(f"\n--- Processing {person_name} / {month_str} ---")
            
            # 出力ディレクトリ作成: output/YYYY-MM/PersonName/
            output_dir = os.path.join(base_output_dir, month_str, person_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"digest_{person_name}_{month_str}_{focus}.mp4")

            final_clips = []
            
            for video_path, detections in video_list:
                if not os.path.exists(video_path):
                    continue

                try:
                    video = VideoFileClip(video_path)
                    duration = video.duration
                    
                    # 重視項目（Focus）に応じたスコアリング関数
                    def get_score_func(f):
                        if f == "Smile":
                            return lambda x: x.get('happy', 0)
                        elif f == "Emotional":
                            return lambda x: x.get('drama', 0)
                        elif f == "Active":
                            return lambda x: x.get('motion', 0)
                        else: # Balance
                            return lambda x: (x.get('happy', 0) + x.get('drama', 0) + (x.get('motion', 0)/10.0)) / 2.0

                    score_func = get_score_func(focus)
                    best_detection = max(detections, key=score_func)
                    best_t = best_detection['t']
                    
                    # 前後1.5秒（計3秒）を切り抜く
                    start = max(0, best_t - 1.5)
                    end = min(duration, best_t + 1.5)
                    
                    print(f"  抽出中: {os.path.basename(video_path)} @ {best_t}s (Focus: {focus}, Score: {score_func(best_detection):.2f})")
                    clip = video.subclip(start, end)
                    
                    # --- 究極の解像度正規化 (1280x720 キャンバス固定) ---
                    # 1. 回転を「無効」にし、MoviePyが読み取った生のピクセルサイズを維持する
                    # (縦長・横長にかかわらず、読み取った w, h をそのまま使う)
                    orig_w, orig_h = clip.size
                    target_w, target_h = 1280, 720
                    
                    # 2. 比率を「絶対に」維持して 1280x720 に収まる倍率を計算
                    scale = min(target_w / orig_w, target_h / orig_h)
                    
                    # 3. リサイズ実行 (倍率指定リサイズはアス比が崩れない)
                    clip = clip.resize(scale)
                    new_w, new_h = clip.size
                    print(f"    [DEBUG] Normalizing: {orig_w}x{orig_h} -> {new_w}x{new_h} (Scale: {scale:.3f})")

                    # 4. 1280x720の黒背景の中央に配置 (CompositeVideoClipでガチガチに固める)
                    from moviepy.editor import ColorClip
                    bg_clip = ColorClip(size=(target_w, target_h), color=(0,0,0)).set_duration(clip.duration)
                    clip = CompositeVideoClip([bg_clip, clip.set_position("center")])
                    
                    # 5. テクニカル同期
                    clip = clip.set_fps(24)
                    
                    # --- 音声と時間の厳密な同期 ---
                    if clip.audio is not None:
                        clip.audio = clip.audio.set_fps(44100)
                    clip = clip.set_duration(3.0)

                    # 顔ぼかし適用
                    if blur_enabled:
                        clip = clip.fl_image(lambda f: apply_blur(f, target_encodings, blur_enabled))

                    # 日付テロップ適用 (メタデータから取得)
                    date_str = ""
                    meta = results.get("metadata", {}).get(video_path, {})
                    v_date = meta.get("date", meta.get("month", ""))
                    
                    if v_date and len(v_date) >= 10:
                        date_str = v_date.split(" ")[0].replace("-", "/")
                    else:
                        # メタデータ不備時のフォールバック: ファイルの更新日時から取得
                        try:
                            mtime = os.path.getmtime(video_path)
                            dt = datetime.datetime.fromtimestamp(mtime)
                            date_str = dt.strftime('%Y/%m/%d')
                        except:
                            date_str = v_date # そのまま使う (YYYY-MMなど)

                    if date_str:
                        # lambdaの遅延バインディング回避のためにデフォルト引数で渡す
                        clip = clip.fl_image(lambda f, ds=date_str: add_date_overlay(f, ds))

                    final_clips.append(clip)
                    
                except Exception as e:
                    print(f"  エラー: {video_path}: {e}")

            if not final_clips:
                processed_groups += 1
                continue

            print(f"  {len(final_clips)} 個のクリップを結合中: {output_path}")
            
            try:
                final_video = concatenate_videoclips(final_clips, method="compose")
                # moviepy 1.0.3 write_videofile has progress bar to stderr, but we can emit our own
                final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                                            fps=24, audio_fps=44100, threads=4,
                                            ffmpeg_params=["-pix_fmt", "yuv420p"])
            except Exception as e:
                print(f"  エラー: {e}")
            finally:
                if 'final_video' in locals(): final_video.close()
                for clip in final_clips: clip.close()
            
            processed_groups += 1
            print(f"進捗: {int((processed_groups / total_groups) * 100)}%")

def add_date_overlay(frame, date_str):
    from PIL import Image, ImageDraw, ImageFont
    
    # Numpy -> PIL
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    
    # フォントロード (オープンソースのNoto Sans JPを使用)
    try:
        font_path = resource_path("assets/fonts/NotoSansJP-Bold.ttf")
        font = ImageFont.truetype(font_path, 40)
    except:
        font = ImageFont.load_default()
    
    # 左下 (位置も少し調整)
    pos = (50, 630)
    
    # 影（薄く、細く）
    shadow_color = (0, 0, 0, 100) # RGBA
    
    # 影描画 (RGBモードでdraw.textを使う場合の簡易的な影)
    offset = 2
    # fillにalphaを含めてもRGBモードのImageでは無視されるか、単色になるため
    # 本当に薄くしたいならRGBAモードのImageを作って合成する必要があるが、
    # ここではシンプルにサイズダウンと「黒の不透明度」を下げる代わりに灰色を使う手もあるが、
    # ひとまずサイズダウンが主目的。影は真っ黒でOKだが控えめに。
    draw.text((pos[0]+offset, pos[1]+offset), date_str, font=font, fill=(0,0,0))
    
    # 本体（白）
    draw.text(pos, date_str, font=font, fill=(255,255,255))
    
    # PIL -> Numpy
    return np.array(img_pil)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="scan_results.json")
    parser.add_argument("--person", default=None)
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--no-blur", action="store_false", dest="blur")
    parser.add_argument("--period", default="All Time")
    parser.add_argument("--focus", default="Balance")
    args = parser.parse_args()

    # コマンドライン引数を優先
    os.environ["DIGEST_BLUR"] = "1" if args.blur else "0"

    create_digest(args.json, target_person_name=args.person, period=args.period, focus=args.focus)
