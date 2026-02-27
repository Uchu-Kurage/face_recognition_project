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
import subprocess
import imageio_ffmpeg
from utils import resource_path, load_config, get_ffprobe_path

def load_scan_results(json_path='scan_results.json'):
    if not os.path.exists(json_path):
        print(f"エラー: スキャン結果ファイルが見つかりません: {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_video_rotation(path):
    """ffprobeを使用して動画の回転メタデータを取得する。"""
    try:
        ffprobe_exe = get_ffprobe_path()
        cmd = [ffprobe_exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream_side_data=rotation", "-of", "json", path]
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams and "side_data_list" in streams[0]:
                for sd in streams[0]["side_data_list"]:
                    if "rotation" in sd: return int(sd["rotation"])
    except: pass
    return 0



from utils import load_config

def apply_blur(frame, target_encodings, blur_enabled):
    if not blur_enabled:
        return frame
    
    # MoviePy frames are often read-only. Create a copy to modify.
    processed_frame = frame.copy()
    
    # 【追加】顔検出用に画像を1/4に縮小（爆速化の要！）
    small_frame = cv2.resize(processed_frame, (0, 0), fx=0.25, fy=0.25)
    
    # 縮小した画像で顔の場所を探す
    small_face_locations = face_recognition.face_locations(small_frame)
    if not small_face_locations:
        return processed_frame
        
    # 見つけた顔の座標を4倍にして、元のサイズに戻す
    face_locations = [
        (int(top*4), int(right*4), int(bottom*4), int(left*4)) 
        for (top, right, bottom, left) in small_face_locations
    ]
    
    # エンコーディング（顔の照合）は元の画質で行う
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
                    print(f"    [DEBUG] Full Path: {video_path}")
                    
                    # メタデータの詳細ログ出力 (1行に集約)
                    try:
                        ffprobe_path = get_ffprobe_path()
                        meta_cmd = [
                            ffprobe_path, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,display_aspect_ratio,pix_fmt:stream_tags:stream_side_data",
                            "-of", "json", video_path
                        ]
                        meta_json = json.loads(subprocess.check_output(meta_cmd).decode('utf-8'))
                        meta_flat = json.dumps(meta_json, separators=(',', ':'))
                        print(f"    [DEBUG] Video Metadata: {meta_flat}")
                    except Exception as me:
                        print(f"    [DEBUG] Could not fetch metadata: {me}")

                    clip = video.subclip(start, end)
                    
                    # --- Robust Normalization (1280x720 Fixed Canvas) ---
                    target_w, target_h = 1280, 720
                    
                    # メタデータから本来の向きを判定
                    rotation = get_video_rotation(video_path)
                    orig_w_pre, orig_h_pre = clip.size # MoviePyが誤認識している枠のサイズ
                    
                    # 横長枠なのに回転メタデータ(-90等)がある場合のみ True になる
                    needs_unsquash = (orig_w_pre > orig_h_pre) and (rotation in [-90, 90, 270, -270])
                    
                    if needs_unsquash:
                        # 【異常な動画用】MoviePyに潰された映像をOpenCVで解毒して強制復元
                        print(f"    [UNSQUASH] Detecting squashed frame {orig_w_pre}x{orig_h_pre}. Applying OpenCV Unsquash...")
                        
                        def format_canvas(frame):
                            # 潰された映像を本来の縦長に引き伸ばす
                            frame = cv2.resize(frame, (orig_h_pre, orig_w_pre))
                            
                            h, w = frame.shape[:2]
                            is_vertical = h > w
                            ratio_diff = abs((w / h) - (target_w / target_h))
                            use_bokeh = is_vertical or ratio_diff > 0.1

                            # 前景（メイン動画）のリサイズ
                            scale_fg = min(target_w / w, target_h / h)
                            new_w, new_h = int(w * scale_fg), int(h * scale_fg)
                            fg_resized = cv2.resize(frame, (new_w, new_h))
                            
                            x_offset = (target_w - new_w) // 2
                            y_offset = (target_h - new_h) // 2
                            
                            if use_bokeh:
                                # 背景（ボカシ）の生成
                                scale_bg = max(target_w / w, target_h / h)
                                new_w_bg = max(target_w, int(w * scale_bg))
                                new_h_bg = max(target_h, int(h * scale_bg))
                                bg_resized = cv2.resize(frame, (new_w_bg, new_h_bg))
                                
                                x_crop = (new_w_bg - target_w) // 2
                                y_crop = (new_h_bg - target_h) // 2
                                bg_cropped = bg_resized[y_crop:y_crop+target_h, x_crop:x_crop+target_w]
                                
                                canvas = cv2.GaussianBlur(bg_cropped, (51, 51), 0)
                            else:
                                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                            
                            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = fg_resized
                            return canvas

                        clip = clip.fl_image(format_canvas)
                        clip.size = (target_w, target_h)
                        
                    else:
                        # 【正常な動画用】安定しているMoviePyネイティブ処理
                        print(f"    [NORMAL] Processing standard video {orig_w_pre}x{orig_h_pre}")
                        
                        orig_w, orig_h = clip.size
                        is_vertical = orig_h > orig_w
                        ratio_diff = abs((orig_w / orig_h) - (target_w / target_h))
                        
                        if is_vertical or ratio_diff > 0.1:
                            bg_scale = max(target_w / orig_w, target_h / orig_h)
                            bg_clip = clip.resize(bg_scale)
                            bg_clip = bg_clip.fl_image(lambda f: cv2.GaussianBlur(f, (51, 51), 0))
                            # 中央でクロップ
                            bg_clip = bg_clip.crop(width=target_w, height=target_h, x_center=bg_clip.size[0]/2, y_center=bg_clip.size[1]/2)
                            
                            fg_scale = min(target_w / orig_w, target_h / orig_h)
                            fg_clip = clip.resize(fg_scale) 
                            clip = CompositeVideoClip([bg_clip, fg_clip.set_position("center")], size=(target_w, target_h))
                        else:
                            scale = min(target_w / orig_w, target_h / orig_h)
                            scaled_clip = clip.resize(scale)
                            bg_clip = ColorClip(size=(target_w, target_h), color=(0,0,0)).set_duration(scaled_clip.duration)
                            clip = CompositeVideoClip([bg_clip, scaled_clip.set_position("center")], size=(target_w, target_h))

                    # 5. テクニカル同期
                    clip = clip.set_fps(24)
                    
                    # --- 音声と時間の厳密な同期 ---
                    if clip.audio is not None:
                        clip.audio = clip.audio.set_fps(44100)
                    clip = clip.set_duration(3.0)

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
                # すべて1280x720に正規化済みのため、重い compose ではなくデフォルト(chaining)で安定化
                final_video = concatenate_videoclips(final_clips)
                
                # 日本語パス等での Broken pipe 回避のため一時ファイルを安全な場所に作成
                import tempfile
                temp_audio_path = os.path.join(tempfile.gettempdir(), f"mpy_temp_audio_{person_name}_{month_str}.m4a")
                
                # moviepy 1.0.3 write_videofile has progress bar to stderr, but we can emit our own
                final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                                            fps=24, audio_fps=44100, threads=4,
                                            temp_audiofile=temp_audio_path, remove_temp=True,
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
    parser.add_argument("--period", default="All Time")
    parser.add_argument("--focus", default="Balance")
    args = parser.parse_args()

    create_digest(args.json, target_person_name=args.person, period=args.period, focus=args.focus)
