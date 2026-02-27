import json
import os
import sys
import pickle
import gc
import random
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import subprocess
import imageio_ffmpeg
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip, ImageClip
from utils import resource_path, load_config, get_user_data_dir, get_ffprobe_path


def get_video_rotation(path):
    """ffprobeを使用して動画の回転メタデータを取得する。
    """
    try:
        ffprobe_exe = get_ffprobe_path()
        
        cmd = [
            ffprobe_exe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_side_data=rotation",
            "-of", "json", path
        ]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo, shell=os.name == 'nt')
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams and "side_data_list" in streams[0]:
                for sd in streams[0]["side_data_list"]:
                    if "rotation" in sd:
                        return int(sd["rotation"])
    except Exception as e:
        print(f"    [DEBUG] Rotation detection fallback triggered: {e}")
        # Final fallback: if everything fails, assume no rotation needed
    return 0






def add_date_overlay(frame, date_str):
    from PIL import Image, ImageDraw, ImageFont
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font_path = resource_path("assets/fonts/NotoSansJP-Bold.ttf")
        font = ImageFont.truetype(font_path, 40)
    except:
        font = ImageFont.load_default()
    pos = (50, 630)
    offset = 2
    draw.text((pos[0]+offset, pos[1]+offset), date_str, font=font, fill=(0,0,0))
    draw.text(pos, date_str, font=font, fill=(255,255,255))
    return np.array(img_pil)

def apply_color_filter(frame, filter_type):
    if filter_type == "None" or not filter_type:
        return frame
    
    img = frame.astype(np.float32)
    if filter_type == "Film":
        img = img * 1.1 - 10
        img[:,:,2] *= 1.1
    elif filter_type == "Sunset":
        img[:,:,0] *= 1.2
        img[:,:,1] *= 1.1
        img[:,:,2] *= 0.8
    elif filter_type == "Cinema":
        img = img * 1.25 - 20
        avg = np.mean(img, axis=2, keepdims=True)
        img = img * 0.7 + avg * 0.3
    elif filter_type == "Nostalgic":
        img[:,:,0] *= 1.15
        img[:,:,1] *= 1.05
        img[:,:,2] *= 0.85
        img = img * 0.9 + 15
    elif filter_type == "Vivid":
        img = (img - 128) * 1.3 + 128
        img *= 1.1
    elif filter_type == "Pastel":
        img = img * 0.6 + 90
        img[:,:,0] *= 1.15
        img[:,:,2] *= 1.10
    
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def add_title_overlay(frame, title_text):
    if not title_text:
        return frame
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.2
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(title_text, font, scale, thickness)
    text_x = (w - tw) // 2
    text_y = (h + th) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (text_x-20, text_y-th-20), (text_x+tw+20, text_y+20), (0,0,0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.putText(img, title_text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_title_card(title_text, subtitle_text="", duration=3.0, font_size=80):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1280, 720
    img_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    try:
        font_path = resource_path("assets/fonts/NotoSansJP-Bold.ttf")
        font_title = ImageFont.truetype(font_path, font_size)
        font_sub = ImageFont.truetype(font_path, 40)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    def draw_centered(text, font, y_offset=0):
        if not text: return
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        y = (height - text_h) // 2 + y_offset
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        return y + text_h
    draw_centered(title_text, font_title, y_offset=-20 if subtitle_text else 0)
    if subtitle_text:
        draw_centered(subtitle_text, font_sub, y_offset=60)
    return ImageClip(np.array(img_pil)).set_duration(duration).set_fps(24)

def render_documentary(playlist_path='story_playlist.json', config_path='config.json', output_dir='output', filter_type=None, bgm_enabled=None, focus=None):
    if not os.path.exists(playlist_path):
        print(f"Error: Playlist not found: {playlist_path}")
        return

    with open(playlist_path, 'r', encoding='utf-8') as f:
        playlist_data = json.load(f)
    
    # 新しい形式（dict）と古い形式（list）の両方に対応
    if isinstance(playlist_data, dict):
        playlist = playlist_data.get("clips", [])
        dominant_vibe = playlist_data.get("dominant_vibe", "穏やか")
    else:
        playlist = playlist_data
        dominant_vibe = "穏やか"

    config = load_config(config_path)
    # 引数、環境変数、Configの順で優先
        
    if bgm_enabled is None:
        bgm_enabled = str(os.environ.get("RENDER_BGM", "0")).lower() in ("1", "true", "yes")

    # BGMのVibeに合わせた自動フィルター設定
    if filter_type == "None" or filter_type is None:
        vibe_to_filter = {
            "感動的": "Cinema",
            "穏やか": "Nostalgic",
            "エネルギッシュ": "Vivid",
            "かわいい": "Pastel"
        }
        auto_filter = vibe_to_filter.get(dominant_vibe)
        if auto_filter:
            print(f"  Vibe ({dominant_vibe}) に基づきフィルター '{auto_filter}' を自動適用します")
            filter_type = auto_filter


    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    f_tag = f"_{focus}" if focus else ""
    output_path = os.path.join(output_dir, f"documentary_{timestamp_str}{f_tag}.mp4")

    final_clips = []
    print(f"\n>>>> ドキュメンタリーをレンダリング中 ({len(playlist)} clips) <<<<")

    for i, item in enumerate(playlist):
        video_path = item["video_path"]
        # NFC/NFD normalization check
        if not os.path.exists(video_path):
            import unicodedata
            video_path = unicodedata.normalize('NFC', item["video_path"])
            if not os.path.exists(video_path):
                video_path = unicodedata.normalize('NFD', item["video_path"])
            
        if not os.path.exists(video_path):
            print(f"  [ERROR] File not found: {item['video_path']}")
            continue
            
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            best_t = item["t"]
            start = max(0, best_t - 1.5)
            end = min(duration, best_t + 1.5)
            
            print(f"  [{i+1}/{len(playlist)}] Processing: {os.path.basename(video_path)} @ {best_t}s")
            print(f"    [DEBUG] Full Path: {video_path}")
            
            # メタデータの詳細ログ出力 (1行に集約)
            try:
                ffprobe_exe = get_ffprobe_path()
                meta_cmd = [ffprobe_exe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
                
                # Use startupinfo and shell=True for Windows
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                meta_json = json.loads(subprocess.check_output(meta_cmd, startupinfo=startupinfo, shell=os.name == 'nt').decode('utf-8'))
                # 冗長なパス情報などを除き、コンパクトに1行で出力
                meta_flat = json.dumps(meta_json, separators=(',', ':'))
                print(f"    [DEBUG] Video Metadata: {meta_flat}")
            except Exception as me:
                print(f"    [DEBUG] Could not fetch metadata: {me}")

            # --- Load and Subclip ---
            raw_clip = video.subclip(start, end)
            
            # --- Robust Normalization (1280x720 Fixed Canvas) ---
            target_w, target_h = 1280, 720
            
            # メタデータから本来の向きを判定
            rotation = get_video_rotation(video_path)
            orig_w_pre, orig_h_pre = raw_clip.size # MoviePyが誤認識している枠のサイズ
            
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

                clip = raw_clip.fl_image(format_canvas)
                clip.size = (target_w, target_h)
                
            else:
                # 【正常な動画用】以前安定稼働していたMoviePyネイティブの処理
                print(f"    [NORMAL] Processing standard video {orig_w_pre}x{orig_h_pre}")
                
                orig_w, orig_h = raw_clip.size
                is_vertical = orig_h > orig_w
                ratio_diff = abs((orig_w / orig_h) - (target_w / target_h))
                
                if is_vertical or ratio_diff > 0.1:
                    bg_scale = max(target_w / orig_w, target_h / orig_h)
                    bg_clip = raw_clip.resize(bg_scale)
                    bg_clip = bg_clip.fl_image(lambda f: cv2.GaussianBlur(f, (51, 51), 0))
                    # 中央でクロップ
                    bg_clip = bg_clip.crop(width=target_w, height=target_h, x_center=bg_clip.size[0]/2, y_center=bg_clip.size[1]/2)
                    
                    fg_scale = min(target_w / orig_w, target_h / orig_h)
                    fg_clip = raw_clip.resize(fg_scale) 
                    clip = CompositeVideoClip([bg_clip, fg_clip.set_position("center")], size=(target_w, target_h))
                else:
                    scale = min(target_w / orig_w, target_h / orig_h)
                    scaled_clip = raw_clip.resize(scale)
                    bg_clip = ColorClip(size=(target_w, target_h), color=(0,0,0)).set_duration(scaled_clip.duration)
                    clip = CompositeVideoClip([bg_clip, scaled_clip.set_position("center")], size=(target_w, target_h))

            # 5. テクニカル同期
            clip = clip.set_fps(24)
            if clip.audio is not None:
                clip.audio = clip.audio.set_fps(44100)
            clip = clip.set_duration(3.0)
            
            # --- 5. Visual Overlays ---
            
            # Apply color filter
            if filter_type and filter_type != "None":
                clip = clip.fl_image(lambda f: apply_color_filter(f, filter_type))

            # Apply Date Overlay
            timestamp = item.get("timestamp", "")
            if timestamp:
                ds = timestamp.split(" ")[0].replace("-", "/")
                clip = clip.fl_image(lambda f, d=ds: add_date_overlay(f, d))

            final_clips.append(clip)
            
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
        
        # 定期的にGCを走らせてメモリ解放
        if i % 5 == 0:
            gc.collect()

    if not final_clips:
        print("Error: No clips to concatenate.")
        return

    # --- Add Opening and Ending ---
    # 人物名と期間を取得（プレイリストに含まれている場合）
    # 現状はプレイリストファイル自体にはメタデータが少ないため、簡易的に実装
    # 実際には create_story.py で metadata を保存するように改修するか、
    # クリップ情報から推測する。ここではシンプルに "Memory Documentary" とするか、
    # クリップがあればその期間を表示。
    
    period_str = ""
    if playlist:
        try:
            dates = [c.get("timestamp", "").split(" ")[0] for c in playlist if c.get("timestamp")]
            dates.sort()
            if dates:
                start_year = dates[0][:4]
                end_year = dates[-1][:4]
                if start_year == end_year:
                    period_str = start_year
                else:
                    period_str = f"{start_year} - {end_year}"
        except:
            pass
            
    # OP: Title + Period
    person_name = ""
    if isinstance(playlist_data, dict):
        person_name = playlist_data.get("person_name", "")
        
    if person_name:
        op_title = f"The Story of {person_name}"
    else:
        op_title = "Memory Documentary"
        
    op_clip = create_title_card(op_title, period_str, duration=3.0).fadein(1.0)
    
    # ED: To Be Continued...
    # ED: Randomized Text
    ed_texts = [
        "The Best is Yet to Come",
        "Life is a Journey",
        "Moments to Treasure",
        "Timeless Memories",
        "To Be Continued...",
        "Every Day is a New Beginning",
        "Cherish Every Moment",
        "Our Story Continues",
        "Always in Our Hearts",
        "Focus on the Good"
    ]
    ed_text = random.choice(ed_texts)
    ed_clip = create_title_card(ed_text, "", duration=4.0, font_size=50).fadein(1.0).fadeout(1.0)
    
    # 結合: OP + Main + ED
    final_clips = [op_clip] + final_clips + [ed_clip]

    if not final_clips:
        print("Error: No clips were successfully processed.")
        return

    print(f"\nConcatenating {len(final_clips)} clips...")
    try:
        # すべて同サイズに正規化済みなので、最速のデフォルトメソッド(chain)を使用
        final_video = concatenate_videoclips(final_clips)
        
        # BGMミキシング
        if bgm_enabled:
            # vibeに応じたプレフィックス (日本語と英語の両方をチェック)
            vibe_prefix_en = {
                "穏やか": "calm",
                "エネルギッシュ": "energetic",
                "感動的": "emotional",
                "かわいい": "cute"
            }
            
            import unicodedata
            
            # Check for manual BGM
            manual_bgm = playlist_data.get("manual_bgm_path", "")
            print(f"DEBUG: Manual BGM Path from playlist: '{manual_bgm}'")
            
            candidates = []
            if manual_bgm:
                # 1. Try exact match
                if os.path.exists(manual_bgm):
                    candidates = [manual_bgm]
                else:
                    # 2. Try Unicode normalization (NFC/NFD)
                    normalized_nfc = unicodedata.normalize('NFC', manual_bgm)
                    normalized_nfd = unicodedata.normalize('NFD', manual_bgm)
                    
                    if os.path.exists(normalized_nfc):
                        candidates = [normalized_nfc]
                        print(f"DEBUG: Found BGM via NFC normalization: {normalized_nfc}")
                    elif os.path.exists(normalized_nfd):
                        candidates = [normalized_nfd]
                        print(f"DEBUG: Found BGM via NFD normalization: {normalized_nfd}")
                    else:
                        # 3. Try finding by filename in the bgm directory (loose match)
                        bgm_dir = os.path.dirname(manual_bgm)
                        bgm_name = os.path.basename(manual_bgm)
                        
                        if os.path.exists(bgm_dir):
                            print(f"DEBUG: Searching in {bgm_dir} for {bgm_name}...")
                            for f in os.listdir(bgm_dir):
                                # Normalize both for comparison
                                if unicodedata.normalize('NFC', f) == unicodedata.normalize('NFC', bgm_name):
                                    found_path = os.path.join(bgm_dir, f)
                                    candidates = [found_path]
                                    print(f"DEBUG: Found BGM via directory search: {found_path}")
                                    break
            
            if candidates:
                # Use the first valid candidate
                manual_bgm = candidates[0]
                print(f"\n>>> Using Manually Selected BGM (Found): {manual_bgm}")
            else:
                if manual_bgm:
                    print(f"DEBUG: Manual BGM path was provided but file not found: {manual_bgm}")
                print(">>> No manual BGM selected. Proceeding without BGM.")
                
            if candidates:
                bgm_file = candidates[0] # Only one candidate
                print(f"\n>>> BGMをミックス中: {bgm_file}")
                try:
                    # Temporary workaround for non-ASCII filenames/metadata issues in MoviePy
                    import shutil
                    import tempfile
                    
                    # Create a temp file for the clean audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        temp_bgm_path = tmp.name
                    
                    # Use ffmpeg to transcode to wav (strips messy metadata and ensures decodeability)
                    # ffmpeg -i input -y output
                    # -vn: disable video, -acodec pcm_s16le: standard wav
                    # Use imageio_ffmpeg to get the binary path
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    
                    cmd = [
                        ffmpeg_exe, 
                        "-y", 
                        "-i", bgm_file, 
                        "-vn", 
                        "-acodec", "pcm_s16le", 
                        "-ar", "44100", 
                        "-ac", "2",
                        "-map_metadata", "-1", # Strip all metadata
                        temp_bgm_path
                    ]
                    
                    print(f"  BGM再変換中 (ffmpeg: {ffmpeg_exe})...")
                    # Run ffmpeg
                    # On Windows, subprocess need proper handling for paths with spaces if not in list, but list is fine.
                    # Also need to ensure no console window for calling ffmpeg if possible, but subprocess.run usually fine.
                    # Startupsinfo to hide console window on Windows
                    startupinfo = None
                    if os.name == 'nt':
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
                    
                    if result.returncode != 0:
                        print(f"  Warning: ffmpeg conversion failed. Using original file copy method.")
                        print(f"  ffmpeg stderr: {result.stderr.decode('utf-8', errors='ignore')}")
                        # Fallback: just copy
                        shutil.copy2(bgm_file, temp_bgm_path)
                    else:
                        print("  ffmpegによるWAV変換・メタデータ削除完了")

                    print(f"  AudioFileClipで読み込み開始: {temp_bgm_path}")
                    bgm_audio = AudioFileClip(temp_bgm_path)
                    print(f"  AudioFileClip読み込み成功. Duration: {bgm_audio.duration}")
                    
                    # BGMを動画の長さに合わせる
                    video_duration = final_video.duration
                    is_special_vibe = dominant_vibe in ["感動的"]
                    
                    if is_special_vibe:
                        # 感動的: ループさせず、20秒以降かつ動画の終わりに合わせて配置
                        # 47秒のBGMが動画の最後に終わるように開始時間を計算
                        # ただし、開始は最低でも20秒後とする
                        bgm_start = max(20.0, video_duration - bgm_audio.duration)
                        bgm_audio = bgm_audio.set_start(bgm_start)
                        # 動画の長さを超える部分はカット
                        if bgm_start + bgm_audio.duration > video_duration:
                            bgm_audio = bgm_audio.subclip(0, video_duration - bgm_start)
                        print(f"  特殊配置適用 (穏やか/感動): 開始={bgm_start:.1f}s")
                    else:
                        # かわいい・元気: 必要に応じてループ
                        if bgm_audio.duration < video_duration:
                            # 既存のループ処理をベースにするが、47秒の素材を想定
                            crossfade_dur = min(3.0, bgm_audio.duration / 3) 
                            loop_audio = bgm_audio.audio_fadeout(crossfade_dur)
                            current_len = bgm_audio.duration
                            
                            while current_len < video_duration + crossfade_dur:
                                print(f"  BGMループを追加中... (現在: {current_len:.1f}s)")
                                next_segment = bgm_audio.audio_fadein(crossfade_dur).audio_fadeout(crossfade_dur)
                                start_time = current_len - crossfade_dur
                                loop_audio = CompositeAudioClip([loop_audio, next_segment.set_start(start_time)])
                                current_len = loop_audio.duration
                            
                            bgm_audio = loop_audio.subclip(0, video_duration)
                        else:
                            bgm_audio = bgm_audio.subclip(0, video_duration)
                    
                    # フェードアウト（最後の2秒）
                    bgm_audio = bgm_audio.audio_fadeout(2.0)
                    
                    # 元の音声とBGMをミックス（BGMは小さめに）
                    if final_video.audio is not None:
                        # 元の音声を保持しつつBGMを追加（BGMは30%の音量）
                        # volumexは副作用がないので新しいクリップを返す
                        mixed_audio = CompositeAudioClip([
                            final_video.audio.volumex(1.0), 
                            bgm_audio.volumex(0.3)
                        ])
                    else:
                        # 元の音声がない場合はBGMのみ
                        mixed_audio = bgm_audio.volumex(0.3)
                    
                    final_video = final_video.set_audio(mixed_audio)
                    print(f"  BGMミキシング完了")
                    
                    # Close bgm_audio after setting it to final_video (final_video keeps a copy or reference)
                    # we close them in final finally
                except Exception as e:
                    print(f"  BGMミキシングエラー: {e}")
                    print(f"  BGMなしで続行します...")
        
        # Generate a safe temp audio path in the system temp directory
        # to avoid Broken Pipe error when output_path contains multi-byte characters.
        import tempfile
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_mpy_{timestamp_str}.m4a")
        
        # --- Absolute Stability: pix_fmt yuv420p, audio_fps, threads ---
        print(f"\n>>> RENDERING FILE: {output_path}")
        print(f"    (Preset: ultrafast, Threads: 4, FPS: 24)")
        
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                                    fps=24, audio_fps=44100, threads=4,
                                    preset='ultrafast',
                                    temp_audiofile=temp_audio_path, remove_temp=True,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"\n>>> DOCUMENTARY GENERATED SUCCESSFULLY: {output_path}")
    except Exception as e:
        print(f"Error during concatenation: {e}")
    finally:
        print("\nCleaning up resources...")
        if 'final_video' in locals() and final_video: 
            try: final_video.close()
            except: pass
        
        # BGM一時ファイルの削除をここ（最後に）移動
        if 'temp_bgm_path' in locals() and temp_bgm_path and os.path.exists(temp_bgm_path):
            try:
                if 'bgm_audio' in locals() and bgm_audio:
                    bgm_audio.close()
                os.remove(temp_bgm_path)
                print(f"  BGM一時ファイルを最終削除しました: {temp_bgm_path}")
            except: pass

        # すべてのサブクリップを明示的に閉じる
        if 'final_clips' in locals():
            for c in final_clips:
                try: c.close()
                except: pass
        
        # 最後にメモリを強制解放
        gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgm", action="store_true")
    parser.add_argument("--no-bgm", action="store_false", dest="bgm")
    args = parser.parse_args()

    # 環境変数にセットして render_documentary 内で参照
    os.environ["RENDER_BGM"] = "1" if args.bgm else "0"

    render_documentary()
