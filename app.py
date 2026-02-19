import sys
import os

# Fix macOS crash when using pygame and tkinter together
if sys.platform == "darwin":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import customtkinter as ctk
from tkinter import filedialog, messagebox
import subprocess
import threading
import time
import math
import queue
import hashlib
import unicodedata
from PIL import Image, ImageTk
import json
import glob
import re
from collections import Counter
from extract_features import register_person, delete_person
import scan_videos
import create_digest
import create_story
import render_story
import generate_bgm
import webbrowser
import pygame
import cv2
from utils import resource_path, get_app_dir

class RedirectText(object):
    def __init__(self, callback):
        self.callback = callback
        self.buffer = []
        self.last_flush_time = time.time()
        self.lock = threading.Lock()

    def write(self, string):
        with self.lock:
            self.buffer.append(string)
            current_time = time.time()
            if current_time - self.last_flush_time > 0.1: # Reduced from 0.2
                self.flush_locked()

    def flush(self):
        with self.lock:
            # Minimal throttle for flush to avoid spamming from tqdm
            if time.time() - self.last_flush_time > 0.02: # Reduced from 0.05
                self.flush_locked()

    def flush_locked(self):
        if self.buffer:
            text = "".join(self.buffer)
            self.callback(text)
            self.buffer = []
            self.last_flush_time = time.time()

    def isatty(self):
        return False
        
    def close(self):
        pass

# アプリ設定
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernDigestApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Omokage")
        self.geometry("1200x900")
        
        # --- Theme Colors (Twilight Memory) ---
        self.COLOR_DEEP_BG = "#0D1B2A"    # Deep Indigo
        self.COLOR_SIDEBAR = "#1B263B"    # Midnight Blue
        self.COLOR_ACCENT = "#E09F3E"     # Twilight Amber
        self.COLOR_HOVER = "#FFB703"      # Bright Amber
        self.COLOR_TEXT = "#E0E1DD"       # Off White
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue") # Base theme
        
        # Set top-level background to deep indigo
        self.configure(fg_color=self.COLOR_DEEP_BG)

        # 変数 (実行ファイルと同階層に保存)
        app_dir = get_app_dir()
        self.CONFIG_FILE = os.path.join(app_dir, "config.json")
        self.SCAN_RESULTS = os.path.join(app_dir, "scan_results.json")
        self.TARGET_PKL = os.path.join(app_dir, "target_faces.pkl")
        self.STORY_PLAYLIST = os.path.join(app_dir, "story_playlist.json")
        self.OUTPUT_DIR = os.path.join(app_dir, "output")
        
        self.config = self.load_config()
        
        self.target_image_path = ctk.StringVar(value=self.config.get("target_path", ""))
        self.video_folder_path = ctk.StringVar(value=self.config.get("video_folder", ""))
        self.is_running = False
        self.scan_stop_event = threading.Event()
        
        # 編集設定
        self.blur_enabled = ctk.BooleanVar(value=self.config.get("blur_enabled", False))
        self.color_filter = ctk.StringVar(value=self.config.get("color_filter", "None"))
        self.selected_period = ctk.StringVar(value="All Time")
        self.force_rescan = ctk.BooleanVar(value=False) # Added this line
        self.selected_focus = ctk.StringVar(value="バランス")
        self.bgm_enabled = ctk.BooleanVar(value=self.config.get("bgm_enabled", False))
        self.hf_token = ctk.StringVar(value=self.config.get("hf_token", ""))
        self.view_mode = "People" # Forced default
        self.clip_view_mode = "list" # "list" or "grid"
        self.selected_clips = set() # set of (video_path, timestamp)
        self.grid_tile_widgets = {} # NEW: Keep track of tile frames for fast updates
        
        # Scan Data Cache
        self.cached_scan_data = None
        self.cached_scan_mtime = 0
        
        # Audio Playback State
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Failed to initialize pygame mixer: {e}")
        self.playing_bgm = None # path of current playing bgm
        
        # Rendering state for progress bar
        self.render_phase = "init"

        # 画像ロード (assetsから)
        self.load_icons()

        # Load Icon
        icon_path_ico = resource_path(os.path.join("assets", "icon.ico"))
        icon_path_png = resource_path(os.path.join("assets", "icon.png"))
        
        try:
            if sys.platform == "win32" and os.path.exists(icon_path_ico):
                self.iconbitmap(icon_path_ico)
            elif os.path.exists(icon_path_png):
                img = Image.open(icon_path_png)
                photo = ImageTk.PhotoImage(img)
                self.wm_iconphoto(True, photo)
        except Exception as e:
            print(f"Failed to load icon: {e}")
            
        # UI構築
        self.create_layout()
        self.refresh_profiles()
        
        # Protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start log polling & music status polling
        self.log_queue = queue.Queue()
        self.check_log_queue()
        self.check_music_status()

        # サムネイルのバックグラウンド生成開始
        self.after(2000, self.start_thumbnail_warmup)

    def start_thumbnail_warmup(self):
        """未作成のサムネイルをバックグラウンドで一括生成する"""
        def warmup_task():
            results = self.load_scan_results()
            if not results or "people" not in results: return
            
            from utils import generate_face_thumbnail, get_app_dir
            app_dir = get_app_dir()
            all_clips = []
            for name, videos in results["people"].items():
                for v_path, clips in videos.items():
                    for clip in clips:
                        all_clips.append((v_path, clip))
            
            # 既にスキャン済みかつサムネイルがないものを優先
            count = 0
            for v_path, clip in all_clips:
                # 重複判定等は generate_face_thumbnail 内で行われる
                res = generate_face_thumbnail(v_path, clip['t'], clip['face_loc'], app_dir)
                if res: count += 1
                if count % 10 == 0:
                    import time
                    time.sleep(0.01) # UI スレッドをブロックしない程度に
            
            if count > 0:
                print(f"Warm-up: {count} thumbnails pre-generated.")

        threading.Thread(target=warmup_task, daemon=True).start()

    def check_log_queue(self):
        try:
            # Fetch batch of items (up to 1000)
            messages = []
            for _ in range(1000):
                try:
                    msg_item = self.log_queue.get_nowait()
                    messages.append(msg_item)
                except queue.Empty:
                    break
            
            if messages:
                current_batch_text = ""
                
                for msg, kwargs in messages:
                    if msg == "__CLEAR__":
                        # 1. Flush accumulated text
                        if current_batch_text:
                            self._update_log_ui_batch(current_batch_text)
                            current_batch_text = ""
                        
                        # 2. Perform Clear
                        self.textbox.configure(state="normal")
                        self.textbox.delete("1.0", "end")
                        self.textbox.configure(state="disabled")
                    
                    elif msg == "__NOTIFY__":
                        # 1. Flush existing text first
                        if current_batch_text:
                            self._update_log_ui_batch(current_batch_text)
                            current_batch_text = ""
                        
                        # 2. Show Dialog
                        title = kwargs.get("title", "通知")
                        message = kwargs.get("message", "")
                        ntype = kwargs.get("type", "info")
                        
                        if ntype == "error":
                            messagebox.showerror(title, message)
                        elif ntype == "warning":
                            messagebox.showwarning(title, message)
                        else:
                            messagebox.showinfo(title, message)

                    else:
                        current_batch_text += str(msg) + kwargs.get("end", "\n")
                
                # 3. Flush remaining text
                if current_batch_text:
                    self._update_log_ui_batch(current_batch_text)
                
                
        finally:
            self.after(100, self.check_log_queue)

    def _update_log_ui_batch(self, text):
        # Strip ANSI escape codes (e.g., \x1b[A, \x1b[K) which cause messy logs like "[A" in GUI
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        
        self.textbox.configure(state="normal")
        
        # Check if scroll is at the bottom before adding text
        # If scroll is near the bottom (thin margin), we'll auto-scroll after insert
        was_at_bottom = self.textbox.yview()[1] > 0.95
        
        if "\r" in text:
            # GUI text widgets always append to the end. \r in a terminal moves the cursor back,
            # but in a GUI, we must convert this to a newline to avoid horizontal bunching.
            
            # If the entire incoming text starts with \r, it's an update to the current line.
            # In a GUI, we force a newline so it doesn't just append to the previous % bar.
            if text.startswith("\r") and "%|" in text:
                text = "\n" + text.lstrip("\r")

            # Handle multiple carriage returns in a batch by only keeping the last update 
            # within each conceptual line. This avoids thousands of lines from tqdm.
            lines = text.split("\n")
            sanitized_lines = []
            for line in lines:
                if "\r" in line:
                    parts = line.split("\r")
                    new_parts = []
                    last_progress = None
                    for p in parts:
                        if "%|" in p:
                            last_progress = p
                        elif p.strip():
                            new_parts.append(p)
                    
                    if last_progress:
                        # Attach progress bar to header if it ends in colon, else separate line
                        if new_parts and new_parts[-1].strip().endswith(":"):
                            new_parts[-1] = new_parts[-1].rstrip() + " " + last_progress.lstrip()
                        else:
                            new_parts.append(last_progress)
                    
                    # If this line segment originally started with \r or has multiple parts,
                    # we use newlines to separate the results for the GUI.
                    sanitized_lines.append("\n".join(new_parts))
                else:
                    sanitized_lines.append(line)
            text = "\n".join(sanitized_lines)
            
        self.textbox.insert("end", text)
        
        # Detect Phase
        if "Writing audio" in text:
            self.render_phase = "audio"
        if "Writing video" in text or "MoviePy" in text: # MoviePy start
            if "Writing video" in text:
                self.render_phase = "video"
        
        # Parse progress from text (e.g. " 86%|")
        try:
            matches = re.findall(r"(\d+)%\|", text)
            if matches:
                last_pct = int(matches[-1])
                
                final_val = 0.0
                if self.render_phase == "audio":
                    # Audio 0-100% -> UI 0-5% (Fast)
                    final_val = (last_pct / 100.0) * 0.05
                elif self.render_phase == "video":
                    # Video 0-100% -> UI 5-100% (Slow, main part)
                    final_val = 0.05 + ((last_pct / 100.0) * 0.95)
                else:
                    # Fallback
                    final_val = last_pct / 100.0
                
                self.progressbar.set(final_val)
        except:
            pass
        
        # Limit history to ~1000 lines to prevent lag
        try:
            num_lines = int(self.textbox.index('end-1c').split('.')[0])
            if num_lines > 1000:
                self.textbox.delete("1.0", f"{num_lines - 800}.0") # Keep last ~800 lines
        except:
            pass
            
        if was_at_bottom:
            self.textbox.see("end")
        
        self.textbox.configure(state="disabled")

    def load_icons(self):
        def load_icon(name, size=(20, 20)):
            try:
                path = resource_path(os.path.join("assets", name))
                return ctk.CTkImage(light_image=Image.open(path), dark_image=Image.open(path), size=size)
            except:
                return None

        self.icon_user = load_icon("icon_user.png")
        self.icon_folder = load_icon("icon_folder.png")
        self.icon_play = load_icon("icon_play.png")
        self.icon_browse_folder = load_icon("icon_browse_folder.png")
        self.icon_browse_photo = load_icon("icon_browse_photo.png")
        self.icon_video_output = load_icon("icon_folder.png")
        self.icon_story = load_icon("icon_projector.png")
        self.icon_video = load_icon("icon_video.png")
        self.icon_refresh = load_icon("icon_refresh.png")
        self.icon_search = load_icon("icon_search.png")
        self.icon_nav_scan = load_icon("icon_nav_scan.png", size=(25, 25))
        self.icon_nav_edit = load_icon("icon_nav_edit.png", size=(25, 25))
        self.icon_nav_info = load_icon("icon_nav_info.png", size=(25, 25))

    def create_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. サイドナビゲーション
        self.sidebar_frame = ctk.CTkFrame(self, width=160, corner_radius=0, fg_color=self.COLOR_DEEP_BG)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1) # Spacer

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="OMO\nKAGE", 
                                       text_color=self.COLOR_ACCENT,
                                       font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=25)

        self.btn_nav_scan = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=45, border_spacing=10, text="スキャン",
                                          fg_color="transparent", text_color=self.COLOR_TEXT, hover_color=self.COLOR_SIDEBAR,
                                          image=self.icon_nav_scan, anchor="w", font=ctk.CTkFont(size=14, weight="bold"),
                                          command=lambda: self.select_frame_by_name("scan"))
        self.btn_nav_scan.grid(row=1, column=0, sticky="ew")

        self.btn_nav_edit = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=45, border_spacing=10, text="編集 / 作成",
                                          fg_color="transparent", text_color=self.COLOR_TEXT, hover_color=self.COLOR_SIDEBAR,
                                          image=self.icon_nav_edit, anchor="w", font=ctk.CTkFont(size=14, weight="bold"),
                                          command=lambda: self.select_frame_by_name("edit"))
        self.btn_nav_edit.grid(row=2, column=0, sticky="ew")

        self.btn_nav_settings = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=45, border_spacing=10, text="管理設定",
                                              fg_color="transparent", text_color=self.COLOR_TEXT, hover_color=self.COLOR_SIDEBAR,
                                              anchor="w", font=ctk.CTkFont(size=14, weight="bold"),
                                              command=lambda: self.select_frame_by_name("settings"))
        self.btn_nav_settings.grid(row=3, column=0, sticky="ew")

        self.btn_nav_about = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=45, border_spacing=10, text="情報",
                                           fg_color="transparent", text_color=self.COLOR_TEXT, hover_color=self.COLOR_SIDEBAR,
                                           anchor="w", font=ctk.CTkFont(size=14, weight="bold"),
                                           command=lambda: self.select_frame_by_name("about"))
        self.btn_nav_about.grid(row=4, column=0, sticky="ew")

        self.btn_open_out = ctk.CTkButton(self.sidebar_frame, text="保存先を開く", image=self.icon_video_output, compound="left",
                                          fg_color=self.COLOR_SIDEBAR, hover_color=self.COLOR_ACCENT, 
                                          text_color=self.COLOR_TEXT, font=ctk.CTkFont(size=12, weight="bold"),
                                          command=self.open_result)
        self.btn_open_out.grid(row=5, column=0, padx=20, pady=20, sticky="ew")

        # 2. メインコンテナ (Page Switcher)
        self.container_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.container_frame.grid(row=0, column=1, sticky="nsew")
        self.container_frame.grid_columnconfigure(0, weight=1)
        self.container_frame.grid_rowconfigure(0, weight=1)
        self.container_frame.grid_rowconfigure(1, weight=0) # Log Area

        # --- A. SCAN PAGE (Material Preparation) ---
        self.scan_frame = ctk.CTkFrame(self.container_frame, fg_color=self.COLOR_DEEP_BG)
        self.scan_frame.grid_columnconfigure(0, weight=1)
        self.scan_frame.grid_columnconfigure(1, weight=4)
        self.scan_frame.grid_rowconfigure(0, weight=1)
        
        # 1. PEOPLE MANAGEMENT SECTION
        self.people_section = ctk.CTkFrame(self.scan_frame, corner_radius=15, fg_color=self.COLOR_SIDEBAR)
        self.people_section.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=10)
        ctk.CTkLabel(self.people_section, text="👤 人物管理", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=15, weight="bold")).pack(pady=10)

        # Profiles List
        self.list_cnt = ctk.CTkFrame(self.people_section, fg_color="transparent")
        self.list_cnt.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.list_header = ctk.CTkFrame(self.list_cnt, fg_color="transparent")
        self.list_header.pack(fill="x", pady=2)
        ctk.CTkLabel(self.list_header, text="登録済みリスト", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        
        self.profile_scroll = ctk.CTkScrollableFrame(self.list_cnt, height=300, fg_color=self.COLOR_DEEP_BG,
                                                     scrollbar_button_color=self.COLOR_ACCENT,
                                                     scrollbar_button_hover_color=self.COLOR_HOVER)
        self.profile_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Registration Button (Bottom)
        self.reg_cnt = ctk.CTkFrame(self.people_section, fg_color="transparent")
        self.reg_cnt.pack(fill="x", pady=(5, 15), padx=10)
        
        self.btn_start_reg = ctk.CTkButton(self.reg_cnt, text="新しい人物を登録", command=self.start_sequential_registration, 
                                          image=self.icon_user, compound="left", fg_color=self.COLOR_ACCENT, 
                                          hover_color=self.COLOR_HOVER, height=35,
                                          text_color="black", font=ctk.CTkFont(size=12, weight="bold"))
        self.btn_start_reg.pack(pady=5, padx=10, fill="x")

        # 2. VIDEO ANALYSIS SECTION
        self.video_area = ctk.CTkFrame(self.scan_frame, corner_radius=15, fg_color=self.COLOR_SIDEBAR)
        self.video_area.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=10)
        ctk.CTkLabel(self.video_area, text="🎥 動画分析", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=15, weight="bold")).pack(pady=10)

        # Scanned List
        self.scanned_cnt = ctk.CTkFrame(self.video_area, fg_color="transparent")
        self.scanned_cnt.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.scanned_header = ctk.CTkFrame(self.scanned_cnt, fg_color="transparent")
        self.scanned_header.pack(fill="x", pady=2)
        ctk.CTkLabel(self.scanned_header, text="スキャン済み動画", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)

        self.scanned_scroll = ctk.CTkScrollableFrame(self.scanned_cnt, height=300, fg_color=self.COLOR_DEEP_BG,
                                                     scrollbar_button_color=self.COLOR_ACCENT,
                                                     scrollbar_button_hover_color=self.COLOR_HOVER)
        self.scanned_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Bulk Action Bar (常駐フッター、選択中のみ中身を表示)
        self.bulk_bar = ctk.CTkFrame(self.scanned_cnt, height=45, fg_color=self.COLOR_SIDEBAR)
        self.bulk_bar.pack(side="bottom", fill="x", padx=5, pady=(0, 5))
        self.bulk_bar.pack_propagate(False) # 高さを固定してガタつきを防止

        # Analysis Controls (Bottom)
        self.right_ctrl_cnt = ctk.CTkFrame(self.video_area, fg_color="transparent")
        self.right_ctrl_cnt.pack(fill="x", pady=(5, 15), padx=10)
        
        self.cb_force = ctk.CTkCheckBox(self.right_ctrl_cnt, text="スキャン済みも再度スキャンして上書き", 
                                        variable=self.force_rescan, font=ctk.CTkFont(size=11),
                                        fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color=self.COLOR_TEXT)
        self.cb_force.pack(pady=5, padx=15, anchor="w")

        self.btn_scan_run = ctk.CTkButton(self.right_ctrl_cnt, text="新規スキャン開始", image=self.icon_search, 
                                          compound="left", command=self.start_sequential_scan, height=35,
                                          font=ctk.CTkFont(size=12, weight="bold"), 
                                          fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black")
        self.btn_scan_run.pack(pady=5, padx=10, fill="x")



        # --- B. EDIT PAGE ---
        self.edit_frame = ctk.CTkFrame(self.container_frame, fg_color=self.COLOR_DEEP_BG)
        self.edit_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="edit_cols")
        self.edit_frame.grid_rowconfigure(0, weight=1)

        # 0. AI BGM事前生成
        self.bgm_section = ctk.CTkFrame(self.edit_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.bgm_section.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(self.bgm_section, text="0. AI BGM 事前生成", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12, 5))
        
        # BGMジャンルボタン (2x2 Grid)
        self.bgm_grid = ctk.CTkFrame(self.bgm_section, fg_color="transparent")
        self.bgm_grid.pack(pady=5, padx=10, fill="x")
        self.bgm_grid.grid_columnconfigure((0, 1), weight=1)
        
        self.btn_gen_calm = ctk.CTkButton(self.bgm_grid, text="穏やか", height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                          fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                          command=lambda: self.generate_single_bgm("穏やか"))
        self.btn_gen_calm.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        
        self.btn_gen_energetic = ctk.CTkButton(self.bgm_grid, text="元気", height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                                fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                                command=lambda: self.generate_single_bgm("エネルギッシュ"))
        self.btn_gen_energetic.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        
        self.btn_gen_emotional = ctk.CTkButton(self.bgm_grid, text="感動", height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                                fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                                command=lambda: self.generate_single_bgm("感動的"))
        self.btn_gen_emotional.grid(row=1, column=0, padx=3, pady=3, sticky="ew")

        self.btn_gen_cute = ctk.CTkButton(self.bgm_grid, text="かわいい", height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                          fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                          command=lambda: self.generate_single_bgm("かわいい"))
        self.btn_gen_cute.grid(row=1, column=1, padx=3, pady=3, sticky="ew")

        ctk.CTkLabel(self.bgm_section, text="生成済みBGM:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20, pady=(5, 0))
        self.bgm_list_frame = ctk.CTkScrollableFrame(self.bgm_section, height=140, fg_color=self.COLOR_DEEP_BG, 
                                                    scrollbar_button_color=self.COLOR_ACCENT,
                                                    scrollbar_button_hover_color=self.COLOR_HOVER)
        self.bgm_list_frame.pack(pady=(5, 12), padx=20, fill="both", expand=True)

        # 1 & 2. 設定 (中央列)
        self.settings_col = ctk.CTkFrame(self.edit_frame, fg_color="transparent")
        self.settings_col.grid(row=0, column=1, sticky="nsew")
        self.settings_col.grid_columnconfigure(0, weight=1)
        self.settings_col.grid_rowconfigure(0, weight=0) # Target
        self.settings_col.grid_rowconfigure(1, weight=1) # Settings
        
        # 1. 対象を選択
        self.target_section = ctk.CTkFrame(self.settings_col, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.target_section.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        ctk.CTkLabel(self.target_section, text="1. 対象を選択", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12, 5))
        
        self.target_person = ctk.StringVar(value="選択してください...")
        self.menu_target = ctk.CTkOptionMenu(self.target_section, variable=self.target_person, values=["選択してください..."], 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          height=30, font=ctk.CTkFont(size=12))
        self.menu_target.pack(pady=(5, 12), padx=20, fill="x")

        # 2. 編集設定
        self.set_section = ctk.CTkFrame(self.settings_col, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.set_section.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        ctk.CTkLabel(self.set_section, text="2. 編集設定", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12, 5))
        
        self.sw_blur = ctk.CTkSwitch(self.set_section, text="顔ぼかし (対象以外)", variable=self.blur_enabled, 
                                     progress_color=self.COLOR_ACCENT, command=self.save_config)
        self.sw_blur.pack(pady=5)
        

        ctk.CTkLabel(self.set_section, text="期間:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)
        self.menu_period = ctk.CTkOptionMenu(self.set_section, values=["All Time"], variable=self.selected_period, 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          height=30, font=ctk.CTkFont(size=12))
        self.menu_period.pack(pady=(2, 5), padx=20, fill="x")

        ctk.CTkLabel(self.set_section, text="重視:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)
        self.menu_focus = ctk.CTkOptionMenu(self.set_section, values=["バランス", "笑顔", "動き", "感動"], variable=self.selected_focus, 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          height=30, font=ctk.CTkFont(size=12))
        self.menu_focus.pack(pady=(2, 5), padx=20, fill="x")

        self.sw_bgm = ctk.CTkSwitch(self.set_section, text="BGMを合成", variable=self.bgm_enabled, 
                                     progress_color=self.COLOR_ACCENT, command=self.save_config)
        self.sw_bgm.pack(pady=(5, 0))
        
        ctk.CTkLabel(self.set_section, text="※ドキュメンタリー機能のみ有効", 
                     text_color="gray75", font=ctk.CTkFont(size=10)).pack(pady=(0, 12))

        # 3. 生成アクション (右列)
        self.gen_section = ctk.CTkFrame(self.edit_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.gen_section.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(self.gen_section, text="3. 動画生成", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12, 5))
        
        self.btn_gen_digest = ctk.CTkButton(self.gen_section, text="月間ダイジェスト", image=self.icon_video, 
                                            compound="left", command=self.start_digest_only, height=35, font=ctk.CTkFont(size=12, weight="bold"),
                                            fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black")
        self.btn_gen_digest.pack(pady=10, padx=20, fill="x")
        
        self.btn_gen_story = ctk.CTkButton(self.gen_section, text="1分ドキュメンタリー", image=self.icon_story, 
                                           compound="left", command=self.generate_documentary, height=35, font=ctk.CTkFont(size=12, weight="bold"),
                                           fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black")
        self.btn_gen_story.pack(pady=5, padx=20, fill="x")

        # 3. STATUS & PROGRESS AREA (Bottom)
        self.status_frame = ctk.CTkFrame(self.container_frame, height=200, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        # Progress Bar logic
        self.progressbar = ctk.CTkProgressBar(self.status_frame, orientation="horizontal", progress_color=self.COLOR_ACCENT)
        self.progressbar.grid(row=0, column=0, padx=20, pady=(15, 0), sticky="ew")
        self.progressbar.set(0)
        
        self.textbox = ctk.CTkTextbox(self.status_frame, height=120, font=ctk.CTkFont(size=11), 
                                      fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                      scrollbar_button_color=self.COLOR_ACCENT,
                                      scrollbar_button_hover_color=self.COLOR_HOVER)
        self.textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.textbox.configure(state="disabled")
        
        # --- C. ABOUT PAGE ---
        self.about_frame = ctk.CTkFrame(self.container_frame, fg_color=self.COLOR_DEEP_BG)
        self.about_frame.grid_columnconfigure(0, weight=1)
        self.about_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.about_frame, text="Omokage", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, pady=(20, 10))
        
        self.about_textbox = ctk.CTkTextbox(self.about_frame, wrap="word", font=ctk.CTkFont(size=12),
                                            fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                            scrollbar_button_color=self.COLOR_ACCENT,
                                            scrollbar_button_hover_color=self.COLOR_HOVER)
        self.about_textbox.grid(row=1, column=0, sticky="nsew", padx=40, pady=(0, 20))
        
        # LICENSE_NOTICE.mdの内容を読み込んで表示
        notice_path = resource_path("LICENSE_NOTICE.md")
        content = "ライセンス情報が見つかりません。"
        if os.path.exists(notice_path):
            try:
                with open(notice_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                pass
        self.about_textbox.insert("0.0", content)
        self.about_textbox.configure(state="disabled") # 編集不可に設定

        # --- D. SETTINGS PAGE ---
        self.settings_frame = ctk.CTkFrame(self.container_frame, fg_color=self.COLOR_DEEP_BG)
        self.settings_frame.grid_columnconfigure(0, weight=1)
        # self.settings_frame.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(self.settings_frame, text="管理設定", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, pady=(20, 10))

        # 2. 管理設定 (Hugging Face / API)
        self.admin_section = ctk.CTkFrame(self.settings_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.admin_section.grid(row=1, column=0, sticky="ew", padx=40, pady=20)
        ctk.CTkLabel(self.admin_section, text="🛠️ モデル設定 (AI BGM用)", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        hf_cnt = ctk.CTkFrame(self.admin_section, fg_color="transparent")
        hf_cnt.pack(pady=5)
        ctk.CTkLabel(hf_cnt, text="Hugging Face Token:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.entry_hf = ctk.CTkEntry(hf_cnt, textvariable=self.hf_token, width=220, 
                                     fg_color=self.COLOR_DEEP_BG, border_color=self.COLOR_ACCENT,
                                     placeholder_text="hf_...", show="*")
        self.entry_hf.pack(side="left", padx=10)
        
        self.btn_save_token = ctk.CTkButton(hf_cnt, text="保存", width=60, height=28, 
                                            fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                            font=ctk.CTkFont(size=11, weight="bold"),
                                            command=lambda: self.save_config(notify=True))
        self.btn_save_token.pack(side="left", padx=5)

        self.btn_get_token = ctk.CTkButton(hf_cnt, text="トークン取得", width=80, fg_color="transparent", 
                                           border_width=1, text_color=("gray10", "gray90"),
                                           command=lambda: webbrowser.open("https://huggingface.co/settings/tokens"))
        self.btn_get_token.pack(side="right", padx=(0, 10))

        ctk.CTkLabel(self.admin_section, text="【Hugging Face Token の入手手順】", 
                     font=ctk.CTkFont(size=11, weight="bold"), text_color=self.COLOR_ACCENT).pack(pady=(10, 5))
        
        # Instructions Container (Centered block, content left-aligned)
        instr_cnt = ctk.CTkFrame(self.admin_section, fg_color="transparent")
        instr_cnt.pack(pady=(0, 10))

        # Step 1
        s1_row = ctk.CTkFrame(instr_cnt, fg_color="transparent")
        s1_row.pack(anchor="w", pady=2)
        ctk.CTkLabel(s1_row, text="1. アカウント作成 (無料): ", font=ctk.CTkFont(size=10), text_color="gray80").pack(side="left")
        btn_hf_reg = ctk.CTkButton(s1_row, text="huggingface.co", width=80, height=18, font=ctk.CTkFont(size=9),
                                   fg_color="transparent", border_width=1, border_color="gray50",
                                   command=lambda: webbrowser.open("https://huggingface.co/join"))
        btn_hf_reg.pack(side="left", padx=5)

        # Step 2
        s2_row = ctk.CTkFrame(instr_cnt, fg_color="transparent")
        s2_row.pack(anchor="w", pady=2)
        ctk.CTkLabel(s2_row, text="2. 利用規約への同意: ", font=ctk.CTkFont(size=10), text_color="gray80").pack(side="left")
        btn_hf_model = ctk.CTkButton(s2_row, text="モデルページを開く", width=100, height=18, font=ctk.CTkFont(size=9),
                                     fg_color="transparent", border_width=1, border_color="gray50",
                                     command=lambda: webbrowser.open("https://huggingface.co/stabilityai/stable-audio-open-1.0"))
        btn_hf_model.pack(side="left", padx=5)
        
        ctk.CTkLabel(instr_cnt, text="   └ 「Agree and access repository」ボタンをクリック", 
                     font=ctk.CTkFont(size=10), text_color="gray60", anchor="w").pack(anchor="w")

        # Step 3
        s3_row = ctk.CTkFrame(instr_cnt, fg_color="transparent")
        s3_row.pack(anchor="w", pady=2)
        ctk.CTkLabel(s3_row, text="3. トークンの作成: ", font=ctk.CTkFont(size=10), text_color="gray80").pack(side="left")
        btn_hf_token = ctk.CTkButton(s3_row, text="Settings > Access Tokens", width=130, height=18, font=ctk.CTkFont(size=9),
                                     fg_color="transparent", border_width=1, border_color="gray50",
                                     command=lambda: webbrowser.open("https://huggingface.co/settings/tokens"))
        btn_hf_token.pack(side="left", padx=5)
        
        ctk.CTkLabel(instr_cnt, text="   └ 「Create new token」から「Type: Read」で作成してコピー", 
                     font=ctk.CTkFont(size=10), text_color="gray60", anchor="w").pack(anchor="w")

        # --- 共通ログ & プログレス (メインの下部に配置) ---
        self.bottom_frame = ctk.CTkFrame(self.container_frame, height=300, fg_color=self.COLOR_DEEP_BG, corner_radius=0)
        self.bottom_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        self.textbox = ctk.CTkTextbox(self.bottom_frame, height=225, font=ctk.CTkFont(family="Consolas", size=12),
                                      scrollbar_button_color=self.COLOR_ACCENT,
                                      scrollbar_button_hover_color=self.COLOR_HOVER)
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.progressbar = ctk.CTkProgressBar(self.bottom_frame)
        self.progressbar.pack(fill="x", padx=10, pady=(0, 10))
        self.progressbar.set(0)

        # 初期表示
        self.select_frame_by_name("scan")
        self.refresh_scanned_files()
        self.refresh_bgm_list()

    def select_frame_by_name(self, name):
        # ボタンの色リセット
        self.btn_nav_scan.configure(fg_color=("gray75", "gray25") if name == "scan" else "transparent")
        self.btn_nav_edit.configure(fg_color=("gray75", "gray25") if name == "edit" else "transparent")
        self.btn_nav_settings.configure(fg_color=("gray75", "gray25") if name == "settings" else "transparent")
        self.btn_nav_about.configure(fg_color=("gray75", "gray25") if name == "about" else "transparent")

        # ページ切り替え
        self.scan_frame.grid_forget()
        self.edit_frame.grid_forget()
        self.settings_frame.grid_forget()
        self.about_frame.grid_forget()

        if name == "scan":
            self.scan_frame.grid(row=0, column=0, sticky="nsew")
        elif name == "edit":
            self.edit_frame.grid(row=0, column=0, sticky="nsew")
            self.update_target_menu()
            self.update_period_menu()
        elif name == "settings":
            self.settings_frame.grid(row=0, column=0, sticky="nsew")
        elif name == "about":
            self.about_frame.grid(row=0, column=0, sticky="nsew")

    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self, notify=False):
        config = {
            "target_path": self.target_image_path.get(),
            "video_folder": self.video_folder_path.get(),
            "blur_enabled": self.blur_enabled.get(),
            "color_filter": self.color_filter.get(),
            "bgm_enabled": self.bgm_enabled.get(),
            "hf_token": self.hf_token.get().strip()
        }
        with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        if notify:
            messagebox.showinfo("完了", "設定を保存しました。")

    def start_sequential_registration(self):
        # Step 1: Browse Image
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return # Cancelled
        
        # Step 2: Input Name
        dialog = ctk.CTkInputDialog(text="人物の名前を入力してください:", title="人物登録")
        name = dialog.get_input()
        
        if not name or not name.strip():
            return # Cancelled or empty
        
        name = name.strip()
        
        # Step 3: Register
        self.log(f"[SYSTEM] Registering {name}...")
        success, reason = register_person(path, name, pkl_path=self.TARGET_PKL)
        
        if success:
            self.log(f"[SUCCESS] Registered {name}")
            self.refresh_profiles()
            messagebox.showinfo("完了", f"{name} を登録しました。")
        else:
            self.log(f"[ERROR] Failed to register {name}: {reason}")
            
            if reason == "NO_FACE":
                msg = f"{name} の顔が検出されませんでした。\n\nお顔が「正面」かつ「大きく（アップで）」写っている写真を選んでください。"
            elif reason == "MULTIPLE_FACES":
                msg = "複数の顔が検出されました。\n\n登録したい方「一人だけ」が写っている写真を選んでください。"
            else:
                msg = f"{name} の登録に失敗しました。\n(理由: {reason})"
                
            messagebox.showerror("登録失敗", msg)

    def delete_click(self, name):
        if messagebox.askyesno("削除", f"プロファイル '{name}' を削除してもよろしいですか？"):
            if delete_person(name, pkl_path=self.TARGET_PKL):
                self.log(f"[SYSTEM] Deleted profile: {name}")
                self.refresh_profiles()
            else:
                self.log(f"[ERROR] Failed to delete: {name}")

    def refresh_profiles(self):
        for child in self.profile_scroll.winfo_children():
            child.destroy()
            
        profile_dir = os.path.join(get_app_dir(), "profiles")
        icons = glob.glob(os.path.join(profile_dir, "*.jpg"))
        for icon_path in sorted(icons):
            p_name = os.path.splitext(os.path.basename(icon_path))[0]
            frame = ctk.CTkFrame(self.profile_scroll, fg_color="transparent")
            frame.pack(fill="x", pady=2)
            
            try:
                img = ctk.CTkImage(light_image=Image.open(icon_path), size=(30, 30))
                lbl_img = ctk.CTkLabel(frame, image=img, text="")
                lbl_img.pack(side="left", padx=5)
            except: pass
            
            lbl_name = ctk.CTkLabel(frame, text=p_name, font=ctk.CTkFont(size=11), anchor="w")
            lbl_name.pack(side="left", padx=5, expand=True, fill="x")

            btn_del = ctk.CTkButton(frame, text="×", width=24, height=24, 
                                    fg_color="#5D6D7E", hover_color="#A93226",
                                    text_color="white", font=ctk.CTkFont(size=14, weight="bold"),
                                    anchor="center",
                                    command=lambda n=p_name: self.delete_click(n))
            btn_del.pack(side="right", padx=5)

    def start_sequential_scan(self):
        if self.is_running: return

        # Step 1: Browse Folder
        folder = filedialog.askdirectory()
        if not folder:
            return # Cancelled
        
        self.video_folder_path.set(folder) # Updates config via trace if I kept trace, but I removed UI trace. 
        # I removed trace UI, so save manually or rely on variable set if logic uses it.
        # Original logic used self.video_folder_path.get()
        self.save_config() 

        self.btn_scan_run.configure(text="中断 (Stop)", fg_color="#CD6155", hover_color="#A93226", command=self.stop_scan)
        self.is_running = True
        self.scan_stop_event.clear()
        self.log(f">>> 動画スキャンを開始: {folder}")
        
        def run():
            try:
                # Direct call instead of subprocess
                current_stdout = sys.stdout
                current_stderr = sys.stderr
                sys.stdout = RedirectText(lambda s: self.log(s, end=""))
                sys.stderr = RedirectText(lambda s: self.log(s, end=""))
                
                try:
                    scan_videos.run_scan(folder, target_pkl=self.TARGET_PKL, output_json=self.SCAN_RESULTS, force=self.force_rescan.get(), stop_event=self.scan_stop_event)
                finally:
                    sys.stdout = current_stdout
                    sys.stderr = current_stderr

                if self.scan_stop_event.is_set():
                    self.log("\n>>> SCAN CANCELLED!")
                    self.log("__NOTIFY__", title="中断", message="動画スキャンを中断しました。")
                else:
                    self.log("\n>>> SCAN COMPLETE!")
                    self.log("__NOTIFY__", title="完了", message="動画スキャンが完了しました。")
                    
                self.after(0, self.refresh_scanned_files)
                self.after(0, self.update_period_menu)
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
            finally:
                self.cached_scan_data = None # Invalidate cache
                self.after(0, self.reset_scan_ui)
        # update dynamic menu
        threading.Thread(target=run, daemon=True).start()

    def stop_scan(self):
        if not self.is_running: return
        self.log(">>> 中断リクエストを送信しました。現在の動画の処理が終わり次第停止します...")
        self.scan_stop_event.set()
        self.btn_scan_run.configure(state="disabled", text="中断中...")

    def refresh_scanned_files(self, show_all=False):
        # Clear and show loading state
        for child in self.scanned_scroll.winfo_children():
            child.destroy()
        
        loading_lbl = ctk.CTkLabel(self.scanned_scroll, text="⌛ Loading database...", text_color="gray")
        loading_lbl.pack(pady=20)
        
        def bg_load():
            if not os.path.exists(self.SCAN_RESULTS):
                self.after(0, lambda: self._finalize_refresh_ui(None, [], loading_lbl))
                return

            try:
                with open(self.SCAN_RESULTS, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                scanned_paths = sorted(data.get("metadata", {}).keys(), reverse=True)
                if not scanned_paths:
                    self.after(0, lambda: self._finalize_refresh_ui(None, [], loading_lbl))
                    return

                # Pre-calculate mapping in background
                video_map = {path: {"people": [], "vibes": set(), "descs": set(), "date": None} for path in sorted(data.get("metadata", {}).keys())}
                people_map = {} # {name: {count: 0, last_seen: str, vibes: set()}}

                for p_name, p_videos in data.get("people", {}).items():
                    # Initialize person stats
                    if p_name not in people_map:
                        people_map[p_name] = {"count": 0, "last_seen": "0000-00-00", "vibes": set(), "dists": []}
                    
                    people_map[p_name]["count"] = len(p_videos)
                    
                    for v_path, detections in p_videos.items():
                        # Video Map Update
                        if v_path in video_map:
                            video_map[v_path]["people"].append(p_name)
                            for d in detections:
                                if d.get("vibe"): 
                                    video_map[v_path]["vibes"].add(d["vibe"])
                                    people_map[p_name]["vibes"].add(d["vibe"])
                                if d.get("dist") is not None:
                                    people_map[p_name]["dists"].append(d["dist"])
                                if d.get("description"): video_map[v_path]["descs"].add(d["description"])
                                if not video_map[v_path]["date"] and d.get("timestamp"):
                                    video_map[v_path]["date"] = d["timestamp"]
                                    
                                # Update Person Last Seen
                                ts = d.get("timestamp", "")
                                if ts and ts > people_map[p_name]["last_seen"]:
                                    people_map[p_name]["last_seen"] = ts

                # Metadata fallback and ensure every entry has a date for sorting
                all_paths = list(video_map.keys())
                for path in all_paths:
                    if not video_map[path]["date"]:
                        meta = data.get("metadata", {}).get(path, {})
                        video_map[path]["date"] = meta.get("date", meta.get("month", "Unknown"))

                # Sort paths by the actual date field (ascending)
                scanned_paths = sorted(all_paths, key=lambda p: str(video_map[p]["date"]), reverse=True)

                # Convert sets to list for easier UI handling later
                for p in video_map:
                    video_map[p]["vibes"] = list(video_map[p]["vibes"])
                    video_map[p]["descs"] = list(video_map[p]["descs"])
                
                for p in people_map:
                    people_map[p]["vibes"] = list(people_map[p]["vibes"])

                self.after(0, lambda: self._finalize_refresh_ui(data, scanned_paths, loading_lbl, video_map, people_map, show_all))
            except Exception as e:
                print(f"Async Load Error: {e}")
                self.after(0, lambda: self._finalize_refresh_ui(None, [], loading_lbl))

        threading.Thread(target=bg_load, daemon=True).start()

    def _finalize_refresh_ui(self, data, scanned_paths, loading_lbl, video_map=None, people_map=None, show_all=False):
        try:
            loading_lbl.destroy()
            
            if not data:
                ctk.CTkLabel(self.scanned_scroll, text="(No results yet)", text_color="gray").pack(pady=20)
                return

            # Single grid container
            table_container = ctk.CTkFrame(self.scanned_scroll, fg_color=self.COLOR_DEEP_BG)
            table_container.pack(fill="x", expand=True, padx=5)
            
            # --- PEOPLE VIEW (Default and Only) ---
            # Configure Grid
            table_container.grid_columnconfigure(0, weight=2)
            table_container.grid_columnconfigure(1, weight=1)
            table_container.grid_columnconfigure(2, weight=1)
            table_container.grid_columnconfigure(3, weight=2)

            # Header
            # テーブルヘッダー (アンバーのアクセント)
            header_bg = ctk.CTkFrame(table_container, fg_color=self.COLOR_DEEP_BG, height=30, corner_radius=5)
            header_bg.grid(row=0, column=0, columnspan=4, sticky="ew", padx=2, pady=2)
            
            headers = ["NAME", "VIDEOS", "LAST SEEN", "CONFIDENCE"]
            for i, h in enumerate(headers):
                lbl = ctk.CTkLabel(table_container, text=h, font=ctk.CTkFont(size=11, weight="bold"), anchor="w")
                lbl.grid(row=0, column=i, padx=10, pady=5, sticky="w")

            # Data Rows
            sorted_people = sorted(people_map.items(), key=lambda x: x[1]["last_seen"], reverse=True)
            if not sorted_people:
                ctk.CTkLabel(table_container, text="No people found in scan results.", text_color="gray").grid(row=1, column=0, columnspan=4, pady=20)
                return

            for i, (p_name, stats) in enumerate(sorted_people):
                grid_row_idx = i + 1
                
                # Check profile image
                icon_path = os.path.join(get_app_dir(), "profiles", f"{p_name}.jpg")
                icon_img = None
                if os.path.exists(icon_path):
                    try:
                        pil_img = Image.open(icon_path)
                        icon_img = ctk.CTkImage(light_image=pil_img, size=(24, 24))
                    except: pass
                
                name_frame = ctk.CTkFrame(table_container, fg_color="transparent")
                name_frame.grid(row=grid_row_idx, column=0, padx=10, pady=5, sticky="w")
                
                if icon_img:
                    ctk.CTkLabel(name_frame, text="", image=icon_img).pack(side="left", padx=(0, 5))
                ctk.CTkLabel(name_frame, text=p_name, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")

                # Column 1: VideoCount
                count_str = f"{stats['count']} videos"
                ctk.CTkLabel(table_container, text=count_str, font=ctk.CTkFont(size=11), text_color="gray70").grid(row=grid_row_idx, column=1, padx=10, pady=5, sticky="w")

                # Column 2: LastSeen
                last_seen_str = stats['last_seen'].split(" ")[0] if stats['last_seen'] != "0000-00-00" else "-"
                ctk.CTkLabel(table_container, text=last_seen_str, font=ctk.CTkFont(size=11), text_color="gray70").grid(row=grid_row_idx, column=2, padx=10, pady=5, sticky="w")

                # Column 1: VideoCount
                count_str = f"{stats['count']} videos"
                ctk.CTkLabel(table_container, text=count_str, font=ctk.CTkFont(size=11), text_color="gray70").grid(row=grid_row_idx, column=1, padx=10, pady=5, sticky="w")

                # Column 2: LastSeen
                last_seen_str = stats['last_seen'].split(" ")[0] if stats['last_seen'] != "0000-00-00" else "-"
                ctk.CTkLabel(table_container, text=last_seen_str, font=ctk.CTkFont(size=11), text_color="gray70").grid(row=grid_row_idx, column=2, padx=10, pady=5, sticky="w")

                # Recognition Rate (Confidence) & Detail Button
                conf_cnt = ctk.CTkFrame(table_container, fg_color="transparent")
                conf_cnt.grid(row=grid_row_idx, column=3, padx=10, pady=5, sticky="ew")

                avg_dist = sum(stats["dists"]) / len(stats["dists"]) if stats["dists"] else None
                conf_str = f"{int((1.0 - avg_dist) * 100)}%" if avg_dist is not None else "-"
                ctk.CTkLabel(conf_cnt, text=conf_str, font=ctk.CTkFont(size=11), anchor="w",
                             text_color=self.COLOR_ACCENT if conf_str != "-" else "gray").pack(side="left")

                # Detail Button (Magnifying glass / List)
                btn_detail = ctk.CTkButton(conf_cnt, text="🔍", width=30, height=24, 
                                           fg_color=self.COLOR_SIDEBAR, hover_color="#34495E",
                                           command=lambda n=p_name: self.show_person_clips(n))
                btn_detail.pack(side="right", padx=5)

                # Separator
                sep = ctk.CTkFrame(table_container, height=1, fg_color=self.COLOR_SIDEBAR, corner_radius=0)
                sep.grid(row=grid_row_idx, column=0, columnspan=4, sticky="sew")


        except Exception as e:
            self.log(f"[ERROR] Loading table: {e}")

    def on_folder_change(self, *args):
        folder = self.video_folder_path.get()
        self.lbl_folder.configure(text=f"動画フォルダ: {folder if folder else '(未選択)'}")
        self.save_config()

    def log(self, msg_val, *args, **kwargs):
        # バックグラウンドスレッドから安全にUI更新するためにキューを使用
        self.log_queue.put((msg_val, kwargs))

    def update_target_menu(self):
        profile_dir = os.path.join(get_app_dir(), "profiles")
        icons = glob.glob(os.path.join(profile_dir, "*.jpg"))
        names = sorted([os.path.splitext(os.path.basename(i))[0] for i in icons])
        if not names:
            names = ["プロフィールが見つかりません"]
        
        # Avoid redundant configuration
        current_values = self.menu_target.cget("values")
        if list(current_values) == names:
            return

        self.menu_target.configure(values=names)
        if self.target_person.get() not in names:
            self.target_person.set(names[0])

    def update_period_menu(self):
        data = self.load_scan_results() # Use cache-aware loader
        if not data:
            return
        try:
            metadata = data.get("metadata", {})
            months = set()
            years = set()
            for path, meta in metadata.items():
                m = meta.get("month")
                if m:
                    months.add(m)
                    years.add(m.split("-")[0])
            
            p_list = ["All Time"] + sorted(list(years), reverse=True) + sorted(list(months), reverse=True)
            
            # Avoid redundant configuration
            current_values = self.menu_period.cget("values")
            if list(current_values) == p_list:
                return

            self.menu_period.configure(values=p_list)
            if self.selected_period.get() not in p_list:
                self.selected_period.set("All Time")
        except Exception as e:
            self.log(f"[ERROR] Updating period menu: {e}")

    def reset_scan_ui(self):
        self.btn_scan_run.configure(state="normal", text="新規スキャン開始", 
                                    fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER,
                                    command=self.start_sequential_scan)
        self.is_running = False

    def start_digest_only(self):
        person = self.target_person.get()
        if person == "選択してください..." or person == "プロフィールが見つかりません":
            self.log("__NOTIFY__", title="警告", message="人物を選択してください。", type="warning")
            return
            
        # 期間のバリデーション (月が選択されているか)
        period = self.selected_period.get()
        if len(period) != 7 or "-" not in period: # YYYY-MM 形式でなければエラー
            self.log("__NOTIFY__", title="エラー", message="月間ダイジェストを作成するには、特定の「月（YYYY-MM）」を選択してください。\n現在は「年」全体が選択されています。", type="error")
            return

        if self.is_running: return

        self.btn_gen_digest.configure(state="disabled", text="生成中...")
        self.is_running = True
        self.log(f">>> {person} のダイジェストを作成中...")
        
        def run():
            try:
                blur_flag = "--blur" if self.blur_enabled.get() else "--no-blur"
                # Direct call instead of subprocess
                current_stdout = sys.stdout
                current_stderr = sys.stderr
                sys.stdout = RedirectText(lambda s: self.log(s, end=""))
                sys.stderr = RedirectText(lambda s: self.log(s, end=""))
                
                try:
                    create_digest.create_digest(
                        self.SCAN_RESULTS, 
                        target_person_name=person,
                        base_output_dir=self.OUTPUT_DIR,
                        blur_enabled=self.blur_enabled.get(),
                        period=self.selected_period.get(),
                        focus=self.selected_focus.get()
                    )
                finally:
                    sys.stdout = current_stdout
                    sys.stderr = current_stderr
                    
                self.log(f">>> {person} のダイジェストを作成完了！")
                self.log("__NOTIFY__", title="完了", message="ダイジェストを保存しました。")
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
            finally:
                self.after(0, self.reset_edit_ui)
        threading.Thread(target=run).start()

    def generate_documentary(self):
        person_name = self.target_person.get()
        if person_name == "選択してください..." or person_name == "プロフィールが見つかりません":
            self.log("__NOTIFY__", title="警告", message="人物を選択してください。", type="warning")
            return
        
        # Check HF Token if BGM is enabled
        if self.bgm_enabled.get():
            if not self.hf_token.get().strip():
                self.log("__NOTIFY__", title="警告", 
                         message="BGM生成にはHugging Faceトークンが必要です。\n設定ページでトークンを入力してください。", type="warning")
                self.select_frame_by_name("about")
                return
            
        if self.is_running: return

        self.btn_gen_story.configure(state="disabled", text="生成中...")
        self.is_running = True
        self.log(f">>> {person_name} のドキュメンタリーを作成中...")
        
        def run():
            try:
                # Direct call instead of subprocess
                current_stdout = sys.stdout
                current_stderr = sys.stderr
                sys.stdout = RedirectText(lambda s: self.log(s, end=""))
                sys.stderr = RedirectText(lambda s: self.log(s, end=""))
                
                try:
                    self.log("--- ステップ 1/2: ストーリーを構成中 ---")
                    create_story.create_story(
                        person_name, 
                        period=self.selected_period.get(), 
                        focus=self.selected_focus.get(), 
                        bgm_enabled=self.bgm_enabled.get(),
                        json_path=self.SCAN_RESULTS,
                        output_playlist_path=self.STORY_PLAYLIST
                    )
                    
                    self.log("\n--- ステップ 2/2: 動画をレンダリング中 ---")
                    render_story.render_documentary(
                        playlist_path=self.STORY_PLAYLIST,
                        output_dir=self.OUTPUT_DIR,
                        blur_enabled=self.blur_enabled.get(),
                        filter_type=self.color_filter.get(),
                        bgm_enabled=self.bgm_enabled.get(),
                        focus=self.selected_focus.get()
                    )
                finally:
                    sys.stdout = current_stdout
                    sys.stderr = current_stderr

                self.log(">>> ドキュメンタリー作成完了！")
                self.log("__NOTIFY__", title="完了", message="ドキュメンタリーを保存しました。")
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
            finally:
                self.after(0, self.reset_edit_ui)

        threading.Thread(target=run).start()

    def reset_edit_ui(self):
        self.btn_gen_digest.configure(state="normal", text="月間ダイジェスト")
        self.btn_gen_story.configure(state="normal", text="1分ドキュメンタリー")
        self.is_running = False

    def open_video_file(self, path):
        """動画ファイルをデフォルトプレイヤーで開く"""
        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', path])
            elif sys.platform == 'win32':
                os.startfile(path)
            else:
                subprocess.run(['xdg-open', path])
        except Exception as e:
            self.log(f"[ERROR] Failed to open video: {e}")

    def reveal_in_finder(self, path):
        """ファイルをFinderで選択状態で表示する (macOS) またはフォルダを開く"""
        try:
            if sys.platform == 'darwin':
                # -R flag reveals the file in finder
                subprocess.run(['open', '-R', path])
            elif sys.platform == 'win32':
                # /select, allows selecting the file in explorer
                subprocess.run(['explorer', '/select,', os.path.normpath(path)])
            else:
                # Fallback to opening the directory
                subprocess.run(['xdg-open', os.path.dirname(path)])
        except Exception as e:
            self.log(f"[ERROR] Failed to reveal folder: {e}")

    def generate_single_bgm(self, vibe):
        if self.is_running:
            self.log("[WARNING] Another process is running. Please wait.")
            return

        # Check HF Token
        if not self.hf_token.get().strip():
            self.log("__NOTIFY__", title="警告", 
                     message="BGM生成にはHugging Faceトークンが必要です。\n設定ページでトークンを入力してください。", type="warning")
            self.select_frame_by_name("about")
            return
            
        self.log(f"\n>>> Starting Pre-Generation for {vibe} BGM...")
        self.log("    (This will take 2-5 minutes. Please check the log.)")
        self.is_running = True
        
        # Disable buttons
        self.btn_gen_digest.configure(state="disabled")
        self.btn_gen_story.configure(state="disabled")
        
        def run():
            try:
                # Direct call instead of subprocess
                current_stdout = sys.stdout
                current_stderr = sys.stderr
                sys.stdout = RedirectText(lambda s: self.log(s, end=""))
                sys.stderr = RedirectText(lambda s: self.log(s, end=""))
                
                try:
                    # Ensure absolute output dir exists
                    os.makedirs(self.OUTPUT_DIR, exist_ok=True)
                    bgm_output_dir = os.path.join(self.OUTPUT_DIR, "bgm")
                    
                    # Pass token and absolute output_dir to generate_bgm
                    success, _ = generate_bgm.generate_bgm(
                        vibe=vibe, 
                        duration_seconds=47, 
                        output_dir=bgm_output_dir,
                        token=self.hf_token.get().strip()
                    )
                    if not success:
                        raise Exception(f"BGM生成に失敗しました ({vibe})。\nトークンと権限を確認してください。")
                finally:
                    sys.stdout = current_stdout
                    sys.stderr = current_stderr
                
                self.log(f"\n>>> {vibe} BGM GENERATED SUCCESSFULLY!")
                self.log("__NOTIFY__", title="完了", message=f"{vibe} のBGM生成が完了しました！")
                self.after(0, self.refresh_bgm_list)
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
            finally:
                self.after(0, self.reset_edit_ui)

        threading.Thread(target=run).start()

    def refresh_bgm_list(self):
        # Clear existing
        for widget in self.bgm_list_frame.winfo_children():
            widget.destroy()
            
        bgm_dir = os.path.join(self.OUTPUT_DIR, "bgm")
        if not os.path.exists(bgm_dir):
            return
            
        try:
            files = sorted([f for f in os.listdir(bgm_dir) if f.endswith(".wav")], reverse=True)
            
            for f in files:
                f_path = os.path.join(bgm_dir, f)
                f_frame = ctk.CTkFrame(self.bgm_list_frame, fg_color="transparent")
                f_frame.pack(fill="x", pady=2)
                
                # Label for filename (truncated if needed)
                display_name = f
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
                ctk.CTkLabel(f_frame, text=display_name, font=ctk.CTkFont(size=10), text_color=self.COLOR_TEXT).pack(side="left", padx=5)
                
                # Delete button (rightmost)
                btn_del = ctk.CTkButton(f_frame, text="×", width=24, height=24, fg_color="#5D6D7E", hover_color="#A93226",
                                         text_color="white", font=ctk.CTkFont(size=14, weight="bold"),
                                         anchor="center",
                                         command=lambda name=f: self.delete_single_bgm(name))
                btn_del.pack(side="right", padx=5)

                # Play/Stop button
                is_playing = (self.playing_bgm == f_path)
                btn_play = ctk.CTkButton(f_frame, text="■" if is_playing else "▶", 
                                          width=24, height=24, 
                                          fg_color=self.COLOR_ACCENT if not is_playing else "#780001", 
                                          hover_color=self.COLOR_HOVER,
                                          text_color="black" if not is_playing else "white", 
                                          font=ctk.CTkFont(size=11, weight="bold"),
                                          command=lambda path=f_path: self.toggle_bgm_playback(path))
                btn_play.pack(side="right", padx=5)

        except Exception as e:
            self.log(f"[ERROR] Failed to refresh BGM list: {e}")

    def check_music_status(self):
        # If music finished playing naturally, reset state and refresh UI
        if self.playing_bgm and not pygame.mixer.music.get_busy():
            self.playing_bgm = None
            self.refresh_bgm_list()
        self.after(500, self.check_music_status)

    def toggle_bgm_playback(self, path):
        try:
            # Normalize paths for comparison
            target_path = os.path.normpath(os.path.abspath(path))
            cp = self.playing_bgm
            current_playing = os.path.normpath(os.path.abspath(cp)) if cp else None

            if current_playing == target_path:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                self.playing_bgm = None
            else:
                if self.playing_bgm:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                self.playing_bgm = path
            
            # Refresh list to update icons immediately
            self.refresh_bgm_list()
        except Exception as e:
            self.log(f"[ERROR] Playback failed: {e}")

    def delete_single_bgm(self, filename):
        if messagebox.askyesno("確認", f"BGM '{filename}' を削除しますか？"):
            try:
                path = os.path.join(self.OUTPUT_DIR, "bgm", filename)
                if os.path.exists(path):
                    os.remove(path)
                self.refresh_bgm_list()
                self.log(f"[INFO] Deleted BGM: {filename}")
            except Exception as e:
                self.log(f"[ERROR] Failed to delete BGM: {e}")

    def run_command(self, command, p_start=0.0, p_end=1.0):
        try:
             # バッファリングなしで実行
            if command[0] == 'python':
                command[0] = sys.executable
                command.insert(1, '-u')

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd()
            )

            for line in process.stdout:
                line_str = line.strip()
                # Check for progress patterns: "PROGRESS: XX%" or "進捗: XX%"
                if line_str.startswith("PROGRESS:") or line_str.startswith("進捗:"):
                    try:
                        # Extract value after colon
                        val_str = line_str.split(":")[1].strip()
                        # Handle potential " (X/Y本目)" suffix in scan_videos logs
                        if "(" in val_str:
                            val_str = val_str.split("(")[0].strip()
                            
                        if val_str.endswith("%"):
                            p_val = float(val_str.replace("%", "")) / 100.0
                        else:
                            p_val = float(val_str)
                            
                        overall_p = p_start + (p_val * (p_end - p_start))
                        self.after(0, lambda p=overall_p: self.progressbar.set(p))
                    except: pass
                    
                    # If it's PROGRESS:, don't log it (user requested removal)
                    # If it's "進捗:", it's the main log, so we continue to log it below
                    if line_str.startswith("PROGRESS:"):
                        continue

                self.log(line_str)
            
            process.wait()
            return process.returncode == 0
        except Exception as e:
            self.log(f"Command Error: {e}")
            return False

    def run_pipeline(self, folder):
        try:
            # Stage 1 is now handled by Profile Registration
            # self.log("[Stage 1] Extracting Faces...")

            self.log("[第2段階] 動画をスキャン中...")
            if not self.run_command(['python', 'scan_videos.py', folder], p_start=0.0, p_end=0.7):
                raise Exception("動画のスキャンに失敗しました。")

            self.log("[第3段階] ダイジェストを作成中...")
            if not self.run_command(['python', 'create_digest.py'], p_start=0.7, p_end=1.0):
                raise Exception("ダイジェストの作成に失敗しました。")

            self.log(">>> すべて完了しました！ output フォルダの final_digest.mp4 を確認してください。")
            self.log("__NOTIFY__", title="完了", message="ダイジェストの作成が完了しました！\n'output' フォルダに保存されました。")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
        finally:
            self.cached_scan_data = None # Invalidate cache
            self.after(0, self.reset_ui)

    def toggle_clip_view(self, mode):
        """リスト表示とグリッド表示を切り替える"""
        if self.clip_view_mode == mode: return
        self.clip_view_mode = mode
        
        # ボタンの見た目更新
        if hasattr(self, 'btn_list_view'):
            self.btn_list_view.configure(fg_color=self.COLOR_ACCENT if mode == "list" else self.COLOR_SIDEBAR,
                                         text_color="black" if mode == "list" else "white")
        if hasattr(self, 'btn_grid_view'):
            self.btn_grid_view.configure(fg_color=self.COLOR_ACCENT if mode == "grid" else self.COLOR_SIDEBAR,
                                         text_color="black" if mode == "grid" else "white")
        
        # 再描画
        self.selected_clips = set()
        self.update_bulk_bar()
        self.render_clips_batch()

    def update_bulk_bar(self):
        """一括操作バーの状態を更新する (ガタつき防止のため常駐・固定高さ化)"""
        if not hasattr(self, 'bulk_bar'): return
        
        # リスト表示時は表示しない (複数選択をサポートしないため)
        if self.clip_view_mode == "list":
            for child in self.bulk_bar.winfo_children():
                child.destroy()
            return
            
        count = len(self.selected_clips)
        
        # UI更新 (中身を一度クリア)
        for child in self.bulk_bar.winfo_children():
            child.destroy()
            
        if count == 0:
            # グリッド表示時のみ、選択なし時のガイドを表示
            if self.clip_view_mode == "grid":
                lbl_hint = ctk.CTkLabel(self.bulk_bar, text="サムネイルをクリックして選択可能", 
                                        font=ctk.CTkFont(size=11), text_color="gray50")
                lbl_hint.pack(pady=10)
            return
            
        # 選択個数表示
        lbl_count = ctk.CTkLabel(self.bulk_bar, text=f"{count}", font=ctk.CTkFont(size=12, weight="bold"), 
                                 text_color=self.COLOR_ACCENT, width=30)
        lbl_count.pack(side="left", padx=(15, 2), pady=0)
        ctk.CTkLabel(self.bulk_bar, text="件を選択中", font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 10))
        
        # ボタン: 解除
        btn_clear = ctk.CTkButton(self.bulk_bar, text="解除", width=50, height=26, fg_color="transparent",
                                  hover_color=self.COLOR_DEEP_BG, font=ctk.CTkFont(size=11),
                                  command=self.clear_selection)
        btn_clear.pack(side="left", padx=2)

        # ボタン: 全選択
        btn_sel_page = ctk.CTkButton(self.bulk_bar, text="全選択", width=60, height=26, 
                                     fg_color=self.COLOR_SIDEBAR, font=ctk.CTkFont(size=11),
                                     command=self.select_current_page)
        btn_sel_page.pack(side="left", padx=2)
        
        # ボタン: 一括削除 (一番右)
        btn_del = ctk.CTkButton(self.bulk_bar, text="一括削除", width=80, height=26,
                                fg_color="#A93226", hover_color="#CB4335", text_color="white",
                                font=ctk.CTkFont(size=11, weight="bold"), command=self.bulk_delete_selected)
        btn_del.pack(side="right", padx=15, pady=0)

    def on_clip_selected(self, video_path, timestamp, is_selected):
        """クリップの選択状態が変更された時の処理 (グリッド表示では高速化のため個別に更新)"""
        key = (video_path, timestamp)
        if is_selected:
            self.selected_clips.add(key)
        else:
            self.selected_clips.discard(key)
        
        # グリッド表示の場合、全描画を避け、該当タイルのみ色を変える
        if self.clip_view_mode == "grid" and key in self.grid_tile_widgets:
            tile = self.grid_tile_widgets[key]
            tile.configure(fg_color=self.COLOR_ACCENT if is_selected else self.COLOR_SIDEBAR)
            # 子要素のラベル（もしあれば）も色を反転させる必要がある場合はここで処理
            # 今回はサムネイルのみなので枠線のみでOK

        self.update_bulk_bar()

    def clear_selection(self):
        self.selected_clips = set()
        self.render_clips_batch()
        self.update_bulk_bar()

    def select_current_page(self):
        """現在表示されているページのクリップをすべて選択状態にする"""
        batch_size = 100 if self.clip_view_mode == "grid" else 20
        start_idx = self.current_clips_page * batch_size
        end_idx = start_idx + batch_size
        batch = self.all_person_clips[start_idx:end_idx]
        
        for item in batch:
            self.selected_clips.add((item['path'], item['t']))
        
        self.render_clips_batch()
        self.update_bulk_bar()

    def on_page_jump(self, page_str, total_pages):
        """ページ番号入力によるジャンプ処理"""
        try:
            target_page = int(page_str)
            if 1 <= target_page <= total_pages:
                self.current_clips_page = target_page - 1
                self.render_clips_batch()
            else:
                self.log(f"[WARNING] ページ番号は 1～{total_pages} の範囲で指定してください")
        except ValueError:
            self.log("[WARNING] 有効な数字を入力してください")

    def load_scan_results(self):
        """Cache-aware loading of scan results with backup recovery"""
        from utils import load_json_safe
        
        # 基本的なキャッシュチェック
        if os.path.exists(self.SCAN_RESULTS):
            mtime = os.path.getmtime(self.SCAN_RESULTS)
            if self.cached_scan_data and mtime == self.cached_scan_mtime:
                return self.cached_scan_data
        
        # 安全な回避策付きロード
        data = load_json_safe(self.SCAN_RESULTS, lambda: None)
        
        if data:
            self.cached_scan_data = data
            if os.path.exists(self.SCAN_RESULTS):
                self.cached_scan_mtime = os.path.getmtime(self.SCAN_RESULTS)
        
        return data

    def get_face_thumbnail(self, video_path, timestamp, face_loc):
        """指定された動画のタイムスタンプ＋座標から、お顔のサムネイルを取得/生成する (utils共通ロジックを使用)"""
        from utils import generate_face_thumbnail, get_app_dir
        return generate_face_thumbnail(video_path, timestamp, face_loc, get_app_dir())

    def show_person_clips(self, person_name, restart=True, target_y=None, target_page=None):
        """特定の人物の全ヒットクリップを表示する (Pagination対応 / スクロール復元対応)"""
        self.target_y_to_restore = target_y
        self.target_page_to_restore = target_page # 0-indexed
        self.last_person_viewed = person_name
        if restart:
            for child in self.scanned_scroll.winfo_children():
                child.destroy()
            
            # Header
            header = ctk.CTkFrame(self.scanned_scroll, fg_color="transparent")
            header.pack(fill="x", padx=10, pady=10)
            btn_back = ctk.CTkButton(header, text="← 戻る", width=80, fg_color=self.COLOR_SIDEBAR, 
                                     command=lambda: [self.selected_clips.clear(), self.update_bulk_bar(), self.refresh_scanned_files(show_all=True)])
            btn_back.pack(side="left")
            lbl_title = ctk.CTkLabel(header, text=f"{person_name} の検出カット一覧", font=ctk.CTkFont(size=14, weight="bold"))
            lbl_title.pack(side="left", padx=20)
            
            self.clips_container = ctk.CTkFrame(self.scanned_scroll, fg_color="transparent")
            self.clips_container.pack(fill="both", expand=True, padx=10)
            
            # ビュー切替ボタンをヘッダーに追加
            view_ctrl = ctk.CTkFrame(header, fg_color="transparent")
            view_ctrl.pack(side="right", padx=10)
            
            self.btn_list_view = ctk.CTkButton(view_ctrl, text="リスト", width=60, height=26,
                                              fg_color=self.COLOR_ACCENT if self.clip_view_mode == "list" else self.COLOR_SIDEBAR,
                                              text_color="black" if self.clip_view_mode == "list" else "white",
                                              command=lambda: self.toggle_clip_view("list"))
            self.btn_list_view.pack(side="left", padx=2)
            
            self.btn_grid_view = ctk.CTkButton(view_ctrl, text="グリッド", width=60, height=26,
                                              fg_color=self.COLOR_ACCENT if self.clip_view_mode == "grid" else self.COLOR_SIDEBAR,
                                              text_color="black" if self.clip_view_mode == "grid" else "white",
                                              command=lambda: self.toggle_clip_view("grid"))
            self.btn_grid_view.pack(side="left", padx=2)

            self.current_clips_page = 0
            self.all_person_clips = []
            self.selected_clips = set() # リセット
            self.update_bulk_bar() # バーを隠す
            
            loading_lbl = ctk.CTkLabel(self.clips_container, text="⌛ データを準備中...", text_color="gray")
            loading_lbl.pack(pady=20)

            def prep_data():
                data = self.load_scan_results()
                if not data: return
                
                clips_dict = data.get("people", {}).get(person_name, {})
                items = []
                for v_path, detections in clips_dict.items():
                    base = os.path.basename(v_path)
                    for d in detections:
                        items.append({
                            "path": v_path,
                            "filename": base,
                            "t": d['t'],
                            "shooting_date": d.get('timestamp', '不明'),
                            "vibe": d.get('vibe', '不明'),
                            "description": d.get('description', '人物が映っているシーン'),
                            "visual_score": d.get('visual_score', '-'),
                            "happy": d.get('happy', 0),
                            "drama": d.get('drama', 0),
                            "motion": d.get('motion', 0),
                            "face_ratio": d.get('face_ratio', 0),
                            "dist": d.get('dist'),
                            "face_loc": d.get('face_loc')
                        })
                items.sort(key=lambda x: (x['path'], x['t']))
                self.all_person_clips = items
                self.after(0, lambda: [loading_lbl.destroy(), self.render_clips_batch()])
            
            threading.Thread(target=prep_data, daemon=True).start()
            return

    def render_clips_batch(self, page_delta=0):
        """指定されたページのクリップを描画する (ページネーション方式)"""
        if not hasattr(self, 'all_person_clips') or not self.all_person_clips:
            return

        # ページ遷移
        new_page = self.current_clips_page + page_delta
        # コンパクトグリッドモード: 1ページ 100件 (10列 x 10行)
        batch_size = 100 if self.clip_view_mode == "grid" else 20
        total_pages = math.ceil(len(self.all_person_clips) / batch_size)
        
        if 0 <= new_page < total_pages:
            self.current_clips_page = new_page
        elif total_pages > 0:
            self.current_clips_page = max(0, total_pages - 1)

        # UIクリア
        for child in self.clips_container.winfo_children():
            child.destroy()

        start_idx = self.current_clips_page * batch_size
        end_idx = start_idx + batch_size
        batch = self.all_person_clips[start_idx:end_idx]
        
        if not batch:
            ctk.CTkLabel(self.clips_container, text="データがありません。").pack(pady=20)
            return

        # 表示モードに応じて描画
        if self.clip_view_mode == "grid":
            self.render_clips_grid(batch)
        else:
            self.render_clips_list(batch)

        # ページネーションUI
        nav_frame = ctk.CTkFrame(self.clips_container, fg_color="transparent")
        nav_frame.pack(pady=20)

        btn_first = ctk.CTkButton(nav_frame, text="« 最初", width=40, fg_color=self.COLOR_SIDEBAR, 
                                  command=lambda: [setattr(self, 'current_clips_page', 0), self.render_clips_batch()])
        btn_first.pack(side="left", padx=5)

        btn_prev = ctk.CTkButton(nav_frame, text="< 前へ", width=60, fg_color=self.COLOR_SIDEBAR, 
                                 command=lambda: self.render_clips_batch(page_delta=-1))
        btn_prev.pack(side="left", padx=5)

        lbl_page = ctk.CTkLabel(nav_frame, text=f"ページ {self.current_clips_page + 1} / {total_pages}", 
                                font=ctk.CTkFont(size=12, weight="bold"))
        lbl_page.pack(side="left", padx=20)

        # ページジャンプ用 入力欄とボタン
        jump_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        jump_frame.pack(side="left", padx=10)
        
        entry_page = ctk.CTkEntry(jump_frame, width=45, height=28, font=ctk.CTkFont(size=12))
        entry_page.insert(0, str(self.current_clips_page + 1))
        entry_page.pack(side="left")
        
        btn_jump = ctk.CTkButton(jump_frame, text="移動", width=45, height=28, fg_color=self.COLOR_SIDEBAR,
                                 command=lambda: self.on_page_jump(entry_page.get(), total_pages))
        btn_jump.pack(side="left", padx=2)
        entry_page.bind("<Return>", lambda e: self.on_page_jump(entry_page.get(), total_pages))

        btn_next = ctk.CTkButton(nav_frame, text="次へ >", width=60, fg_color=self.COLOR_SIDEBAR, 
                                 command=lambda: self.render_clips_batch(page_delta=1))
        btn_next.pack(side="left", padx=5)

        btn_last = ctk.CTkButton(nav_frame, text="最後 »", width=40, fg_color=self.COLOR_SIDEBAR, 
                                 command=lambda: [setattr(self, 'current_clips_page', total_pages - 1), self.render_clips_batch()])
        btn_last.pack(side="left", padx=5)

        # スクロール位置の管理
        if self.target_y_to_restore is not None:
            self.after(200, lambda: self.scanned_scroll._parent_canvas.yview_moveto(self.target_y_to_restore))
            self.target_y_to_restore = None
        else:
            self.after(100, lambda: self.scanned_scroll._parent_canvas.yview_moveto(0))

    def render_clips_list(self, batch):
        """詳細リスト形式で描画する"""
        row_widgets = {}
        for item in batch:
            row = ctk.CTkFrame(self.clips_container, fg_color=self.COLOR_SIDEBAR)
            row.pack(fill="x", pady=5, padx=5)
            
            # Key for identification (though not used for selection in list mode now)
            key = (item['path'], item['t'])
            
            lbl_img = ctk.CTkLabel(row, text="⌛", width=80, height=80, fg_color="black")
            lbl_img.pack(side="left", padx=10, pady=5)
            row_widgets[key] = lbl_img
            
            # 右側ボタン（コンパクト化）
            btns_frame = ctk.CTkFrame(row, fg_color="transparent")
            btns_frame.pack(side="right", padx=5)
            top_btns = ctk.CTkFrame(btns_frame, fg_color="transparent")
            top_btns.pack(side="top", pady=(0, 2))

            btn_play_file = ctk.CTkButton(top_btns, text="", image=self.icon_play, width=20, height=20, 
                                          fg_color="transparent", hover_color=self.COLOR_DEEP_BG,
                                          command=lambda p=item['path']: self.open_video_file(p))
            btn_play_file.pack(side="left", padx=0)
            btn_reveal_file = ctk.CTkButton(top_btns, text="", image=self.icon_folder, width=20, height=20, 
                                            fg_color="transparent", hover_color=self.COLOR_DEEP_BG,
                                            command=lambda p=item['path']: self.reveal_in_finder(p))
            btn_reveal_file.pack(side="left", padx=0)

            btn_del = ctk.CTkButton(btns_frame, text="削除", fg_color="#5D6D7E", hover_color="#A93226",
                                    width=40, height=20, font=ctk.CTkFont(size=10, weight="bold"),
                                    command=lambda v=item['path'], t=item['t'], r=row: self.delete_scan_clip(self.last_person_viewed, v, t, r))
            btn_del.pack(side="top")

            info_frame = ctk.CTkFrame(row, fg_color="transparent")
            info_frame.pack(side="left", padx=10, expand=True, fill="both")
            
            # Vibe準備
            vibe_val = item.get('vibe', '')
            
            conf = item.get('dist')
            conf_txt = f" (識別率: {int((1.0 - conf) * 100)}%)" if conf is not None else ""
            lbl_desc = ctk.CTkLabel(info_frame, text=f"{item['description']}{conf_txt}", 
                                   font=ctk.CTkFont(size=13, weight="bold"), text_color=self.COLOR_ACCENT, anchor="w")
            lbl_desc.pack(fill="x")

            # ファイル名とVibeタグを一列に
            file_row = ctk.CTkFrame(info_frame, fg_color="transparent")
            file_row.pack(fill="x")
            
            lbl_filename = ctk.CTkLabel(file_row, text=f"📂 {item['filename']}", font=ctk.CTkFont(size=11), 
                                        text_color="gray60", anchor="w")
            lbl_filename.pack(side="left")
            
            if vibe_val:
                lbl_vibe = ctk.CTkLabel(file_row, text=f"#{vibe_val}", font=ctk.CTkFont(size=10, weight="bold"),
                                        text_color="#FFBF00", fg_color="#444444", corner_radius=12, height=16)
                lbl_vibe.pack(side="left", padx=5)

            meta_strip = ctk.CTkFrame(info_frame, fg_color="transparent")
            meta_strip.pack(fill="x")
            meta_txt = f"撮影日: {item['shooting_date']}  |  時間: {item['t']}s"
            lbl_meta = ctk.CTkLabel(meta_strip, text=meta_txt, font=ctk.CTkFont(size=11), anchor="w", text_color="gray80")
            lbl_meta.pack(side="left")

            happy_pct = int(item['happy'] * 100) if isinstance(item['happy'], (int, float)) else 0
            drama_pct = int(item['drama'] * 100) if isinstance(item['drama'], (int, float)) else 0
            
            # 動きと顔サイズ (取得できない詳細は0や-で埋める)
            motion_val = item.get('motion', 0)
            face_ratio_val = item.get('face_ratio', 0) * 100
            
            metrics_txt = f"画質: {item['visual_score']}  |  笑顔: {happy_pct}%  |  ドラマ: {drama_pct}%  |  動き: {motion_val}  |  顔サイズ: {face_ratio_val:.1f}%"
            lbl_metrics = ctk.CTkLabel(info_frame, text=metrics_txt, font=ctk.CTkFont(size=10), anchor="w", text_color="gray70")
            lbl_metrics.pack(fill="x")

        def load_thumbs():
            for itm in batch:
                path, ts, loc = itm['path'], itm['t'], itm['face_loc']
                thumb_path = self.get_face_thumbnail(path, ts, loc)
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        img = ctk.CTkImage(light_image=Image.open(thumb_path), size=(80, 80))
                        def update_ui(k=(path, ts), i=img):
                            if k in row_widgets:
                                w = row_widgets[k]
                                if w.winfo_exists():
                                    w.configure(image=i, text="")
                        self.after(0, update_ui)
                    except: pass
        threading.Thread(target=load_thumbs, daemon=True).start()

    def render_clips_grid(self, batch):
        """タイル形式で描画する (アトミック・サムネイルのみ・超密集版)"""
        grid_frame = ctk.CTkFrame(self.clips_container, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True)
        
        self.grid_tile_widgets = {} # タイル参照をクリア
        cols = 10
        
        # コンテナの幅に合わせて動的にサイズを決定 (10列固定)
        # 1200x900のウィンドウだと、clips_containerの幅は約850-900前後になるはず
        self.update_idletasks() # 最新の幅を取得するために更新
        container_w = self.clips_container.winfo_width()
        
        # デフォルト・最小・最大値を設定
        if container_w <= 1: container_w = 800 
        
        # スクロールバーや余白を考慮して1列あたりの幅を計算
        # 各タイルは padx=1 (左右計2px), 画像は内部で padx=2 (左右計4px)
        spacing_per_tile = 2 + 4 + 2 # tile padding + img padding + margin
        tile_w = (container_w - 30) // cols # 30pxはスクロールバー等のマージン
        thumb_size = max(30, min(150, tile_w - 6))
        
        grid_widgets = {}
        for i, item in enumerate(batch):
            r, c = divmod(i, cols)
            key = (item['path'], item['t'])
            is_selected = key in self.selected_clips
            
            # 間隔を極限まで詰める (padx/pady=1)
            tile = ctk.CTkFrame(grid_frame, width=thumb_size + 4, height=thumb_size + 4, corner_radius=2,
                                fg_color=self.COLOR_ACCENT if is_selected else self.COLOR_SIDEBAR)
            tile.grid(row=r, column=c, padx=1, pady=1)
            tile.grid_propagate(False) # サイズ指定を強制
            self.grid_tile_widgets[key] = tile # 参照保存
            
            def toggle_sel(event, k=key):
                # 状態を反転
                target_sel = k not in self.selected_clips
                self.on_clip_selected(k[0], k[1], target_sel)
            tile.bind("<Button-1>", toggle_sel)
            
            lbl_img = ctk.CTkLabel(tile, text="", width=thumb_size, height=thumb_size, fg_color="black", corner_radius=1)
            lbl_img.pack(expand=True, padx=2, pady=2)
            lbl_img.bind("<Button-1>", toggle_sel)
            grid_widgets[key] = lbl_img

        def load_thumbs():
            for itm in batch:
                path, ts, loc = itm['path'], itm['t'], itm['face_loc']
                thumb_path = self.get_face_thumbnail(path, ts, loc)
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        img = ctk.CTkImage(light_image=Image.open(thumb_path), size=(thumb_size, thumb_size))
                        def update_ui(k=(path, ts), i=img):
                            # 消去済みのウィジェットへのアクセスを防ぐ (エラー回避)
                            if k in grid_widgets:
                                w = grid_widgets[k]
                                if w.winfo_exists():
                                    w.configure(image=i, text="")
                        self.after(0, update_ui)
                    except: pass
        threading.Thread(target=load_thumbs, daemon=True).start()

    def bulk_delete_selected(self):
        """選択されたクリップを一括削除する"""
        count = len(self.selected_clips)
        if count == 0: return
        if not messagebox.askyesno("確認", f"選択された {count} 件を削除しますか？"): return
        try:
            from utils import load_json_safe, save_json_atomic
            data = load_json_safe(self.SCAN_RESULTS, lambda: {"people": {}, "metadata": {}})
            person_name = self.last_person_viewed
            if person_name in data["people"]:
                person_data = data["people"][person_name]
                for vp, ts in list(self.selected_clips):
                    if vp in person_data:
                        person_data[vp] = [d for d in person_data[vp] if abs(d['t'] - ts) > 0.01]
                        if not person_data[vp]: del person_data[vp]
                save_json_atomic(self.SCAN_RESULTS, data)
                self.all_person_clips = [c for c in self.all_person_clips if (c['path'], c['t']) not in self.selected_clips]
                self.selected_clips = set()
                self.cached_scan_data = None
                self.render_clips_batch()
                self.update_bulk_bar()
        except Exception as e:
            self.log(f"[ERROR] Bulk delete failed: {e}")

    def delete_scan_clip(self, person_name, video_path, timestamp, row_widget=None):
        """特定の検出カットを削除する"""
        if not messagebox.askyesno("確認", "このカットを削除しますか？"): return
        try:
            from utils import load_json_safe, save_json_atomic
            data = load_json_safe(self.SCAN_RESULTS, lambda: {"people": {}, "metadata": {}})
            if person_name in data["people"] and video_path in data["people"][person_name]:
                data["people"][person_name][video_path] = [d for d in data["people"][person_name][video_path] if abs(d['t'] - timestamp) > 0.01]
                if not data["people"][person_name][video_path]: del data["people"][person_name][video_path]
                save_json_atomic(self.SCAN_RESULTS, data)
                self.all_person_clips = [c for c in self.all_person_clips if not (c['path'] == video_path and abs(c['t'] - timestamp) < 0.01)]
                self.cached_scan_data = None
                if row_widget: row_widget.destroy()
                if not self.all_person_clips[self.current_clips_page*20 : (self.current_clips_page+1)*20]:
                    self.render_clips_batch(page_delta=-1)
        except Exception as e:
            self.log(f"[ERROR] Delete failed: {e}")

    def open_result(self):
        output_dir = self.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if os.path.exists(output_dir):
            if sys.platform == 'darwin':
                subprocess.run(['open', output_dir])
            elif sys.platform == 'win32':
                os.startfile(output_dir)
            else: # Linux and other Unix-like systems
                subprocess.run(['xdg-open', output_dir])
        else:
            messagebox.showinfo("INFO", "Output folder not found.")

    def reset_ui(self):
        self.progressbar.stop()
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(1)
        self.btn_run.configure(state="normal", text="START")
        self.is_running = False

    def on_closing(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
        self.destroy()

if __name__ == "__main__":
    try:
        app = ModernDigestApp()
        app.mainloop()
    except Exception as e:
        import traceback
        try:
            from utils import get_app_dir
            crash_log = os.path.join(get_app_dir(), "crash_log.txt")
            with open(crash_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- Crash at {time.ctime()} ---\n")
                traceback.print_exc(file=f)
        except:
            pass
        raise
