import customtkinter as ctk
from tkinter import filedialog, messagebox
import subprocess
import threading
import time
import queue
import sys
import os
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
        
        # 編集設定
        self.blur_enabled = ctk.BooleanVar(value=self.config.get("blur_enabled", False))
        self.color_filter = ctk.StringVar(value=self.config.get("color_filter", "None"))
        self.selected_period = ctk.StringVar(value="All Time")
        self.selected_focus = ctk.StringVar(value="バランス")
        self.bgm_enabled = ctk.BooleanVar(value=self.config.get("bgm_enabled", False))
        self.hf_token = ctk.StringVar(value=self.config.get("hf_token", ""))
        self.view_mode = "People" # Forced default
        
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
        
        # Start log polling on main thread
        self.log_queue = queue.Queue()
        self.check_log_queue()

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
                
                self.textbox.see("end")
                
        finally:
            self.after(100, self.check_log_queue)

    def _update_log_ui_batch(self, text):
        self.textbox.configure(state="normal")
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
        
        # Limit history to ~1000 lines to prevent lag
        try:
            num_lines = int(self.textbox.index('end-1c').split('.')[0])
            if num_lines > 1000:
                self.textbox.delete("1.0", f"{num_lines - 800}.0") # Keep last ~800 lines
        except:
            pass
            
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

        self.btn_nav_about = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=45, border_spacing=10, text="設定 / 情報",
                                           fg_color="transparent", text_color=self.COLOR_TEXT, hover_color=self.COLOR_SIDEBAR,
                                           anchor="w", font=ctk.CTkFont(size=14, weight="bold"),
                                           command=lambda: self.select_frame_by_name("about"))
        self.btn_nav_about.grid(row=3, column=0, sticky="ew")

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
        
        # 1. PEOPLE MANAGEMENT SECTION
        self.people_section = ctk.CTkFrame(self.scan_frame, corner_radius=15, fg_color=self.COLOR_SIDEBAR)
        self.people_section.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.people_section, text="👤 人物管理", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=15, weight="bold")).pack(pady=10)

        self.prof_split = ctk.CTkFrame(self.people_section, fg_color=self.COLOR_SIDEBAR)
        self.prof_split.pack(fill="x", padx=10, pady=(0, 10))
        self.prof_split.grid_columnconfigure(0, weight=1)
        self.prof_split.grid_columnconfigure(1, weight=0)

        # Profiles List (Left)
        self.list_cnt = ctk.CTkFrame(self.prof_split, fg_color=self.COLOR_SIDEBAR)
        self.list_cnt.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.list_header = ctk.CTkFrame(self.list_cnt, fg_color="transparent")
        self.list_header.pack(fill="x", pady=5)
        ctk.CTkLabel(self.list_header, text="登録済みリスト", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        
        self.profile_scroll = ctk.CTkScrollableFrame(self.list_cnt, height=180, fg_color=self.COLOR_DEEP_BG)
        self.profile_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Registration Form (Right)
        self.reg_cnt = ctk.CTkFrame(self.prof_split, width=220, fg_color=self.COLOR_SIDEBAR)
        self.reg_cnt.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self.btn_start_reg = ctk.CTkButton(self.reg_cnt, text="新しい人物を登録", command=self.start_sequential_registration, 
                                          image=self.icon_user, compound="left", fg_color=self.COLOR_ACCENT, 
                                          hover_color=self.COLOR_HOVER, height=35,
                                          text_color="black", font=ctk.CTkFont(size=12, weight="bold"))
        self.btn_start_reg.pack(pady=(45, 15), padx=10, fill="x")

        # 2. VIDEO ANALYSIS SECTION
        self.video_area = ctk.CTkFrame(self.scan_frame, corner_radius=15, fg_color=self.COLOR_SIDEBAR)
        self.video_area.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(self.video_area, text="🎥 動画分析", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=15, weight="bold")).pack(pady=10)

        # Folder Selection & Scanned File List (Horizontal Split)
        self.video_split = ctk.CTkFrame(self.video_area, fg_color=self.COLOR_SIDEBAR)
        self.video_split.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.video_split.grid_columnconfigure(0, weight=1)
        self.video_split.grid_columnconfigure(1, weight=0)

        # Scanned List (Left) - Data
        self.scanned_cnt = ctk.CTkFrame(self.video_split, fg_color=self.COLOR_SIDEBAR)
        self.scanned_cnt.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Header Area with Title and View Switcher
        self.scanned_header = ctk.CTkFrame(self.scanned_cnt, fg_color="transparent")
        self.scanned_header.pack(fill="x", pady=5)
        
        ctk.CTkLabel(self.scanned_header, text="スキャン済み動画", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)

        self.scanned_scroll = ctk.CTkScrollableFrame(self.scanned_cnt, height=180, fg_color=self.COLOR_DEEP_BG)
        self.scanned_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Source & Actions (Right) - Controls
        self.right_ctrl_cnt = ctk.CTkFrame(self.video_split, width=220, fg_color=self.COLOR_SIDEBAR)
        self.right_ctrl_cnt.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Action Buttons
        self.btn_scan_run = ctk.CTkButton(self.right_ctrl_cnt, text="新規スキャン開始", image=self.icon_search, 
                                          compound="left", command=self.start_sequential_scan, height=35,
                                          font=ctk.CTkFont(size=12, weight="bold"), 
                                          fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black")
        self.btn_scan_run.pack(pady=(45, 15), padx=10, fill="x")



        # --- B. EDIT PAGE ---
        self.edit_frame = ctk.CTkFrame(self.container_frame, fg_color=self.COLOR_DEEP_BG)
        self.edit_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="edit_cols")
        self.edit_frame.grid_rowconfigure(0, weight=1)

        # 1. ターゲット選択
        self.target_section = ctk.CTkFrame(self.edit_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.target_section.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(self.target_section, text="1. 対象を選択", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.target_person = ctk.StringVar(value="選択してください...")
        self.menu_target = ctk.CTkOptionMenu(self.target_section, variable=self.target_person, values=["選択してください..."], 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          height=32, font=ctk.CTkFont(size=12))
        self.menu_target.pack(pady=10, padx=20, fill="x")

        # 2. 設定
        self.set_section = ctk.CTkFrame(self.edit_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.set_section.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(self.set_section, text="2. 編集設定", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.sw_blur = ctk.CTkSwitch(self.set_section, text="顔ぼかし (対象以外)", variable=self.blur_enabled, 
                                     progress_color=self.COLOR_ACCENT, command=self.save_config)
        self.sw_blur.pack(pady=10)
        
        self.filter_cnt = ctk.CTkFrame(self.set_section, fg_color="transparent")
        self.filter_cnt.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(self.filter_cnt, text="フィルター:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10)
        self.menu_filt = ctk.CTkOptionMenu(self.filter_cnt, values=["None", "Film", "Sunset"], variable=self.color_filter, 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          command=lambda _: self.save_config(), height=32, font=ctk.CTkFont(size=12))
        self.menu_filt.pack(pady=5, padx=10, fill="x")

        # 追加: 期間とこだわり
        ctk.CTkLabel(self.set_section, text="期間:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)
        self.menu_period = ctk.CTkOptionMenu(self.set_section, values=["All Time"], variable=self.selected_period, 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          height=32, font=ctk.CTkFont(size=12))
        self.menu_period.pack(pady=5, padx=20, fill="x")

        ctk.CTkLabel(self.set_section, text="重視:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)
        self.menu_focus = ctk.CTkOptionMenu(self.set_section, values=["バランス", "笑顔", "動き", "感動"], variable=self.selected_focus, 
                                          button_color=self.COLOR_ACCENT, button_hover_color=self.COLOR_HOVER,
                                          fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT,
                                          height=32, font=ctk.CTkFont(size=12))
        self.menu_focus.pack(pady=5, padx=20, fill="x")

        self.sw_bgm = ctk.CTkSwitch(self.set_section, text="BGMを合成", variable=self.bgm_enabled, 
                                     progress_color=self.COLOR_ACCENT, command=self.save_config)
        self.sw_bgm.pack(pady=5)

        # 追加: BGM事前生成ボタン
        ctk.CTkLabel(self.set_section, text="AI BGM 事前生成:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20, pady=(10, 0))
        self.bgm_btn_frame = ctk.CTkFrame(self.set_section, fg_color="transparent")
        self.bgm_btn_frame.pack(pady=5, padx=10, fill="x")
        
        self.btn_gen_calm = ctk.CTkButton(self.bgm_btn_frame, text="穏やか", width=60, height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                          fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                          command=lambda: self.generate_single_bgm("穏やか"))
        self.btn_gen_calm.pack(side="left", padx=5, expand=True)
        
        self.btn_gen_energetic = ctk.CTkButton(self.bgm_btn_frame, text="元気", width=60, height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                                fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                                command=lambda: self.generate_single_bgm("エネルギッシュ"))
        self.btn_gen_energetic.pack(side="left", padx=5, expand=True)
        
        self.btn_gen_emotional = ctk.CTkButton(self.bgm_btn_frame, text="感動", width=60, height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                                fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                                command=lambda: self.generate_single_bgm("感動的"))
        self.btn_gen_emotional.pack(side="left", padx=5, expand=True)
        
        self.btn_gen_cute = ctk.CTkButton(self.bgm_btn_frame, text="かわいい", width=60, height=28, font=ctk.CTkFont(size=11, weight="bold"),
                                          fg_color=self.COLOR_ACCENT, hover_color=self.COLOR_HOVER, text_color="black",
                                          command=lambda: self.generate_single_bgm("かわいい"))
        self.btn_gen_cute.pack(side="left", padx=5, expand=True)

        # 3. 生成アクション
        self.gen_section = ctk.CTkFrame(self.edit_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.gen_section.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(self.gen_section, text="3. 動画生成", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.btn_gen_digest = ctk.CTkButton(self.gen_section, text="ダイジェスト", image=self.icon_video, 
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
                                      fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT)
        self.textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.textbox.configure(state="disabled")
        
        # --- C. ABOUT PAGE ---
        self.about_frame = ctk.CTkFrame(self.container_frame, fg_color=self.COLOR_DEEP_BG)
        self.about_frame.grid_columnconfigure(0, weight=1)
        self.about_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.about_frame, text="Omokage", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, pady=(20, 10))
        
        self.about_textbox = ctk.CTkTextbox(self.about_frame, wrap="word", font=ctk.CTkFont(size=12),
                                            fg_color=self.COLOR_DEEP_BG, text_color=self.COLOR_TEXT)
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

        # 2. 管理設定 (Hugging Face / API)
        self.admin_section = ctk.CTkFrame(self.about_frame, fg_color=self.COLOR_SIDEBAR, corner_radius=15)
        self.admin_section.grid(row=2, column=0, sticky="ew", padx=40, pady=20)
        ctk.CTkLabel(self.admin_section, text="🛠️ 管理設定", text_color=self.COLOR_ACCENT, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        hf_cnt = ctk.CTkFrame(self.admin_section, fg_color="transparent")
        hf_cnt.pack(pady=5) # Removed fill="x" to allow centering
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

        ctk.CTkLabel(self.admin_section, text="* Stable Audio Open 1.0 (Hugging Face) の利用規約同意が必要です。", 
                     font=ctk.CTkFont(size=10), text_color="gray70").pack(pady=(5, 0))
        
        ctk.CTkLabel(self.admin_section, text="その後、Read権限のトークンを作成してここに貼り付けて保存してください。", 
                     font=ctk.CTkFont(size=10), text_color="gray70").pack(pady=(0, 10))

        # --- 共通ログ & プログレス (メインの下部に配置) ---
        self.bottom_frame = ctk.CTkFrame(self.container_frame, height=200, fg_color=self.COLOR_DEEP_BG, corner_radius=0)
        self.bottom_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        self.textbox = ctk.CTkTextbox(self.bottom_frame, height=150, font=ctk.CTkFont(family="Consolas", size=12))
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.progressbar = ctk.CTkProgressBar(self.bottom_frame)
        self.progressbar.pack(fill="x", padx=10, pady=(0, 10))
        self.progressbar.set(0)

        # 初期表示
        self.select_frame_by_name("scan")
        self.refresh_scanned_files()

    def select_frame_by_name(self, name):
        # ボタンの色リセット
        self.btn_nav_scan.configure(fg_color=("gray75", "gray25") if name == "scan" else "transparent")
        self.btn_nav_edit.configure(fg_color=("gray75", "gray25") if name == "edit" else "transparent")
        self.btn_nav_about.configure(fg_color=("gray75", "gray25") if name == "about" else "transparent")

        # ページ切り替え
        self.scan_frame.grid_forget()
        self.edit_frame.grid_forget()
        self.about_frame.grid_forget()

        if name == "scan":
            self.scan_frame.grid(row=0, column=0, sticky="nsew")
        elif name == "edit":
            self.edit_frame.grid(row=0, column=0, sticky="nsew")
            self.update_target_menu()
            self.update_period_menu()
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

            btn_del = ctk.CTkButton(frame, text="X", width=20, height=20, 
                                    fg_color="#CD6155", hover_color="#A93226",
                                    text_color="white",
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

        self.btn_scan_run.configure(state="disabled", text="スキャン中...")
        self.is_running = True
        self.log(f">>> 動画スキャンを開始: {folder}")
        
        def run():
            try:
                # Direct call instead of subprocess
                current_stdout = sys.stdout
                current_stderr = sys.stderr
                sys.stdout = RedirectText(lambda s: self.log(s, end=""))
                sys.stderr = RedirectText(lambda s: self.log(s, end=""))
                
                try:
                    scan_videos.run_scan(folder, target_pkl=self.TARGET_PKL)
                finally:
                    sys.stdout = current_stdout
                    sys.stderr = current_stderr

                self.log(">>> SCAN COMPLETE!")
                self.after(0, self.refresh_scanned_files)
                self.after(0, self.update_period_menu)
                self.log("__NOTIFY__", title="完了", message="動画スキャンが完了しました。\nすでに処理済みのファイルは自動的にスキップされました。")
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
            finally:
                self.after(0, self.reset_scan_ui)
        # update dynamic menu
        threading.Thread(target=run).start()

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
                        people_map[p_name] = {"count": 0, "last_seen": "0000-00-00", "vibes": set()}
                    
                    people_map[p_name]["count"] = len(p_videos)
                    
                    for v_path, detections in p_videos.items():
                        # Video Map Update
                        if v_path in video_map:
                            video_map[v_path]["people"].append(p_name)
                            for d in detections:
                                if d.get("vibe"): 
                                    video_map[v_path]["vibes"].add(d["vibe"])
                                    people_map[p_name]["vibes"].add(d["vibe"])
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
            table_container.grid_columnconfigure(3, weight=3)

            # Header
            # テーブルヘッダー (アンバーのアクセント)
            header_bg = ctk.CTkFrame(table_container, fg_color=self.COLOR_DEEP_BG, height=30, corner_radius=5)
            header_bg.grid(row=0, column=0, columnspan=4, sticky="ew", padx=2, pady=2)
            
            headers = ["NAME", "VIDEOS", "LAST SEEN", "COMMON VIBES"]
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
                
                vibes = Counter(stats["vibes"]).most_common(3)
                vibes_str = ", ".join([v[0] for v in vibes]) if vibes else "-"
                
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

                ctk.CTkLabel(table_container, text=f"{stats['count']} clips", font=ctk.CTkFont(size=11)).grid(row=grid_row_idx, column=1, padx=10, pady=5, sticky="w")
                ctk.CTkLabel(table_container, text=stats["last_seen"], font=ctk.CTkFont(size=11)).grid(row=grid_row_idx, column=2, padx=10, pady=5, sticky="w")
                ctk.CTkLabel(table_container, text=vibes_str, font=ctk.CTkFont(size=10), text_color="gray70").grid(row=grid_row_idx, column=3, padx=10, pady=5, sticky="w")

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
        names = [os.path.splitext(os.path.basename(i))[0] for i in icons]
        if not names:
            names = ["プロフィールが見つかりません"]
        
        self.menu_target.configure(values=names)
        if self.target_person.get() not in names:
            self.target_person.set(names[0])

    def update_period_menu(self):
        if not os.path.exists(self.SCAN_RESULTS):
            return
        try:
            with open(self.SCAN_RESULTS, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            months = set()
            years = set()
            for path, meta in metadata.items():
                m = meta.get("month")
                if m:
                    months.add(m)
                    years.add(m.split("-")[0])
            
            p_list = ["All Time"] + sorted(list(years), reverse=True) + sorted(list(months), reverse=True)
            self.menu_period.configure(values=p_list)
            if self.selected_period.get() not in p_list:
                self.selected_period.set("All Time")
        except:
            pass

    def reset_scan_ui(self):
        self.btn_scan_run.configure(state="normal", text="新規スキャン開始")
        self.is_running = False

    def start_digest_only(self):
        person = self.target_person.get()
        if person == "選択してください..." or person == "プロフィールが見つかりません":
            self.log("__NOTIFY__", title="警告", message="人物を選択してください。", type="warning")
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
                        filter_type=self.color_filter.get(),
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
                        bgm_enabled=self.bgm_enabled.get()
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
        self.btn_gen_digest.configure(state="normal", text="ダイジェスト")
        self.btn_gen_story.configure(state="normal", text="1分ドキュメンタリー")
        self.is_running = False

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
                        duration_seconds=60, 
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
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.log("__NOTIFY__", title="エラー", message=str(e), type="error")
            finally:
                self.after(0, self.reset_edit_ui)

        threading.Thread(target=run).start()

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
                if line_str.startswith("PROGRESS:"):
                    try:
                        p_val = float(line_str.split(":")[1].strip())
                        overall_p = p_start + (p_val * (p_end - p_start))
                        self.after(0, lambda p=overall_p: self.progressbar.set(p))
                    except: pass
                else:
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
            self.after(0, self.reset_ui)

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
