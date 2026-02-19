import sys
import os
import json
import logging
import hashlib
import cv2
import numpy as np
from datetime import datetime

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_app_dir():
    """ Get the directory of the executable or script """
    if getattr(sys, 'frozen', False):
        # Bundled executable
        exe_dir = os.path.dirname(sys.executable)
        # On macOS, the executable is inside Omokage.app/Contents/MacOS/
        # We want to save configs/output next to Omokage.app
        if sys.platform == 'darwin' and ".app/Contents/MacOS" in exe_dir:
            return os.path.abspath(os.path.join(exe_dir, "../../.."))
        return exe_dir
    else:
        # Normal script
        return os.path.dirname(os.path.abspath(__file__))

def get_user_data_dir():
    """Returns a writable user data directory (Documents/Omokage)"""
    app_name = "Omokage"
    if sys.platform == "win32":
        # Use %APPDATA%
        base_dir = os.getenv('APPDATA') or os.path.expanduser("~")
        data_dir = os.path.join(base_dir, app_name)
    elif sys.platform == "darwin":
        # User requested to use Documents/Omokage
        base_dir = os.path.expanduser("~/Documents")
        data_dir = os.path.join(base_dir, app_name)
    else:
        base_dir = os.path.expanduser("~")
        data_dir = os.path.join(base_dir, f".{app_name.lower()}")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    return data_dir

def save_json_atomic(file_path, data):
    """ Save JSON to a temporary file and then replace the target file atomically. """
    temp_path = file_path + ".tmp"
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, file_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def load_json_safe(file_path, default_factory):
    """ Load JSON file with backup recovery. Returns default if both fail. """
    bk_path = file_path.replace(".json", "_bk.json")
    
    # 1. Try primary file
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load primary JSON ({file_path}): {e}")
            
    # 2. Try backup file
    if os.path.exists(bk_path):
        try:
            with open(bk_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Info: Recovered data from backup ({bk_path})")
                return data
        except Exception as e:
            print(f"Error: Failed to load backup JSON ({bk_path}): {e}")
            
    if callable(default_factory):
        return default_factory()
    return default_factory

def generate_face_thumbnail(video_path, timestamp, face_loc, output_dir):
    """
    指定された動画のタイムスタンプ＋座標から、お顔のサムネイルを取得/生成する (共通化用)
    face_loc: [top, right, bottom, left]
    """
    if not face_loc or len(face_loc) < 4:
        return None
        
    thumb_dir = os.path.join(output_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    
    # ハッシュでファイル名を一意にする
    h = hashlib.md5(f"{video_path}_{timestamp}".encode()).hexdigest()
    thumb_path = os.path.join(thumb_dir, f"thumb_{h}.jpg")
    
    if os.path.exists(thumb_path):
        return thumb_path
        
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return None
        
        # クロップ
        h_orig, w_orig = frame.shape[:2]
        t, r, b, l = face_loc
        
        # 少しマージンを持たせる (30%)
        pad_h = int((b - t) * 0.3)
        pad_w = int((r - l) * 0.3)
        
        t = max(0, t - pad_h)
        b = min(h_orig, b + pad_h)
        l = max(0, l - pad_w)
        r = min(w_orig, r + pad_w)
        
        face_img = frame[t:b, l:r]
        if face_img.size == 0: return None
        
        # 保存前にリサイズ (80px四方程度で十分)
        face_img = cv2.resize(face_img, (80, 80))
        cv2.imwrite(thumb_path, face_img)
        return thumb_path
    except:
        return None
