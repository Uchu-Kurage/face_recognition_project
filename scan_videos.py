import face_recognition
import cv2
import pickle
import sys
import os
import json
import glob
import datetime
import os

import numpy as np

# 感情分析用ライブラリを遅延インポート
deepface_ready = False
def init_deepface():
    global DeepFace, deepface_ready
    if not deepface_ready:
        print("  ... Initializing DeepFace ...")
        from deepface import DeepFace
        deepface_ready = True

def calculate_visual_score(frame):
    """画質の良さや構図を簡易評価 (1-10)"""
    # 鮮明度 (Laplacian)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 0-500程度を 1-7 にマッピング
    sharpness = min(7.0, blur_score / 70.0)
    
    # 彩度
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].mean() / 255.0 * 2.0 # 0.0-2.0
    
    # コントラスト
    contrast = gray.std() / 128.0 # 0.0-1.0
    
    score = 1.0 + sharpness + saturation + contrast
    return round(min(10.0, score), 1)

def infer_description_vibe(emotion_data, motion=0.0, face_ratio=0.0, visual_score=0.0):
    """感情、動き、構図データからシーンの解説と雰囲気を推論"""
    dominant = max(emotion_data, key=emotion_data.get)
    happy = emotion_data.get('happy', 0)
    surprise = emotion_data.get('surprise', 0)
    drama = (emotion_data.get('surprise', 0) + emotion_data.get('sad', 0) + 
             emotion_data.get('angry', 0) + emotion_data.get('fear', 0))

    # --- Description Logic ---
    if happy > 70 and face_ratio > 10:
        desc = "満面の笑みが際立つベストショット"
    elif happy > 40 and motion > 5:
        desc = "楽しそうにはしゃいでいる瞬間"
    elif drama > 50:
        desc = "感情が動くドラマチックなカット"
    elif motion > 7:
        desc = "躍動感あふれるダイナミックなシーン"
    elif face_ratio < 2:
        desc = "風景の中に溶け込む自然な佇まい"
    elif visual_score > 8 and happy > 20:
        desc = "映画のワンシーンのような美しい日常"
    elif happy > 40:
        desc = "幸せそうな笑顔あふれるシーン"
    elif surprise > 30:
        desc = "驚きのある印象的な瞬間"
    elif dominant in ['angry', 'fear']:
        desc = "表情豊かな力強いシーン"
    else:
        desc = "自然な表情の日常シーン"

    # --- Vibe Logic ---
    if motion > 6:
        vibe = "エネルギッシュ"
    elif visual_score > 8.5:
        vibe = "シネマティック"
    elif happy > 50:
        vibe = "ハッピー"
    elif drama > 40:
        vibe = "エモーショナル"
    else:
        vibe = "ナチュラル"

    return desc, vibe

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

from utils import get_user_data_dir

def load_processed_files(scan_results_path):
    if os.path.exists(scan_results_path):
        try:
            with open(scan_results_path, 'r') as f:
                data = json.load(f)
                return set(data.keys())
        except:
            pass
    return set()

def main(video_folder):
    if not os.path.exists(video_folder):
        print(f"Error: Video folder not found: {video_folder}")
        return

    # --- PATH SETUP ---
    user_data_dir = get_user_data_dir()
    target_faces_path = os.path.join(user_data_dir, "target_faces.pkl")
    scan_results_path = os.path.join(user_data_dir, "scan_results.json")
    
    # Check Target Faces
    if not os.path.exists(target_faces_path):
        print("Error: Target faces not registered.")
        return

def load_target_encodings(pkl_path='target_faces.pkl'):
    """保存された顔特徴辞書データをロードする"""
    if not os.path.exists(pkl_path):
        print(f"エラー: 特徴データファイルが見つかりません: {pkl_path}")
        sys.exit(1)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict):
        # 旧バージョン互換性（単一人物）
        data = {"target": data}
        
    print(f"特徴データをロードしました: {list(data.keys())} ({len(data)} 人)")
    return data

def scan_video(video_path, target_data, check_interval_sec=0.5, resize_scale=0.5, stop_event=None):
    """
    1本の動画をスキャンし、各人物の出現タイムスタンプを辞書形式で返す。
    target_data: { "Name": encoding }
    """
    # { "Name": [timestamps...] }
    results_per_person = {name: [] for name in target_data.keys()}
    init_deepface() # 感情分析の準備
    
    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  警告: 動画を開けませんでした: {video_path}")
        return {name: [] for name in target_data.keys()}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = frame_width * frame_height if frame_width > 0 else 1
    
    # 動画名と長さを表示
    print(f"  スキャン中: {os.path.basename(video_path)} ({video_duration:.1f}秒, {fps:.1f}fps)")

    # チェックするフレーム間隔
    frame_step = int(fps * check_interval_sec)
    if frame_step < 1:
        frame_step = 1

    # 前後1.5秒（3秒カット用）はスキャン対象外とする
    start_margin = int(fps * 1.5)
    end_limit = total_frames - int(fps * 1.5)
    
    current_frame_index = start_margin
    prev_frame_gray = None # 動き解析用
    last_detections = {} # { name: (detection_dict, already_added_to_results) }
    
    while True:
        if stop_event and stop_event.is_set():
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # 動き（モーション）スコアの計算
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if prev_frame_gray is not None:
            # 前のチェックフレームとの差分（簡易的）
            diff = cv2.absdiff(gray, prev_frame_gray)
            motion_score = np.mean(diff) / 25.5 # 0-10にスケーリング
        prev_frame_gray = gray

        # BGR(OpenCV) -> RGB(face_recognition)
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 顔検出
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        current_frame_matches = set()
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for i, enc in enumerate(face_encodings):
                # 精度向上のため compare_faces ではなく face_distance (最短距離) を使用
                best_dist = 1.0
                best_name = None
                
                for name, enc_list in target_data.items():
                    # 互換性のため、単一エンコーディングの場合はリストとして扱う
                    if not isinstance(enc_list, list):
                        enc_list = [enc_list]
                    
                    # 登録されている全写真との距離を計算し、最も近いものを採用
                    distances = face_recognition.face_distance(enc_list, enc)
                    min_d = min(distances) if len(distances) > 0 else 1.0
                    
                    if min_d < best_dist:
                        best_dist = min_d
                        best_name = name
                
                # 精度向上のための厳格化: 0.45 -> 0.42
                if best_name and best_dist < 0.42:
                    # 顔の大きさ（クローズアップ度）
                    top, right, bottom, left = face_locations[i]
                    inv_scale = 1.0 / resize_scale
                    face_w = (right - left) * inv_scale
                    face_h = (bottom - top) * inv_scale
                    face_ratio = (face_w * face_h) / total_pixels * 100.0 # ％表記
                    
                    # 誤検知抑制: 小さすぎる顔は背景の他人の可能性が高いため無視 (1.2%未満)
                    if face_ratio < 1.2:
                        continue
                        
                    current_frame_matches.add(best_name)
                    timestamp = current_frame_index / fps
                    
                    mtime = os.path.getmtime(video_path)
                    date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

                    det = {
                        "t": round(float(timestamp), 2),
                        "motion": round(float(motion_score), 2),
                        "face_ratio": round(float(face_ratio), 2),
                        "dist": round(float(best_dist), 4),
                        "face_loc": [int(top*inv_scale), int(right*inv_scale), int(bottom*inv_scale), int(left*inv_scale)],
                        "timestamp": date_str
                    }

                    # --- 連続検知フィルタ (Temporal Filter) ---
                    # 1回だけの検知は誤認の可能性があるため、2回連続して検知された場合のみ記録
                    if best_name in last_detections:
                        prev_det, added = last_detections[best_name]
                        
                        # 記録が確定したタイミングで、重たい解析（DeepFace/Thumb）を一度だけ実行
                        def enrich_detection(d):
                            if "happy" in d: return d # すでに解析済み
                            try:
                                t_top, t_right, t_bottom, t_left = d["face_loc"]
                                face_img = frame[t_top:t_bottom, t_left:t_right]
                                analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                                if isinstance(analysis, list): analysis = analysis[0]
                                emo = analysis['emotion']
                                
                                v_score = calculate_visual_score(frame)
                                desc, vibe = infer_description_vibe(emo, motion=d["motion"], face_ratio=d["face_ratio"], visual_score=v_score)
                                
                                d["happy"] = round(float(emo.get('happy', 0)) / 100.0, 3)
                                d["drama"] = round((float(emo.get('surprise', 0)) + float(emo.get('sad', 0)) + 
                                               float(emo.get('angry', 0)) + float(emo.get('fear', 0))) / 100.0, 3)
                                d["description"] = desc
                                d["vibe"] = vibe
                                d["visual_score"] = v_score
                            except:
                                d["happy"], d["drama"], d["description"], d["vibe"], d["visual_score"] = 0, 0, "人物が映っているシーン", "ナチュラル", 5.0
                            
                            # Generate thumbnail for UI (using user profile dir)
                            from utils import generate_face_thumbnail, get_user_data_dir
                            profile_dir = os.path.join(get_user_data_dir(), "profiles")
                            generate_face_thumbnail(video_path, d["t"], d["face_loc"], profile_dir)
                            return d

                        if not added:
                            results_per_person[best_name].append(enrich_detection(prev_det))
                        results_per_person[best_name].append(enrich_detection(det))
                        last_detections[best_name] = (det, True)
                    else:
                        last_detections[best_name] = (det, False)

        # このフレームで見つからなかった人物の履歴をクリア
        for name in list(last_detections.keys()):
            if name not in current_frame_matches:
                del last_detections[name]

        # 次のフレームへ (Seeking vs Reading)
        current_frame_index += frame_step
        if current_frame_index >= end_limit:
            break
        
        # 1フレーム飛ばし程度なら cap.read() の方が速い場合があるが、
        # 大抵の interval (0.5s) では cap.set() の方が効率的
        # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
            
    cap.release()
    return results_per_person

def run_scan(video_folder, target_pkl='target_faces.pkl', output_json=None, force=False, stop_event=None):
    if output_json is None:
        from utils import get_user_data_dir
        output_json = os.path.join(get_user_data_dir(), 'scan_results.json')
        
    # 特徴量ロード
    target_data = load_target_encodings(target_pkl)

    # 動画ファイル検索
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.MP4', '*.MOV']
    video_files = []
    
    print(f"フォルダを検索中 (再帰的): {video_folder}...")
    for ext in video_extensions:
        # 再帰的に検索 (サブフォルダも含む)
        path_pattern = os.path.join(video_folder, "**", ext)
        video_files.extend(glob.glob(path_pattern, recursive=True))
    
    # 重複排除とソート
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print("動画ファイルが見つかりませんでした。")
        sys.exit(0)

    print(f"動画ファイル {len(video_files)} 本を対象に処理を開始します。")

    # --- 既存の結果をロード ---
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from utils import load_json_safe
    
    # 既存の結果をロード
    default_results = {
            "people": {name: {} for name in target_data.keys()},
            "metadata": {}
        }
    results = load_json_safe(output_json, default_results)
    
    # 未登録の人物を同期
    for name in target_data.keys():
        if name not in results["people"]:
            results["people"][name] = {}

    # スキャン対象を決定
    to_scan = []
    for video_path in video_files:
        if not force and video_path in results["metadata"]:
            continue
        to_scan.append(video_path)

    if not to_scan:
        print("スキャン対象の新しい動画はありません。")
        return

    print(f"スキャン開始: {len(to_scan)} 本の動画を処理します。")
    
    # プロセス数はCPU芯数の半分程度に制限（メモリ消費対策）
    max_workers = max(1, multiprocessing.cpu_count() // 2)
    if max_workers > 4: max_workers = 4 # メモリを大量に使うため最大4程度に制限
    
    from utils import save_json_atomic
    
    # ProcessPoolExecutor では stop_event (threading.Event) は渡せないので注意
    # (マルチプロセス用のマネージャが必要になるが、ここではシンプルに1本終わるごとにチェックする)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # scan_video は引数が多いため partial や wrapper を使う
        future_to_video = {
            executor.submit(scan_video, v_path, target_data): v_path 
            for v_path in to_scan
        }
        
        completed_count = 0
        for future in as_completed(future_to_video):
            if stop_event and stop_event.is_set():
                print("\nユーザーによる中断がリクエストされました。")
                # 残りのタスクはキャンセル（Executor終了時に破棄される）
                break
                
            video_path = future_to_video[future]
            completed_count += 1
            
            try:
                results_per_person = future.result()
                
                # --- マージ保存 ---
                final_data = load_json_safe(output_json, lambda: results)
                mtime = os.path.getmtime(video_path)
                dt = datetime.datetime.fromtimestamp(mtime)
                month_str = dt.strftime('%Y-%m')
                date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                final_data["metadata"][video_path] = {"month": month_str, "date": date_str}
                
                for name, ts_list in results_per_person.items():
                    if name not in final_data["people"]:
                        final_data["people"][name] = {}
                    if ts_list:
                        final_data["people"][name][video_path] = ts_list
                    elif video_path in final_data["people"][name]:
                        del final_data["people"][name][video_path]
                
                save_json_atomic(output_json, final_data)
                results = final_data
                
                # 進捗報告
                pct = int((completed_count / len(to_scan)) * 100)
                print(f"進捗: {pct}% ({completed_count}/{len(to_scan)}本完了) - {os.path.basename(video_path)}")
                sys.stdout.flush()
                
            except Exception as e:
                print(f"  エラー ({os.path.basename(video_path)}): {e}")

    print(f"\n処理完了。結果詳細を保存しました: {output_json}")
    if results:
        print("検出された動画:")
        for path in results.keys():
            print(f"- {os.path.basename(path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder")
    parser.add_argument("target_pkl", nargs="?", default='target_faces.pkl')
    parser.add_argument("--force", action="store_true", help="既スキャン動画を再スキャン")
    args = parser.parse_args()
    
    run_scan(args.video_folder, args.target_pkl, force=args.force)
