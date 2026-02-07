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

def infer_description_vibe(emotion_data):
    """感情データからシーンの解説と雰囲気を推論"""
    dominant = max(emotion_data, key=emotion_data.get)
    happy = emotion_data.get('happy', 0)
    
    if happy > 50:
        return "幸せそうな笑顔あふれるシーン", "穏やか"
    elif dominant == 'surprise':
        return "驚きのあるドラマチックな瞬間", "感動的"
    elif dominant in ['angry', 'fear']:
        return "表情豊かな力強いシーン", "エネルギッシュ"
    else:
        return "自然な表情の日常シーン", "穏やか"

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

def scan_video(video_path, target_data, check_interval_sec=1.0, resize_scale=0.25):
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
    
    while True:
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
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for i, enc in enumerate(face_encodings):
                names = list(target_data.keys())
                known_encodings = [target_data[n] for n in names]
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = names[first_match_index]
                    timestamp = current_frame_index / fps
                    
                    # 顔の大きさ（クローズアップ度）
                    top, right, bottom, left = face_locations[i]
                    inv_scale = 1.0 / resize_scale
                    face_w = (right - left) * inv_scale
                    face_h = (bottom - top) * inv_scale
                    face_ratio = (face_w * face_h) / total_pixels * 100.0 # ％表記
                    
                    happy_score = 0
                    drama_score = 0
                    description = ""
                    vibe = ""
                    try:
                        face_img = frame[int(top*inv_scale):int(bottom*inv_scale), int(left*inv_scale):int(right*inv_scale)]
                        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                        if isinstance(analysis, list): analysis = analysis[0]
                        emo = analysis['emotion']
                        
                        happy_score = float(emo['happy']) / 100.0
                        # ドラマスコア: 驚き、悲しみ、怒り、恐怖を合算
                        drama_score = (float(emo['surprise']) + float(emo['sad']) + float(emo['angry']) + float(emo['fear'])) / 100.0
                        description, vibe = infer_description_vibe(emo)
                    except:
                        description, vibe = "人物が映っているシーン", "穏やか"

                    v_score = calculate_visual_score(frame)
                    mtime = os.path.getmtime(video_path)
                    date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

                    results_per_person[name].append({
                        "t": round(float(timestamp), 2),
                        "happy": round(happy_score, 3),
                        "drama": round(drama_score, 3),
                        "motion": round(float(motion_score), 2),
                        "face_ratio": round(float(face_ratio), 2),
                        "description": description,
                        "vibe": vibe,
                        "visual_score": v_score,
                        "timestamp": date_str
                    })

        current_frame_index += frame_step
        if current_frame_index >= end_limit:
            break
            
    cap.release()
    return results_per_person

def run_scan(video_folder, target_pkl='target_faces.pkl', output_json=None):
    if output_json is None:
        from utils import get_app_dir
        output_json = os.path.join(get_app_dir(), 'scan_results.json')
        
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

    # 既存の結果をロード
    output_json = 'scan_results.json'
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # 登録人物リストに合わせてpeople構造を同期
            for name in target_data.keys():
                if name not in results["people"]:
                    results["people"][name] = {}
            # 登録されていない人物のデータを削除（オプション: ユーザー要望次第だが、現状維持が安全か）
        except:
            results = {
                "people": {name: {} for name in target_data.keys()},
                "metadata": {}
            }
    else:
        results = {
            "people": {name: {} for name in target_data.keys()},
            "metadata": {}
        }

    for i, video_path in enumerate(video_files):
        # スキップ判定: 既にメタデータが存在する場合はスキャン済みとみなす
        if video_path in results["metadata"]:
            already_scanned_all_people = True
            # もし登録されて間もない新人物がいる場合、その人物のデータがこのパスにない可能性がある
            for name in target_data.keys():
                if video_path not in results["people"][name]:
                    # ここで「いや、この人物分だけスキャンしなきゃ」とするか、
                    # ユーザーの言う通り「ファイルのスキャンはスキップ」を優先するか。
                    # シンプルに「スキャン済みなら何もしない」とする。
                    pass
            
            print(f"  スキップ (スキャン済み): {os.path.basename(video_path)}")
            continue

        # 進捗率
        progress = (i) / len(video_files)
        pct = int(progress * 100)
        print(f"進捗: {pct}% ({i+1}/{len(video_files)}本目)")
        print(f"PROGRESS: {pct}%")
        sys.stdout.flush()

        results_per_person = scan_video(video_path, target_data)
        
        # メタデータ取得
        mtime = os.path.getmtime(video_path)
        dt = datetime.datetime.fromtimestamp(mtime)
        month_str = dt.strftime('%Y-%m')
        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        results["metadata"][video_path] = {"month": month_str, "date": date_str}
        
        hit_any = False
        for name, ts in results_per_person.items():
            if ts:
                results["people"][name][video_path] = ts
                hit_any = True
        
        if hit_any:
            print(f"  => ヒットあり (Date: {month_str})")
        else:
            print(f"  => 検出なし (Date: {month_str})")
        
        # 1ファイルごとに結果を保存（途中終了時のデータ保護）
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
    # 最後は 100% を出力
    print(f"進捗: 100% ({len(video_files)}/{len(video_files)}本目)")
    print(f"PROGRESS: 100%")
    sys.stdout.flush()

    # 結果保存
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n処理完了。結果詳細を保存しました: {output_json}")
    if results:
        print("検出された動画:")
        for path in results.keys():
            print(f"- {os.path.basename(path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python scan_videos.py <video_folder_path> [target_pkl_path]")
        sys.exit(1)
    
    video_folder = sys.argv[1]
    target_pkl = sys.argv[2] if len(sys.argv) > 2 else 'target_faces.pkl'
    run_scan(video_folder, target_pkl)
