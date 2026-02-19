import face_recognition
import pickle
import sys
import os
import glob
import cv2

def register_person(image_path, name, pkl_path='target_faces.pkl', profile_dir=None):
    """
    1枚の画像から人物を登録。顔をクロップしてアイコン保存し、エンコーディングを保存する。
    """
    if profile_dir is None:
        from utils import get_user_data_dir
        profile_dir = os.path.join(get_user_data_dir(), "profiles")

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return False, "FILE_NOT_FOUND"

    try:
        # ディレクトリ準備
        os.makedirs(profile_dir, exist_ok=True)
        
        # ロード
        print(f"[{name}] 処理中...")
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            print("  => 顔が検出されませんでした。")
            return False, "NO_FACE"
            
        if len(face_locations) > 1:
            print(f"  => 複数の顔が検出されました ({len(face_locations)}人)。")
            return False, "MULTIPLE_FACES"

        # 特徴量抽出
        encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
        if not encodings:
            return False, "ENCODING_ERROR"
            
        # 既存データの読み込み (互換性のため、常にリスト形式で保存)
        data = {}
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
            except:
                data = {}
        
        # データの追加/更新 (常にリストに格納)
        if name in data:
            if isinstance(data[name], list):
                data[name].append(encodings[0])
            else:
                # 旧データが単一エンコーディングの場合、リストに変換して追加
                data[name] = [data[name], encodings[0]]
        else:
            data[name] = [encodings[0]]

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
            
        # アイコンの保存 (OpenCV)
        # face_recognition の locations は (top, right, bottom, left)
        t, r, b, l = face_locations[0]
        
        # 余白を少し持たせる
        h, w, _ = image.shape
        pad = int((b - t) * 0.2)
        t_pad = max(0, t - pad)
        b_pad = min(h, b + pad)
        l_pad = max(0, l - pad)
        r_pad = min(w, r + pad)
        
        face_chip = image[t_pad:b_pad, l_pad:r_pad]
        face_chip_bgr = cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR)
        
        icon_path = os.path.join(profile_dir, f"{name}.jpg")
        cv2.imwrite(icon_path, face_chip_bgr)
        
        print(f"  => 登録完了: {name} (Icon: {icon_path})")
        return True, "SUCCESS"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in register_person: {e}")
        return False, str(e)

def delete_person(name, pkl_path='target_faces.pkl'):
    """
    指定した名前の人物を削除する。
    """
    try:
        # pkl から削除
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if name in data:
                del data[name]
                with open(pkl_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"  => データの削除完了: {name}")

        # アイコンを削除
        from utils import get_user_data_dir
        icon_path = os.path.join(get_user_data_dir(), "profiles", f"{name}.jpg")
        if os.path.exists(icon_path):
            os.remove(icon_path)
            print(f"  => アイコンの削除完了: {icon_path}")
            
        return True
    except Exception as e:
        print(f"Error deleting person {name}: {e}")
        return False

def extract_faces_from_folder(folder_path, output_path='target_faces.pkl'):
    # (既存のフォルダバルクスキャン機能も一応残しておく)
    # ... 省略 ...
    pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用法: python extract_features.py <image_path> <name>")
        sys.exit(1)
    
    img = sys.argv[1]
    name = sys.argv[2]
    register_person(img, name)
