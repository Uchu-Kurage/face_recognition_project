import json
import os
import sys
import cv2
from datetime import datetime



from utils import get_user_data_dir

def load_scan_results(json_path='scan_results.json'):
    from utils import load_json_safe
    return load_json_safe(json_path, lambda: {"people": {}, "metadata": {}})

def main():
    # Unused main method, keeping for future script logic or CLI extension
    pass
def create_story(person_name, period="All Time", focus="Balance", bgm_enabled=False, json_path='scan_results.json', output_playlist_path='story_playlist.json', manual_bgm_path=""):
    print(f"DEBUG: create_story received manual_bgm_path = '{manual_bgm_path}'")
    results = load_scan_results(json_path)
    if not results or person_name not in results.get("people", {}):
        print(f"Error: No data found for {person_name}")
        return

    video_map = results["people"][person_name]
    metadata = results.get("metadata", {})
    all_clips = []
    valid_videos_cache = {} # 読み込み可否のキャッシュ

    # Flatten the results into a list of clips with metadata
    for video_path, detections in video_map.items():
        # ファイルの実在と読み込み可否をチェック
        if video_path not in valid_videos_cache:
            if os.path.exists(video_path):
                # OpenCVでヘッダーが読み込めるか試行 (I/Oエラー対策)
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    valid_videos_cache[video_path] = True
                    cap.release()
                else:
                    print(f"  Warning: Video file exists but is unreadable (I/O error): {video_path}")
                    valid_videos_cache[video_path] = False
            else:
                valid_videos_cache[video_path] = False

        if not valid_videos_cache[video_path]:
            continue

        # 期間フィルタリング
        month = metadata.get(video_path, {}).get('month', 'unknown')
        if period != "All Time":
            if period.count("-") == 1: # YYYY-MM
                if month != period: continue
            else: # YYYY
                if month.split("-")[0] != period: continue

        for det in detections:
            all_clips.append({
                "video_path": video_path,
                "t": det["t"],
                "happy": det.get("happy", 0),
                "visual_score": det.get("visual_score", 5.0),
                "vibe": det.get("vibe", "穏やか"),
                "drama": det.get("drama", 0),
                "motion": det.get("motion", 0),
                "face_ratio": det.get("face_ratio", 0),
                "timestamp": det.get("timestamp", ""),
                "overlay_text": "" # 後で追加
            })

    if not all_clips:
        print(f"Error: No clips found for {person_name}")
        return

    # Sort by timestamp first
    try:
        all_clips.sort(key=lambda x: datetime.strptime(x["timestamp"], '%Y-%m-%d %H:%M:%S') if x["timestamp"] else datetime.min)
    except:
        pass

    # すでに使ったシーン、動画ファイル、日付を記録する
    used_scenes = set() # (video_path, t) を記録
    used_videos = set()
    used_dates = set()

    import random

    def pick_unique(candidates, count, key_func, reverse=True):
        """重複を避けつつ、バリエーション豊かな候補からランダム性を考慮して選択する。"""
        
        # スコアに揺らぎ（ノイズ）を加える
        def noisy_key(x):
            val = key_func(x)
            # マトリックススコアリング対応 (total_score, breakdown) のタプルが返る場合
            if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], dict):
                total, breakdown = val
                x["_score_breakdown"] = breakdown
                x["_total_score"] = total
                return total * random.uniform(0.8, 1.2)
            
            # 従来形式の場合
            # 数値の場合は ±20% の揺らぎを加える
            if isinstance(val, (int, float)):
                return val * random.uniform(0.8, 1.2)
            # タプルの場合は各要素に ±10% の揺らぎ（数値のみ）
            if isinstance(val, tuple):
                return tuple((v * random.uniform(0.9, 1.1) if isinstance(v, (int, float)) else v) for v in val)
            return val

        # 揺らぎを加えたスコアでソート
        sorted_cands = sorted(candidates, key=noisy_key, reverse=reverse)
        
        # 候補プールを大幅に広げる（必要数の15倍、または全候補の半分）
        pool_size = max(min(len(sorted_cands), count * 15), len(sorted_cands) // 2)
        pool = sorted_cands[:pool_size]
        
        # 絶対に同じシーンは選ばない (STRICT)
        pool = [c for c in pool if (c["video_path"], c["t"]) not in used_scenes]
        
        picked = []
        
        # 同一動画内での時間的分散（近くのシーンを連続して選ばない）
        def is_temporally_dispersed(c):
            # 同じ動画から既に選んでいる場合、それらと一定時間(15秒)以上離れているか
            picked_ts = [p["t"] for p in picked if p["video_path"] == c["video_path"]]
            # 他のフェーズで選ばれたシーンとも比較
            global_picked_ts = [s[1] for s in used_scenes if s[0] == c["video_path"]]
            all_nearby_ts = picked_ts + global_picked_ts
            
            for pt in all_nearby_ts:
                if abs(pt - c["t"]) < 15.0:
                    return False
            return True

        # Phase 1: 未使用の日付 & 未使用のビデオ & 時間的分散
        p1 = [c for c in pool if c["video_path"] not in used_videos and c["timestamp"].split(" ")[0] not in used_dates and is_temporally_dispersed(c)]
        random.shuffle(p1)
        for c in p1:
            if len(picked) >= count: break
            picked.append(c)
            used_scenes.add((c["video_path"], c["t"]))
            used_videos.add(c["video_path"])
            used_dates.add(c["timestamp"].split(" ")[0])
        
        # Phase 2: 未使用のビデオ & 時間的分散
        if len(picked) < count:
            p2 = [c for c in pool if c["video_path"] not in used_videos and (c["video_path"], c["t"]) not in used_scenes and is_temporally_dispersed(c)]
            random.shuffle(p2)
            for c in p2:
                if len(picked) >= count: break
                picked.append(c)
                used_scenes.add((c["video_path"], c["t"]))
                used_videos.add(c["video_path"])
                used_dates.add(c["timestamp"].split(" ")[0])

        # Phase 3: 条件を緩めて選ぶ (ただしシーン重複は絶対にNG)
        if len(picked) < count:
            remaining = [c for c in pool if (c["video_path"], c["t"]) not in used_scenes]
            random.shuffle(remaining)
            for c in remaining:
                if len(picked) >= count: break
                picked.append(c)
                used_scenes.add((c["video_path"], c["t"]))
                        
        return picked

    # Focus の正規化 (UIは日本語、内部は英語で判定していたため)
    focus_map = {
        "バランス": "Balance",
        "笑顔": "Smile",
        "動き": "Active",
        "感動": "Emotional"
    }
    focus = focus_map.get(focus, focus)

    # --- Step 0.5: 統計情報の出力 (各Focusへの該当件数を計算) ---
    count_smile = len([c for c in all_clips if c["happy"] >= 0.5])
    count_emotional = len([c for c in all_clips if c["drama"] >= 0.5])
    count_active = len([c for c in all_clips if c["motion"] >= 3.0])
    
    print(f"\n--- 素材統計 (全 {len(all_clips)} シーン) ---")
    print(f"  😊 笑顔 (Smile): {count_smile} シーン")
    print(f"  🎬 感動 (Emotional): {count_emotional} シーン")
    print(f"  ⚡ 動き (Active): {count_active} シーン")
    print(f"  ⚖️ 全体 (Total): {len(all_clips)} シーン")
    print(f"----------------------------------------\n")

    # --- Step 1: ユーザーの重視項目（Focus）による事前フィルタリング ---
    filtered_clips = []
    if focus == "Smile":
        filtered_clips = [c for c in all_clips if c["happy"] >= 0.5]
        filter_msg = "笑顔率 50%以上"
    elif focus == "Emotional":
        filtered_clips = [c for c in all_clips if c["drama"] >= 0.5]
        filter_msg = "ドラマ度 50%以上"
    elif focus == "Active":
        filtered_clips = [c for c in all_clips if c["motion"] >= 3.0]
        filter_msg = "動き 3.0以上"
    else: # Balance
        # 他の3つの条件（笑顔、感動、動き）のいずれにも該当しない「日常」シーンを抽出
        filtered_clips = [c for c in all_clips if not (c["happy"] >= 0.5 or c["drama"] >= 0.5 or c["motion"] >= 3.0)]
        filter_msg = "日常シーン（特徴的なクリップ以外）"

    # --- Step 1.5: フォールバック処理 (クリップが少なすぎる場合) ---
    # Balance の場合も、フィルタリングの結果少なすぎれば全クリップに戻す
    if len(filtered_clips) < 20:
        print(f"  Warning: フィルタリング後の素材が {len(filtered_clips)} 件と少なすぎるため、全クリップを使用します。")
        filtered_clips = all_clips
    elif focus != "Balance":
        print(f"  Info: '{filter_msg}' により {len(all_clips)} 件 -> {len(filtered_clips)} 件に絞り込みました。")
    else:
        print(f"  Info: 'バランス'設定により日常シーン（{len(filtered_clips)}件）を対象にします。")

    # --- Step 2: 絞り込まれたリストを時系列で再計算し、起承転結に分割 ---
    total_count = len(filtered_clips)
    idx_ki = max(1, int(total_count * 0.2))
    idx_sho = max(idx_ki + 1, int(total_count * 0.65))
    idx_ten = max(idx_sho + 1, int(total_count * 0.9))

    ki_segment = filtered_clips[:idx_ki]
    sho_segment = filtered_clips[idx_ki:idx_sho]
    ten_segment = filtered_clips[idx_sho:idx_ten]
    ketsu_segment = filtered_clips[idx_ten:]

    # Focusに応じた選択ロジック (構造 × スタイルのマトリックススコアリング)
    def get_key_func(part):
        def score_clip(x):
            # 1. 構造によるベーススコア (0.0 ~ 1.0)
            base = x.get("visual_score", 5.0) / 10.0
            
            # 2. 構造上の役割に応じた重み付け
            struct_weight = 1.0
            if part == "起":
                if x.get("vibe") != "穏やか": struct_weight *= 0.3
            elif part == "結":
                ratio = x.get("face_ratio", 1.0)
                if ratio > 3.0: struct_weight *= 1.5
                if x.get("vibe") != "穏やか": struct_weight *= 0.5

            # 3. Focus Style による加点 (フィルタ済みだが、その中でもより良いものを選ぶ)
            style_bonus = 0.0
            if focus == "Balance":
                # バランスの場合はスコアリングせず一律（noisy_keyにより実質ランダム選択）
                style_bonus = 1.0
            elif focus == "Smile":
                style_bonus = x.get("happy", 0) * 2.0
            elif focus == "Active":
                style_bonus = (x.get("motion", 0) / 5.0)
            elif focus == "Emotional":
                style_bonus = x.get("drama", 0) + (x.get("face_ratio", 0) / 10.0)
            else:
                # 予備
                style_bonus = (x.get("happy", 0) + x.get("drama", 0) + (x.get("motion", 0)/10.0)) / 1.5

            total_score = (base * struct_weight) + style_bonus
            return total_score, {
                "base": round(base, 2),
                "struct": round(struct_weight, 2),
                "style": round(style_bonus, 2)
            }
        return score_clip

    # [起] Intro: 2 clips
    ki = pick_unique(ki_segment, 2, get_key_func("起"))
    # [承] Development: 10 clips
    sho = pick_unique(sho_segment, 10, get_key_func("承"))
    # [転] Twist/Climax: 6 clips
    ten = pick_unique(ten_segment, 6, get_key_func("転"))
    # [結] Conclusion: 2 clips
    ketsu = pick_unique(ketsu_segment, 2, get_key_func("結"))

    # 最終的なプレイリスト案
    playlist_draft = ki + sho + ten + ketsu
    
    # 最終的に時系列で再ソート
    try:
        playlist = sorted(playlist_draft, key=lambda x: datetime.strptime(x["timestamp"], '%Y-%m-%d %H:%M:%S') if x["timestamp"] else datetime.min)
    except:
        playlist = playlist_draft

    # --- Chapter Titles (物語への文字入れ) ---
    # 起・承・転・結の各開始ポイントにタイトルを付与
    phase_titles = {
        "起": "Chapter 1: The Beginning",
        "承": "Chapter 2: Daily Life",
        "転": "Chapter 3: Best Smiles",
        "結": "Chapter 4: Memories"
    }
    
    # 実際は vibe などに合わせて日本語で情緒的に
    for i, clip in enumerate(playlist):
        tag = ""
        if clip in ki: tag = "起"
        elif clip in sho: tag = "承"
        elif clip in ten: tag = "転"
        elif clip in ketsu: tag = "結"
        
        # 各フェーズの最初の1秒間（またはクリップ）に表示
        if i == 0:
            clip["overlay_text"] = "The Story of " + person_name
        elif tag == "承" and playlist[i-1] in ki:
            clip["overlay_text"] = "穏やかな日常"
        elif tag == "転" and playlist[i-1] in sho:
            clip["overlay_text"] = "最高の笑顔"
        elif tag == "結" and playlist[i-1] in ten:
            clip["overlay_text"] = "いつまでも、この瞬間を"

    # --- BGM Recommendation ---
    # Automatic BGM selection is removed. Only manual selection is used.
    dominant_vibe = "None (Auto-selection removed)"
    bgm_suggestion = "Manual Selection Only"

    # --- Output ---
    print(f"\n========================================")
    print(f"🎬 1-MINUTE DOCUMENTARY PLAN: {person_name}")
    print(f"========================================\n")
    
    print(f"🎵 SUGGESTED BGM: {bgm_suggestion}\n")
    
    print(f"📋 PLAYLIST (Total: {len(playlist)} clips, approx 60s):")
    print("(Chronologically ordered for a smooth narrative flow)\n")
    for i, clip in enumerate(playlist):
        # Find which phase the clip belongs to based on original segment lists
        phase = "?"
        if clip in ki: phase = "起"
        elif clip in sho: phase = "承"
        elif clip in ten: phase = "転"
        elif clip in ketsu: phase = "結"
        
        print(f"[{phase}] {os.path.basename(clip['video_path'])} @ {clip['t']}s (Time: {clip['timestamp']}, Happy: {clip['happy']})")

    # Save playlist to a file for render_story to read
    playlist_data = {
        "person_name": person_name,
        "clips": playlist,
        "dominant_vibe": dominant_vibe,
        "suggested_bgm": bgm_suggestion,
        "manual_bgm_path": manual_bgm_path
    }
    
    with open(output_playlist_path, 'w', encoding='utf-8') as f:
        json.dump(playlist_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nPlaylist data saved to '{output_playlist_path}'")
    print(f"Dominant vibe: {dominant_vibe}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("person")
    parser.add_argument("--period", default="All Time")
    parser.add_argument("--focus", default="Balance")
    parser.add_argument("--bgm", action="store_true")
    parser.add_argument("--no-bgm", action="store_false", dest="bgm")
    args = parser.parse_args()

    create_story(args.person, period=args.period, focus=args.focus, bgm_enabled=args.bgm)
