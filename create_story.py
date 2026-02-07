import json
import os
import sys
from datetime import datetime

def load_scan_results(json_path='scan_results.json'):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_story(person_name, period="All Time", focus="Balance", bgm_enabled=False, json_path='scan_results.json', output_playlist_path='story_playlist.json'):
    results = load_scan_results(json_path)
    if not results or person_name not in results.get("people", {}):
        print(f"Error: No data found for {person_name}")
        return

    video_map = results["people"][person_name]
    metadata = results.get("metadata", {})
    all_clips = []

    # Flatten the results into a list of clips with metadata
    for video_path, detections in video_map.items():
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
        """重複を避けつつ、上位候補からランダムに選択する。"""
        # スコア順にソート
        sorted_cands = sorted(candidates, key=key_func, reverse=reverse)
        
        # 候補プールを広げる（必要な数の5倍程度まで）
        pool_size = min(len(sorted_cands), count * 5)
        pool = sorted_cands[:pool_size]
        
        # 絶対に同じシーンは選ばない (STRICT)
        pool = [c for c in pool if (c["video_path"], c["t"]) not in used_scenes]
        
        picked = []
        
        # Phase 1: 未使用の日付 & 未使用のビデオ から選ぶ (バリエーション優先)
        p1 = [c for c in pool if c["video_path"] not in used_videos and c["timestamp"].split(" ")[0] not in used_dates]
        random.shuffle(p1)
        for c in p1:
            if len(picked) >= count: break
            picked.append(c)
            used_scenes.add((c["video_path"], c["t"]))
            used_videos.add(c["video_path"])
            used_dates.add(c["timestamp"].split(" ")[0])
        
        # Phase 2: 未使用のビデオ から選ぶ (日付重複は許容)
        if len(picked) < count:
            p2 = [c for c in pool if c["video_path"] not in used_videos and (c["video_path"], c["t"]) not in used_scenes]
            random.shuffle(p2)
            for c in p2:
                if len(picked) >= count: break
                picked.append(c)
                used_scenes.add((c["video_path"], c["t"]))
                used_videos.add(c["video_path"])
                used_dates.add(c["timestamp"].split(" ")[0])

        # Phase 3: ビデオ/日付重複を許容して選ぶ (ただしシーン重複は絶対にNG)
        if len(picked) < count:
            p3 = [c for c in pool if (c["video_path"], c["t"]) not in used_scenes]
            random.shuffle(p3)
            for c in p3:
                if len(picked) >= count: break
                picked.append(c)
                used_scenes.add((c["video_path"], c["t"]))
                        
        return picked

    # タイムラインを4つのセグメントに分割
    total_count = len(all_clips)
    idx_ki = max(1, int(total_count * 0.2))
    idx_sho = max(idx_ki + 1, int(total_count * 0.65))
    idx_ten = max(idx_sho + 1, int(total_count * 0.9))

    ki_segment = all_clips[:idx_ki]
    sho_segment = all_clips[idx_ki:idx_sho]
    ten_segment = all_clips[idx_sho:idx_ten]
    ketsu_segment = all_clips[idx_ten:]

    # Focus の正規化 (UIは日本語、内部は英語で判定していたため)
    focus_map = {
        "バランス": "Balance",
        "笑顔": "Smile",
        "動き": "Active",
        "感動": "Emotional"
    }
    focus = focus_map.get(focus, focus) # 日本語なら英語に、そうでなければそのまま

    # Focusに応じた選択ロジック関数の切り替え
    def get_key_func(part):
        if focus == "Smile":
            # 笑顔スコア優先
            return lambda x: x["happy"]
            
        elif focus == "Active":
            # エネルギッシュ優先、なければVisual
            return lambda x: (x["vibe"] == "エネルギッシュ", x["visual_score"])
            
        elif focus == "Emotional":
            # 感動的優先、なければ穏やか + Visual
            return lambda x: (x["vibe"] == "感動的", x["visual_score"])
            
        else: # Balance (Default)
            if part == "起": return lambda x: (x["vibe"] == "穏やか", x["visual_score"])
            elif part == "承": return lambda x: x["visual_score"]
            elif part == "転": return lambda x: x["happy"]
            elif part == "結": return lambda x: (x["vibe"] == "穏やか", x["visual_score"])

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
    vibes = [c["vibe"] for c in playlist]
    vibe_counts = {v: vibes.count(v) for v in set(vibes)}
    # デフォルトはクリップからの判定
    dominant_vibe = max(vibe_counts, key=vibe_counts.get) if vibe_counts else "穏やか"

    # Focusによる強力なオーバーライド
    if focus == "Smile": dominant_vibe = "かわいい"
    elif focus == "Active": dominant_vibe = "エネルギッシュ"
    elif focus == "Emotional": dominant_vibe = "感動的"
    elif focus == "Balance": dominant_vibe = "穏やか"

    bgm_map = {
        "穏やか": "Lo-fi / Acoustic (Soft and warm)",
        "エネルギッシュ": "Upbeat / Pop (High energy and bright)",
        "感動的": "Cinematic / Piano (Dramatic and emotional)",
        "かわいい": "Gentle Lofi / Nostalgic (Cute and relaxing)"
    }
    bgm_suggestion = bgm_map.get(dominant_vibe, "Lo-fi / Cinematic MIX")

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
        "suggested_bgm": bgm_suggestion
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
