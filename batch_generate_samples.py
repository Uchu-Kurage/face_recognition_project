import os
import sys
from generate_bgm import generate_bgm, VIBE_PROMPTS_MAP
from utils import load_config

def batch_generate():
    config = load_config('config.json')
    token = config.get("hf_token") or os.getenv("HF_TOKEN")
    
    output_dir = "output/bgm_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting batch BGM generation...")
    print(f"Output directory: {output_dir}")
    
    # 4 vibes x 3 prompts x 3 samples = 36 files
    for vibe, prompts in VIBE_PROMPTS_MAP.items():
        for p_idx in range(len(prompts)):
            for s_idx in range(3):
                # 10秒のサンプル
                s_tag = f"_S{s_idx+1}"
                print(f"\n[Vibe: {vibe}] Prompt #{p_idx+1} (Sample {s_idx+1}/3)")
                
                # generate_bgmを呼び出す
                # ファイル名は generate_bgm 側で Vibe_PX_Timestamp.wav になるが、
                # サンプル比較用に少し調整が必要かもしれない。
                # 現状の generate_bgm は timestamp を含むので重複はしない。
                success, path = generate_bgm(
                    vibe=vibe,
                    duration_seconds=10,
                    output_dir=output_dir,
                    token=token,
                    prompt_index=p_idx
                )
                
                if success:
                    print(f"  Success: {path}")
                else:
                    print(f"  Failed to generate sample.")

if __name__ == "__main__":
    batch_generate()
