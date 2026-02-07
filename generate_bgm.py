import torch

# Monkey patch for systems without xpu support to avoid AttributeError
if not hasattr(torch, "xpu"):
    class DummyXPU:
        def is_available(self):
            return False
        def empty_cache(self):
            pass
        def device_count(self):
            return 0
        def current_device(self):
            return "cpu"
        def synchronize(self):
            pass
        def __getattr__(self, name):
            # Fallback for any other attribute to prevent crash
            def method(*args, **kwargs):
                return None
            return method
    torch.xpu = DummyXPU()

import torch.distributed
if not hasattr(torch.distributed, "device_mesh"):
    class DummyDeviceMeshModule:
        class DeviceMesh:
            def __init__(self, *args, **kwargs):
                pass
    torch.distributed.device_mesh = DummyDeviceMeshModule()

import os

import scipy.io.wavfile as wavfile
import numpy as np
from datetime import datetime

# Stable Audio Open 1.0 は diffusers を使用します
# Note: このモデルはGatedなため、配布時はユーザーにHFトークンを入力してもらう必要があります。

def generate_bgm(vibe="穏やか", duration_seconds=30, output_dir="bgm", token=None):
    """
    Stable Audio Open 1.0 を使用してBGMを生成し、WAVとして保存する。
    """
    from diffusers import StableAudioPipeline
    
    import random
    
    # Vibeを英語プロンプトのリストにマッピング (バリエーション強化)
    vibe_prompts_map = {
        "穏やか": [
            "beautifully melodic acoustic guitar and piano, warm and peaceful, emotionally evolving, studio recording",
            "ambient ethereal soundscape, soft pads and distant bells, serene and calm, meditative, high quality",
            "gentle solo piano, emotional and nostalgic, soft touch, peaceful atmosphere, professional production",
            "lo-fi acoustic chill, mellow vibes, relaxing beats with warm guitar, cozy and serene"
        ],
        "エネルギッシュ": [
            "uplifting energetic pop, bright synths and driving drums, catchy melodic hooks, high-energy, professional production",
            "fast-paced synthwave, neon vibes, rhythmic and driving electronic beats, energetic and bold",
            "funky upbeat rhythm, groovy bassline and bright horns, danceable and happy, high quality",
            "inspiring corporate pop, motivational and bright, rhythmic guitar and percussion, positive energy"
        ],
        "感動的": [
            "cinematic orchestral masterpiece, soaring expressive violin, rich emotional piano, dramatic and powerful",
            "epic cinematic piano and strings, building tension and emotional release, evocative and grand",
            "heartfelt solo cello and piano, deep emotional resonance, slowly evolving beautiful melody, high quality",
            "atmospheric cinematic soundscape, emotional swells, ethereal vocals and lush strings, evocative"
        ],
        "かわいい": [
            "playful whimsical melody, plucky strings and mallets, bright and cheerful, bouncy and lighthearted",
            "cute 8-bit chiptune, happy and adventurous, retro game music, catchy and playful electronics",
            "kawaii future bass, bright and bubbly synths, sweet melody, energetic and cute rhythm",
            "gentle toy piano and woodwinds, nursery rhyme style, innocent and sweet, playful atmosphere"
        ]
    }

    choices = vibe_prompts_map.get(vibe, vibe_prompts_map["穏やか"])
    prompt = random.choice(choices)
    
    # プロンプトの一部を抜粋してスタイル名として使用（ファイル名用）
    style_keywords = ["piano", "guitar", "synth", "violin", "lofi", "8-bit", "orchestra", "ambient"]
    detected_style = "music"
    for kw in style_keywords:
        if kw in prompt.lower():
            detected_style = kw
            break

    # Vibeに応じたベースファイル名
    base_name = vibe
    filename = f"{base_name}_{detected_style}_{timestamp}.wav"
    
    # Ensure absolute path if it looks relative
    if not os.path.isabs(output_dir):
        from utils import get_app_dir
        output_dir = os.path.join(get_app_dir(), output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, filename)

    print(f"\n>>> AI BGM生成を開始します...")
    print(f"  雰囲気: {vibe}")
    print(f"  プロンプト: {prompt}")
    print(f"  長さ: {duration_seconds}秒 (Stable Audio制限: 最大47秒)")
    print(f"  出力先: {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac M1/M2/M3 の場合は mps を優先
    if torch.backends.mps.is_available():
        device = "mps"
    
    # mps は float16 または float32
    dtype = torch.float16 if device != "cpu" else torch.float32
    
    try:
        print(f"  モデルをロード中... ({device}, {dtype})")
        # モデルのロード
        # token が指定されている場合はそれを使用
        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            torch_dtype=dtype,
            token=token
        )
        pipe = pipe.to(device)
        
        # 生成
        # Stable Audio Open 1.0 は最大47秒まで対応
        audio_duration = min(duration_seconds, 47.0)
        
        print(f"  音楽を生成中...")
        # 生成実行
        output = pipe(
            prompt, 
            num_inference_steps=50, 
            audio_end_in_s=audio_duration
        ).audios
        
        # output[0] は第一生成サンプル (channels, samples)
        # scipy.io.wavfile.write のために NumPy 配列に変換
        audio_data = output[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()
        
        # 44.1kHz で保存
        # scipy.io.wavfile.write は float16 をサポートしていないため float32 に変換
        # また、(samples, channels) を期待するため転置
        audio_data_t = audio_data.T.astype(np.float32)
        
        wavfile.write(output_path, 44100, audio_data_t)
        
        print(f">>> BGM生成完了: {output_path}")
        return True, output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during BGM generation: {e}")
        if "gated" in str(e).lower() or "not found" in str(e).lower() or "unauthorized" in str(e).lower():
            print("ヒント: このモデルには Hugging Face トークンが必要です。設定を確認してください。")
        return False, None

if __name__ == "__main__":
    import sys
    test_token = os.getenv("HF_TOKEN")
    success, _ = generate_bgm(vibe="穏やか", duration_seconds=10, token=test_token)
    sys.exit(0 if success else 1)
