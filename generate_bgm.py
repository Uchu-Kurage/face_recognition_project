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
    
# Vibeを英語プロンプトのリストにマッピング (ユーザー指定のプロンプトに刷新)
VIBE_PROMPTS_MAP = {
    "穏やか": [
        "Format: Solo. Instrument: Grand Piano. Genre: Minimalist Classical, Neoclassical. Mood: Peaceful, Harmonious, Consonant, Serene, Uplifting. Details: High fidelity, studio recording, pristine sound, clear melody, simple arrangement, lyrical phrasing, reverb. BPM: 80."
    ],
    "エネルギッシュ": [
        "Format: Instrumental. Genre: Corporate, Easy Listening. Instruments: Bright Grand Piano lead. Mood: Motivational, Inspiring, Positive, Success, Optimistic. Style: Advertising, Commercial, Podcast Intro. Details: Clean mix, balanced, simple melody, catchy hook, rhythmic but harmonious. BPM: 110."
    ],
    "感動的": [
        "Format: Solo. Instrument: Steinway Grand Piano. Genre: Cinematic, Pop Ballad. Mood: Emotional, Heartfelt, Hopeful, Euphoric, Touching. Style: Movie Soundtrack, Film Score. Details: Featured melody, expressive performance, dynamic, well-arranged, harmonious, lush reverb. BPM: 90."
    ],
    "かわいい": [
        "Children's Music, Kawaii, Upright Piano, Toy Piano, Cute, Playful, Whimsical, 110 BPM, C Major, High Octave, Simple Melody, Bouncy, Crisp, Dry Mix, Seamless Loop" 
    ]
}

# Default negative prompt to exclude drums/percussion/jazz/dissonance
DEFAULT_NEGATIVE_PROMPT = "Drums, Percussion, Jazz, Dissonance, Atonal, Complex, Muddy, low quality, noise"

def generate_bgm(vibe="穏やか", duration_seconds=30, output_dir="bgm", token=None, prompt_index=None, num_inference_steps=50, negative_prompt=None):
    """
    Stable Audio Open 1.0 を使用してBGMを生成し、WAVとして保存する。
    """
    from diffusers import StableAudioPipeline
    
    import random
    
    choices = VIBE_PROMPTS_MAP.get(vibe, VIBE_PROMPTS_MAP["穏やか"])
    if prompt_index is not None and 0 <= prompt_index < len(choices):
        idx = prompt_index
    else:
        idx = random.randint(0, len(choices) - 1)
    prompt = choices[idx]
    
    # Use default negative prompt if none provided
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
    
    # プロンプトの一部を抜粋してスタイル名として使用（ファイル名用）
    style_keywords = ["piano", "guitar", "synth", "violin", "lofi", "8-bit", "orchestra", "ambient"]
    detected_style = "music"
    for kw in style_keywords:
        if kw in prompt.lower():
            detected_style = kw
            break

    # Vibeに応じたベースファイル名 (番号 _P1, _P2 などを含める)
    base_name = vibe
    p_tag = f"_P{idx+1}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}{p_tag}_{timestamp}.wav"
    
    # Ensure absolute path if it looks relative
    if not os.path.isabs(output_dir):
        from utils import get_app_dir
        output_dir = os.path.join(get_app_dir(), output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Stable Audio Open 1.0 は最大47秒まで対応
    # 47秒を超える動画（例: 67秒）の場合、半分の長さ（33.5秒）を2回ループさせる方が
    # 1つの長いセグメントを末尾で無理やり繋ぐより音楽的に自然になりやすい。
    is_looped = False
    if duration_seconds > 47.0:
        audio_duration = duration_seconds / 2.0
        is_looped = True
    else:
        audio_duration = min(duration_seconds, 47.0)

    print(f"\n" + "="*50)
    print(f"🎬 AI BGM GENERATION: {vibe}")
    print(f"="*50)
    print(f"  - Style:    {detected_style}")
    if is_looped:
        print(f"  - Length:   {duration_seconds}s -> {audio_duration:.1f}s (Loop-optimized)")
    else:
        print(f"  - Length:   {audio_duration:.1f}s (Model Limit: 47s)")
    print(f"  - Output:   {os.path.basename(output_path)}")
    print(f"  - Prompt:   {prompt}")
    if negative_prompt:
        print(f"  - Negative: {negative_prompt}")
    print(f"-"*50)

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

        # Use EulerDiscreteScheduler for stability
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # 生成
        print(f"  音楽を生成中...")
        # 生成実行
        output = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            audio_end_in_s=audio_duration
        ).audios
        print() # Force newline after tqdm progress bar
        
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
