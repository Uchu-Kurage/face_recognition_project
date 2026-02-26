import face_recognition_models
from PyInstaller.utils.hooks import collect_all
import os
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 10)

# Platform check
is_mac = sys.platform == 'darwin'
is_win = sys.platform == 'win32'

models_path = os.path.dirname(face_recognition_models.__file__)

datas = [('assets', 'assets'), (models_path, 'face_recognition_models'), ('LICENSE_NOTICE.md', '.')]
binaries = []
hiddenimports = [
    'moviepy.editor',
    'moviepy.video.fx.all',
    'moviepy.audio.fx.all',
    'moviepy.video.VideoClip',
    'moviepy.video.compositing.CompositeVideoClip',
    'moviepy.audio.AudioClip',
    'decorator',
    'decorator',
    'onnxruntime'
]

# Collect all (datas, binaries, hiddenimports) for tricky packages
for pkg in ['imageio', 'moviepy', 'customtkinter', 'requests', 'face_recognition', 'dlib', 'onnxruntime']:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'tensorflow', 'keras', 'deepface', 'transformers', 'diffusers', 'torch', 'h5py', 
              'libpng', 'libjpeg', 'libtiff', 'libz', 'libwebp', 'lzma', '_lzma'],
    noarchive=False,
)

pyz = PYZ(a.pure)

pyz = PYZ(a.pure)

# PATCH: Replace system's incompatible liblzma with PIL's bundled version if available
import PIL
pil_path = os.path.dirname(PIL.__file__)
pil_lzma = os.path.join(pil_path, '.dylibs', 'liblzma.5.dylib')

if os.path.exists(pil_lzma):
    print(f"--- Patching liblzma with compatible version from: {pil_lzma} ---")
    new_binaries = []
    for name, path, typecode in a.binaries:
        if 'liblzma' in name:
            print(f"  Replacing {name} (was {path})")
            new_binaries.append((name, pil_lzma, typecode))
        else:
            new_binaries.append((name, path, typecode))
    a.binaries = new_binaries
else:
    print(f"--- Info: PIL-bundled liblzma not found at {pil_lzma}, using default ---")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Omokage',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if is_win else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Omokage',
)

if is_mac:
    app = BUNDLE(
        coll,
        name='Omokage.app',
        icon='assets/icon.icns',
        bundle_identifier='com.omokage.app',
        info_plist={
            'LSMinimumSystemVersion': '12.0',
            'NSHighResolutionCapable': 'True'
        }
    )
