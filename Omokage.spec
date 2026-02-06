import face_recognition_models
from PyInstaller.utils.hooks import collect_all
import os
import sys

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
    'decorator'
]

# Collect all (datas, binaries, hiddenimports) for tricky packages
for pkg in ['imageio', 'moviepy', 'customtkinter']:
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
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

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
        icon=None,
        bundle_identifier='com.omokage.app',
    )
