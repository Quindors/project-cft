# cft.spec — PyInstaller build spec for project-cft
#
# Build:
#   pyinstaller cft.spec --noconfirm
#
# Produces dist/cft/cft.exe (onedir). Distribute via installer.iss.

import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None


# --- Bundled, read-only resources -------------------------------------------
# Format: (source_path, dest_subdir_in_bundle)
datas = [
    ('prompt-main.txt',   '.'),
    ('prompt-critic.txt', '.'),
]

# The focus-dashboard production build, if build.ps1 staged it into ./dashboard.
# Optional: when absent, the tray falls back to a hosted DASHBOARD_URL.
if os.path.isdir('dashboard'):
    datas.append(('dashboard', 'dashboard'))


# --- Hidden imports ---------------------------------------------------------
# supabase, openai, and friends use dynamic imports that PyInstaller's static
# analysis sometimes misses.
hiddenimports = []
hiddenimports += collect_submodules('supabase')
hiddenimports += collect_submodules('supabase_auth')
hiddenimports += collect_submodules('supabase_functions')
hiddenimports += collect_submodules('postgrest')
hiddenimports += collect_submodules('gotrue')
hiddenimports += collect_submodules('storage3')
hiddenimports += collect_submodules('realtime')
hiddenimports += collect_submodules('openai')
hiddenimports += ['h2', 'hpack', 'hyperframe']
hiddenimports += ['pystray._win32', 'PIL.ImageDraw']


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # NOT excluded: tkinter. The floating timer widget and the full-screen
        # focus alert (monitor/tk_host.py) are Tk, so the build needs it.
        'matplotlib',
        'pytest',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cft',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,         # windowed: no terminal; boot logs go to cft.log (run.py)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,             # add 'cft.ico' here if you want a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='cft',
)
