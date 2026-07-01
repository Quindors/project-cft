# cft.spec — PyInstaller build spec for project-cft
#
# Build:
#   pyinstaller cft.spec --noconfirm
#
# Produces dist/cft/cft.exe (onedir). Distribute via installer.iss.

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None


# --- Bundled, read-only resources -------------------------------------------
# Format: (source_path, dest_subdir_in_bundle)
datas = [
    ('prompt-main.txt',   '.'),
    ('prompt-critic.txt', '.'),
]


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


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'tkinter',
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
    console=True,          # kept visible for [boot] logs; launcher hides the window
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
