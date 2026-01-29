# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for PCDS Installer
Build with: pyinstaller PCDS_Setup.spec
This bundles the tray agent exe inside the installer.
"""

a = Analysis(
    ['installer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dist/pcds_tray_agent.exe', '.'),  # Bundle the tray agent inside
    ],
    hiddenimports=[],
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
    a.binaries,
    a.datas,
    [],
    name='PCDS_Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    uac_admin=False,  # No admin required for tray app
)
