# -*- mode: python ; coding: utf-8 -*-
import os
import scipy

block_cipher = None
hiddenimports = []

a = Analysis(['HawkeyeGUI.py'],
             pathex=['C:\\Troy\\Work\\Kodukula Associates\\20191118 Hawkeye',
					f'{os.path.dirname(scipy.__file__)}\extra-dll'],
             binaries=[],
             datas=[('hawkeye.ico', '.'),('help_docs/*.xml','help_docs')],
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='HawkeyeGUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False , icon='hawkeye.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='HawkeyeGUI')
