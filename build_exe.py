import PyInstaller.__main__
import os

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'equipment_analyzer.py',
    '--onefile',
    '--windowed',
    '--name=GVO_Equipment_Analyzer',
    f'--add-data={os.path.join(script_dir, "equipment.csv")}{os.pathsep}.',
    '--icon=NONE',
    '--clean',
    '--noconfirm',
    # Optimize for size and startup time
    '--strip',
    '--noupx'
]) 