import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gui.interface import run_gui

if __name__ == "__main__":
    run_gui()
