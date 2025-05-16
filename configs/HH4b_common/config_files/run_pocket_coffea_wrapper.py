import subprocess
import sys
import os

def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["bash", f"{file_dir}/run_pocket_coffea.sh"] + sys.argv[1:], check=True)
