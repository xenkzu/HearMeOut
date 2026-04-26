import subprocess, sys, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def build_frontend():
    frontend_dir = BASE_DIR / "frontend"
    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists():
        print("[*] Building frontend...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, shell=True)
        subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True, shell=True)
        print("[OK] Frontend built")
    else:
        print("[OK] Frontend already built")

def start_backend():
    print("[*] Starting HearMeOut...")
    # Make sure we're using the correct python executable
    python_cmd = sys.executable
    subprocess.run([python_cmd, "src-python/main.py"], cwd=BASE_DIR, check=True)

if __name__ == "__main__":
    build_frontend()
    start_backend()
