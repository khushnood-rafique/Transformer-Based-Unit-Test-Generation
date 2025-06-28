import os
import subprocess
import sys

VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"

def create_virtualenv():
    if not os.path.isdir(VENV_DIR):
        print("[+] Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("[✓] Virtual environment already exists.")

def activate_virtualenv():
    # Activation is shell-specific, handled outside Python script
    print("[ℹ️] Please activate your virtual environment manually:")
    print(f"    On Windows PowerShell: .\\{VENV_DIR}\\Scripts\\Activate.ps1")
    print(f"    On CMD:                .\\{VENV_DIR}\\Scripts\\activate.bat")
    print(f"    On macOS/Linux:        source {VENV_DIR}/bin/activate")

def install_dependencies():
    python_exe = os.path.join(
        VENV_DIR,
        "Scripts" if os.name == "nt" else "bin",
        "python"
    )
    if os.path.exists(REQUIREMENTS_FILE):
        print("[+] Installing dependencies from requirements.txt...")
        subprocess.check_call([os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python"),
                               "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        print("[✓] Dependencies installed.")
    else:
        print("[!] No requirements.txt found. Creating one...")
        try:
            with open(REQUIREMENTS_FILE, "w") as f:
                subprocess.check_call([python_exe, "-m", "pip", "freeze"], stdout=f)
            print(f"[✓] Created {REQUIREMENTS_FILE}")
        except subprocess.CalledProcessError as e:
            print("[✗] pip freeze failed:", e)

def main():
    create_virtualenv()
    install_dependencies()
    activate_virtualenv()

if __name__ == "__main__":
    main()
