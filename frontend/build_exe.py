import os
import subprocess
import sys
import shutil

def run_command(command, cwd=None):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root_dir)
    backend_dir = os.path.join(project_root, "backend")

    # 1. Verify frontend dist folder exists
    dist_dir = os.path.join(root_dir, "dist-app")
    if not os.path.exists(dist_dir) or not os.listdir(dist_dir):
        print("Error: React static files 'dist-app' folder not found or empty.")
        print("Please build the frontend first using: npm run build")
        sys.exit(1)
        
    print("Found React static assets in 'dist-app' directory.")

    # 2. Path to PyInstaller inside the backend virtual environment
    venv_pyinstaller = os.path.join(backend_dir, "venv", "Scripts", "pyinstaller.exe")
    if not os.path.exists(venv_pyinstaller):
        # Fallback to system pyinstaller if virtualenv doesn't have it
        venv_pyinstaller = "pyinstaller"
        print("Warning: PyInstaller not found in venv. Using global 'pyinstaller' command.")

    # 3. Assemble PyInstaller arguments
    # We add the static files folder 'dist-app' from the root into the EXE
    # On Windows, PyInstaller's --add-data takes the format "source_path;dest_relative_path"
    add_data_arg = f'"{os.path.join(root_dir, "dist-app")};dist"'
    
    pyinstaller_args = [
        f'"{venv_pyinstaller}"',
        "--onefile",
        f'--add-data {add_data_arg}',
        # Backend runtime files, resolved relative to the bundled modules (sys._MEIPASS)
        '--add-data "app/templates;app/templates"',
        '--add-data "assets;assets"',  # bundled models: pose, face, person detector (efficientdet_lite0.tflite)
        "--collect-all mediapipe",
        # Hidden imports needed by Uvicorn server to run properly when compiled
        "--hidden-import uvicorn.logging",
        "--hidden-import uvicorn.loops",
        "--hidden-import uvicorn.loops.auto",
        "--hidden-import uvicorn.protocols",
        "--hidden-import uvicorn.protocols.http",
        "--hidden-import uvicorn.protocols.http.auto",
        "--hidden-import uvicorn.protocols.websockets",
        "--hidden-import uvicorn.protocols.websockets.auto",
        "--hidden-import uvicorn.lifespan",
        "--hidden-import uvicorn.lifespan.on",
        # Ensure we have the fastapi components
        "--hidden-import fastapi",
        "--name virtual-trial-room",
        "main.py"
    ]
    
    pyinstaller_cmd = " ".join(pyinstaller_args)
    print("\n--- Compiling FastAPI and Frontend React Assets into single EXE ---")
    run_command(pyinstaller_cmd, cwd=backend_dir)
    
    # 4. Copy output exe + backend .env into frontend/release (own folder: frontend/.env is Vite's, not the backend's)
    compiled_exe_path = os.path.join(backend_dir, "dist", "virtual-trial-room.exe")
    release_dir = os.path.join(root_dir, "release")
    final_exe_path = os.path.join(release_dir, "virtual-trial-room.exe")

    if os.path.exists(compiled_exe_path):
        print(f"\nSuccess! Copying compiled EXE to {release_dir}...")
        os.makedirs(release_dir, exist_ok=True)
        shutil.copy2(compiled_exe_path, final_exe_path)
        print(f"Executable is available at: {final_exe_path}")
        # load_dotenv() in the exe reads .env from the launch folder; never bundle the API key into the exe
        env_path = os.path.join(backend_dir, ".env")
        if os.path.exists(env_path):
            shutil.copy2(env_path, os.path.join(release_dir, ".env"))
            print("Copied backend .env next to the EXE. Launch the EXE from that folder.")
        else:
            print("Warning: backend/.env not found. Put a .env with GEMINI_API_KEY next to the EXE.")
    else:
        print("\nError: Executable compilation succeeded but the output file was not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
