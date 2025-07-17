import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    try:
        # Change to the app directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
        
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 SunScore Analytics stopped")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
