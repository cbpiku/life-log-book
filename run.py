import subprocess
import sys

# Replace 'app.py' with the path to your Streamlit app
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502", "--server.fileWatcherType=none"])
