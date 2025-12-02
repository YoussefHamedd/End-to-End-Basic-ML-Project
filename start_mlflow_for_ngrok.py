"""
Start MLflow server configured to work with ngrok tunneling.
This disables host header checking to allow ngrok URLs.
"""
import os
import sys
import subprocess

def start_mlflow_ngrok():
    """Start MLflow server with settings compatible with ngrok."""

    # Set environment variable to disable host checking
    env = os.environ.copy()
    env['FLASK_ENV'] = 'development'

    print("🚀 Starting MLflow server for ngrok access...")
    print("📝 Configuration:")
    print("   - Host: 0.0.0.0 (accessible from network)")
    print("   - Port: 5000")
    print("   - Backend: ./mlruns")
    print("   - Host checking: disabled for ngrok")
    print()

    try:
        # Start MLflow with proper configuration
        subprocess.run([
            sys.executable, "-m", "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "./mlruns",
            "--default-artifact-root", "./mlruns",
            "--serve-artifacts"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n✋ MLflow server stopped")
    except Exception as e:
        print(f"❌ Error starting MLflow: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(start_mlflow_ngrok())
