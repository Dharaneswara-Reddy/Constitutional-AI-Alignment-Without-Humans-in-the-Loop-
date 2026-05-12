"""
Diagnostic script to test TensorBoard startup
"""
import subprocess
import time
import socket
import sys

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            result = s.connect_ex(('localhost', port))
            return result == 0
        except:
            return False

def test_tensorboard():
    print("=" * 70)
    print("TENSORBOARD DIAGNOSTIC TEST")
    print("=" * 70)
    
    # Check if tensorboard is installed
    print("\n1. Checking if TensorBoard is installed...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "tensorboard"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ TensorBoard is installed")
            # Extract version
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    print(f"   Version: {line.split(':')[1].strip()}")
        else:
            print("   ❌ TensorBoard is NOT installed")
            print("   Run: pip install tensorboard")
            return
    except Exception as e:
        print(f"   ❌ Error checking TensorBoard: {e}")
        return
    
    # Check if port 6006 is already in use
    print("\n2. Checking if port 6006 is available...")
    if is_port_in_use(6006):
        print("   ⚠️  Port 6006 is already in use!")
        print("   TensorBoard might already be running.")
        print("   Try opening: http://localhost:6006")
        return
    else:
        print("   ✅ Port 6006 is available")
    
    # Check if logs directory exists
    print("\n3. Checking logs directory...")
    from pathlib import Path
    log_dir = Path("logs/tensorboard")
    if log_dir.exists():
        print(f"   ✅ Log directory exists: {log_dir.absolute()}")
        # Count files
        files = list(log_dir.rglob("*"))
        print(f"   Found {len(files)} files in log directory")
    else:
        print(f"   ⚠️  Log directory does not exist: {log_dir.absolute()}")
        print("   Creating it...")
        log_dir.mkdir(parents=True, exist_ok=True)
        print("   ✅ Created log directory")
    
    # Try to start TensorBoard
    print("\n4. Starting TensorBoard...")
    print("   Command: tensorboard --logdir logs/tensorboard --port 6006")
    print("   (This will run for 10 seconds, then stop)")
    
    try:
        # Start TensorBoard process
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", "logs/tensorboard",
             "--port", "6006",
             "--host", "localhost"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for it to start
        print("\n   Waiting for TensorBoard to start...")
        for i in range(15):
            time.sleep(1)
            if is_port_in_use(6006):
                print(f"   ✅ TensorBoard started successfully after {i+1} seconds!")
                print(f"\n   🌐 Open in browser: http://localhost:6006")
                print("\n   Press Ctrl+C to stop TensorBoard...")
                
                # Keep it running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n   Stopping TensorBoard...")
                    process.terminate()
                    process.wait()
                    print("   ✅ TensorBoard stopped")
                return
            
            # Check if process died
            if process.poll() is not None:
                print(f"\n   ❌ TensorBoard process died!")
                stdout, stderr = process.communicate()
                print(f"\n   STDOUT:\n{stdout}")
                print(f"\n   STDERR:\n{stderr}")
                return
        
        # Timeout
        print("\n   ❌ TensorBoard did not start within 15 seconds")
        process.terminate()
        stdout, stderr = process.communicate()
        print(f"\n   STDOUT:\n{stdout}")
        print(f"\n   STDERR:\n{stderr}")
        
    except Exception as e:
        print(f"\n   ❌ Error starting TensorBoard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tensorboard()
