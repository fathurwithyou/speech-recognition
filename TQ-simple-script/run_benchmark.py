#!/usr/bin/env python3
"""
Main benchmark runner script with setup validation and guided execution.
"""

import os
import sys
import subprocess
import time

def check_python_deps():
    """Check if required Python dependencies are installed."""
    required_packages = ['pika', 'websockets', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} missing")
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Docker is available")
            return True
    except:
        pass
    
    print("✗ Docker not available")
    return False

def check_rabbitmq_container():
    """Check if RabbitMQ container is running."""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=my-rabbitmq'], 
                              capture_output=True, text=True, timeout=10)
        if 'my-rabbitmq' in result.stdout:
            print("✓ RabbitMQ container is running")
            return True
    except:
        pass
    
    print("✗ RabbitMQ container not running")
    return False

def start_rabbitmq():
    """Start RabbitMQ container."""
    print("Starting RabbitMQ container...")
    try:
        # Stop existing container if running
        subprocess.run(['docker', 'stop', 'my-rabbitmq'], 
                      capture_output=True, timeout=10)
        subprocess.run(['docker', 'rm', 'my-rabbitmq'], 
                      capture_output=True, timeout=10)
        
        # Start new container
        result = subprocess.run([
            'docker', 'run', '--name', 'my-rabbitmq', 
            '-p', '5672:5672', '-p', '15672:15672', 
            '-d', 'rabbitmq:3-management'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ RabbitMQ container started")
            print("⏳ Waiting for RabbitMQ to be ready...")
            time.sleep(10)  # Give RabbitMQ time to start
            return True
        else:
            print(f"✗ Failed to start RabbitMQ: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error starting RabbitMQ: {e}")
        return False

def check_timit_files():
    """Check if TIMIT evaluation files are available."""
    timit_path = "../timit_eval"
    if not os.path.exists(timit_path):
        print(f"✗ TIMIT files not found at {timit_path}")
        return False
    
    wav_files = [f for f in os.listdir(timit_path) if f.endswith('.wav')]
    if len(wav_files) < 10:
        print(f"✗ Not enough TIMIT files (found {len(wav_files)}, need at least 10)")
        return False
    
    print(f"✓ Found {len(wav_files)} TIMIT WAV files")
    return True

def run_system_check():
    """Run complete system check."""
    print("🔍 SYSTEM CHECK")
    print("=" * 30)
    
    all_good = True
    
    # Check Python dependencies
    if not check_python_deps():
        all_good = False
    
    # Check Docker
    docker_available = check_docker()
    if not docker_available:
        print("   Note: Docker needed for RabbitMQ benchmark")
    
    # Check TIMIT files
    if not check_timit_files():
        all_good = False
    
    # Check RabbitMQ if Docker available
    if docker_available:
        if not check_rabbitmq_container():
            print("   RabbitMQ will be started automatically")
    
    return all_good, docker_available

def run_multiprocessing_only():
    """Run multiprocessing-only benchmark."""
    print("\n🚀 RUNNING MULTIPROCESSING BENCHMARK")
    print("=" * 40)
    
    try:
        result = subprocess.run([sys.executable, 'benchmark.py', 'mp-only'], 
                              timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⚠️  Benchmark timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def run_full_benchmark():
    """Run full benchmark including RabbitMQ."""
    print("\n🚀 RUNNING FULL BENCHMARK")
    print("=" * 40)
    
    # Start RabbitMQ if needed
    if not check_rabbitmq_container():
        if not start_rabbitmq():
            print("Failed to start RabbitMQ - running multiprocessing only")
            return run_multiprocessing_only()
    
    # Start RabbitMQ components in background
    print("Starting RabbitMQ producer and worker...")
    
    try:
        # Start producer
        producer_process = subprocess.Popen([sys.executable, 'src/producer.py'])
        time.sleep(3)  # Give producer time to start
        
        # Start worker
        worker_process = subprocess.Popen([sys.executable, 'src/worker.py'])
        time.sleep(3)  # Give worker time to start
        
        # Run benchmark
        print("Running benchmark...")
        result = subprocess.run([sys.executable, 'benchmark.py'], timeout=600)
        benchmark_success = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⚠️  Benchmark timed out after 10 minutes")
        benchmark_success = False
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        benchmark_success = False
    finally:
        # Clean up processes
        try:
            producer_process.terminate()
            worker_process.terminate()
            time.sleep(2)
            producer_process.kill()
            worker_process.kill()
        except:
            pass
    
    return benchmark_success

def main():
    """Main execution function."""
    print("🎯 SPEECH RECOGNITION BENCHMARK RUNNER")
    print("=" * 50)
    
    # System check
    system_ok, docker_available = run_system_check()
    
    if not system_ok:
        print("\n❌ System check failed. Please fix the issues above.")
        return 1
    
    print("\n✅ System check passed!")
    
    # Choose benchmark type
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        if docker_available:
            print("\nChoose benchmark mode:")
            print("1. Full benchmark (RabbitMQ + Multiprocessing)")
            print("2. Multiprocessing only")
            choice = input("Enter choice (1/2) [1]: ").strip()
            mode = "mp-only" if choice == "2" else "full"
        else:
            mode = "mp-only"
            print("\nRunning multiprocessing benchmark (Docker not available)")
    
    # Run benchmark
    if mode == "mp-only":
        success = run_multiprocessing_only()
    else:
        success = run_full_benchmark()
    
    if success:
        print("\n🎉 Benchmark completed successfully!")
        
        # Find and show results
        import glob
        result_files = glob.glob("benchmark_results_*.json")
        if result_files:
            latest_result = max(result_files, key=os.path.getctime)
            print(f"\n📊 Results saved to: {latest_result}")
            print(f"📈 View analysis with: python visualize_benchmark.py {latest_result}")
        
        return 0
    else:
        print("\n❌ Benchmark failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())