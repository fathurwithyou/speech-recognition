#!/bin/bash

# Docker-based Whisper Parallel Processing Benchmark
# Implements high-resource environment as specified in CLAUDE.md Task 4

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are available
check_requirements() {
    print_status "Checking requirements..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    # Check if TIMIT data exists
    if [ ! -d "../timit_eval" ]; then
        print_error "TIMIT evaluation data not found at ../timit_eval/"
        print_status "Please ensure you have TIMIT dataset available"
        exit 1
    fi

    # Check available system resources
    print_status "Checking system resources..."

    # Get total memory in GB
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))

    # Get CPU core count
    CPU_CORES=$(nproc)

    print_status "System resources detected:"
    print_status "  CPU Cores: $CPU_CORES"
    print_status "  Total RAM: ${TOTAL_MEM_GB}GB"

    if [ "$TOTAL_MEM_GB" -lt 16 ]; then
        print_warning "System has less than 16GB RAM. Benchmark may not run optimally."
    fi

    if [ "$CPU_CORES" -lt 8 ]; then
        print_warning "System has less than 8 CPU cores. Consider using a machine with more cores for comprehensive testing."
    fi

    print_success "Requirements check passed"
}

# Build and start services
start_services() {
    print_status "Building and starting Docker services..."

    # Create results directory
    mkdir -p ./benchmark_results

    # Build and start services
    docker-compose up -d --build

    # Wait for services to be healthy
    print_status "Waiting for services to be ready..."

    # Wait for RabbitMQ
    timeout=60
    counter=0
    while [ $counter -lt $timeout ]; do
        if docker-compose exec -T rabbitmq rabbitmq-diagnostics check_port_connectivity &> /dev/null; then
            print_success "RabbitMQ is ready"
            break
        fi
        sleep 2
        counter=$((counter + 2))
    done

    if [ $counter -ge $timeout ]; then
        print_error "RabbitMQ failed to start within $timeout seconds"
        exit 1
    fi

    # Wait for producer service
    sleep 5
    if docker-compose ps whisper-producer | grep -q "Up"; then
        print_success "Producer service is running"
    else
        print_error "Producer service failed to start"
        exit 1
    fi

    print_success "All services are ready"
}

# Run benchmark in container
run_benchmark() {
    local test_type=${1:-"full"}
    local pool_sizes=${2:-"1 2 4 8 16"}
    local num_files=${3:-"20"}

    print_status "Running benchmark with the following configuration:"
    print_status "  Test type: $test_type"
    print_status "  Pool sizes: $pool_sizes"
    print_status "  Number of files: $num_files"

    # Convert pool_sizes to array format for Python
    pool_sizes_arg=$(echo "$pool_sizes" | tr ' ' ' ')

    print_status "Starting benchmark in container..."

    docker-compose exec whisper-benchmark bash -c "
        cd /app &&
        uv run python benchmark.py \\
            --test-type $test_type \\
            --pool-sizes $pool_sizes_arg \\
            --num-files $num_files \\
            --audio-dir /app/timit_eval \\
            --server-uri ws://whisper-producer:8765
    "

    if [ $? -eq 0 ]; then
        print_success "Benchmark completed successfully"

        # Copy results from container
        print_status "Copying results from container..."
        docker-compose exec whisper-benchmark bash -c "
            cp *.png *.csv /app/benchmark_results/ 2>/dev/null || true
        "

        if [ -n "$(ls -A ./benchmark_results/*.png 2>/dev/null)" ]; then
            print_success "Results saved to ./benchmark_results/"
            ls -la ./benchmark_results/
        fi
    else
        print_error "Benchmark failed"
        exit 1
    fi
}

# Interactive benchmark runner
run_interactive() {
    print_status "Running interactive benchmark..."

    docker-compose exec whisper-benchmark bash -c "
        cd /app && bash
    "
}

# Monitor resources during benchmark
monitor_resources() {
    print_status "Starting resource monitoring..."

    # Run monitoring in background
    {
        while true; do
            echo "$(date): CPU: $(docker stats --no-stream --format 'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}' | grep whisper-benchmark)"
            sleep 5
        done
    } &

    MONITOR_PID=$!
    echo $MONITOR_PID > /tmp/monitor.pid

    print_status "Resource monitoring started (PID: $MONITOR_PID)"
}

# Stop monitoring
stop_monitoring() {
    if [ -f "/tmp/monitor.pid" ]; then
        MONITOR_PID=$(cat /tmp/monitor.pid)
        kill $MONITOR_PID 2>/dev/null || true
        rm -f /tmp/monitor.pid
        print_status "Resource monitoring stopped"
    fi
}

# Clean up services
cleanup() {
    print_status "Cleaning up services..."

    stop_monitoring

    docker-compose down

    # Optional: remove volumes (uncomment if needed)
    # docker-compose down -v

    print_success "Cleanup completed"
}

# Show service logs
show_logs() {
    local service=${1:-""}

    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Main script logic
main() {
    local command=${1:-"run"}

    case $command in
        "check")
            check_requirements
            ;;
        "start")
            check_requirements
            start_services
            ;;
        "run")
            check_requirements
            start_services
            run_benchmark "${2:-full}" "${3:-1 2 4 8 16}" "${4:-20}"
            ;;
        "interactive")
            check_requirements
            start_services
            run_interactive
            ;;
        "monitor")
            monitor_resources
            ;;
        "logs")
            show_logs "$2"
            ;;
        "stop")
            cleanup
            ;;
        "clean")
            cleanup
            docker system prune -f
            ;;
        *)
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  check                           - Check system requirements"
            echo "  start                          - Start Docker services only"
            echo "  run [type] [pools] [files]     - Run benchmark (default: full '1 2 4 8 16' 20)"
            echo "  interactive                    - Start interactive session"
            echo "  monitor                        - Start resource monitoring"
            echo "  logs [service]                 - Show service logs"
            echo "  stop                           - Stop and cleanup services"
            echo "  clean                          - Stop services and clean Docker"
            echo ""
            echo "Examples:"
            echo "  $0 run full '1 2 4 8' 15      - Run full benchmark with specific pool sizes"
            echo "  $0 run baseline '' 10          - Run baseline test only"
            echo "  $0 run scaling '2 4 8 16' 25   - Run scaling test with 25 files"
            echo "  $0 interactive                 - Open interactive shell in container"
            echo "  $0 logs whisper-benchmark      - Show benchmark service logs"
            ;;
    esac
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"