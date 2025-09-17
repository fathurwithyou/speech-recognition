# Whisper Parallel Processing Benchmark

This benchmark tool implements Task 4 from CLAUDE.md to evaluate parallel processing resource scaling for the Whisper transcription service.

## Overview

The benchmark measures how adding more CPU cores and memory affects transcription throughput by testing different multiprocessing pool sizes. It provides insights into the optimal configuration for scaling the service.

## Prerequisites

Install required dependencies using `uv`:

```bash
uv sync
```

The required dependencies are already added to `pyproject.toml`:
- `psutil>=6.1.0` - System resource monitoring
- `matplotlib>=3.9.0` - Performance visualization
- `pandas>=2.2.0` - Data analysis and export

Ensure you have:
- RabbitMQ server running locally
- TIMIT evaluation dataset in `../timit_eval/` directory
- At least 10-20 WAV files for testing

## Usage

### Quick Start

Run the full benchmark suite:

```bash
uv run python benchmark.py
```

### Command Line Options

```bash
uv run python benchmark.py [OPTIONS]

Options:
  --test-type {baseline,scaling,full}
                        Type of benchmark to run (default: full)
  --pool-sizes POOL_SIZES [POOL_SIZES ...]
                        Pool sizes to test (default: 1 2 4 8)
  --num-files NUM_FILES
                        Number of files to test per configuration (default: 20)
  --audio-dir AUDIO_DIR
                        Directory containing audio files (default: ../timit_eval)
  --server-uri SERVER_URI
                        WebSocket server URI (default: ws://localhost:8765)
```

### Examples

**Baseline test only** (single process):
```bash
uv run python benchmark.py --test-type baseline --num-files 10
```

**Scaling test** with specific pool sizes:
```bash
uv run python benchmark.py --test-type scaling --pool-sizes 2 4 8 16 --num-files 25
```

**Custom audio directory**:
```bash
uv run python benchmark.py --audio-dir /path/to/audio/files --num-files 15
```

## What the Benchmark Tests

### 4.1 - Parallel Processing vs. CPU Core and Memory Scaling

1. **Baseline Test (Single Process)**
   - Tests with `--pool-size 1`
   - Establishes performance baseline
   - Confirms single process doesn't benefit from multiple cores

2. **Parallel Scaling Tests**
   - Tests multiple pool sizes (2, 4, 8, 16, etc.)
   - Measures total completion time for all transcriptions
   - Monitors CPU and memory usage throughout

3. **Resource Monitoring**
   - Tracks CPU utilization percentage
   - Monitors memory usage in GB
   - Records system load average
   - Identifies resource contention points

## Output and Results

### Console Output
- Real-time progress for each test
- Performance metrics per configuration
- Resource usage statistics
- Optimal configuration recommendations

### Generated Files
- `whisper_benchmark_YYYYMMDD_HHMMSS.png` - Performance visualizations
- `whisper_benchmark_data_YYYYMMDD_HHMMSS.csv` - Raw benchmark data
- Console report with scaling recommendations

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Processing Rate** | Audio-to-processing time ratio (1.0x = real-time) |
| **Total Time** | Wall-clock time to complete all transcriptions |
| **Success Rate** | Percentage of files successfully processed |
| **Peak CPU %** | Maximum CPU utilization during test |
| **Peak Memory** | Maximum memory usage in GB |
| **Efficiency** | Speedup divided by number of processes |

## Interpreting Results

### Performance Analysis

1. **Sweet Spot Identification**
   - Look for pool size with best processing rate
   - Consider resource usage vs. performance gain
   - Note point of diminishing returns

2. **Resource Contention**
   - High CPU usage (>90%) may indicate over-subscription
   - Memory usage growth shows model loading overhead
   - Load average indicates system stress

3. **Scaling Recommendations**
   - Optimal pool size for current hardware
   - Vertical scaling limits (adding more cores/RAM)
   - Horizontal scaling suggestions (more machines)

### Example Interpretation

```
Pool Size 4: Best performance (3.2x real-time, 85% CPU usage)
Pool Size 8: Diminishing returns (3.1x real-time, 95% CPU usage)
Pool Size 16: Resource contention (2.8x real-time, 98% CPU usage)

Recommendation: Use pool size 4 for optimal efficiency
```

## Production Scaling Strategy

Based on benchmark results, the tool provides:

1. **Optimal Configuration**
   - Recommended pool size per server
   - Expected throughput per machine

2. **Scaling Strategy**
   - Vertical scaling limits
   - Horizontal scaling approach
   - Cost-performance analysis

3. **Capacity Planning**
   - Users per server estimate
   - Infrastructure requirements
   - Load balancing considerations

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Ensure RabbitMQ is running
   sudo systemctl start rabbitmq-server

   # Check if WebSocket server is accessible
   curl -I http://localhost:8765
   ```

2. **Audio Files Not Found**
   ```bash
   # Check audio directory
   ls -la ../timit_eval/*.wav

   # Use custom directory
   python benchmark.py --audio-dir /path/to/audio/files
   ```

3. **Memory Issues**
   - Reduce `--num-files` for testing
   - Monitor system memory before testing
   - Close other applications to free RAM

4. **Process Cleanup**
   - Benchmark automatically cleans up processes
   - Manual cleanup if needed: `pkill -f mp_consumer_worker`

### Performance Tips

1. **System Preparation**
   - Close unnecessary applications
   - Ensure stable power supply
   - Monitor system temperature

2. **Test Configuration**
   - Start with smaller file counts
   - Gradually increase pool sizes
   - Use representative audio samples

3. **Resource Monitoring**
   - Watch system resources during tests
   - Identify bottlenecks early
   - Consider I/O limitations

## Understanding the Results

The benchmark helps answer key questions:

- **How many parallel processes can my system handle efficiently?**
- **What's the optimal balance between resource usage and performance?**
- **When should I scale horizontally vs. vertically?**
- **What throughput can I expect in production?**

Use the results to make informed decisions about infrastructure scaling and capacity planning for your Whisper transcription service.