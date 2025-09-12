# TIMIT Speech Recognition with RabbitMQ

A distributed speech recognition system using TIMIT audio files, RabbitMQ for task queuing, and WebSockets for client communication.

## Features

- Real TIMIT audio file processing (200 WAV files: 0.wav to 199.wav)
- RabbitMQ-based distributed task processing
- WebSocket interface for real-time communication
- Audio feature extraction and basic speech-to-text
- Scalable worker architecture

## Setup

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install websockets pika numpy torch
```

### 3. Run RabbitMQ with Docker

```bash
docker run --name my-rabbitmq -p 5672:5672 -p 15672:15672 -d rabbitmq:3-management
```

Access RabbitMQ Management UI at: http://localhost:15672 (guest/guest)

## Running the Scripts

### Standard Single-threaded Workers
```bash
# Terminal 1: Start single-threaded worker
python src/worker.py

# Terminal 2: Start producer (WebSocket server)
python src/producer.py

# Terminal 3+: Additional workers (optional)
python src/worker.py
```

### Multiprocessing Consumer Workers (NEW!)
```bash
# Terminal 1: Start multiprocessing worker (4 processes, batch size 5)
python src/mp_consumer_worker.py --pool-size 4 --batch-size 5

# Terminal 2: Start producer
python src/producer.py

# Terminal 3: Additional MP worker with different config
python src/mp_consumer_worker.py --pool-size 8 --batch-size 8 --batch-timeout 1.0
```

### Multiprocessing Consumer Options
- `--pool-size N`: Number of worker processes in the pool (default: 4)
- `--batch-size N`: Batch size for processing tasks (default: 5) 
- `--batch-timeout T`: Timeout in seconds to process partial batches (default: 2.0)

## Testing

Test the speech recognition system using:

### Basic Speech Recognition Test
```bash
python speech_client.py
```

### Batch Processing Test
```bash
python speech_client.py batch
```

### Original Simple Client
```bash
python client.py
```

### Manual Testing
Connect to WebSocket at `ws://localhost:8765` and send TIMIT file IDs (0-199) to get transcriptions.

## File Structure

```
TQ-simple-script/
├── src/
│   ├── producer.py       # WebSocket server + RabbitMQ producer
│   ├── worker.py         # Speech recognition worker
│   └── audio_utils.py    # Audio processing utilities
├── speech_client.py      # TIMIT-specific test client
├── client.py            # Simple test client
└── README.md
```

## How It Works

1. **Client** sends TIMIT file ID (0-199) via WebSocket
2. **Producer** receives request, creates task, sends to RabbitMQ queue
3. **Worker** processes audio file, extracts features, generates transcription
4. **Worker** sends result back via RabbitMQ to client-specific queue
5. **Producer** forwards transcription result to WebSocket client

## Benchmarking & Performance Testing

### Quick Start Benchmark
```bash
# Automated benchmark runner with system checks
python run_benchmark.py
```

### Manual Benchmark Options

#### Compare RabbitMQ vs Multiprocessing (Full)
```bash
python benchmark.py
```

#### Multiprocessing Only
```bash
python benchmark.py mp-only
```

#### RabbitMQ Only (requires running system)
```bash
python benchmark.py rabbitmq-only
```

#### Quick Test (10 files)
```bash
python benchmark.py quick
```

### Analyze Results
```bash
# Analyze most recent benchmark
python visualize_benchmark.py

# Analyze specific results file
python visualize_benchmark.py benchmark_results_20240912_143022.json
```

### Performance Comparison

The benchmark tests compare:

- **RabbitMQ Distributed Queue**: WebSocket → RabbitMQ → Workers → Results
- **Python Multiprocessing**: Direct multiprocessing.Pool with 1, 2, 4, 8 workers

**Metrics Measured:**
- Total processing time
- Throughput (tasks/second)
- Average processing time per task
- Success/failure rates
- Worker scaling efficiency

**Expected Results:**
- Multiprocessing: Better for single-machine, CPU-bound tasks
- RabbitMQ: Better for distributed systems, network resilience, scalability

### Advanced Benchmarking

#### Hybrid Worker Comparison
```bash
# Compare single-threaded vs multiprocessing workers
python src/hybrid_benchmark.py

# Quick hybrid test
python src/hybrid_benchmark.py --quick
```

#### Scaling Analysis  
```bash
# Test how different worker configurations scale
python scaling_test.py

# Quick scaling test
python scaling_test.py --quick
```

### Benchmark Files Structure
```
TQ-simple-script/
├── benchmark.py                    # Main benchmark coordinator  
├── run_benchmark.py                # Automated runner with checks
├── visualize_benchmark.py          # Results analysis and charts
├── scaling_test.py                 # Worker scaling analysis
├── src/
│   ├── worker.py                   # Single-threaded RabbitMQ worker
│   ├── mp_consumer_worker.py       # Multiprocessing RabbitMQ worker  
│   ├── mp_worker.py                # Pure multiprocessing implementation
│   ├── hybrid_benchmark.py         # Hybrid worker comparison
│   ├── rabbitmq_benchmark.py       # RabbitMQ benchmark client
│   └── ...
└── *_results_*.json                # Generated result files
```

## Consumer-side Multiprocessing Architecture

The new `mp_consumer_worker.py` implements a **hybrid approach**:

```
RabbitMQ Queue → Consumer Worker → Multiprocessing Pool → Audio Processing
                      ↓                    ↓
                 Batch Collection    Parallel Execution
                      ↓                    ↓  
                 Result Publishing ← Results Collection
```

### Key Features:
- **Batch Processing**: Collects multiple tasks before processing
- **Multiprocessing Pool**: Parallel audio processing within each worker
- **Smart Batching**: Time-based and size-based batch triggers
- **Resource Efficiency**: Better CPU utilization than single-threaded workers
- **Scalability**: Can run multiple multiprocessing workers simultaneously

### Performance Comparison:

| Method | Best Use Case | Pros | Cons |
|--------|---------------|------|------|
| Single Workers | Simple deployment | Low memory usage | Limited CPU utilization |  
| MP Consumers | High-throughput single machine | Full CPU utilization | Higher memory usage |
| Multiple MP Workers | Distributed high-throughput | Combines benefits | Complex configuration |

### Recommended Configurations:

```bash
# For 4-core machine:
python src/mp_consumer_worker.py --pool-size 4 --batch-size 4

# For 8-core machine:  
python src/mp_consumer_worker.py --pool-size 8 --batch-size 8 --batch-timeout 1.0

# For distributed setup (2 workers on 8-core):
python src/mp_consumer_worker.py --pool-size 4 --batch-size 5  # Terminal 1
python src/mp_consumer_worker.py --pool-size 4 --batch-size 5  # Terminal 2
```

## OpenAI Model Integration 🤖

The system now uses a **real OpenAI speech recognition model** instead of placeholder transcriptions.

### Model Features:
- **Quantized INT8 model** for efficient inference
- **Automatic fallback** if model loading fails
- **GPU/CPU detection** with automatic device selection
- **Confidence scoring** for transcription quality
- **Real-time factor** measurement for performance

### Model Loading:
```bash
# Test model loading and inference
python src/model_loader.py

# Run comprehensive model benchmark
python model_benchmark.py

# Quick model test
python model_benchmark.py --quick
```

### Model Path:
The model is automatically loaded from: `../exp/quant_outputs_openai/openai_model_int8_sd.pt`

### Enhanced Output:
Workers now provide:
- **Real transcriptions** from the OpenAI model
- **Confidence scores** (0.0 to 1.0)
- **Model type used** (openai_quantized, fallback, etc.)
- **Inference time** measurements
- **Real-time factor** for audio processing