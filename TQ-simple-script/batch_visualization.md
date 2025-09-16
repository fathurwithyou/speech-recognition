# Visualisasi Batch Processing dalam Speech Recognition System

## 1. Skema Arsitektur Batch Processing

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐
│   WebSocket     │    │    RabbitMQ      │    │  Multiprocessing Pool   │
│    Client       │────│     Queue        │────│    (Worker Processes)   │
└─────────────────┘    └──────────────────┘    └─────────────────────────┘
         │                        │                           │
         │                        │                           │
    Audio Files              Task Batching              Batch Processing
      (1-by-1)                (Group Tasks)            (Parallel Execution)
```

## 2. Batch Size Flow Visualization

### Tanpa Batch Processing (batch_size = 1)
```
Task Flow: [T1] → [T2] → [T3] → [T4] → [T5] → [T6]
Workers:   W1     W2     W3     W4     W1     W2
Timeline:  ├──────┼──────┼──────┼──────┼──────┼──────►
           0s    3s    6s    9s   12s   15s   18s

Karakteristik:
- Setiap task diproses segera saat tiba
- Workers bisa idle menunggu task baru
- Latency rendah per task
- Throughput rendah secara keseluruhan
```

### Dengan Batch Processing (batch_size = 3)
```
Batch 1: [T1, T2, T3] → Process Parallel
Batch 2: [T4, T5, T6] → Process Parallel

Workers:  W1   W2   W3   W4      W1   W2   W3
Tasks:   [T1] [T2] [T3] idle   [T4] [T5] [T6]
Timeline: ├────────────┼────────────┼─────────►
          0s          4s          8s       12s

Karakteristik:
- Tasks dikumpulkan dulu sebelum diproses
- Semua workers aktif bersamaan
- Latency sedikit lebih tinggi per task
- Throughput tinggi secara keseluruhan
```

## 3. Perbandingan Kinerja Berdasarkan Batch Size

### Batch Size = 1 (No Batching)
```
Advantages:
✓ Latency minimal per request
✓ Tidak ada delay waiting
✓ Simple processing logic

Disadvantages:
✗ Worker utilization rendah
✗ Overhead tinggi per task
✗ Context switching frequency tinggi
```

### Batch Size = 5 (Medium Batching)
```
Advantages:
✓ Balanced latency vs throughput
✓ Good worker utilization
✓ Reduced overhead per task
✓ Better resource efficiency

Disadvantages:
✗ Slight delay saat menunggu batch penuh
✗ Complexity increase
```

### Batch Size = 20 (Large Batching)
```
Advantages:
✓ Maximum worker utilization
✓ Minimal overhead per task
✓ Highest throughput

Disadvantages:
✗ High latency untuk task pertama dalam batch
✗ Memory usage lebih tinggi
✗ Jika ada failure, banyak task terdampak
```

## 4. Batch Timeout Mechanism

```
Batch Formation with Timeout (batch_size=5, timeout=2s):

Scenario 1: Batch Penuh
[T1] → [T2] → [T3] → [T4] → [T5] → PROCESS (batch penuh)
 0s    0.2s   0.4s   0.6s   0.8s

Scenario 2: Timeout Triggered
[T1] → [T2] → [T3] → .... → .... → PROCESS (2s timeout)
 0s    0.2s   0.4s   1.8s   2.0s
                             ↑
                        timeout hits
```

## 5. Worker Pool dengan Batch Processing

### Current Configuration
```
Pool Size: 4 workers
Batch Size: 5 tasks
Batch Timeout: 2.0 seconds
Threads per Model: 4 threads

┌─────────────────────────────────────────────────────────┐
│                    Worker Pool                          │
├─────────────┬─────────────┬─────────────┬─────────────┤
│   Worker 1  │   Worker 2  │   Worker 3  │   Worker 4  │
│             │             │             │             │
│ Model       │ Model       │ Model       │ Model       │
│ Loaded      │ Loaded      │ Loaded      │ Loaded      │
│             │             │             │             │
│ 4 threads   │ 4 threads   │ 4 threads   │ 4 threads   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## 6. Skenario Praktis Batch Processing

### Input: 15 Audio Files
```
Files: [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]

Batch Formation (batch_size=5):
Batch 1: [A1, A2, A3, A4, A5]
Batch 2: [A6, A7, A8, A9, A10]
Batch 3: [A11, A12, A13, A14, A15]

Processing Timeline:
Time 0s:  Batch 1 → Workers [W1:A1, W2:A2, W3:A3, W4:A4, W1:A5]
Time 4s:  Batch 2 → Workers [W1:A6, W2:A7, W3:A8, W4:A9, W1:A10]
Time 8s:  Batch 3 → Workers [W1:A11, W2:A12, W3:A13, W4:A14, W1:A15]
Time 12s: All Complete
```

## 7. Keuntungan Batch Processing dalam Speech Recognition

### 1. **Resource Utilization**
```
Without Batching: CPU Usage = 25-40% (underutilized)
With Batching:    CPU Usage = 80-95% (fully utilized)
```

### 2. **Memory Efficiency**
```
Model Loading:
- Setiap worker load model sekali saat startup
- Model tetap di memory selama worker hidup
- Tidak ada reload overhead per request
```

### 3. **Throughput Improvement**
```
Sequential Processing: 1 task per 3s = 20 tasks/minute
Batch Processing:      5 tasks per 4s = 75 tasks/minute
Improvement:           275% increase
```

### 4. **Queue Management**
```
RabbitMQ Queue:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│T1 │T2 │T3 │T4 │T5 │T6 │T7 │T8 │T9 │T10│
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ↑                   ↑
Batch 1            Batch 2
(prefetch 10)
```

## 8. Tuning Parameters untuk Optimal Performance

### Small Files (< 2s audio):
```
batch_size = 8
batch_timeout = 1.0s
pool_size = 6
```

### Medium Files (2-5s audio):
```
batch_size = 5
batch_timeout = 2.0s
pool_size = 4
```

### Large Files (> 5s audio):
```
batch_size = 3
batch_timeout = 3.0s
pool_size = 4
```

## Kesimpulan

Batch processing memberikan:
1. **Efisiensi Resource**: Worker utilization maksimal
2. **Throughput Tinggi**: Multiple tasks processed simultaneously
3. **Cost Effectiveness**: Reduced overhead per task
4. **Scalability**: Easy to adjust based on workload

Trade-off yang perlu dipertimbangkan:
- Latency vs Throughput
- Memory usage vs Processing speed
- Complexity vs Performance gain