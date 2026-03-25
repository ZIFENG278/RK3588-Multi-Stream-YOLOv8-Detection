# RK3588 Multi-Stream YOLOv8 Detection

Multi-stream YOLOv8 object detection on RK3588 NPU using Python pipeline architecture.

## Features

- **Multi-stream processing**: Support 6+ video streams simultaneously
- **3-core NPU acceleration**: Fully utilize RK3588's 3 NPU cores
- **Pipeline architecture**: Decode → NPU Inference → Postprocess+Draw → Display
- **Real-time display**: Grid display with FPS statistics (Video FPS + Inference FPS)
- **Video recording**: Save detection grid as MP4 (`--save-video`)
- **Letterbox preprocessing**: Maintain aspect ratio for detection accuracy

## Requirements

- RK3588 development board
- RKNN toolkit-lite-2 installed
- OpenCV for Python (`pip install opencv-python`)
- Video files for testing

## Installation

```bash
# Install dependencies
pip install opencv-python
pip install rknn-toolkit-lite2
```

## Usage

### Basic Run (6 streams, 3 cores, display)

```bash
python main_rknn_pipeline.py
```

### Save Video Without Display (headless recording)

```bash
python main_rknn_pipeline.py --no-display --save-video --max-frames 300
```

### Custom Parameters

```bash
python main_rknn_pipeline.py \
    --num-streams 9 \
    --num-cores 3 \
    --num-postprocess 1 \
    --model yolov8n-i8-3588.rknn \
    --video-dir video \
    --conf-threshold 0.4 \
    --iou-threshold 0.45 \
    --save-video \
    --output-dir output
```

## Command Line Options

| Parameter          | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `--num-streams`    | 6       | Number of video streams                  |
| `--num-cores`      | 3       | NPU cores to use (1-3)                   |
| `--num-postprocess`| 3       | Postprocess worker threads (recommend 1)  |
| `--model`          | `yolov8n-i8-3588.rknn` | RKNN model file path        |
| `--video-dir`      | video   | Directory containing test videos          |
| `--conf-threshold` | 0.4     | Confidence threshold for detections       |
| `--iou-threshold`  | 0.45    | NMS IoU threshold                        |
| `--no-display`     | False   | Disable window display                   |
| `--save-video`     | False   | Save detection grid as MP4               |
| `--output-dir`     | output  | Output directory for saved videos        |
| `--max-frames`     | None    | Limit total frames processed             |

## Architecture

```
DecodeWorker[N]          # Decode video + letterbox preprocess
         │
         ▼
   npu_input_queue           # maxsize=20
         │
         ▼
NPUWorker[1-6]             # 3 cores run independently in parallel (NPU)
         │
         ▼
   npu_output_queue          # maxsize=20
         │
         ▼
PostProcessWorker (x3)      # postprocess + draw bounding boxes
         │          
         │          
         ▼
  postprocess_queue          # maxsize=20
         │
         ▼
Main Thread                  # Collect results, build grid, display/save
```

**Note on PostProcess Workers**: Python's GIL serializes all Python bytecode execution. Multiple postprocess workers compete for the GIL without gaining parallelism — 1 worker is faster than N workers. The `--num-postprocess` default of 3 exists for compatibility; change to 1 for best performance.

## Performance

### 9 Streams, 3 NPU Cores, 3 PostProcess Worker

| Stage        | Time  |
|--------------|-------|
| Decode       | ~2 ms |
| Preprocess   | ~3 ms |
| NPU Inference| ~35 ms |
| Postprocess  | ~23 ms |
| Draw         | ~1 ms  |
| **Total E2E**| **~65 ms** |

**Throughput: ~9-10 fps per stream**

## Project Structure

```
yolo_multi/
├── main_rknn_pipeline.py       # Main pipeline entry point
├── src/
│   ├── config.py               # Configuration dataclass
│   ├── worker.py               # DecodeWorker, NPUWorker, PostProcessWorker
│   └── visualization.py        # Visualizer, GridDisplay, VideoWriter
├── video/                      # Test videos
├── output/                     # Saved video output
└── *.rknn                     # RKNN model files
```

## Display Info

- **Vid**: Original video FPS (fixed value from video file)
- **Inf**: Real-time inference FPS (updated every second per stream)

## Models

Place your RKNN models in the project root. Available models:

| Model File             | Size  | Description    |
|------------------------|-------|---------------|
| `yolov8n-i8-3588.rknn`| Nano  | Fastest, lowest accuracy |
| `yolov8s-i8-3588.rknn`| Small | Balanced          |
| `yolov8m-i8-3588.rknn`| Medium| Higher accuracy     |

## License

MIT License
