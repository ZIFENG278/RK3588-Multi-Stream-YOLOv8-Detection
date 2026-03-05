# RK3588 多路 YOLOv8 目标检测 - 项目架构

## 1. 项目概览

```
yolo_multi/
├── main_rknn_pipeline.py     # 主程序 (Pipeline 架构) [当前使用]
├── src_rknn/               # RKNN 推理模块
│   ├── __init__.py
│   ├── rknn_inference.py   # 核心推理引擎 (DFL, NMS)
│   └── config.py            # 配置类
├── src/                    # 可视化模块
│   ├── visualization.py     # 网格显示 (Visualizer, GridDisplay)
│   └── config.py
├── py_utils/               # 工具模块
│   ├── __init__.py
│   └── rknn_executor.py    # NPU 模型加载
├── docs/
│   └── ARCHITECTURE.md     # 架构文档
├── dx_app/                 # 历史示例代码 (保留)
├── video/                  # 测试视频
└── *.rknn                 # 模型文件
```

## 2. 整体架构 (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Main Thread                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Result Collection Loop                        │    │
│  │  - 从 result_queue 获取结果                                      │    │
│  │  - 绘制检测框 (draw_detections)                                 │    │
│  │  - 组帧显示 (Grid)                                              │    │
│  │  - 统计 FPS (process_fps 实时, video_fps 固定)                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↑
                            result_queue
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                    NPU Worker Threads (3)                          │
    │  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
    │  │  Worker 0  │    │  Worker 1  │    │  Worker 2  │           │
    │  │  ────────  │    │  ────────  │    │  ────────  │           │
    │  │ Inference  │ →  │ Inference  │ →  │ Inference  │           │
    │  │     ↓     │    │     ↓     │    │     ↓     │           │
    │  │Postprocess│    │Postprocess│    │Postprocess│           │
    │  └────────────┘    └────────────┘    └────────────┘           │
    └──────────────────────────────────────────────────────────────────┘
                                    ↑
                            task_queue
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                   Decode Worker Threads (N)                        │
    │  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
    │  │ Decode 0   │    │ Decode 1   │    │ Decode N-1 │           │
    │  │  ────────  │    │  ────────  │    │  ────────  │           │
    │  │   Decode   │ →  │   Decode   │ →  │   Decode   │           │
    │  │     ↓     │    │     ↓     │    │     ↓     │           │
    │  │Preprocess │    │Preprocess │    │Preprocess │           │
    │  └────────────┘    └────────────┘    └────────────┘           │
    └──────────────────────────────────────────────────────────────────┘
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                      Video Sources (N)                            │
    │   video/output_000.mp4  video/output_001.mp4  ...              │
    └──────────────────────────────────────────────────────────────────┘
```

## 3. 核心组件

### 3.1 FrameTask - 任务数据单元

```python
class FrameTask:
    stream_id: int           # 流 ID
    frame: np.ndarray       # 原始帧
    orig_shape: tuple       # 原始尺寸 (H, W)
    processed: np.ndarray   # 预处理后图像 (640x640 RGB)
    video_fps: float        # 原始视频帧率 (固定)
    detections: list        # 检测结果

    # 耗时统计 (ms)
    decode_time: float
    preprocess_time: float
    infer_time: float
    postprocess_time: float
```

### 3.2 DecodeWorker - 视频解码线程

```
职责:
  1. 循环读取视频帧 (cv2.VideoCapture)
  2. Letterbox 预处理 (保持宽高比缩放)
  3. BGR → RGB 转换
  4. 放入 task_queue

耗时:
  - Decode: ~10ms/帧
  - Preprocess: ~15ms/帧
```

### 3.3 NPUWorker - NPU 推理线程

```
职责:
  1. 从 task_queue 获取任务
  2. NPU 推理 (绑定特定核心)
  3. Postprocess 后处理 (DFL + NMS + 坐标映射)
  4. 放入 result_queue

耗时:
  - Inference: ~30-35ms/帧
  - Postprocess: ~14-18ms/帧
```

### 3.4 PipelineDetector - 主控类

```python
class PipelineDetector:
    config: Config
    video_paths: List[str]
    num_streams: int
    num_cores: int

    models: List[YOLOv8RKNN]   # 每核心一个模型
    task_queue: Queue           # 任务队列
    result_queue: Queue         # 结果队列
    stop_event: Event           # 停止信号

    decode_workers: List[DecodeWorker]
    npu_workers: List[NPUWorker]
    grid_display: GridDisplay
```

## 4. 数据流

```
输入: 视频文件
  │
  ▼
DecodeWorker 0 ──→ task_queue ──→ NPUWorker 0 ──→ result_queue ──→ Main Thread
  │                                            │
  ▼                                            ▼
DecodeWorker 1 ──→ task_queue ──→ NPUWorker 1 ──→ result_queue ──→ 绘制 + 显示
  │                                            │
  ▼                                            ▼
DecodeWorker N ──→ task_queue ──→ NPUWorker 2 ──→ result_queue ──→ 统计 FPS
```

## 5. 性能数据

### 9 流 3 核心测试结果

| 阶段 | 平均耗时 | 占比 |
|------|---------|------|
| Decode | 9.5 ms | 12% |
| Preprocess | 15 ms | 19% |
| Inference | 32 ms | 41% |
| Postprocess | 15 ms | 19% |
| Draw | 7 ms | 9% |
| **总计** | **78 ms** | 100% |

**FPS ≈ 12-13 FPS**

### NPU 利用率分析

```
理论: 3 核心并行
实际: ~33% 利用率

原因:
  - Postprocess 是 Python CPU 操作，受 GIL 限制
  - 同一时刻只能有一个线程执行 Python 代码
  - Postprocess 处理 8400 个预测框 (DFL + NMS)
```

## 6. 依赖关系

```
main_rknn_pipeline.py
    │
    ├── src_rknn/rknn_inference.py
    │   └── YOLOv8RKNN
    │       ├── rknn.api.RKNN
    │       ├── py_utils.rknn_executor.RKNN_model_container
    │       └── post_process (NMS, DFL)
    │
    ├── src_rknn/config.py
    │   └── Config
    │
    ├── src/visualization.py
    │   ├── Visualizer (draw_detections, draw_stream_label)
    │   └── GridDisplay (create_grid, show)
    │
    └── OpenCV (视频解码, 图像处理)
```

## 7. 运行命令

```bash
# 基础运行 (9 流, 3 核心)
python main_rknn_pipeline.py

# 无显示模式 (适合性能测试)
python main_rknn_pipeline.py --no-display --max-frames 100

# 自定义参数
python main_rknn_pipeline.py \
    --num-streams 6 \
    --num-cores 3 \
    --model yolov8n.rknn \
    --video-dir video \
    --conf-threshold 0.5 \
    --iou-threshold 0.45
```

## 8. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --num-streams | 9 | 视频流数量 |
| --num-cores | 3 | NPU 核心数量 (1-3) |
| --model | yolov8n.rknn | 模型文件 |
| --conf-threshold | 0.4 | 置信度阈值 |
| --iou-threshold | 0.45 | NMS IoU 阈值 |
| --no-display | False | 无显示模式 |
| --max-frames | None | 限制帧数 |

## 9. 显示信息说明

- **Vid**: 原始视频帧率 (固定值，从 cv2.CAP_PROP_FPS 读取)
- **Inf**: 实时推理帧率 (计算公式: 1.0 / dt)

## 10. 优化记录

### 已完成优化
1. ~~重复 draw_detections~~ - create_grid 直接使用已绘制帧
2. ~~draw_stream_label 冗余 copy~~ - 减少一次 frame.copy()

### 待优化方向
1. **Postprocess 加速**
   - 使用 Numba 加速 DFL/NMS
   - 预期提升: 15ms → 5ms

2. **FP16 模型**
   - 使用 FP16 量化，NPU 加速 DFL 计算
   - 预期减少 postprocess 时间

3. **Batch Inference**
   - 多帧批量推理
   - 提高 NPU 利用率
