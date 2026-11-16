# Model comparison

URFD columns use each model's thresholds frozen on CAUCAFall; 'literature' uses the draft's default thresholds. Event level = 'fall detected'.

| Model | COCO box mAP50-95 | COCO pose mAP50-95 | CAUCAFall F1 (calib.) | URFD F1 | URFD Acc | URFD Recall | URFD Spec. | URFD frame F1 | URFD F1 (literature) | Detector FPS | Detector ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-pose | 0.696 | 0.506 | 0.786 | 0.744 | 0.710 | 0.967 | 0.513 | 0.688 | 0.623 | 90.700 | 11.025 |
| yolo11n-pose | 0.705 | 0.503 | 0.807 | 0.771 | 0.768 | 0.900 | 0.667 | 0.638 | 0.719 | 76.010 | 13.157 |
| yolo26n-pose | 0.669 | 0.569 | 0.800 | 0.698 | 0.725 | 0.733 | 0.718 | 0.473 | 0.567 | 75.290 | 13.282 |
