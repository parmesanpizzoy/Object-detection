# Custom Vehicle Object Detection: YOLOv8 vs. YOLOv11

A computer vision project that explores custom vehicle object detection using **YOLOv8s** and **YOLO11s**. The project builds a dataset pipeline, generates initial annotations through pseudo-labeling, trains both models, evaluates their performance, and compares their detection results.

## Project Overview

This project focuses on detecting and classifying vehicles into three custom categories:

| Class | Description |
|---|---|
| `roda2` | Motorcycles and bicycles |
| `roda3` | Bajaj, rickshaws, tuk-tuks, and similar three-wheeled vehicles |
| `roda4` | Cars, buses, and trucks |

The notebook compares YOLOv8s and YOLO11s using the same dataset and training configuration.

## Important Dataset Note

The original dataset contains images organized into folders, but **does not include bounding-box annotations**.

To prepare the dataset for object detection, this project uses an automatic pseudo-labeling approach with a pretrained YOLO11x COCO model:

- `roda2`: Uses detected bicycles and motorcycles.
- `roda3`: Selects the largest detected vehicle because these custom vehicle types are not directly represented in the COCO classes used.
- `roda4`: Uses detected cars, buses, and trucks.
- If no suitable detection is found, a fallback bounding box is generated to keep the pipeline running.

**Important:** These automatically generated labels are preliminary annotations, not verified ground truth. Some boxes may be inaccurate, especially fallback boxes. Manual annotation review and correction using tools such as Roboflow, CVAT, or LabelImg are recommended before drawing conclusions about model accuracy.

## Tech Stack

- **Python** — main programming language
- **Ultralytics YOLO** — object detection and model training
- **YOLOv8s** — first model evaluated
- **YOLO11s** — second model evaluated
- **YOLO11x** — pretrained teacher model for pseudo-label generation
- **Pandas** — dataset reporting and metric comparison
- **NumPy** — numerical operations
- **Matplotlib** — visualization of annotations, training curves, and predictions
- **Pillow** — image processing
- **Google Colab / Jupyter Notebook** — development environment

## Project Workflow

1. **Dataset Preparation**
   - Extract the dataset ZIP file.
   - Collect training and testing images.
   - Organize images into the three vehicle classes.

2. **Automatic Annotation**
   - Use a pretrained YOLO11x model to generate initial bounding boxes.
   - Convert detections into YOLO annotation format.
   - Create stratified training and validation splits.
   - Save a label report as `label_report.csv`.
   - Visualize sample annotations for inspection.

3. **Model Training**
   - Train YOLOv8s and YOLO11s.
   - Use pretrained weights and image augmentations.
   - Training configuration:
     - Epochs: 10
     - Image size: 640
     - Batch size: 8
     - Random seed: 42

4. **Model Evaluation**
   - Evaluate both models on validation and testing splits.
   - Compare:
     - Precision
     - Recall
     - mAP@50
     - mAP@50–95

5. **Prediction and Visualization**
   - Run predictions on the testing images.
   - Visualize detected bounding boxes.
   - Save prediction outputs for both models.

6. **Optional Export**
   - Select the model with the higher test mAP@50–95.
   - Export the selected model to ONNX format.
   - The notebook also includes example commands for image, video, and webcam inference.

## Repository Structure

```text
Object-detection/
├── README.md
├── yolo_kendaraan_v8_v11_(2).ipynb
└── dataset kendaraan.zip
```

The notebook generates its working files in a `work_yolo/` directory, including the processed YOLO dataset, label report, training runs, evaluation outputs, and prediction results.

## Getting Started

### Option 1: Google Colab (Recommended)

1. Clone or download this repository.
2. Open `yolo_kendaraan_v8_v11_(2).ipynb` in Google Colab.
3. Upload `dataset kendaraan.zip` to the Colab environment.
4. Make sure the ZIP file is available at:

```text
/content/dataset kendaraan.zip
```

5. Run the notebook cells from top to bottom.

The notebook installs its required Python packages in the first setup cell:

```bash
pip install -q ultralytics pillow-avif-plugin pandas matplotlib pyyaml
```

### Option 2: Local Environment

Install the dependencies:

```bash
pip install ultralytics pillow-avif-plugin pandas matplotlib pyyaml
```

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Then open `yolo_kendaraan_v8_v11_(2).ipynb`.

Update the `ZIP_PATH` variable in the notebook to match the location of your dataset ZIP file:

```python
ZIP_PATH = r"/path/to/dataset kendaraan.zip"
```

Run the notebook cells sequentially.

## Evaluation Metrics

The notebook compares both models using the following metrics:

| Metric | Description |
|---|---|
| Precision | Measures how many predicted detections are correct |
| Recall | Measures how many actual objects are detected |
| mAP@50 | Mean Average Precision at an IoU threshold of 0.50 |
| mAP@50–95 | Mean Average Precision across IoU thresholds from 0.50 to 0.95 |

The comparison table and training visualizations are generated when the notebook runs. Actual metric values are intentionally not hard-coded in this README.

## Output Files

The notebook automatically generates:

- `label_report.csv` — report of generated annotations, including fallback labels.
- `yolo_dataset/` — processed images, labels, and `data.yaml`.
- `runs/` — model training and evaluation outputs.
- `predict/` — prediction images for YOLOv8s and YOLO11s.
- ONNX export of the selected model.

All generated files are stored inside `work_yolo/`.

## Limitations

- The original dataset does not contain manually verified bounding-box annotations.
- Pseudo-labeling may introduce incorrect or incomplete bounding boxes.
- The `roda3` category is approximated using the largest detected vehicle because its vehicle types are not directly represented in the selected COCO classes.
- The dataset is relatively small, so model performance may not generalize well to unseen images.
- Evaluation metrics should be interpreted carefully because the generated annotations are not verified ground truth.

Improving annotation quality and expanding the dataset are important next steps.

## Future Improvements

- Manually review and correct bounding-box annotations.
- Expand the dataset with more diverse vehicle images.
- Experiment with different YOLO model sizes and training configurations.
- Compare inference speed and model size alongside detection metrics.
- Test the trained model on additional real-world images and videos.

## Author

**Sofie Dearni**

Computer Science Undergraduate | Computer Vision | IoT & Software Engineering

GitHub: [@parmesanpizzoy](https://github.com/parmesanpizzoy)
