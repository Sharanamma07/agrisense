# AgriSense — Plant Disease Detection System

Image-based leaf disease classifier using transfer learning (MobileNetV2),
served via a Flask API + simple web UI.

## Project Structure
```
agrisense/
├── data/                   # Dataset (PlantVillage) goes here
│   ├── train/
│   └── val/
├── models/                 # Saved trained models (.h5 / .keras)
├── notebooks/              # EDA / experimentation notebooks
├── static/
│   └── uploads/            # Temp storage for user-uploaded images
├── templates/
│   └── index.html          # Upload UI
├── train.py                # Model training script
├── app.py                  # Flask app (inference API + UI)
├── disease_info.json       # Disease -> treatment advice lookup
├── requirements.txt
└── README.md
```

## Setup

1. **Get the dataset**
   Download PlantVillage from Kaggle:
   https://www.kaggle.com/datasets/emmarex/plantdisease
   Extract into `data/` and split into `data/train/` and `data/val/`
   (80/20 split, one subfolder per class — Keras' `image_dataset_from_directory`
   expects this layout).

2. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Train**
   ```bash
   python train.py
   ```
   This fine-tunes MobileNetV2 and saves the best model to `models/agrisense_model.keras`.

4. **Run the app**
   ```bash
   python app.py
   ```
   Visit http://localhost:5000, upload a leaf image, get a diagnosis.

## Next steps / stretch goals
- Add Grad-CAM heatmap overlay on predictions (great demo feature)
- Test on PlantDoc (real-world field images) to report generalization honestly
- Deploy to HuggingFace Spaces or Render free tier
