# Plant Disease Detection

Welcome to **Plant Disease Detection**, a modern AI-powered application that automatically classifies plant leaf health from images.

## 🌿 What This Project Does

This project uses a high-performance ensemble of deep learning models to detect and classify leaf conditions into four categories:
- **Healthy**
- **Multiple Diseases**
- **Rust**
- **Scab**

The application is delivered through a beautiful and responsive **Streamlit web app** with visual probability feedback, prediction history, and export options.

## ✨ Key Features

- Accurate ensemble model built from **Xception + DenseNet121**
- Streamlit-based **interactive user interface**
- **Live image upload** and instant disease prediction
- **Probability bars** for every class
- **Prediction history** and analytics support
- **CSV export** for recorded predictions
- Built-in **feedback buttons** to mark results as helpful or incorrect

## 🚀 Quick Start

### 1. Create a fresh Python environment

Windows:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```
macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal.

## 📁 Recommended Project Structure

```
Plant-Disease-Detection/
├── app.py
├── utils.py
├── train_model.py
├── prepare_data.py
├── test_app.py
├── requirements.txt
├── model.h5
├── README.md
└── install_guide.md
```

> Note: Local virtual environment folders like `.venv` should not be committed to the repository.

## 🧠 Model Architecture

The prediction engine is built as an ensemble:

1. **Xception backbone** for strong feature extraction
2. **DenseNet121 backbone** for complementary representation
3. **Averaged outputs** from both models for robust final prediction

The app input is resized to **512×512 pixels**, normalized, and evaluated by the ensemble before producing the final scores.

## 📦 Dataset Layout

Use this directory structure when preparing your own training data:

```
dataset/
├── healthy/
├── multiple_diseases/
├── rust/
└── scab/
```

Each class folder should contain JPG or PNG leaf images.

## 💻 Usage

### Run the web application

```bash
python -m streamlit run app.py
```

### Test dependencies

```bash
python test_app.py
```

### Train a new model

1. Prepare your data with `prepare_data.py`
2. Configure training options in `train_model.py`
3. Run:
```bash
python train_model.py
```

## ✅ Requirements

- Python 3.8–3.11
- TensorFlow 2.13+
- Streamlit 1.28+
- Pillow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## 🔧 Notes

- If you use local virtual environments, keep them out of version control.
- Keep `model.h5` in the project root for the app to load predictions instantly.

## 💡 Tips for Best Results

- Use clear, well-lit images
- Keep the leaf centered in the frame
- Avoid heavy shadows and noise
- Use 512×512 resized input for the best model performance

## 🤝 Contributing

1. Fork this repo
2. Create a branch
3. Add your enhancements
4. Submit a pull request

## 📣 Support

If you enjoy this project, please star the repository and share any issues or improvement ideas.

---

This README is designed for clarity, speed, and real-world usage of the Plant Disease Detection app.