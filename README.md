# 🌿 FreshScan — AI Fruit & Vegetable Freshness Detector

<div align="center">

![FreshScan Banner](https://img.shields.io/badge/FreshScan-AI%20Freshness%20Detector-4ADE80?style=for-the-badge&logo=leaf&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-orange?style=flat-square)](https://keras.io/api/applications/mobilenet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

<br/>

> **An AI-powered web application that detects whether fruits and vegetables are fresh or rotten — in real time using your camera or by uploading an image — built by an IT student learning deep learning from scratch.**

<br/>

![Demo Preview](https://img.shields.io/badge/🎥%20Live%20Camera%20Detection-Working-4ADE80?style=for-the-badge)
![Demo Preview](https://img.shields.io/badge/🖼️%20Image%20Upload-Working-4ADE80?style=for-the-badge)
![Demo Preview](https://img.shields.io/badge/📊%20Nutrition%20Info-13%20Produces-FBBF24?style=for-the-badge)

</div>

---

## 👨‍💻 About Me & This Project

Hey! I'm an **IT student** and this is one of my most challenging and rewarding personal projects so far.

I built **FreshScan** because I wanted to go beyond textbook assignments and build something real — something that actually works in the physical world. The idea of pointing a camera at a fruit and having an AI instantly tell you if it's fresh or rotten felt like actual magic to me when I first thought of it. So I decided to build it myself.

This project pushed me hard. I had **zero experience with deep learning** before starting. I had to learn what CNNs are, how transfer learning works, why my model kept giving wrong predictions, and how to even get a camera feed running inside a Python notebook. There were moments I genuinely wanted to give up — but getting that live camera detection to finally work made every frustrating hour completely worth it.

This is a fully working, end-to-end AI project — from raw dataset to trained model to a deployed Streamlit web UI — built entirely by me as a solo IT student.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image Upload Detection** | Upload any fruit/vegetable photo and get instant freshness analysis |
| 🎥 **Live Camera Detection** | Real-time detection through your webcam with on-screen overlay |
| 📊 **Nutrition Information** | Detailed nutritional data for 13 fruits & vegetables |
| 💾 **Storage Tips** | How and where to store each produce for maximum shelf life |
| ⏳ **Shelf Life Guide** | Room temp vs refrigerator storage duration |
| ⚠️ **Spoilage Signs** | Visual and smell indicators for freshness vs rot |
| 🎯 **Top-5 Predictions** | Confidence scores for top 5 predictions with visual bars |
| 🌿 **13 Produce Types** | Apple, Banana, Orange, Mango, Strawberry, Tomato, Potato, Carrot & more |

---

## 🧠 How It Works

```
Your Image / Camera Feed
         │
         ▼
   Resize to 224×224
         │
         ▼
  MobileNetV2 Backbone
  (Pre-trained ImageNet)
         │
         ▼
  Custom Classification Head
  (Dense → Dropout → Softmax)
         │
         ▼
  Fresh / Rotten + Produce Type
         │
         ▼
  Nutrition & Storage Info Display
```

The model uses **Transfer Learning** with **MobileNetV2** as the backbone — a lightweight, fast CNN architecture originally trained on 1.4 million ImageNet images. Instead of training from scratch (which would take days and a huge dataset), the pre-trained weights give the model a massive head start at recognizing visual patterns, while the custom top layers learn to specifically distinguish fresh vs rotten produce.

Training happens in **two phases:**
- **Phase 1** — Base frozen, only top classification layers train
- **Phase 2** — Top 30 layers of MobileNetV2 unfrozen for fine-tuning

---

## 📁 Project Structure

```
FreshScan/
│
├── 📓 FruitVeg_Freshness_Detector.ipynb   ← Training notebook (Jupyter)
├── 🖥️  app.py                              ← Streamlit web UI
├── 📋 requirements.txt                    ← Python dependencies
│
├── 💾 freshness_model.h5                  ← Saved trained model
├── 🏷️  class_names.json                   ← Class index → label mapping
│
└── 📁 dataset/
    ├── train/
    │   ├── freshapples/
    │   ├── rottenapples/
    │   ├── freshbanana/
    │   ├── rottenbanana/
    │   └── ... (other classes)
    └── test/
        └── ... (same structure)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/freshscan.git
cd freshscan
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Get the dataset from Kaggle:
- 🔗 [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

Extract and place it as `dataset/` in the project root folder.

### 4. Train the Model (Jupyter Notebook)

```bash
jupyter notebook FruitVeg_Freshness_Detector.ipynb
```

Run all cells top to bottom. Training saves:
- `freshness_model.h5`
- `class_names.json`

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. Done! 🎉

---

## 🖥️ App Screenshots

> *(Add your screenshots here after running the app)*

| Image Detection Tab | Live Camera Tab |
|---|---|
| ![Image Tab](screenshot_image.jpg) | ![Camera Tab](screenshot_camera.jpg) |

---

## 🍎 Supported Produce

| # | Produce | Type | Classes |
|---|---------|------|---------|
| 1 | 🍎 Apple | Fruit | freshapples / rottenapples |
| 2 | 🍌 Banana | Fruit | freshbanana / rottenbanana |
| 3 | 🍊 Orange | Fruit | freshoranges / rottenoranges |
| 4 | 🥭 Mango | Fruit | freshmango / rottenmango |
| 5 | 🍓 Strawberry | Fruit | freshstrawberry / rottenstrawberry |
| 6 | 🍇 Grapes | Fruit | freshgrapes / rottengrapes |
| 7 | 🍅 Tomato | Vegetable | freshtomato / rottentomato |
| 8 | 🥔 Potato | Vegetable | freshpotato / rottenpotato |
| 9 | 🥕 Carrot | Vegetable | freshcarrot / rottencarrot |
| 10 | 🫑 Bell Pepper | Vegetable | freshbellpepper / rottenbellpepper |
| 11 | 🥒 Cucumber | Vegetable | freshcucumber / rottencucumber |
| 12 | 🥬 Spinach | Vegetable | freshspinach / rottenspinach |
| 13 | 🍎 Pomegranate | Fruit | freshpomegranate / rottenpomegranate |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **TensorFlow / Keras** | Deep learning framework |
| **MobileNetV2** | Pre-trained CNN backbone |
| **OpenCV** | Live camera capture & frame processing |
| **Streamlit** | Web application UI |
| **Pillow** | Image loading & preprocessing |
| **NumPy** | Array operations |
| **Matplotlib / Seaborn** | Training plots & confusion matrix |
| **Scikit-learn** | Evaluation metrics |
| **Jupyter Notebook** | Model training environment |

---

## 😤 My Challenges (The Honest Story)

I want to be real here — this project was **not easy** for me. Here's what actually happened behind the scenes:

### 🧠 Challenge 1: Understanding Deep Learning From Zero
When I started, I didn't really understand what a CNN was. I kept reading about "layers", "filters", "feature maps" and it didn't click for weeks. I had to watch multiple YouTube videos, re-read articles, and just start coding even when I didn't fully understand — and slowly things started making sense. Transfer learning especially confused me at first. Why would we use someone else's model? But once I understood that MobileNetV2 already "knows" what edges, textures, and shapes look like from training on millions of images — it finally clicked.

### 📉 Challenge 2: Model Not Giving Accurate Results
This was genuinely frustrating. My first few training runs gave me models that were barely better than guessing. The model would predict "fresh apple" on a clearly rotten banana. I went through a lot of trial and error:
- My images weren't being normalized correctly
- I wasn't using enough data augmentation
- The learning rate was too high and training was unstable
- I wasn't using validation data properly

Eventually I learned about two-phase training — freeze the base, train the top, then fine-tune — and that made a huge difference in accuracy.

### 🎥 Challenge 3: Getting Live Camera to Work
This was the hardest part for me practically. OpenCV inside Jupyter kept freezing. The `cv2.imshow()` window wouldn't respond to keypresses. I tried multiple approaches — running the loop in a background thread, using `ipywidgets` to display frames inside the notebook, and finally building the Streamlit UI where the camera feed renders directly in the browser. Each approach taught me something new about how Python threading and UI rendering works.

---

## 💪 What I'm Proud Of

Out of everything in this project, I'm most proud of getting the **live camera detection working end-to-end**.

It sounds simple, but think about what's actually happening:
- Your webcam captures a frame
- It gets resized, normalized, and fed into a neural network
- The network returns probabilities across 13+ classes
- The top prediction gets mapped to a produce name
- Nutrition info, freshness status, and a confidence bar all update — live — in the browser

The first time I pointed my camera at an apple and saw "🍎 APPLE — ✅ FRESH — 94.2%" appear on screen in real time, I genuinely felt like I'd done something. That moment made all the frustrating debugging sessions worth it.

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 + Custom Head |
| Input Size | 224 × 224 × 3 |
| Training Phases | 2 (Frozen + Fine-tune) |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Augmentation | Rotation, Flip, Zoom, Brightness, Shear |

> Actual accuracy depends on your dataset size and quality. With the recommended Kaggle dataset, expect **90–96% validation accuracy**.

---

## 🔮 Future Improvements

Things I want to add next:

- [ ] **More produce types** — Lettuce, Broccoli, Grapes, Pineapple
- [ ] **Confidence threshold warning** — Alert user when model is uncertain
- [ ] **Multi-object detection** — Detect multiple fruits in one frame using YOLOv8
- [ ] **Mobile app version** — React Native or Flutter wrapper
- [ ] **Expiry date estimator** — Estimate days remaining before spoilage
- [ ] **History log** — Save past detections with timestamps

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! This is a personal learning project but I'd love feedback and suggestions.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Kaggle** — for the fruits fresh/rotten dataset
- **TensorFlow & Keras team** — for MobileNetV2 and the amazing framework
- **Streamlit** — for making it easy to build web UIs in pure Python
- **OpenCV** — for camera capture capabilities
- **Claude AI** — helped me debug code, understand concepts, and structure the project when I was stuck
- Every Stack Overflow answer that saved me at 2am 😄

---

<div align="center">

**Built with 💚 and a lot of debugging by an IT student who refused to give up**

⭐ If this project helped or inspired you, consider giving it a star!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/freshscan?style=social)](https://github.com/yourusername/freshscan)

</div>
