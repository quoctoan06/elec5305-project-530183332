# Music Genre Classification

A deep learning project that classifies short audio clips into one of ten music genres using a CNN trained on mel-spectrogram representations.

---

## Project Overview

### Problem

Classify short music clips into one of ten genres:
`blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`.

### Key Components

* **Feature Extraction:** Mel-spectrograms generated from 4-second overlapping audio segments.
* **Model:** Convolutional Neural Network (CNN) consisting of stacked Conv2D → MaxPool → Dropout → Dense layers.
* **Training Setup:** 80/20 train–test split with categorical cross-entropy loss and Adam optimizer.
* **Evaluation Metric:** Overall accuracy (additional metrics planned: precision, recall, F1-score).

---

## Dataset

### Structure

The dataset has the following directory layout:

```
data/audio/
    blues/*.wav
    classical/*.wav
    country/*.wav
    ...
    rock/*.wav
```

Each `.wav` file should belong to one of the ten genre folders. Example filenames:

```
rock.00000.wav, jazz.00001.wav, ...
```

### Supported Genres

```python
CLASSES = ['blues','classical','country','disco','hiphop',
           'jazz','metal','pop','reggae','rock']
```

---

## Notebook Execution (Google Colab)

### How to Run

1. Open the notebook in Google Colab.
2. Upload the `data.zip` file containing the folder structure above.
3. Run all cells sequentially to preprocess data, train the model, and view results.

---

## Data Preprocessing

* **Duration:** Each clip is split into 4-second chunks (50% overlap).
* **Sampling:** Loaded using `librosa.load()` (default 22,050 Hz).
* **Feature Extraction:** Compute mel-spectrograms → convert to dB scale → resize to `150×150`.
* **Labels:** Inferred automatically from parent folder names.

---

## Model Architecture

The CNN consists of three convolutional blocks followed by dense layers:

```
[Conv2D(32) → Conv2D(32) → MaxPool]
[Conv2D(64) → Conv2D(64) → MaxPool]
[Conv2D(128) → Conv2D(128) → MaxPool → Dropout(0.3)]
Flatten → Dense(1200, relu) → Dropout(0.45) → Dense(10, softmax)
```

The model is compiled with:

```python
optimizer = Adam(learning_rate=1e-4)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
```

---

## Output

* Trained model: `trained_model.h5`
* Performance plots: training vs. validation accuracy and loss
* Future work: confusion matrix, per-genre performance metrics, and audio demos.
