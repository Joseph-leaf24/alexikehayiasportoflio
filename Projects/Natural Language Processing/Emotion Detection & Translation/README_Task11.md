## **Emotion Classification Pipeline- Group 19(Feta Team)**

- This directory contains the pipeline code that processes reality tv series from YouTube, transcribing the audio, translating the transcribed text from Greek to English, and then mapping each sentence with one of the 7 emotions: 

1. Anger.
2. Disgust.
3. Fear.
4. Happiness.
5. Neutral.
6. Sadness.
7. Surprise.

### **Overview**

The pipeline performs the following steps:
1. **Video Loading**: Converts a YouTube video to audio and loads it into memory to avoid saving it to disk.
2. **Transcription**: Uses Whisper(large-v3) for transcribing audio to text.
3. **Emotion Prediction**: Predicts emotions and their intensities using the Emotion Classification Transformer model, then maping each emotion's intensity using an OpenAI API.
4. **Translation**: Translates Greek text to English using a fine-tuned HuggingFace Greek to English translation model.

#### **Environment Setup**

##### **Prerequisites**

- Anaconda environment.
- CUDA 11.2.
- cuDNN 8.1.
- Python 3.9.21.
- PyTorch with CUDA 11.8 support.

##### **Installation**

1. **Create a Conda environment**:
```bash
conda create --name emotion_pipeline python=3.9.21
conda activate emotion_pipeline
```

2. **Install PyTorch with CUDA support**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK resource**

```python
import nltk
nltk.download('punkt')
```

5. **Install Whisper**

    1. Install FFmpeg (required by Whisper):
      - conda install -c conda-forge ffmpeg
      + Alternatively, download FFmpeg from:
      - https://ffmpeg.org/download.html

    2. Install OpenAI Whisper via pip:
      - pip install git+https://github.com/openai/whisper.git

    3. Verify the installation by running:
      - whisper --help


### **Running the Pipeline**

1. **Download Translation and Emotion Classification Models**
- Insert the video link from YouTube


- In the pipeline.py script, in section: Predict- Emotion Classification Model, make sure to adjust the file path to the location of the downloaded model, and also adjust the OpenAI API Key.
- [Emotion Classification Mode](https://edubuas-my.sharepoint.com/:f:/g/personal/232230_buas_nl/ErnNz9lnbJBMhdG1vGdBJVQBO-gqCTc5DgLyysiLy3ddxQ?e=pCLfW6)

- [Greek to English Translation Model](https://edubuas-my.sharepoint.com/:f:/g/personal/232230_buas_nl/ElrVCHLn02xGofK19xBqQw4BDgy7J3uDuzyCQoxy1hBJtQ?)
- In the pipeline.py script, in section: Translate- Translation Model, make sure to adjust the file path to the location of the downloaded model.


2. **Execute the pipeline**
- Run the following command:
```bash
python pipeline.py
```

3. **Output**
- Results will be saved to `Emotion_Translation.csv` in the same directory as the code is run in.

---


