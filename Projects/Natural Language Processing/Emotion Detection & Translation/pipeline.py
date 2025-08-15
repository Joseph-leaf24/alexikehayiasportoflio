from helper_functions import video_functions, predict_functions, translate_functions
from tqdm import tqdm

# Define emotion names
emotion_names = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# ---------------------------------------------------------------------------------------------------------------------------------------
# Load the video
# ---------------------------------------------------------------------------------------------------------------------------------------

# YouTube link
link = "https://www.youtube.com/watch?v=XpKl3BnjnV0"

# Load the video and convert to audio
samples, sample_rate = video_functions.youtube_converter(link)

print('Video loaded')

# ---------------------------------------------------------------------------------------------------------------------------------------
# Transcribe
# ---------------------------------------------------------------------------------------------------------------------------------------

print('Beginning transcription')

# Transcribe the audio
transcription_df = video_functions.transcribe_and_save_from_array(samples, sample_rate=44100, chunk_length=30)

print('Audio Transcribed')

# ---------------------------------------------------------------------------------------------------------------------------------------
# Predict- Emotion Classification Model
# ---------------------------------------------------------------------------------------------------------------------------------------

print('Beginning emotion prediction')

# Predict the dataset
emotion_df = predict_functions.predict_dataset(transcription_df, 'Sentence', 
                                               # Make sure to adjust this before running the script, with your own OpenAI API key
                                               api_key="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",
                                                # and the path to your model directory
                                               model_dir=r"C:\Users\Testing 2\Downloads\Transformer_Models\emotion_classifier_model_v10.7_Bilingual")

print('Emotions predicted')

# ---------------------------------------------------------------------------------------------------------------------------------------
# Translate- Translation Model
# ---------------------------------------------------------------------------------------------------------------------------------------

print('Beginning translation')

# Enabling tqdm for pandas
tqdm.pandas() 

# Apply the translation function with a progress bar
emotion_df['Translation'] = emotion_df['Sentence'].progress_apply(translate_functions.translate_to_english,
                                                                # Adjust the path to your model directory before running the script
                                                                  model_dir=r"C:\Users\Testing 2\Downloads\Translation Model")

print('Translation completed')  

# ---------------------------------------------------------------------------------------------------------------------------------------
# Reorganize the dataset
# ---------------------------------------------------------------------------------------------------------------------------------------

# Re organize the dataset
emotion_df = emotion_df[['Start Time', 'End Time', 'Sentence', 'Translation', 'Emotion', 'Intensity Score']]

emotion_df['Start Time'] = emotion_df['Start Time'].apply(translate_functions.seconds_to_time)
emotion_df['End Time'] = emotion_df['End Time'].apply(translate_functions.seconds_to_time)

# Show a preview & save!

print(emotion_df.head(1))

print('Saving results')
emotion_df.to_csv("Emotion_Translation.csv", index=False)

