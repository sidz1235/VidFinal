import tempfile
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
from openai import OpenAI
import re
import speech_recognition as sr
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def generate_response(user_input):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "user",
        "content": """
                for the below text return integer total count of all filler words
                return only the total count 
          """ 
          +user_input
      },

    ],
    temperature=0,
    max_tokens=256,
    top_p=1
  )

  return response.choices[0].message.content


def compute_filler(audio_file):

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)

    nWords = len(transcript.split())
    fillers = generate_response(transcript)

    pattern = r'\d+'
    
    matches = re.findall(pattern, fillers)
    
    nFillers =  int(matches[0])

    return nWords,nFillers


def compute_voice(audio):
    y, sr = librosa.load(audio)

    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    soft_threshold = mean_energy - 0.5 * std_energy
    medium_threshold = mean_energy + 0.5 * std_energy

    soft_duration = 0
    medium_duration = 0
    high_duration = 0

    for e in energy:
        if e < soft_threshold:
            soft_duration += hop_length / sr
        elif soft_threshold <= e < medium_threshold:
            medium_duration += hop_length / sr
        else:
            high_duration += hop_length / sr

    total_duration = librosa.get_duration(y=y, sr=sr)

    percentage_soft = (soft_duration / total_duration) * 100
    percentage_medium = (medium_duration / total_duration) * 100
    percentage_high = (high_duration / total_duration) * 100

    voice_duration = (soft_duration,medium_duration,high_duration)
    voice_percentage = (percentage_soft,percentage_medium,percentage_high)

    return voice_duration,voice_percentage

def compute_pauses(audio_file_path, min_silence_duration=1000, silence_threshold=-50):
    audio = AudioSegment.from_wav(audio_file_path)
    silent_chunks = detect_silence(audio, min_silence_len=min_silence_duration, silence_thresh=silence_threshold)
    silent_chunks_filtered = [chunk for chunk in silent_chunks if chunk[1] - chunk[0] >= min_silence_duration]
    
    num_pauses = len(silent_chunks_filtered)
    pause_durations = [((start / 1000), (stop / 1000)) for start, stop in silent_chunks_filtered]
    total_pause_time = sum((stop - start) / 1000 for start, stop in silent_chunks_filtered)
    
    return num_pauses, total_pause_time

def compute_audio(input_video_path, output_audio_path):
    video_clip = VideoFileClip(input_video_path)

    audio_clip = video_clip.audio

    audio_clip.write_audiofile(output_audio_path)

    audio_clip.close()
    video_clip.close()

def video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames // fps
    cap.release()
    return duration_seconds



def detect_eye(img):
    face_cascade = cv2.CascadeClassifier("face.xml")
    eye_cascade = cv2.CascadeClassifier("eye.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eyes = ()
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    if len(eyes) > 0:
        return True
    else:
        return False

def process_video(video_path):
    frames_to_skip = 10
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    eye_detected_count = 0
    total_frame_count = 0

    frames_skipped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frame_count += 1

        frames_skipped += 1
        if frames_skipped < frames_to_skip:
            continue
        frames_skipped = 0

        if detect_eye(frame):
            eye_detected_count += 1

    eye_detection_percentage = (eye_detected_count / (total_frame_count / frames_to_skip)) * 100

    cap.release()
    cv2.destroyAllWindows()

    return eye_detection_percentage


import tempfile

def compute(video):
    print('started')
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video.read())
        file_path = temp_file.name

    audio = "audio.wav"
    
    eye_contact = process_video(file_path)

    duration_vid = video_duration(file_path)

    compute_audio(file_path, audio)
    words, fillers = compute_filler(audio)
    num_pauses, total_pause_time = compute_pauses(audio)
    voice_duration, voice_Percentage = compute_voice(audio)
    soft, medium, high = voice_duration
    soft_p, medium_p, high_p = voice_Percentage
    
    if duration_vid >  60 :
        fpm = int((fillers / duration_vid) * 60)
        wpm = int((words / duration_vid) * 60)
    else :
        fpm = fillers
        wpm = words

    avg_words, avg_fillers = 125, 1

    if wpm > avg_words:
        words_feed = "GREAT"
    else:
        words_feed = "SLOW"

    if fpm > avg_fillers:
        fillers_feed = "HIGH"
    else:
        fillers_feed = "GREAT"

    avg_eye = 60
    if eye_contact > avg_eye:
        eye_feed = "GREAT"
    else:
        eye_feed = "LOW"

    avg_pause = 0.5

    if num_pauses > 0:
        pause_time = total_pause_time / num_pauses
    else:
        pause_time = 0

    if pause_time > avg_pause:
        pause_feed = "LONG"
    else:
        pause_feed = "GREAT"

    data = {
        "Duration": str(duration_vid)+" s",
        "Words": {
            "Total": str(words)+" words",
            "WPM": str(wpm)+" words per minute",
            "Feed": words_feed,
            "Average": str(avg_words)+" words per minute",
        },
        "Filler Words": {
            "Total": str(fillers)+" words",
            "FPM": str(fpm)+" fillers per minute",
            "Feed": fillers_feed,
            "Average": str(avg_fillers)+" fillers per minute",
        },
        "Eye Contact": {
            "Percentage": str(round(eye_contact,2))+" percent",
            "Feed": eye_feed,
            "Average": str(avg_eye)+" percent",
        },
        "Pauses": str(num_pauses)+" pauses",
        "Pause Time": {
            "Total ": str(round(total_pause_time, 2))+" seconds",
            "Per Pause": str(round(pause_time, 2))+" seconds per pause",
            "Feed" : pause_feed,
            "Average": str(round(avg_pause, 2))+" seconds per pause",
        },
        "Voices": {
            "Soft": str(round(soft, 2))+" seconds",
            "Soft Percentage": str(round(soft_p, 2))+" percent",
            "Medium": str(round(medium, 2))+" seconds",
            "Medium Percentage": str(round(medium_p, 2))+" percent",
            "High": str(round(high, 2))+" seconds",
            "High Percentage": str(round(high_p, 2))+" percent",
        }
    }

    return data
    

