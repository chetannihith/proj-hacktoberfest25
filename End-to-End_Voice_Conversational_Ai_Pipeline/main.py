import os
import sounddevice as sd
import numpy as np
import torch
import whisper
import genai
from dotenv import load_dotenv
from scipy.io.wavfile import write
import soundfile as sf
from google.genai.errors import ClientError
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    print("GEMINI_API_KEY not found.")



def record_audio(filename="input.wav", duration=5, fs=16000):
    print(f"Recording for {duration} seconds.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    write(filename, fs, audio)
    print(f"Recording saved to {filename}")
    return filename

def transcribe_audio(audio_path: str) -> str:
    print("Beginning")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)
    print("Success")
    return result["text"]

def query_gemini(prompt: str) -> str:
    print("Waiting")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        response = client.generate_content(
            model="models/gemini-1.5-flash",
            contents=prompt,
        )
        print("Gemini has responded.")
        return response.text
    except Exception as e:
        return f"Error: {e.message}"

def text_to_speech(text, voice="expresso/ex04-ex02_desire_001_channel1_657s.wav",
                   output_file="output.wav", device=None):
    print("Creating speech...")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=device)
    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(voice)
    condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
    pcms = []
    def _on_frame(frame):
        if (frame != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))
    with tts_model.mimi.streaming(1):
        tts_model.generate([entries], [condition_attributes], on_frame=_on_frame)
    audio = np.concatenate(pcms, axis=-1)
    audio_int16 = (audio * 32767).astype(np.int16)
    write(output_file, tts_model.mimi.sample_rate, audio_int16)
    print(f"✅ Speech Created")
    return output_file

def play_audio(file_path):
    print(f"Playing audio reply...")
    data, fs = sf.read(file_path, dtype='float32')
    sd.play(data, fs)
    sd.wait()
    print("Playback complete.")

def main():
    print("--- Starting AI Pipleine ---")
    
    input_file = record_audio(duration=6)
    
    user_text = transcribe_audio(input_file)
    print("\nYour Input ---\n")
    print(user_text)
    
    ai_reply = query_gemini(user_text)
    print("\nGemini's Output ---\n")
    print(ai_reply)
    
    output_file = text_to_speech(ai_reply, output_file="reply.wav")
    
    play_audio(output_file)
    
    print("\n--- Replied Successfully ---")

if __name__ == "__main__":
    main()
