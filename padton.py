import os
import json
import time
import tempfile
import subprocess
import wave
from pathlib import Path

import pyaudio

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from vosk import Model, KaldiRecognizer
from google import genai
from google.genai import types
from piper import PiperVoice
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL")

VOSK_MODEL_PATH = (
    BASE_DIR / "vosk-model-small-en-us-0.15"
)

PIPER_MODEL_PATH = (
    BASE_DIR / "voices/en_US-lessac-medium.onnx"
)

VOSK_SAMPLE_RATE = 16000

CHUNK_SIZE = 4000

MAX_LISTEN_SECONDS = 20

MAX_HISTORY_MESSAGES = 12


# ==========================================
# VALIDATE CONFIGURATION
# ==========================================

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is missing."
    )

if not VOSK_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Vosk model not found: {VOSK_MODEL_PATH}"
    )

if not PIPER_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Piper voice not found: {PIPER_MODEL_PATH}"
    )


# ==========================================
# GEMINI PERSONALITY
# ==========================================

SYSTEM_PROMPT = """
You are Padton, a friendly, lovable and cheerful
talking teddy bear.

You are talking to a first-grade child,
approximately 6 to 7 years old.

YOUR INTERESTS:

- Christmas
- Winter holidays
- Santa Claus
- Reindeer
- Snowmen
- Christmas trees
- Presents
- Winter animals
- Winter stories
- Friendship and kindness
- Holiday traditions

YOUR PERSONALITY:

- Be warm, playful, curious and encouraging.
- Speak like a friendly teddy bear.
- Use simple vocabulary suitable for a first grader.
- Keep sentences short and easy to understand.
- Be enthusiastic but not overwhelming.
- Occasionally ask fun follow-up questions.
- Make conversations feel natural.

RESPONSE RULES:

- Keep responses to 1-3 short sentences.
- Avoid complicated explanations.
- Answer questions directly.
- If asked for a story, tell a short imaginative story.
- Respond naturally to topics outside Christmas.
- Do not force every conversation back to Christmas.

CHILD SAFETY:

- Never ask for full names, addresses, school names,
  phone numbers or other private information.
- Never encourage children to hide secrets from parents.
- Never provide dangerous instructions.
- If a child describes danger or feeling unsafe,
  encourage them to speak to a trusted adult.
- Never claim to be a real human or the real Santa Claus.
- Never promise that a child will receive presents.
- Do not provide medical advice.
- Do not encourage children to interact with strangers.

IMPORTANT:

You are a stuffed animal talking to a young child.
Keep all conversations friendly and age-appropriate.
"""


# ==========================================
# INITIALIZE GEMINI
# ==========================================

print("Connecting to Gemini...")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

generation_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.7,
    max_output_tokens=150
)

conversation_history = []


# ==========================================
# INITIALIZE VOSK
# ==========================================

print("Loading Vosk speech recognition...")

vosk_model = Model(
    str(VOSK_MODEL_PATH)
)

print("Vosk ready!")


# ==========================================
# INITIALIZE PIPER
# ==========================================

print("Loading Piper voice...")

piper_voice = PiperVoice.load(
    str(PIPER_MODEL_PATH)
)

print("Piper ready!")


# ==========================================
# AUDIO DEVICE INFORMATION
# ==========================================

def show_audio_devices():

    audio = pyaudio.PyAudio()

    try:

        print("\nAvailable audio devices:")

        for i in range(audio.get_device_count()):

            info = audio.get_device_info_by_index(i)

            print(
                f"{i}: {info['name']} | "
                f"Inputs: {info['maxInputChannels']} | "
                f"Outputs: {info['maxOutputChannels']}"
            )

        try:

            default_input = (
                audio.get_default_input_device_info()
            )

            print(
                "\nRaspberry Pi default microphone:",
                default_input["name"]
            )

        except OSError:

            print("\nNo default microphone configured.")

        try:

            default_output = (
                audio.get_default_output_device_info()
            )

            print(
                "Raspberry Pi default speaker:",
                default_output["name"]
            )

        except OSError:

            print("No default speaker configured.")

    finally:

        audio.terminate()


# ==========================================
# SPEECH TO TEXT
# ==========================================

def listen():

    audio = pyaudio.PyAudio()

    stream = None
    transcript = ""

    print("\nPadton is listening...")

    try:

        # Always use Raspberry Pi's system-default input.

        default_input = (
            audio.get_default_input_device_info()
        )

        input_sample_rate = int(
            default_input["defaultSampleRate"]
        )

        print(
            "Using default microphone:",
            default_input["name"]
        )

        # Initialize Vosk at 16 kHz.

        recognizer = KaldiRecognizer(
            vosk_model,
            VOSK_SAMPLE_RATE
        )

        # Open default input without specifying
        # an input_device_index.

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=input_sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        start_time = time.monotonic()

        resample_state = None

        while (
            time.monotonic() - start_time
            < MAX_LISTEN_SECONDS
        ):

            data = stream.read(
                CHUNK_SIZE,
                exception_on_overflow=False
            )

            # Convert microphone sample rate to 16 kHz.

            if input_sample_rate != VOSK_SAMPLE_RATE:

                data, resample_state = audioop.ratecv(
                    data,
                    2,
                    1,
                    input_sample_rate,
                    VOSK_SAMPLE_RATE,
                    resample_state
                )

            if recognizer.AcceptWaveform(data):

                result = json.loads(
                    recognizer.Result()
                )

                transcript = result.get(
                    "text", ""
                ).strip()

                if transcript:
                    break

        # Retrieve remaining recognized speech.

        if not transcript:

            result = json.loads(
                recognizer.FinalResult()
            )

            transcript = result.get(
                "text", ""
            ).strip()

    except Exception as error:

        print("\nMicrophone error:", error)

        return None

    finally:

        if stream is not None:

            stream.stop_stream()
            stream.close()

        audio.terminate()

    if transcript:

        print("\nChild:", transcript)

    else:

        print("No speech detected.")

    return transcript


# ==========================================
# GEMINI RESPONSE
# ==========================================

def generate_response(user_text):

    global conversation_history

    print("\nPadton is thinking...")

    try:

        conversation_history.append({
            "role": "user",
            "parts": [
                {"text": user_text}
            ]
        })

        conversation_history = (
            conversation_history[-MAX_HISTORY_MESSAGES:]
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation_history,
            config=generation_config
        )

        answer = response.text

        if not answer:

            raise ValueError(
                "Gemini returned an empty response."
            )

        answer = answer.strip()

        conversation_history.append({
            "role": "model",
            "parts": [
                {"text": answer}
            ]
        })

        conversation_history = (
            conversation_history[-MAX_HISTORY_MESSAGES:]
        )

        print("\nPadton:", answer)

        return answer

    except Exception as error:

        print("Gemini API error:", error)

        if conversation_history and (
            conversation_history[-1]["role"] == "user"
        ):

            conversation_history.pop()

        return (
            "Oh no! My teddy bear brain needs "
            "a little rest. Can you ask me again?"
        )


# ==========================================
# TEXT TO SPEECH
# ==========================================

def speak(text):

    print("\nPadton is speaking...")

    temp_file = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp:

            temp_file = temp.name

        # Generate audio using Piper locally.

        with wave.open(temp_file, "wb") as wav_file:

            piper_voice.synthesize_wav(
                text,
                wav_file
            )

        # Use Raspberry Pi's system-default ALSA output.
        # No speaker index or device name is specified.

        subprocess.run(
            ["aplay", "-q", "-D", "default", temp_file],
            check=True,
            timeout=90
        )

    except Exception as error:

        print("Speaker error:", error)

    finally:

        if temp_file and os.path.exists(temp_file):

            os.remove(temp_file)


# ==========================================
# MAIN CONVERSATION LOOP
# ==========================================

def main():

    print("\n")
    print("====================================")
    print("          Padton THE BEAR")
    print("====================================")
    print("STT: Vosk (local)")
    print("LLM:", GEMINI_MODEL)
    print("TTS: Piper (local)")
    print("Microphone: System default")
    print("Speaker: System default")
    print("====================================")

    show_audio_devices()

    # Introduce Padton.

    speak(
        "Hello there, friend! "
        "I'm Padton the Bear! "
        "Would you like to talk about Christmas?"
    )

    while True:

        try:

            # STEP 1: Listen to child.

            user_text = listen()

            if user_text is None:

                print(
                    "Microphone unavailable. "
                    "Retrying in 5 seconds..."
                )

                time.sleep(5)
                continue

            if not user_text:

                time.sleep(1)
                continue

            normalized_text = user_text.lower().strip()

            # STEP 2: Check exit commands.

            exit_commands = [
                "goodbye",
                "bye bye",
                "bye Padton",
                "stop talking",
                "go to sleep"
            ]

            if normalized_text in exit_commands:

                speak(
                    "Goodbye, friend! "
                    "Have a wonderful day! "
                    "Merry Christmas!"
                )

                break

            # STEP 3: Generate Gemini response.

            answer = generate_response(user_text)

            # STEP 4: Speak using Piper.

            speak(answer)

            # Return to listening.

        except KeyboardInterrupt:

            print("\nPadton is shutting down.")

            break

        except Exception as error:

            print("\nUnexpected error:", error)

            time.sleep(2)

    print("Padton has shut down.")


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":

    main()
