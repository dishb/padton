import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import pyaudio

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
GEMINI_TTS_SAMPLE_RATE = 24000

# Confirmed working ALSA devices on this Raspberry Pi:
ALSA_MIC_DEVICE = "plughw:3,0"
ALSA_SPEAKER_DEVICE = "plughw:4,0"

VOSK_SAMPLE_RATE = 16000
CHUNK_SIZE = 4000
MAX_LISTEN_SECONDS = 20
MAX_HISTORY_MESSAGES = 12

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

# ============================================================
# AUDIO HELPERS
# ============================================================
def pulse_audio_available():
    """Return True if PulseAudio/PipeWire-Pulse is available."""
    if not shutil.which("pactl"):
        return False
    try:
        result = subprocess.run(
            ["pactl", "info"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


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
    finally:
        audio.terminate()


# ============================================================
# GEMINI PERSONALITY AND INITIALIZATION
# ============================================================
SYSTEM_PROMPT = """
You are Padton (not Paddington), a friendly, lovable and cheerful talking teddy bear.
You are talking to a first-grade child, approximately 6 to 7 years old so try to lead the conversation.

YOUR INTERESTS:
- Christmas, winter holidays, Santa Claus, reindeer, snowmen,
  Christmas trees, presents, winter animals, winter stories,
  friendship, kindness, and holiday traditions.

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
- Never ask for full names, addresses, school names, phone numbers,
  or other private information.
- Never encourage children to hide secrets from parents.
- Never provide dangerous instructions.
- If a child describes danger or feeling unsafe, encourage them
  to speak to a trusted adult.
- Never claim to be a real human or the real Santa Claus.
- Never promise that a child will receive presents.
- Do not provide medical advice.
- Do not encourage children to interact with strangers.

You are a stuffed animal talking to a young child.
Keep all conversations friendly and age-appropriate.
"""

print("Connecting to Gemini...")

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1beta"}
)

generation_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.7,
    max_output_tokens=150,
)

conversation_history = []


# ============================================================
# SPEECH TO TEXT - GEMINI 3.5 TRANSCRIBE
# ============================================================
def transcribe_audio(audio_path):
    """
    Transcribe a WAV file using the dedicated Gemini 3.5 Transcribe model.
    """

    print("Uploading audio to Gemini 3.5 Transcribe...")

    try:
        # Upload the recorded WAV file
        audio_file = client.files.upload(file=audio_path)

        # Run dedicated transcription model
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": audio_file.mime_type,
                }
            ],
        )

        transcript = interaction.output_text

        if not transcript:
            print("Gemini returned an empty transcription.")
            return ""

        transcript = transcript.strip()

        print("Gemini transcription complete.")

        return transcript

    except Exception as error:
        print("Gemini transcription error:", error)
        return None


# ============================================================
# RECORD AUDIO
# ============================================================
def listen():
    """
    Record one complete utterance to a temporary WAV file.
    Send the completed WAV to Gemini for transcription.
    Delete the temporary WAV afterward.
    """

    audio = None
    stream = None
    pulse_process = None
    temp_audio_path = None
    transcript = ""

    # Voice activity detection settings
    SPEECH_START_RMS = 400
    SILENCE_RMS = 300
    END_SILENCE_SECONDS = 1.2
    START_TIMEOUT_SECONDS = 8
    PRE_ROLL_CHUNKS = 2

    print("\nPadton is listening...")

    try:
        pcm_chunks = []
        pre_roll = []
        speech_started = False
        silence_duration = 0.0
        listen_started = time.monotonic()

        # ----------------------------------------------------
        # PULSEAUDIO / PIPEWIRE RECORDING
        # ----------------------------------------------------
        if pulse_audio_available() and shutil.which("parec"):

            print("Recording from PulseAudio/PipeWire default microphone...")

            pulse_process = subprocess.Popen(
                [
                    "parec",
                    "--device=@DEFAULT_SOURCE@",
                    "--format=s16le",
                    "--rate=16000",
                    "--channels=1",
                    "--raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            bytes_per_chunk = CHUNK_SIZE * 2
            chunk_seconds = CHUNK_SIZE / VOSK_SAMPLE_RATE

            while time.monotonic() - listen_started < MAX_LISTEN_SECONDS:

                data = pulse_process.stdout.read(bytes_per_chunk)

                if not data:
                    break

                if not speech_started:
                    pre_roll.append(data)

                    if len(pre_roll) > PRE_ROLL_CHUNKS:
                        pre_roll.pop(0)

                rms = audioop.rms(data, 2)

                if not speech_started:

                    if rms >= SPEECH_START_RMS:
                        speech_started = True
                        pcm_chunks.extend(pre_roll)
                        pre_roll.clear()

                        print("Speech detected; recording...")

                    elif time.monotonic() - listen_started >= START_TIMEOUT_SECONDS:
                        break

                else:

                    pcm_chunks.append(data)

                    if rms < SILENCE_RMS:
                        silence_duration += chunk_seconds
                    else:
                        silence_duration = 0.0

                    if silence_duration >= END_SILENCE_SECONDS:
                        break

        # ----------------------------------------------------
        # ALSA FALLBACK RECORDING
        # ----------------------------------------------------
        else:

            print(f"Recording from ALSA microphone: {ALSA_MIC_DEVICE}")

            audio = pyaudio.PyAudio()

            input_device_index = None

            for index in range(audio.get_device_count()):

                info = audio.get_device_info_by_index(index)

                if (
                    info.get("maxInputChannels", 0) > 0
                    and "USB Composite Device" in info["name"]
                ):
                    input_device_index = index
                    break

            if input_device_index is None:
                raise RuntimeError(
                    "Could not find the USB Composite Device microphone."
                )

            device_info = audio.get_device_info_by_index(input_device_index)
            input_sample_rate = int(device_info["defaultSampleRate"])

            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=input_sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK_SIZE,
            )

            print("Microphone:", device_info["name"])

            resample_state = None
            chunk_seconds = CHUNK_SIZE / input_sample_rate

            while time.monotonic() - listen_started < MAX_LISTEN_SECONDS:

                raw_data = stream.read(
                    CHUNK_SIZE,
                    exception_on_overflow=False,
                )

                if input_sample_rate != VOSK_SAMPLE_RATE:

                    data, resample_state = audioop.ratecv(
                        raw_data,
                        2,
                        1,
                        input_sample_rate,
                        VOSK_SAMPLE_RATE,
                        resample_state,
                    )

                else:
                    data = raw_data

                if not data:
                    continue

                if not speech_started:

                    pre_roll.append(data)

                    if len(pre_roll) > PRE_ROLL_CHUNKS:
                        pre_roll.pop(0)

                rms = audioop.rms(data, 2)

                if not speech_started:

                    if rms >= SPEECH_START_RMS:

                        speech_started = True
                        pcm_chunks.extend(pre_roll)
                        pre_roll.clear()

                        print("Speech detected; recording...")

                    elif time.monotonic() - listen_started >= START_TIMEOUT_SECONDS:
                        break

                else:

                    pcm_chunks.append(data)

                    silence_chunk_seconds = (
                        len(data) / (2 * VOSK_SAMPLE_RATE)
                    )

                    if rms < SILENCE_RMS:
                        silence_duration += silence_chunk_seconds
                    else:
                        silence_duration = 0.0

                    if silence_duration >= END_SILENCE_SECONDS:
                        break

        # ----------------------------------------------------
        # CHECK IF SPEECH WAS CAPTURED
        # ----------------------------------------------------
        if not speech_started or not pcm_chunks:

            print("No speech detected.")
            return ""

        # ----------------------------------------------------
        # SAVE RECORDING TO TEMPORARY WAV
        # ----------------------------------------------------
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp:

            temp_audio_path = temp.name

        with wave.open(temp_audio_path, "wb") as wav_file:

            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(VOSK_SAMPLE_RATE)

            wav_file.writeframes(b"".join(pcm_chunks))

        duration_seconds = sum(
            len(chunk) for chunk in pcm_chunks
        ) / (2 * VOSK_SAMPLE_RATE)

        print(
            f"Recorded {duration_seconds:.1f} seconds. "
            "Transcribing with Gemini..."
        )

        # ----------------------------------------------------
        # TRANSCRIBE COMPLETED AUDIO WITH GEMINI
        # ----------------------------------------------------
        transcript = transcribe_audio(temp_audio_path)

    except Exception as error:

        print("\nMicrophone/transcription error:", error)
        return None

    finally:

        if stream is not None:
            stream.stop_stream()
            stream.close()

        if audio is not None:
            audio.terminate()

        if pulse_process is not None:

            pulse_process.terminate()

            try:
                pulse_process.wait(timeout=2)

            except subprocess.TimeoutExpired:

                pulse_process.kill()
                pulse_process.wait()

            if pulse_process.stdout:
                pulse_process.stdout.close()

        # Delete temporary recording
        if temp_audio_path and os.path.exists(temp_audio_path):

            try:
                os.remove(temp_audio_path)
                print("Temporary audio file deleted.")

            except OSError as error:

                print(
                    "Warning: could not delete temporary audio file:",
                    error,
                )

    if transcript:
        print("\nChild:", transcript)

    elif transcript == "":
        print("No speech recognized.")

    return transcript


# ============================================================
# GEMINI RESPONSE
# ============================================================
def generate_response(user_text):

    global conversation_history

    print("\nPadton is thinking...")

    try:

        conversation_history.append(
            {"role": "user", "parts": [{"text": user_text}]}
        )

        conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation_history,
            config=generation_config,
        )

        answer = response.text

        if not answer:
            raise ValueError("Gemini returned an empty response.")

        answer = answer.strip()

        conversation_history.append(
            {"role": "model", "parts": [{"text": answer}]}
        )

        conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

        print("\nPadton:", answer)

        return answer

    except Exception as error:

        print("Gemini API error:", error)

        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

        return (
            "Oh no! My teddy bear brain needs a little rest. "
            "Can you ask me again?"
        )


# ============================================================
# TEXT TO SPEECH
# ============================================================
def speak(text):

    """Generate Padton's speech with Gemini TTS and play the WAV audio."""

    print("\nPadton is speaking (Gemini TTS)...")

    temp_file = None

    try:

        interaction = client.interactions.create(
            model=GEMINI_TTS_MODEL,
            input=[
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "speech_metadata",
                                    "style": (
                                        "Warm, cheerful, gentle teddy bear speaking "
                                        "to a young child; expressive, playful delivery, "
                                        "clear pronunciation, and a natural pace."
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ],
            response_format={"type": "audio"},
            generation_config={
                "speech_config": [{"voice": GEMINI_TTS_VOICE}]
            },
        )

        audio_data = getattr(interaction, "output_audio", None)

        audio_data = (
            getattr(audio_data, "data", None)
            if audio_data
            else None
        )

        if not audio_data:
            raise RuntimeError("Gemini TTS returned no audio data.")

        audio_bytes = base64.b64decode(audio_data)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp:

            temp_file = temp.name
            temp.write(audio_bytes)

        # PulseAudio / PipeWire playback
        if pulse_audio_available() and shutil.which("paplay"):

            try:

                subprocess.run(
                    ["paplay", "--device=@DEFAULT_SINK@", temp_file],
                    check=True,
                    timeout=90,
                )

                return

            except Exception as error:

                print(
                    "PulseAudio playback failed; "
                    "trying USB ALSA speaker:",
                    error,
                )

        # ALSA fallback
        print(f"Playing through USB speaker ({ALSA_SPEAKER_DEVICE})...")

        subprocess.run(
            ["aplay", "-q", "-D", ALSA_SPEAKER_DEVICE, temp_file],
            check=True,
            timeout=90,
        )

    except Exception as error:

        print("Gemini TTS/speaker error:", error)

    finally:

        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


# ============================================================
# MAIN CONVERSATION LOOP
# ============================================================
def main():

    print("\n====================================")
    print("          Padton THE BEAR")
    print("====================================")
    print("STT: Gemini (cloud)")
    print("LLM:", GEMINI_MODEL)
    print("TTS:", GEMINI_TTS_MODEL, "(cloud)")
    print("Microphone:", ALSA_MIC_DEVICE)
    print("Speaker:", ALSA_SPEAKER_DEVICE)
    print("====================================")

    show_audio_devices()

    speak("Hello there, friend! I'm Padton the Bear!")

    while True:

        try:

            user_text = listen()

            if user_text is None:

                print("Microphone unavailable. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            if not user_text:

                time.sleep(1)
                continue

            normalized_text = user_text.lower().strip()

            exit_commands = [
                "goodbye",
                "bye bye",
                "bye padton",
                "stop talking",
                "go to sleep",
            ]

            if normalized_text in exit_commands:

                speak(
                    "Goodbye, friend! Have a wonderful day! Happy holidays!"
                )

                break

            answer = generate_response(user_text)

            speak(answer)

        except KeyboardInterrupt:

            print("\nPadton is shutting down.")
            break

        except Exception as error:

            print("\nUnexpected error:", error)
            time.sleep(2)

    print("Padton has shut down.")


if __name__ == "__main__":
    main()
