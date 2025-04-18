import re
import time
import speech_recognition as sr

# ─── Try to import & init TTS ─────────────────────────────────────
try:
    import pyttsx3
    _engine = pyttsx3.init()
    def _speak(text: str):
        _engine.say(text)
        _engine.runAndWait()
except Exception:
    # No TTS available → silent fallback
    def _speak(text: str):
        pass

# ─── Setup recognizer & mic ────────────────────────────────────────
_recognizer = sr.Recognizer()
_mic = sr.Microphone()

def _listen(timeout=5, phrase_time_limit=5) -> str | None:
    """Listen once and return lowercased transcript, or None if nothing understood."""
    with _mic as source:
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return _recognizer.recognize_google(audio).lower()
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
            return None

def collect_user_voice_input() -> tuple[dict, dict]:
    """
    Uses server‑side speech_recognition to collect all fields.  
    Returns (user_data, transcript) or ({}, {"left":[], "right":[]}) on abort.
    """

    # Field specs: key, prompt, parse_fn, validate_fn, error_msg
    fields = [
        ("age",
         "Please tell your age in years.",
         lambda s: int(re.sub(r"\D+", "", s)),
         lambda v: 1 <= v <= 120,
         "That doesn't look like a real age—please try again."),

        ("height",
         "What is your height in centimeters?",
         lambda s: int(re.sub(r"\D+", "", s)),
         lambda v: 50 <= v <= 250,
         "I need a height between 50 and 250 cm."),

        ("weight",
         "Tell me your weight in kilograms.",
         lambda s: int(re.sub(r"\D+", "", s)),
         lambda v: 10 <= v <= 300,
         "Please give a weight between 10 and 300 kg."),

        ("ap_hi",
         "What is your systolic blood pressure?",
         lambda s: int(re.sub(r"\D+", "", s)),
         lambda v: 70 <= v <= 250,
         "Systolic BP should be between 70 and 250."),

        ("ap_lo",
         "Now your diastolic blood pressure?",
         lambda s: int(re.sub(r"\D+", "", s)),
         lambda v: 40 <= v <= 150,
         "Diastolic BP should be between 40 and 150."),

        ("gender",
         "Say your gender: male or female.",
         lambda s: 2 if "male" in s else 1,
         lambda v: v in (1, 2),
         "Please say either male or female."),

        ("cholesterol",
         "Rate your cholesterol: 1 for normal, 2 for above, 3 for well above.",
         lambda s: 3 if "three" in s else 2 if re.search(r"\btwo\b|\b2\b", s) else 1,
         lambda v: v in (1, 2, 3),
         "Say one, two, or three."),

        ("gluc",
         "Rate your glucose: 1 for normal, 2 for above, 3 for well above.",
         lambda s: 3 if "three" in s else 2 if re.search(r"\btwo\b|\b2\b", s) else 1,
         lambda v: v in (1, 2, 3),
         "Say one, two, or three."),

        ("smoke",
         "Do you smoke? Say yes or no.",
         lambda s: 1 if "yes" in s else 0,
         lambda v: v in (0, 1),
         "Please answer yes or no."),

        ("alco",
         "Do you consume alcohol? Say yes or no.",
         lambda s: 1 if "yes" in s else 0,
         lambda v: v in (0, 1),
         "Please answer yes or no."),

        ("active",
         "Are you physically active? Say yes or no.",
         lambda s: 1 if "yes" in s else 0,
         lambda v: v in (0, 1),
         "Please answer yes or no.")
    ]

    user_data = {}
    transcript = {"left": [], "right": []}

    for key, prompt, parser, validator, err_msg in fields:
        attempts = 0
        while True:
            attempts += 1
            if attempts > 3:
                _speak("Too many failed attempts. Aborting.")
                return {}, {"left": [], "right": []}

            # ask
            transcript["left"].append(prompt)
            _speak(prompt)

            # listen
            heard = _listen()
            if not heard:
                _speak("I didn't catch that. Please repeat.")
                continue

            transcript["right"].append(heard)

            # parse
            try:
                val = parser(heard)
            except Exception:
                _speak(err_msg)
                continue

            # validate
            if not validator(val):
                _speak(err_msg)
                continue

            user_data[key] = val
            time.sleep(0.2)
            break

    return user_data, transcript
