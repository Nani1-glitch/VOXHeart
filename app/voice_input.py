# voice_input.py

import re
import time
import speech_recognition as sr

# ─── Tiny number‐word map ──────────────────────────────────────────────────────
NUMBER_MAP = {
    "zero":0, "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
    "eleven":11, "twelve":12, "thirteen":13, "fourteen":14,
    "fifteen":15, "sixteen":16, "seventeen":17, "eighteen":18,
    "nineteen":19, "twenty":20, "thirty":30, "forty":40,
    "fifty":50, "sixty":60, "seventy":70, "eighty":80, "ninety":90
}

def words_to_number(s: str) -> int:
    s = s.lower().replace("-", " ")
    total = 0
    for token in s.split():
        if token.isdigit():
            return int(token)
        if token in NUMBER_MAP:
            total += NUMBER_MAP[token]
    if total:
        return total
    digits = re.sub(r"[^\d]", "", s)
    if digits:
        return int(digits)
    raise ValueError(f"Cannot parse a number from “{s}”")

def parse_category_1to3(s: str) -> int:
    sl = s.lower()
    if re.search(r"\b(well above|3|three)\b", sl):
        return 3
    if re.search(r"\b(above|2|two)\b", sl):
        return 2
    if re.search(r"\b(normal|1|one)\b", sl):
        return 1
    raise ValueError(f"Cannot parse category from “{s}”")

# ─── TTS (pyttsx3) or silent fallback ─────────────────────────────────────────
try:
    import pyttsx3
    _tts = pyttsx3.init()
    def _speak(text: str):
        try:
            _tts.say(text)
            _tts.runAndWait()
        except Exception:
            pass
except Exception:
    # catches ImportError, NameError, driver errors, etc.
    def _speak(text: str):
        pass

# ─── Recognizer & mic ─────────────────────────────────────────────────────────
_recognizer = sr.Recognizer()
_mic = sr.Microphone()

def _listen(timeout=6, phrase_time_limit=6) -> str | None:
    with _mic as source:
        _recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return _recognizer.recognize_google(audio).lower()
        except Exception:
            return None

# ─── Field definitions ─────────────────────────────────────────────────────────
_FIELDS = [
    ("age", "Please tell your age in years.", words_to_number, (1, 120),
     "Age must be between 1 and 120."),

    ("height", "What is your height in centimeters?", words_to_number, (50, 250),
     "Height must be between 50 and 250 cm."),

    ("weight", "Tell me your weight in kilograms.", words_to_number, (10, 300),
     "Weight must be between 10 and 300 kg."),

    ("ap_hi", "What is your systolic blood pressure?", words_to_number, (70, 250),
     "Systolic BP must be between 70 and 250 mmHg."),

    ("ap_lo", "What is your diastolic blood pressure?", words_to_number, (40, 150),
     "Diastolic BP must be between 40 and 150 mmHg."),

    ("gender", "Say your gender: male or female.",
     lambda s: 2 if "male" in s else (1 if "female" in s else (_ for _ in ()).throw(ValueError())),
     (1, 2), "Please say male or female."),

    ("cholesterol", "Rate your cholesterol: normal, above normal, or well above normal.",
     parse_category_1to3, (1, 3),
     "Cholesterol must be normal (1), above (2), or well above (3)."),

    ("gluc", "Rate your glucose: normal, above normal, or well above normal.",
     parse_category_1to3, (1, 3),
     "Glucose must be normal (1), above (2), or well above (3)."),

    ("smoke", "Do you smoke? Say yes or no.",
     lambda s: 1 if "yes" in s else (0 if "no" in s else (_ for _ in ()).throw(ValueError())),
     (0, 1), "Please answer yes or no."),

    ("alco", "Do you consume alcohol? Say yes or no.",
     lambda s: 1 if "yes" in s else (0 if "no" in s else (_ for _ in ()).throw(ValueError())),
     (0, 1), "Please answer yes or no."),

    ("active", "Are you physically active? Say yes or no.",
     lambda s: 1 if "yes" in s else (0 if "no" in s else (_ for _ in ()).throw(ValueError())),
     (0, 1), "Please answer yes or no."),
]

def collect_user_voice_input() -> tuple[dict, dict]:
    """
    Prompt for each field, parse & validate.
    Returns (user_data, transcript).
    transcript = {"left":[prompts/warnings], "right":[heard phrases]}
    """
    user_data = {}
    transcript = {"left": [], "right": []}

    for key, prompt, parser, (mn, mx), bad_msg in _FIELDS:
        attempts = 0
        while True:
            attempts += 1
            if attempts > 3:
                abort = "Too many failed attempts—aborting voice input."
                transcript["left"].append(abort)
                _speak(abort)
                return {}, {"left": [], "right": []}

            transcript["left"].append(prompt)
            _speak(prompt)

            heard = _listen()
            if not heard:
                retry = "I didn't catch that, please repeat."
                transcript["left"].append(retry)
                _speak(retry)
                continue

            transcript["right"].append(heard)

            try:
                val = parser(heard)
            except Exception:
                err = "Sorry, I didn't get that—please try again."
                transcript["left"].append(err)
                _speak(err)
                continue

            if not (mn <= val <= mx):
                warn = f"This is a real-time health app—{bad_msg}"
                transcript["left"].append(warn)
                _speak(warn)
                continue

            user_data[key] = val
            time.sleep(0.2)
            break

    return user_data, transcript
