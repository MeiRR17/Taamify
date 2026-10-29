"""Validation and Hebrew mark handling shared by the phrase API.

Human performed labels are deliberately absent from a capture request. They
belong to a separate reviewed annotation, never to the text the UI expects.
"""
import hashlib
import json
import re
import unicodedata

CONSENT_VERSION = "taamify-local-research-v1"
TRADITION = "sephardic-jerusalem"
CONSENT_SCOPES = ["local_research", "training"]
MAX_WORDS = 200
MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_AUDIO_SECONDS = 45


def letter_clusters(text):
    """Attach all combining marks to their base letter, independent of order."""
    out = []
    for char in unicodedata.normalize("NFD", text):
        if "\u05d0" <= char <= "\u05ea":
            out.append([char, set()])
        elif out and unicodedata.combining(char):
            out[-1][1].add(char)
        else:
            # A separator must not attach following marks to a previous word.
            out.append([char, set()])
    return [(base, frozenset(marks)) for base, marks in out
            if "\u05d0" <= base <= "\u05ea"]


def phrase_inputs(words, taamim):
    try:
        w, t = json.loads(words), json.loads(taamim)
    except (ValueError, TypeError) as exc:
        raise ValueError("המילים והטעמים צריכים להישלח כרשימות תקינות") from exc
    if not isinstance(w, list) or not isinstance(t, list):
        raise ValueError("המילים והטעמים צריכים להישלח כרשימות")
    if not 1 <= len(w) <= MAX_WORDS or len(w) != len(t):
        raise ValueError("נדרש קטע עם מספר זהה של מילים וטעמים, עד 200 מילים")
    if any(not isinstance(x, str) or not x.strip() or len(x) > 160 for x in w):
        raise ValueError("כל מילה צריכה להיות טקסט לא ריק")
    if any(x is not None and (not isinstance(x, str) or len(x) > 80) for x in t):
        raise ValueError("כל טעם צריך להיות שם תקין או ערך ריק")
    return w, t


def capture_metadata(consent, voice_id, session_id, consent_version, consent_scope, tradition):
    if consent.lower() in ("", "0", "false", "no"):
        return None
    if consent.lower() not in ("1", "true", "yes"):
        raise ValueError("ערך ההסכמה אינו תקין")
    if consent_version != CONSENT_VERSION:
        raise ValueError("נדרש אישור שמירה מעודכן")
    valid_id = re.compile(r"[A-Za-z0-9_-]{8,80}\Z")
    if not valid_id.fullmatch(voice_id) or not valid_id.fullmatch(session_id):
        raise ValueError("נדרשים מזהה קורא ומזהה מפגש לצורך שמירה")
    try:
        scopes = json.loads(consent_scope)
    except (ValueError, TypeError) as exc:
        raise ValueError("הרשאות השמירה אינן תקינות") from exc
    if not isinstance(scopes, list) or sorted(scopes) != sorted(CONSENT_SCOPES):
        raise ValueError("נדרשת הסכמה מפורשת לשמירה מקומית ולשימוש עתידי באימון")
    if tradition != TRADITION:
        raise ValueError("איסוף המחקר מוגדר לנוסח ספרדי־ירושלמי")
    return {"schema_version": 2, "consent": True,
            "consent_version": CONSENT_VERSION, "consent_scope": scopes,
            "voice_id": voice_id, "session_id": session_id, "tradition": tradition}


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()
