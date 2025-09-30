import re
from typing import Dict

# ---------------------
# Helpers
# ---------------------
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


# Cache of loaded indexes: {index_dir: TranscriptRAGIndex}
INDEX_CACHE: Dict[str, object] = {}


# --- Recency heuristics (NEW) ---
RECENCY_PATTERNS = [
    r"\b(now|today|these days|currently|as of|latest|most recent|up[- ]to[- ]date|news|update[s]?)\b",
    r"\b(maintenant|aujourd'hui|actuel(?:le|s)?|derni(?:er|ères?)|réc(?:ent|entes?)|mise[s]?\s*à\s*jour)\b",
    r"\bas of\s*20\d{2}\b", r"\bin\s*20\d{2}\b",  # "as of 2025", "in 2025"
]


# Heuristics for AUTO routing
VISUAL_PATTERNS = [
    r"\b(logo|brand|marque|embl[eè]me|badge|maillot|jersey)s?\b",
    r"\b(scoreboard|score|tableau d'affichage)\b",
    r"\b(color|couleur|couleurs)\b",
    r"\b(camera|cam[ée]ra|angle|shot|plan|transition|cut|montage)\b",
    r"\b(onscreen|on[- ]screen|à l'?écran|texte à l'?écran|OCR)\b",
    r"\b(scenes?|sc[eè]nes?|frame|cadre|image|diapo|slide|UI|interface)\b",
    r"\b(number on (the )?jersey|num[eé]ro sur (le )?maillot)\b",
    r"\b(what is shown|qu'est[- ]ce qui est montr[eé])\b",
    r"\b(at\s*\d{1,2}:\d{2}(:\d{2})?(,\d{3})?)\b",  # timestamps often imply visual inspection
]


TRANSCRIPT_PATTERNS = [
    r"\b(said|say|says|mention|quote|dialog(ue)?|conversation|talk|speech|narrat(?:ion|or))\b",
    r"\b(a dit|dit|disent|mentionne|citation|dialogue|conversation|parle|voix off|audio)\b",
    r"\b(subtitle|subtitles|caption|srt|transcript(ion)?)\b",
    r"\b(what does .* (say|mean)|que (dit|signifie))\b",
]


# ---------- Entity search helpers (robust over spelling/variants) ----------
ENTITY_INTENT_PATTERNS = [
    r"\b(?:all|toutes?|every|liste(?:r)?|montre(?:r)?)\s+(?:the\s+)?scenes?\s+(?:where|with|containing)\s+(?P<name>.+?)\b",
    r"\b(?:donne(?:z)?|give)\s+(?:moi\s+)?toutes?\s+les?\s+sc[eè]nes?\s+o[uù]\s+(?P<name>.+?)\s+(?:appara[iî]t|figure|est (?:pr[eé]sent[e]?)|se voit)\b",
    r"\b(?:find|trouve[r]?)\s+(?:all\s+)?scenes?\s+(?:with|featuring)\s+(?P<name>.+?)\b",
    r"\bscenes?\s+(?:with|de|où)\s+(?P<name>.+?)\b",
]


SC_LINE_RE = re.compile(r"^\s*SCENE\s*N[°o]\s*(\d+)\s*:\s*([0-9:,–\- ]+)?", re.I | re.M)


ALLOWED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}  # add more if needed
