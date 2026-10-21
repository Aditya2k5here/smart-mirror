"""Natural-language time, duration and number parsing for voice commands.

Speech recognition returns text like "remind me to call mom in twenty minutes"
or "set a timer for an hour and a half", so everything here works on plain,
lower-cased sentences and is deliberately forgiving.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta
import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_ABBR["sept"] = 9


def normalize_numbers(text: str) -> str:
    """Replace spelled-out numbers with digits: 'twenty five' -> '25'."""
    tokens = re.findall(r"[\w'.:%/-]+|[^\w\s]", text.lower())
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in _TENS:
            value = _TENS[tok]
            if nxt in _UNITS and 0 < _UNITS[nxt] < 10:
                value += _UNITS[nxt]
                i += 1
            out.append(str(value))
        elif tok in _UNITS:
            out.append(str(_UNITS[tok]))
        else:
            out.append(tok)
        i += 1
    joined = " ".join(out)
    return re.sub(r"\s+([,.!?;:])", r"\1", joined)


_DURATION_UNITS = {
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
}
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
)


def parse_duration(text: str) -> int | None:
    """Return a duration in seconds, or None if the text has no duration."""
    t = normalize_numbers(text)
    t = re.sub(r"\bhalf\s+(?:an?\s+)?hour\b", "30 minutes", t)
    t = re.sub(r"\bquarter\s+(?:of\s+)?(?:an?\s+)?hour\b", "15 minutes", t)
    t = re.sub(r"\ban?\s+(hour|minute|second)", r"1 \1", t)
    # "1 and a half hours" / "1 hour and a half"
    half = re.search(r"(\d+)\s*(hours?|minutes?)\s*and\s*a\s*half\b", t)
    if not half:
        half = re.search(r"(\d+)\s*and\s*a\s*half\s*(hours?|minutes?)", t)
    total = 0.0
    found = False
    if half:
        unit = _DURATION_UNITS[half.group(2) if half.group(2) in _DURATION_UNITS else "m"]
        total += (float(half.group(1)) + 0.5) * unit
        found = True
        t = t.replace(half.group(0), " ")
    for amount, unit in _DURATION_RE.findall(t):
        total += float(amount) * _DURATION_UNITS[unit]
        found = True
    if not found or total <= 0:
        return None
    return int(round(total))


def format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not hours and secs and minutes < 5:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " and ".join(parts) if parts else "0 seconds"


def parse_time_of_day(text: str, now: datetime | None = None) -> tuple[int, int] | None:
    """Parse '6pm', '6:30 am', '18:00', 'noon' into (hour, minute)."""
    now = now or datetime.now()
    t = normalize_numbers(text)
    if re.search(r"\bnoon\b", t):
        return 12, 0
    if re.search(r"\bmidnight\b", t):
        return 0, 0
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)", t)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        meridiem = m.group(3).replace(".", "")
        if hour > 12 or minute > 59:
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return hour, minute
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\bat\s+(\d{1,2})(?:\s*o'?clock)?\b(?!\s*(?:minutes?|mins?|hours?|%|kg|pounds?))", t)
    if m:
        hour = int(m.group(1))
        if not 0 <= hour <= 23:
            return None
        if 1 <= hour <= 11:
            # Ambiguous: pick whichever of am/pm is next.
            for candidate in (hour, hour + 12):
                if now.replace(hour=candidate, minute=0, second=0, microsecond=0) > now:
                    return candidate, 0
            return hour, 0
        return hour, 0
    return None


def _add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


_PART_OF_DAY = {"morning": (8, 0), "afternoon": (15, 0), "evening": (19, 0), "tonight": (21, 0), "night": (21, 0)}


def parse_when(text: str, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Find a point in time in `text`.

    Returns (datetime or None, text with the matched time phrase removed).
    Handles relative ("in 20 minutes"), clock ("at 6pm"), day-relative
    ("tomorrow at 9", "tonight", "friday") and calendar dates ("by december 31").
    """
    now = now or datetime.now()
    t = normalize_numbers(text)
    working = t

    def cut(pattern: str) -> None:
        nonlocal working
        working = re.sub(pattern, " ", working, count=1)

    # --- relative: "in 20 minutes", "in 2 weeks", "in 3 months"
    m = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*(months?|weeks?|days?)\b", working)
    if m:
        amount, unit = float(m.group(1)), m.group(2)
        if unit.startswith("month"):
            target = _add_months(now.date(), int(amount))
            result = datetime.combine(target, now.time())
        elif unit.startswith("week"):
            result = now + timedelta(weeks=amount)
        else:
            result = now + timedelta(days=amount)
        cut(re.escape(m.group(0)))
        return result, _tidy(working)

    m = re.search(r"\bin\s+((?:an?\s+|half\s+|\d)[^,.]*?(?:hours?|hrs?|minutes?|mins?|seconds?|secs?))\b", working)
    if m:
        seconds = parse_duration(m.group(1))
        if seconds:
            cut(re.escape(m.group(0)))
            return now + timedelta(seconds=seconds), _tidy(working)

    # --- calendar dates: "december 31", "31 december", "31st of december"
    month_pat = "|".join(sorted({*MONTHS, *_MONTH_ABBR}, key=len, reverse=True))
    m = re.search(rf"\b(?:by|on|before|until)?\s*({month_pat})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b", working)
    day_first = False
    if not m:
        m = re.search(rf"\b(?:by|on|before|until)?\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_pat})\.?(?:,?\s*(\d{{4}}))?\b", working)
        day_first = bool(m)
    if m:
        if day_first:
            day, month_name, year = int(m.group(1)), m.group(2), m.group(3)
        else:
            month_name, day, year = m.group(1), int(m.group(2)), m.group(3)
        month = _MONTH_ABBR.get(month_name[:3]) or (MONTHS.index(month_name) + 1 if month_name in MONTHS else None)
        if month and 1 <= day <= 31:
            y = int(year) if year else now.year
            try:
                target = date(y, month, min(day, monthrange(y, month)[1]))
            except ValueError:
                target = None
            if target:
                if not year and target < now.date():
                    target = date(y + 1, month, min(day, monthrange(y + 1, month)[1]))
                tod = parse_time_of_day(working, now)
                hour, minute = tod if tod else (9, 0)
                cut(re.escape(m.group(0)))
                return datetime(target.year, target.month, target.day, hour, minute), _tidy(working)

    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", working)
    if iso:
        try:
            result = datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)), 9, 0)
            cut(re.escape(iso.group(0)))
            return result, _tidy(working)
        except ValueError:
            pass

    # --- day-relative
    base_day: date | None = None
    default_time: tuple[int, int] | None = None

    if re.search(r"\bday after tomorrow\b", working):
        base_day = now.date() + timedelta(days=2)
        cut(r"\bday after tomorrow\b")
    elif re.search(r"\btomorrow\b", working):
        base_day = now.date() + timedelta(days=1)
        cut(r"\btomorrow\b")
    elif re.search(r"\btonight\b", working):
        base_day = now.date()
        default_time = _PART_OF_DAY["tonight"]
        cut(r"\btonight\b")
    elif re.search(r"\bnext week\b", working):
        base_day = now.date() + timedelta(days=7)
        default_time = (9, 0)
        cut(r"\bnext week\b")
    elif re.search(r"\bnext month\b", working):
        base_day = _add_months(now.date(), 1)
        default_time = (9, 0)
        cut(r"\bnext month\b")
    else:
        wd = re.search(r"\b(?:(next|on|this)\s+)?(" + "|".join(WEEKDAYS) + r")\b", working)
        if wd:
            target_idx = WEEKDAYS.index(wd.group(2))
            delta = (target_idx - now.weekday()) % 7
            if delta == 0 or wd.group(1) == "next":
                delta = delta or 7
            base_day = now.date() + timedelta(days=delta)
            cut(re.escape(wd.group(0)))

    part = re.search(r"\b(?:this|in the)\s+(morning|afternoon|evening)\b", working)
    if part:
        default_time = _PART_OF_DAY[part.group(1)]
        base_day = base_day or now.date()
        cut(re.escape(part.group(0)))

    tod = parse_time_of_day(working, now)
    if tod:
        cut(r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|o'?clock)?|\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?!\w)|\b\d{1,2}:\d{2}\b|\bnoon\b|\bmidnight\b")
        hour, minute = tod
    elif default_time:
        hour, minute = default_time
    elif base_day is not None:
        hour, minute = 9, 0
    else:
        return None, _tidy(text.lower())

    day = base_day or now.date()
    result = datetime(day.year, day.month, day.day, hour, minute)
    if base_day is None and result <= now:
        result += timedelta(days=1)
    return result, _tidy(working)


def _tidy(text: str) -> str:
    text = re.sub(r"\b(?:at|by|on|in|for|this)\s*$", " ", text.strip())
    text = re.sub(r"^(?:to|that)\s+", "", text.strip())
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;")


def humanize_when(when: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now()
    delta = (when - now).total_seconds()
    if 0 <= delta < 3600:
        minutes = max(1, int(round(delta / 60)))
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    clock = when.strftime("%I:%M %p").lstrip("0").replace(":00", "")
    if when.date() == now.date():
        return f"today at {clock}"
    if when.date() == now.date() + timedelta(days=1):
        return f"tomorrow at {clock}"
    if (when.date() - now.date()).days < 7:
        return f"{WEEKDAYS[when.weekday()].title()} at {clock}"
    return f"{when.strftime('%B')} {when.day} at {clock}"
