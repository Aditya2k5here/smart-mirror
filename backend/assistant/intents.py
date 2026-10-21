"""Rule-based intent parsing for the voice/text fast path.

Anything matched here runs instantly and offline. Everything else goes to the LLM,
which can emit the very same intent names as structured actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable
import re

from core.timeparse import (
    normalize_numbers, parse_duration, parse_time_of_day, parse_when,
)
from assistant.routines import find_breathing, find_routine
from memory.foods import estimate_meal
from workout.exercises import find_exercise

GLASS_ML, CUP_ML, BOTTLE_ML = 250, 250, 500
KG_PER_LB = 0.45359237

_UNIT = r"(kg|kgs|kilo|kilos|kilogram|kilograms|lb|lbs|pound|pounds)"
_NUM = r"(\d+(?:\.\d+)?)"


@dataclass
class Intent:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class ParseContext:
    now: datetime
    units: str = "metric"
    routine_active: bool = False
    breathing_active: bool = False
    timer_ringing: bool = False
    is_guest: bool = False


def clean(text: str) -> str:
    t = normalize_numbers(text)
    t = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1.\2", t)
    t = re.sub(r"[^\w\s.,'%:/-]", " ", t)
    t = re.sub(r"^(?:please|hey|ok|okay|um|uh|so|well|mirror)[ ,]+", "", t.strip())
    t = re.sub(r"^(?:can you|could you|would you|will you|i want you to|i'd like you to)\s+", "", t)
    t = re.sub(r"\s+(?:please|thanks|thank you)$", "", t)
    return re.sub(r"\s+", " ", t).strip(" .,!?")


def to_kg(value: float, unit: str | None, units: str) -> float:
    if unit:
        return value * KG_PER_LB if unit.startswith(("lb", "pound")) else value
    return value * KG_PER_LB if units == "imperial" else value


def _plausible_weight(kg: float) -> bool:
    return 25 <= kg <= 400


# ---------------------------------------------------------------------------
# individual parsers: (clean_text, ctx) -> Intent | None
# ---------------------------------------------------------------------------
Parser = Callable[[str, ParseContext], "Intent | None"]


def p_stop(t: str, c: ParseContext) -> Intent | None:
    if re.fullmatch(r"(?:stop|quiet|be quiet|shut up|enough|silence|never ?mind|cancel that|stop talking|that's enough|okay stop|ok stop)", t):
        return Intent("stop_speaking")
    return None


def p_routine_control(t: str, c: ParseContext) -> Intent | None:
    if not c.routine_active:
        return None
    if re.fullmatch(r"(?:next|next step|done|finished|i'm done|i am done|ok next|okay next|go on|continue|ready|got it|skip|skip it|skip this)", t):
        return Intent("routine_next")
    if re.search(r"\b(?:stop|cancel|end|exit|quit)\b.*\broutine\b|\broutine\b.*\b(?:stop|cancel|over)\b|^(?:stop|cancel)$", t):
        return Intent("routine_stop")
    return None


def p_breathing_control(t: str, c: ParseContext) -> Intent | None:
    if c.breathing_active and re.search(r"\b(?:stop|cancel|end|enough|i'm good|i am good|better)\b", t):
        return Intent("breathing_stop")
    return None


def p_timer_ring(t: str, c: ParseContext) -> Intent | None:
    if c.timer_ringing and re.fullmatch(r"(?:stop|ok|okay|thanks|got it|dismiss|snooze|enough|quiet|stop it)", t):
        return Intent("timer_dismiss")
    return None


def p_privacy(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\bprivacy mode\b.*\b(?:off|disable|end|exit)\b|\b(?:off|disable|exit|end)\b.*\bprivacy mode\b|\bresume (?:the )?camera\b|\bturn (?:the )?camera on\b|\bstart watching\b|\bcamera on\b", t):
        return Intent("privacy", {"on": False})
    if re.search(r"\bprivacy mode\b|\bstop watching( me)?\b|\b(?:pause|turn off|disable) (?:the )?camera\b|\bcamera off\b|\bi need privacy\b", t):
        return Intent("privacy", {"on": True})
    return None


def p_forget(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:forget|delete|erase|wipe)\b.*\b(?:everything|all)\b.*\b(?:about me|you know|my data)\b|\bdelete (?:all )?my data\b|\bforget me\b", t):
        return Intent("forget_all")
    m = re.match(r"(?:forget|delete|remove) (?:that |the fact that |what i said about )?(.{4,})$", t)
    if m and not re.match(r"(?:the )?(?:timer|task|reminder|goal|routine|meal|last meal|note|list)", m.group(1)):
        return Intent("forget_fact", {"phrase": m.group(1)})
    return None


_NAME_STOP = {
    "yes", "no", "nothing", "nobody", "none", "fine", "good", "okay", "ok", "sure", "hello", "hi", "hey", "thanks",
    "here", "back", "tired", "hungry", "happy", "sad", "ready", "done", "stressed", "sorry", "not", "the", "a",
}


def extract_name(text: str, strict: bool = True) -> str | None:
    """Pull a first name out of 'my name is Priya' / 'call me Pri' / (non-strict) bare 'Priya'."""
    t = re.sub(r"[^\w\s'-]", " ", text.strip())
    m = re.search(
        r"(?:my name is|my name's|i am called|i'm called|call me|they call me|people call me|it's|it is|this is|i am|i'm|name is)\s+([A-Za-z][A-Za-z'-]{1,20})(?:\s+([A-Za-z][A-Za-z'-]{1,20}))?\b",
        t, re.I,
    )
    candidate = None
    if m:
        first = m.group(1)
        candidate = first
        second = m.group(2)
        # keep a surname-ish second word only if it is not a verb-ish filler
        if second and second.lower() not in _NAME_STOP and second.lower() not in {"and", "from", "but", "i", "so", "how", "can", "please"}:
            if re.search(r"(?:my name is|call me|name is)", text, re.I):
                candidate = f"{first} {second}"
    elif not strict:
        words = t.split()
        if 1 <= len(words) <= 2:
            candidate = " ".join(words)
    if not candidate:
        return None
    parts = candidate.split()
    if any(p.lower() in _NAME_STOP for p in parts):
        return None
    return " ".join(p.capitalize() for p in parts)


def p_enroll(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:that's not me|that is not me|i'm not|i am not)\b", t) and re.search(r"\b(?:not me|not [a-z]+)\b", t):
        if re.search(r"that'?s not me|that is not me|i(?:'m| am) not (?!tired|hungry|sure|ready|feeling|sleepy|well|good|okay|ok)", t):
            return Intent("not_me")
    if re.search(r"\b(?:my name is|call me|they call me|i am called|i'm called|people call me)\b", t):
        name = extract_name(t)
        if name:
            return Intent("set_name", {"name": name})
    return None


def p_weight_log(t: str, c: ParseContext) -> Intent | None:
    m = re.search(
        r"\b(?:i weigh|i weighed|i am weighing|i'm weighing|my weight is|my weight today is|my weight was|weight is|weight today is|"
        r"weighed in at|weigh in at|weigh-in|log (?:my )?weight(?: as| of| at)?|record (?:my )?weight(?: as| at)?|add (?:my )?weight)\s*"
        r"(?:in at |about |around |at |roughly |is )?" + _NUM + r"\s*" + _UNIT + r"?", t)
    unit = None
    if m:
        value, unit = float(m.group(1)), m.group(2)
    else:
        m = re.search(r"\bi(?:'m| am)\s+" + _NUM + r"\s*" + _UNIT + r"\b", t)
        if not m:
            return None
        value, unit = float(m.group(1)), m.group(2)
    kg = to_kg(value, unit, c.units)
    if not _plausible_weight(kg):
        return None
    return Intent("log_weight", {"value_kg": round(kg, 2)})


def _deadline(t: str, now: datetime) -> date | None:
    when, _ = parse_when(t, now)
    if when and when.date() > now.date():
        return when.date()
    return None


def p_weight_goal(t: str, c: ParseContext) -> Intent | None:
    lose = re.search(
        r"\b(?:want to|need to|aim to|trying to|plan to|hoping to|would like to|wanna|have to|going to|goal is to|goal to|set (?:a )?(?:goal|target) (?:to|of)|help me|let's) "
        r"(?:to )?(?:lose|drop|shed|cut|losing)\s+" + _NUM + r"\s*" + _UNIT + r"?", t)
    gain = re.search(
        r"\b(?:want to|need to|aim to|trying to|plan to|hoping to|would like to|wanna|goal is to|goal to|set (?:a )?(?:goal|target) (?:to|of)|help me)\s+(?:to )?(?:gain|add|build)\s+" + _NUM + r"\s*" + _UNIT, t)
    target = re.search(
        r"\b(?:want to|need to|aim to|trying to|would like to|goal is to|goal to|wanna|hoping to)\s+(?:weigh|get down to|get to|reach|be|hit|get under)\s+(?:about |around )?" + _NUM + r"\s*" + _UNIT + r"?", t) \
        or re.search(r"\b(?:goal|target) weight (?:is|of)\s+" + _NUM + r"\s*" + _UNIT + r"?", t) \
        or re.search(r"\bset (?:a )?(?:goal|target) (?:weight )?(?:of|at)\s+" + _NUM + r"\s*" + _UNIT + r"?", t)
    by = _deadline(t, c.now)
    if lose:
        delta = to_kg(float(lose.group(1)), lose.group(2), c.units)
        if 0.5 <= delta <= 80:
            return Intent("set_weight_goal", {"delta_kg": round(delta, 2), "direction": "down", "by": by.isoformat() if by else None})
    if gain:
        delta = to_kg(float(gain.group(1)), gain.group(2), c.units)
        if 0.5 <= delta <= 60:
            return Intent("set_weight_goal", {"delta_kg": round(delta, 2), "direction": "up", "by": by.isoformat() if by else None})
    if target:
        kg = to_kg(float(target.group(1)), target.group(2), c.units)
        if _plausible_weight(kg):
            return Intent("set_weight_goal", {"target_kg": round(kg, 2), "by": by.isoformat() if by else None})
    return None


def p_generic_goal(t: str, c: ParseContext) -> Intent | None:
    m = re.match(r"(?:my goal is to|set (?:a |my )?(?:new )?goal (?:to|for|of)|new goal[: ]|my new goal is to|add (?:a )?goal (?:to|of)) (.{4,100})$", t)
    if m:
        by = _deadline(t, c.now)
        title = parse_when(m.group(1), c.now)[1] if by else m.group(1)
        return Intent("set_goal", {"title": title[:1].upper() + title[1:], "by": by.isoformat() if by else None})
    return None


def _height_cm(t: str) -> float | None:
    m = re.search(r"\b(\d{3})\s*(?:cm|centimeters|centimetres)\b", t)
    if m and 120 <= int(m.group(1)) <= 230:
        return float(m.group(1))
    m = re.search(r"\b(\d)\s*(?:feet|foot|ft|')\s*(\d{1,2})?(?:\s*(?:inches|inch|in|\"))?", t)
    if m and 3 <= int(m.group(1)) <= 7:
        return round((int(m.group(1)) * 12 + int(m.group(2) or 0)) * 2.54, 1)
    return None


def _clock_for(t: str, kind: str, now: datetime) -> str | None:
    tod = parse_time_of_day(t.replace(" at ", " at "), now)
    if not tod:
        m = re.search(r"\b(\d{1,2})\b", t)
        if not m:
            return None
        tod = (int(m.group(1)), 0)
    hour, minute = tod
    if hour <= 12 and not re.search(r"\b(?:a\.?m|p\.?m)\b", t):
        if kind == "wake" and 1 <= hour <= 11:
            pass  # am
        if kind == "bed" and 6 <= hour <= 11:
            hour += 12
    if not (0 <= hour <= 23):
        return None
    return f"{hour:02d}:{minute:02d}"


def p_profile(t: str, c: ParseContext) -> Intent | None:
    fields: dict[str, Any] = {}
    if re.search(r"\b(?:i am|i'm|my height is|height is|i stand|my height's)\b", t) or re.search(r"\b\d{3}\s*cm\b", t):
        height = _height_cm(t)
        if height:
            fields["height_cm"] = height
    m = re.search(r"\b(?:i am|i'm|my age is|age is)\s+(\d{2})(?:\s*(?:years|year|yrs)(?:\s*old)?)?\b(?!\s*(?:kg|kilo|lb|pound|cm))", t)
    if m and (re.search(r"years? old|yrs|my age", t) or re.fullmatch(r"(?:i am|i'm)\s+\d{2}", t)):
        age = int(m.group(1))
        if 10 <= age <= 100:
            fields["age"] = age
    m = re.search(r"\bi turned (\d{2})\b", t)
    if m:
        fields["age"] = int(m.group(1))
    m = re.search(r"\bi(?:'m| am) (?:a |an )?(male|female|man|woman|guy|girl)\b", t)
    if m:
        fields["sex"] = "male" if m.group(1) in ("male", "man", "guy") else "female"
    m = re.search(r"\bactivity level (?:is )?(sedentary|light|moderate|active|athlete|very active)", t)
    if m:
        fields["activity_level"] = "athlete" if m.group(1) == "very active" else m.group(1)
    elif re.search(r"\bi(?:'m| am) (?:very|super|really) active\b", t):
        fields["activity_level"] = "active"
    elif re.search(r"\bi(?:'m| am) (?:moderately|fairly) active\b|\bi(?:'m| am) active\b", t):
        fields["activity_level"] = "moderate"
    elif re.search(r"\bi(?:'m| am) (?:lightly|slightly) active\b|\bi sit (?:all day|a lot)\b|\bdesk job\b", t):
        fields["activity_level"] = "light"
    m = re.search(r"\bi (?:work ?out|exercise|train|go to the gym) (?:about |around )?(\d) times? a week", t)
    if m:
        n = int(m.group(1))
        fields["activity_level"] = "sedentary" if n == 0 else "light" if n <= 2 else "moderate" if n <= 4 else "active" if n <= 6 else "athlete"
    m = re.search(r"\bi (?:usually |normally |always )?(?:go to bed|go to sleep|turn in|sleep)(?: at| around| by)?\s+(.{1,12})$", t)
    if m:
        clock = _clock_for(m.group(1), "bed", c.now)
        if clock:
            fields["bed_time"] = clock
    m = re.search(r"\bi (?:usually |normally |always )?(?:wake up|get up|wake)(?: at| around| by)?\s+(.{1,12})$", t)
    if m:
        clock = _clock_for(m.group(1), "wake", c.now)
        if clock:
            fields["wake_time"] = clock
    if re.search(r"\b(?:use|switch to|change to) (?:pounds|imperial|lbs)\b", t):
        fields["units"] = "imperial"
    elif re.search(r"\b(?:use|switch to|change to) (?:kilos|kilograms|metric|kg)\b", t):
        fields["units"] = "metric"
    return Intent("update_profile", fields) if fields else None


def p_water(t: str, c: ParseContext) -> Intent | None:
    if not re.search(r"\bwater\b", t) and not re.search(r"\b(?:ml|millilit(?:er|re)s?|lit(?:er|re)s?)\b", t):
        return None
    if re.search(r"\b(?:how (?:much|many)|what|remind|reminder)\b", t):
        return None
    ml = None
    m = re.search(_NUM + r"\s*(?:ml|millilit(?:er|re)s?)\b", t)
    if m:
        ml = float(m.group(1))
    else:
        m = re.search(r"(?:" + _NUM + r"|an?|another|one|a couple of|some)?\s*(glass(?:es)?|cups?|bottles?|lit(?:er|re)s?|l)\b(?: of)?(?: water)?", t)
        if m and re.search(r"water", t):
            qty = float(m.group(1)) if m.group(1) else 1.0
            if m.group(0).strip().startswith("a couple"):
                qty = 2
            unit = m.group(2)
            per = GLASS_ML if unit.startswith("glass") else CUP_ML if unit.startswith("cup") else BOTTLE_ML if unit.startswith("bottle") else 1000
            ml = qty * per
    if ml is None and re.search(r"\b(?:drank|drink|had|finished|just had|sipped)\b.*\bwater\b", t):
        ml = GLASS_ML
    if ml is None and re.search(r"\blog (?:some )?water\b", t):
        ml = GLASS_ML
    if ml and 30 <= ml <= 5000 and re.search(r"\b(?:drank|drink|drunk|had|finished|log|logged|add|sipped|water)\b", t):
        return Intent("log_water", {"ml": ml})
    return None


_MEAL_START = re.compile(
    r"^(?:i |we )?(?:just |only )?(?:had|ate|eaten|have had|am having|i'm having|having|eating|grabbed|snacked on|log(?:ged)?(?: a| my)?(?: meal| breakfast| lunch| dinner| snack)?(?: of| as| was)?|add(?: a)? meal)\b|^(?:for (?:breakfast|lunch|dinner|brunch|a snack),? i (?:had|ate|have))")


def p_meal(t: str, c: ParseContext) -> Intent | None:
    m = re.search(r"\b(?:that was|that's|it was|it's|make that|actually)\s+" + _NUM + r"\s*(?:calories|cals|kcal)\b", t)
    if m:
        return Intent("fix_meal_calories", {"calories": float(m.group(1))})
    if re.search(r"\b(?:undo|remove|delete|scratch)\b.*\b(?:last |that )?meal\b|\b(?:undo|remove) that\b.*\b(?:food|snack|meal)\b", t):
        return Intent("undo_meal")
    m = re.search(_NUM + r"\s*(?:calories|cals|kcal)\b", t)
    if m and re.search(r"\b(?:had|ate|log|eat|consumed)\b", t):
        return Intent("log_meal", {"description": re.sub(r"\s+", " ", re.sub(_NUM + r"\s*(?:calories|cals|kcal)", "", t)).strip() or "meal",
                                   "calories": float(m.group(1))})
    if _MEAL_START.search(t) or re.match(r"(?:log|add) (?:a )?(?:meal|breakfast|lunch|dinner|snack)", t):
        if re.search(r"\bwater\b", t):
            return None
        # "I had a great day" is not a meal: need recognised food or an explicit meal word.
        if estimate_meal(t)["items"] or re.search(r"\b(?:breakfast|lunch|dinner|brunch|snack|meal)\b", t):
            return Intent("log_meal", {"description": t})
    return None


def p_timer(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:remind me|reminder)\b", t):
        return None
    if re.search(r"\b(?:cancel|stop|clear|delete|remove|kill)\b.*\btimers?\b", t):
        label = re.sub(r"\b(?:cancel|stop|clear|delete|remove|kill|the|my|timers?|all|that)\b", "", t).strip()
        return Intent("cancel_timer", {"label": label})
    if re.search(r"\b(?:how (?:much )?(?:time|long)|time left|what's left|status)\b.*\btimers?\b|\btimers?\b.*\b(?:left|remaining)\b|\bhow long (?:is )?left\b", t):
        return Intent("timer_status")
    if re.search(r"\b(?:pomodoro|focus (?:session|mode|time)|deep work)\b", t):
        minutes = 25
        dur = parse_duration(t)
        if dur:
            minutes = max(5, min(120, dur // 60))
        return Intent("start_focus", {"minutes": minutes})
    if re.search(r"\btimer\b|\bcount ?down\b|\bset an? alarm for\b", t) or re.search(r"\b(?:start|set|run)\b.*\b(?:minute|hour|second)s?\b.*\b(?:timer|clock)\b", t):
        dur = parse_duration(t)
        if dur:
            label = ""
            lm = re.search(r"\bfor (?:the |my )?([a-z ]{3,30})$", t)
            if lm and not parse_duration(lm.group(1)):
                label = lm.group(1).strip()
            return Intent("set_timer", {"seconds": dur, "label": label})
    return None


def p_reminder(t: str, c: ParseContext) -> Intent | None:
    m = re.match(r"remind me every (day|morning|evening|night|afternoon|weekday)(?: at (.+?))? (?:to|about) (.+)$", t)
    if m:
        times = {"morning": "08:00", "afternoon": "15:00", "evening": "19:00", "night": "21:30"}
        when = times.get(m.group(1), "09:00")
        if m.group(2):
            tod = parse_time_of_day("at " + m.group(2), c.now)
            if tod:
                when = f"{tod[0]:02d}:{tod[1]:02d}"
        return Intent("add_habit", {"name": m.group(3).strip(), "remind_times": when, "target": 1})
    m = re.match(r"(?:remind me|set a reminder|add a reminder|create a reminder)\b(.*)$", t)
    if m:
        rest = m.group(1).strip()
        when, text = parse_when(rest, c.now)
        text = re.sub(r"^(?:to|about|that|for)\s+", "", text).strip()
        text = re.sub(r"\s+(?:to|about)$", "", text).strip()
        if not text:
            return Intent("add_reminder", {"text": "", "when": when.isoformat() if when else None})
        return Intent("add_reminder", {"text": text[:1].upper() + text[1:], "when": when.isoformat() if when else None})
    m = re.match(r"(?:add|create|new) (?:a )?(?:task|todo|to-do)(?: to| for)?[: ]+(.{3,})$", t)
    if m:
        return Intent("add_reminder", {"text": m.group(1)[:1].upper() + m.group(1)[1:], "when": None})
    if re.search(r"\b(?:complete|finish|finished|done with|mark)\b.*\b(?:task|reminder)\b|\bmark (?:it|that) (?:as )?done\b|\btask (?:is )?done\b", t):
        return Intent("complete_task")
    if re.search(r"\b(?:what(?:'s| are| is)|show|read|list)\b.*\b(?:my )?(?:tasks|reminders|to-?dos?|to do)\b", t):
        return Intent("list_tasks")
    return None


def p_routine(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:breath(?:e|ing)|calm me down|help me (?:relax|calm down)|i need to (?:relax|calm down)|box breathing|4 7 8|4-7-8)\b", t) and not re.search(r"\bhow\b.*\bbreath", t):
        if not re.search(r"\b(?:stop|cancel)\b", t):
            return Intent("start_breathing", {"pattern": find_breathing(t)})
    m = re.search(r"\b(?:start|begin|run|do|let's do|let's start|open|lets do)\b.*?\b(?:my |the |a )?(.*?)(?:routine|session)\b", t)
    if m:
        key = find_routine(m.group(1) or t) or find_routine(t)
        if key:
            return Intent("start_routine", {"name": key})
        if re.search(r"\broutine\b", t):
            return Intent("start_routine", {"name": "morning"})
    if re.search(r"\b(?:time to brush|brush my teeth|start brushing)\b", t):
        return Intent("start_routine", {"name": "brushing"})
    if re.search(r"\b(?:start|begin|do)\b.*\b(?:stretch(?:ing)?|skincare|skin care)\b", t):
        return Intent("start_routine", {"name": find_routine(t) or "stretch"})
    return None


def p_workout(t: str, c: ParseContext) -> Intent | None:
    stop = re.search(
        r"\b(?:stop|end|finish|exit|done with|that's it for)\b.*\b(?:workout|training|exercise|set)\b|\bturn off training(?: mode)?\b|\bworkout (?:over|done|finished)\b", t)
    if stop:
        return Intent("stop_workout")
    if re.search(
            r"\b(?:start|begin|turn on|let's do|time for|lets do)\b.*\b(?:workout|training|exercise|curls?|squats?|push ?ups?|posture mode|lunges?|jumping jacks?|shoulder press|planks?)\b"
            r"|\b(?:workout|training|curl|squat|push ?up|posture) mode\b", t):
        return Intent("start_workout", {"exercise": find_exercise(t)})
    if re.search(r"\bhow (?:many )?(?:workouts|did i work ?out|active|much did i (?:work ?out|exercise))\b|\bworkout (?:summary|history|stats)\b", t):
        return Intent("query", {"topic": "workouts"})
    return None


def p_music(t: str, c: ParseContext) -> Intent | None:
    if re.match(r"(?:pause|stop|mute)(?: the)?(?: music| song| spotify| track| playback)$", t):
        return Intent("music", {"command": "pause"})
    if re.match(r"(?:resume|unpause|continue)(?: the)?(?: music| song| spotify| track| playback)?$|(?:play|start)(?: the)?(?: music| spotify)$", t):
        return Intent("music", {"command": "play"})
    if re.match(r"(?:next|skip)(?: this)?(?: song| track)$|^skip$", t) and not c.routine_active:
        return Intent("music", {"command": "next"})
    if re.match(r"(?:previous|last|go back a?)(?: song| track)$|play (?:the )?previous", t):
        return Intent("music", {"command": "previous"})
    m = re.match(r"(?:turn |make it |set )?(?:the )?(?:volume|music|sound)(?: is)?(?: a bit| a little)? (up|down|louder|quieter|lower|higher)$|(?:louder|quieter|turn it up|turn it down)$", t)
    if m:
        up = bool(re.search(r"up|louder|higher", t))
        return Intent("music", {"command": "volume_up" if up else "volume_down"})
    m = re.match(r"(?:play|put on|queue)\s+(?!the music|music$)(.{3,60})$", t)
    if m and not re.search(r"\b(?:workout|routine|timer|game)\b", m.group(1)):
        query = re.sub(r"^(?:some|something|a few)\s+", "", m.group(1)).strip()
        query = re.sub(r"\s+(?:on spotify|please)$", "", query)
        return Intent("music", {"command": "search", "query": query})
    return None


def p_lights(t: str, c: ParseContext) -> Intent | None:
    m = re.search(r"\b(?:turn|switch|put)\s+(on|off)\s+(?:the |my )?([a-z ]{0,20}?)\s*(?:lights?|lamp)\b|\b(?:turn|switch) (?:the |my )?([a-z ]{0,20}?)\s*(?:lights?|lamp) (on|off)\b|\b(?:lights?|lamp) (on|off)\b", t)
    if m:
        state = m.group(1) or m.group(4) or m.group(5)
        room = (m.group(2) or m.group(3) or "").strip()
        return Intent("lights", {"command": state, "room": room})
    if re.search(r"\b(?:dim|brighten|lower|raise)\b.*\b(?:lights?|lamp)\b", t):
        return Intent("lights", {"command": "dim" if re.search(r"dim|lower", t) else "brighten", "room": ""})
    m = re.search(r"\b(?:activate|set|start|turn on) (?:the )?([a-z ]{3,25}?) (?:scene|mode)\b", t)
    if m and m.group(1).strip() not in ("privacy", "training", "workout", "settings", "diagnostics", "posture", "curl", "squat", "push up", "pushup"):
        return Intent("lights", {"command": "scene", "room": m.group(1).strip()})
    return None


def p_outfit(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\bhow do i look\b|\boutfit check\b|\bhow(?:'s| is) my (?:outfit|look|fit)\b|\brate my (?:outfit|look|fit)\b|\bdo i look (?:okay|ok|good|alright)\b|\bwhat do you see\b|\blook at me\b", t):
        return Intent("outfit_check", {"question": t})
    return None


def p_notes(t: str, c: ParseContext) -> Intent | None:
    m = re.match(r"(?:add|put) (.{2,60}?) (?:to|on|onto|in) (?:my |the )?(shopping|grocery|groceries|to-?do|todo|packing|errands?)(?: list)?$", t)
    if m:
        name = {"groceries": "shopping", "grocery": "shopping", "todo": "to-do", "to-do": "to-do", "errand": "errands"}.get(m.group(2), m.group(2))
        return Intent("add_note", {"text": m.group(1), "list": name})
    m = re.match(r"(?:what(?:'s| is)(?: on)?|show|read|list)(?: me)?(?: out)? (?:my |the )?(shopping|grocery|groceries|packing|errands?|notes)(?: list)?$", t)
    if m:
        name = {"groceries": "shopping", "grocery": "shopping", "errand": "errands"}.get(m.group(1), m.group(1))
        return Intent("show_list", {"list": name})
    m = re.match(r"(?:clear|empty) (?:my |the )?(shopping|grocery|groceries|packing|errands?|notes)(?: list)?$", t)
    if m:
        return Intent("clear_list", {"list": {"groceries": "shopping", "grocery": "shopping"}.get(m.group(1), m.group(1))})
    m = re.match(r"(?:take a note|note down|make a note|note)[: ,]*(?:that )?(.{3,})$", t)
    if m:
        return Intent("add_note", {"text": m.group(1), "list": "notes"})
    m = re.match(r"(?:remember|don't forget|keep in mind|note) that (.{4,})$", t)
    if m:
        return Intent("remember", {"text": m.group(1)})
    return None


_MOODS = {
    "stressed": "stressed", "stressed out": "stressed", "anxious": "anxious", "nervous": "anxious", "worried": "anxious",
    "overwhelmed": "overwhelmed", "tired": "tired", "exhausted": "tired", "sleepy": "tired", "drained": "tired",
    "sad": "sad", "down": "sad", "low": "sad", "lonely": "lonely", "angry": "angry", "frustrated": "frustrated",
    "annoyed": "frustrated", "irritated": "frustrated", "bored": "bored", "happy": "happy", "great": "happy",
    "good": "happy", "amazing": "happy", "fantastic": "happy", "excited": "excited", "motivated": "motivated",
    "pumped": "motivated", "proud": "proud", "calm": "calm", "relaxed": "calm", "sick": "unwell", "unwell": "unwell",
    "fine": "okay", "okay": "okay", "alright": "okay", "meh": "okay", "burnt out": "overwhelmed", "burned out": "overwhelmed",
}


def p_mood(t: str, c: ParseContext) -> Intent | None:
    m = re.match(r"(?:i(?:'m| am)|i feel|i'm feeling|i am feeling|feeling|i have been feeling|i've been feeling|today i feel|i'm just)\s+(?:so |really |very |a bit |a little |kind of |kinda |pretty |quite |just |super |feeling |extremely )*([a-z ]{3,16})$", t)
    if m:
        word = m.group(1).strip()
        for phrase in sorted(_MOODS, key=len, reverse=True):
            if word == phrase or word.endswith(phrase):
                return Intent("log_mood", {"mood": _MOODS[phrase]})
    return None


def p_sleep(t: str, c: ParseContext) -> Intent | None:
    m = re.search(r"\bi (?:only )?(?:slept|got)\s+(?:about |around |roughly |like )?" + _NUM + r"\s*(?:hours?|hrs?)(?: of sleep)?(?: last night)?", t)
    if m and 0.5 <= float(m.group(1)) <= 16:
        return Intent("log_sleep", {"hours": float(m.group(1))})
    if re.search(r"\b(?:good night|goodnight|going to bed|heading to bed|time for bed|i'm off to bed|off to sleep|going to sleep|i'm going to sleep)\b", t):
        return Intent("goodnight")
    return None


def p_habit(t: str, c: ParseContext) -> Intent | None:
    m = re.match(r"(?:track|add|start tracking|create)(?: a| my)?(?: new)? habit(?: of| called| to)?\s+(.{3,40})$|track (?:my )?(.{3,30}) (?:daily|every day)$", t)
    if m:
        name = (m.group(1) or m.group(2)).strip()
        return Intent("add_habit", {"name": name[:1].upper() + name[1:], "target": 1})
    m = re.match(r"(?:i )?(?:just )?(?:took|did|finished|completed|logged|checked off) (?:my |the |a )?(.{3,30})$", t)
    if m and not re.search(r"\b(?:water|calories|breakfast|lunch|dinner|workout)\b", m.group(1)):
        return Intent("log_habit", {"name": m.group(1).strip()})
    m = re.match(r"(?:stop|remove|delete) (?:tracking )?(?:my )?habit (.{3,40})$", t)
    if m:
        return Intent("remove_habit", {"name": m.group(1).strip()})
    return None


_QUERIES: list[tuple[str, str]] = [
    (r"\bhow (?:am i|are we) doing\b|\bhow(?:'s| is) my progress\b|\bam i on track\b|\bprogress report\b|\bhow(?:'s| is) my (?:weight|goal)\b|\bwhat(?:'s| is) my weight\b|\bgoal progress\b|\bhow far (?:along|am i)\b", "progress"),
    (r"\bhow many calories\b|\bcalories? (?:left|remaining)\b|\bcalorie (?:budget|count|target)\b|\bhow much can i eat\b|\bmy calories\b", "calories"),
    (r"\bhow much water\b|\bwater (?:so far|today|left)\b", "water"),
    (r"\bwhat (?:did i|have i) (?:eat|eaten)\b|\bwhat i ate\b|\bmy meals\b", "meals"),
    (r"\bhow was my week\b|\bweekly (?:report|review|summary)\b|\bweek in review\b|\bhow(?:'s| is) my week\b", "week"),
    (r"\bwhat do you (?:know|remember) about me\b|\bwhat have you learned\b|\bwhat do you know\b|\bwhat(?:'s| is) in your memory\b|\bshow (?:me )?(?:my )?memor(?:y|ies)\b", "facts"),
    (r"\bwhat(?:'s| is) (?:on )?(?:my )?(?:calendar|agenda|schedule|plan)\b|\bwhat do i have (?:today|on)\b|\bwhat(?:'s| is) next\b|\bany (?:meetings|events)\b", "agenda"),
    (r"\bwhat time is it\b|\bwhat(?:'s| is) the time\b|\btell me the time\b|\bcurrent time\b", "time"),
    (r"\bwhat(?:'s| is) (?:the |today's )?date\b|\bwhat day is it\b|\btoday's date\b", "date"),
    (r"\bwhat(?:'s| is) the weather\b|\bhow(?:'s| is) the weather\b|\bweather (?:today|now|outside)\b|\bwill it rain\b|\bdo i need an umbrella\b", "weather"),
    (r"\bwhat(?:'s| is) my streak\b|\bhow are my habits\b|\bmy habits\b", "habits"),
    (r"\bwhat can you do\b|\bhelp me with\b.*\bcommands\b|\bwhat are your (?:features|commands|skills)\b|^help$|\bhow do i use you\b", "help"),
    (r"\bhow(?:'s| is) my sleep\b|\bhow (?:much|well) did i sleep\b|\bmy sleep\b", "sleep"),
    (r"\bhow(?:'s| is) my mood\b|\bmood (?:this week|trend)\b|\bhow have i been feeling\b", "mood"),
]


def p_query(t: str, c: ParseContext) -> Intent | None:
    for pattern, topic in _QUERIES:
        if re.search(pattern, t):
            return Intent("query", {"topic": topic})
    return None


def p_journal(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:let me journal|i want to journal|journal(?:ing)? (?:time|mode)|log my day|write in my journal|start (?:a )?journal|reflect on my day|daily reflection)\b", t):
        return Intent("start_journal")
    m = re.match(r"(?:journal|dear diary)[: ,]+(.{6,})$", t)
    if m:
        return Intent("journal", {"text": m.group(1)})
    return None


def p_briefing(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:brief me|daily brief|briefing|what(?:'s| is) my day|how(?:'s| is) my day|good morning|start my day|catch me up|what(?:'s| is) up today|summary of my day)\b", t):
        return Intent("briefing")
    return None


def p_show(t: str, c: ParseContext) -> Intent | None:
    if re.search(r"\b(?:settings mode|show settings|hide settings)\b", t):
        return Intent("show", {"panel": "settings", "visible": "hide" not in t})
    if re.search(r"\b(?:diagnostics mode|show diagnostics|hide diagnostics)\b", t):
        return Intent("show", {"panel": "diagnostics", "visible": "hide" not in t})
    m = re.match(r"(?:show|open|display|bring up) (?:me )?(?:my |the )?(progress|goals?|stats|nutrition|calories|memory|memories|week|weekly report|notes|habits|chart|graph)(?: chart| graph| report)?$", t)
    if m:
        panel = {"goal": "progress", "goals": "progress", "stats": "progress", "chart": "progress", "graph": "progress",
                 "calories": "nutrition", "memories": "memory", "weekly report": "week"}.get(m.group(1), m.group(1))
        return Intent("show", {"panel": panel, "visible": True})
    if re.match(r"(?:hide|close|dismiss|clear)(?: the)?(?: panel| that| this| overlay| screen)?$|^go back$|^close that$", t):
        return Intent("show", {"panel": "none", "visible": False})
    if re.search(r"\bdemo mode\b|\bshow me what you can do\b", t):
        return Intent("query", {"topic": "help"})
    return None


PARSERS: list[Parser] = [
    p_stop, p_timer_ring, p_routine_control, p_breathing_control, p_privacy, p_forget, p_enroll,
    p_weight_log, p_weight_goal, p_generic_goal, p_profile, p_water, p_timer, p_reminder,
    p_routine, p_workout, p_music, p_lights, p_outfit, p_notes, p_sleep, p_mood, p_habit,
    p_meal, p_journal, p_briefing, p_query, p_show,
]


def parse_intent(text: str, ctx: ParseContext) -> Intent | None:
    t = clean(text)
    if not t:
        return None
    for parser in PARSERS:
        intent = parser(t, ctx)
        if intent:
            intent.raw = text
            return intent
    return None
