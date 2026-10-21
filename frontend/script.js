/* Smart Mirror UI.
 *
 * The browser is only a display. The Python backend owns all state and pushes a full
 * snapshot over a WebSocket (ws://localhost:8765) whenever something changes. This file:
 *   1. keeps that socket connected (and reconnects if the backend restarts),
 *   2. renders each snapshot into the page (index.html),
 *   3. sends what you type or click back to the backend as small JSON messages:
 *        {"type": "say", "text": "how am I doing"}
 *        {"type": "action", "name": "complete_task", "id": 3}
 */
"use strict";

const WS_PORT = new URLSearchParams(location.search).get("ws") || "8765";
const WS_HOST = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) ? location.hostname : "localhost";

const CAPTION_HOLD_MS = 14000;   // matches CAPTION_HOLD_SECONDS in runtime.py
const NUDGE_HOLD_MS = 18000;     // matches NUDGE_HOLD_SECONDS
const CELEBRATE_HOLD_MS = 6000;  // matches CELEBRATE_HOLD_SECONDS
const RING_GOAL = 2 * Math.PI * 52;     // circle r=52 in the goal ring
const RING_ROUTINE = 2 * Math.PI * 70;  // circle r=70 in the routine ring

const $ = (id) => document.getElementById(id);
const el = {
    body: document.body,
    screen: $("screen"),
    ambientArt: $("ambient-art"),
    celebrate: $("celebrate-layer"),
    time: $("time"),
    date: $("date"),
    connection: $("connection-status"),
    chipAi: $("chip-ai"),
    chipPrivacy: $("chip-privacy"),
    chipDemo: $("chip-demo"),
    nudge: $("nudge"),
    nudgeKind: $("nudge-kind"),
    nudgeText: $("nudge-text"),
    weatherTemp: $("weather-temp"),
    weatherExtra: $("weather-extra"),
    weatherDesc: $("weather-desc"),
    weatherCity: $("weather-city"),
    leftStack: document.querySelector(".left-stack"),
    rightStack: document.querySelector(".right-stack"),
    calendarPanel: $("calendar-panel"),
    calendar: $("calendar"),
    tasksPanel: $("tasks-container"),
    tasks: $("tasks"),
    timersPanel: $("timers-panel"),
    timers: $("timers"),
    habitsPanel: $("habits-panel"),
    habits: $("habits"),
    presenceHint: $("presence-hint"),
    goalPanel: $("goal-panel"),
    goalTitle: $("goal-title"),
    goalStatus: $("goal-status"),
    goalRingFill: $("goal-ring-fill"),
    goalPercent: $("goal-percent"),
    goalCurrent: $("goal-current"),
    goalUnit: $("goal-unit"),
    goalTarget: $("goal-target"),
    goalRate: $("goal-rate"),
    goalSpark: $("goal-spark"),
    goalHeadline: $("goal-headline"),
    todayPanel: $("today-panel"),
    calText: $("cal-text"),
    calBar: $("cal-bar"),
    proteinText: $("protein-text"),
    proteinBar: $("protein-bar"),
    waterText: $("water-text"),
    waterGlasses: $("water-glasses"),
    todayFoot: $("today-foot"),
    memoryPanel: $("memory-panel"),
    memoryFact: $("memory-fact"),
    insightPanel: $("insight-panel"),
    insight: $("insight"),
    workoutPanel: $("workout-container"),
    workoutExercise: $("workout-exercise"),
    workoutReps: $("workout-reps"),
    workoutRepsLabel: $("workout-reps-label"),
    workoutTimer: $("workout-timer"),
    workoutFeedback: $("workout-feedback"),
    systemPanel: $("system-panel"),
    healthSummary: $("health-summary"),
    healthItems: $("health-items"),
    settingsPanel: $("settings-panel"),
    settingsItems: $("settings-items"),
    diagnosticsPanel: $("diagnostics-panel"),
    diagnosticsItems: $("diagnostics-items"),
    spotify: $("spotify-container"),
    spotifyArt: $("spotify-art"),
    spotifyTitle: $("spotify-title"),
    spotifyArtist: $("spotify-artist"),
    greeting: $("greeting"),
    quote: $("quote"),
    captionUser: $("caption-user"),
    captionMirror: $("caption-mirror"),
    orb: $("orb"),
    assistantState: $("assistant-state"),
    newsStrip: $("news-strip"),
    newsTicker: $("news-ticker"),
    focus: $("focus"),
    focusClose: $("focus-close"),
    focusBody: $("focus-body"),
    breathing: $("breathing"),
    breathFill: $("breath-fill"),
    breathLabel: $("breath-label"),
    breathTitle: $("breath-title"),
    routine: $("routine"),
    routineTitle: $("routine-title"),
    routineRingFill: $("routine-ring-fill"),
    routineCount: $("routine-count"),
    routineText: $("routine-text"),
    routineSteps: $("routine-steps"),
    alarm: $("alarm"),
    command: $("command"),
    commandInput: $("command-input"),
    suggestions: $("suggestions"),
};

let ws = null;
let reconnectDelay = 1000;
let state = null;          // last snapshot from the backend
let skewMs = 0;            // server clock minus browser clock
let renderQueued = false;
let memoryIndex = 0;
let memoryRotatedAt = 0;
let lastCelebrateAt = 0;
let focusDismissedAt = 0;  // the focus "at" the user closed locally
const cache = {};          // last rendered JSON per section, to skip identical re-renders

/* ------------------------------------------------------------------ helpers */

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function setHidden(element, hidden) {
    if (element) element.classList.toggle("is-hidden", Boolean(hidden));
}

function changed(key, value) {
    const json = JSON.stringify(value ?? null);
    if (cache[key] === json) return false;
    cache[key] = json;
    return true;
}

function serverNow() {
    return Date.now() + skewMs;
}

function fmtNumber(n) {
    return Math.round(Number(n) || 0).toLocaleString("en-IN");
}

function fmtClock(totalSeconds) {
    const s = Math.max(0, Math.round(totalSeconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const mm = String(m).padStart(h ? 2 : 1, "0");
    return h ? `${h}:${mm}:${String(sec).padStart(2, "0")}` : `${mm}:${String(sec).padStart(2, "0")}`;
}

function fmtSigned(n, digits = 1) {
    const v = Number(n) || 0;
    const text = String(parseFloat(Math.abs(v).toFixed(digits)));  // 0.50 -> "0.5"
    return `${v > 0 ? "+" : v < 0 ? "\u2212" : ""}${text}`;
}

function fmtDate(iso) {
    if (!iso) return "";
    const d = new Date(`${String(iso).slice(0, 10)}T12:00:00`);
    if (Number.isNaN(d.getTime())) return String(iso);
    const sameYear = d.getFullYear() === new Date().getFullYear();
    return d.toLocaleDateString([], sameYear ? { day: "numeric", month: "short" } : { day: "numeric", month: "short", year: "numeric" });
}

function titleCase(value) {
    return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function isGuest(name) {
    return !name || name === "Unknown" || name === "Guest";
}

function firstName(name) {
    return isGuest(name) ? "" : String(name).split(" ")[0];
}

/* Run one render step; a bug in one panel must never blank the whole mirror. */
function safe(fn, ...args) {
    try {
        fn(...args);
    } catch (error) {
        console.error(`[mirror] ${fn.name} failed:`, error);
    }
}

/* --------------------------------------------------------------- websocket */

function send(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
        return true;
    }
    return false;
}

function sendAction(name, extra = {}) {
    return send({ type: "action", name, ...extra });
}

function sendSay(text) {
    const clean = String(text || "").trim();
    if (!clean) return false;
    return send({ type: "say", text: clean });
}

function setConnectionStatus(label) {
    el.connection.textContent = label;
    el.connection.dataset.status = label;
    if (label !== "online") {
        el.screen.classList.add("active");
        el.greeting.textContent = "Smart Mirror";
        el.quote.textContent = label === "reconnecting" ? "Reconnecting to the mirror backend…" : "Starting up.";
    }
}

function connectWebSocket() {
    ws = new WebSocket(`ws://${WS_HOST}:${WS_PORT}`);
    setConnectionStatus("connecting");

    ws.onopen = () => {
        reconnectDelay = 1000;
        setConnectionStatus("online");
    };

    ws.onmessage = (event) => {
        let payload;
        try {
            payload = JSON.parse(event.data);
        } catch (error) {
            console.error("[mirror] bad websocket payload", error);
            return;
        }
        if (payload.server_time_ms) skewMs = payload.server_time_ms - Date.now();
        queueRender(payload);
    };

    ws.onerror = () => setConnectionStatus("offline");

    ws.onclose = () => {
        setConnectionStatus("reconnecting");
        const delay = reconnectDelay;
        reconnectDelay = Math.min(reconnectDelay * 1.5, 8000);
        window.setTimeout(connectWebSocket, delay);
    };
}

function queueRender(payload) {
    state = payload;
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
        renderQueued = false;
        render(state);
    });
}

/* ------------------------------------------------------------------ render */

function render(s) {
    if (!s) return;
    el.screen.classList.add("active");
    safe(renderBody, s);
    safe(renderChips, s);
    safe(renderWeather, s.weather);
    safe(renderPresence, s);
    safe(renderCaptions, s);
    safe(renderNudge, s);
    safe(renderAssistant, s);

    const present = Boolean(s.present);
    const personal = s.personal || {};
    safe(renderCalendar, present ? s.events || [] : []);
    safe(renderTasks, present ? s.tasks || [] : []);
    safe(renderTimers, s.timers || []);
    safe(renderHabits, present ? personal.habits || [] : []);
    safe(renderWorkout, s.workout || {}, Boolean(s.demo));
    safe(renderGoal, present ? (personal.goals || [])[0] : null);
    safe(renderToday, present ? personal.today : null);
    safe(renderMemory, present ? personal.facts || [] : []);
    safe(renderInsight, present ? s.insight || "" : "");
    safe(renderHealth, s.health, present);
    safe(renderSettings, s.settings, s.show_settings);
    safe(renderDiagnostics, s.diagnostics, s.show_diagnostics);
    safe(renderSpotify, present ? s.track : null);
    safe(renderNews, present ? s.news || [] : []);
    safe(renderFocus, s);
    safe(renderRoutine, s.routine || {});
    safe(renderBreathing, s.breathing || {});
    safe(renderAlarm, s);
    safe(renderCelebrate, s.celebrate);
    safe(renderSuggestions, s);
    safe(fitStacks);
}

function renderBody(s) {
    const ctx = s.context || {};
    const scene = ctx.scene || {};
    const voice = s.voice_state || "idle";
    const classes = [
        `mode-${ctx.day_part || "night"}`,
        `tone-${ctx.tone || "focused"}`,
        scene.name ? `scene-${scene.name}` : "",
        s.present ? "presence-active" : "system-active",
        s.workout && s.workout.active ? "workout-active" : "",
        s.night_mode ? "night-mode" : "",
        s.privacy ? "privacy-on" : "",
        s.routine && s.routine.active ? "routine-active" : "",
        `voice-${voice}`,
    ];
    if (s.conversation_open && voice === "idle" && !s.demo) classes.push("voice-open");
    if (!el.command.classList.contains("is-hidden")) classes.push("command-open");
    el.body.className = classes.filter(Boolean).join(" ");
}

function renderChips(s) {
    const caps = s.capabilities || {};
    const providers = caps.llm_providers || [];
    el.chipAi.textContent = providers.length ? `AI · ${providers.join(" + ")}` : "AI offline";
    el.chipAi.classList.toggle("is-off", providers.length === 0);
    setHidden(el.chipAi, false);
    setHidden(el.chipPrivacy, !s.privacy);
    setHidden(el.chipDemo, !s.demo);
}

function renderWeather(weather) {
    if (!changed("weather", weather)) return;
    if (!weather) {
        el.weatherTemp.textContent = "--";
        el.weatherExtra.textContent = "-- / --";
        el.weatherDesc.textContent = "weather offline";
        el.weatherCity.textContent = "";
        return;
    }
    const hasTemp = weather.temperature !== null && weather.temperature !== undefined;
    el.weatherTemp.textContent = hasTemp ? `${weather.temperature}°C` : "--";
    el.weatherExtra.textContent =
        weather.high != null && weather.low != null ? `H ${weather.high}° / L ${weather.low}°` : "-- / --";
    el.weatherDesc.textContent = weather.stale ? `${weather.description || ""} (cached)` : weather.description || "weather offline";
    el.weatherCity.textContent = weather.city || "";
}

function greetingFor(s) {
    const part = (s.context && s.context.day_part) || "evening";
    const name = firstName(s.name);
    if (s.night_mode) return `Good night${name ? ", " + name : ""}`;
    const hello = { morning: "Good morning", afternoon: "Good afternoon", evening: "Good evening", night: "Rest easy" }[part] || "Hello";
    return name ? `${hello}, ${name}` : `${hello}`;
}

function renderPresence(s) {
    const ctx = s.context || {};
    if (s.present) {
        el.greeting.textContent = greetingFor(s);
        let quote = ctx.ambient_message || "";
        // In the morning the ambient line is just "Good morning, Alex." which repeats the greeting.
        if (!quote || /^good (morning|afternoon|evening)/i.test(quote)) quote = (ctx.scene && ctx.scene.message) || "You are on track.";
        el.quote.textContent = s.privacy ? "Privacy mode. The camera is off." : quote;
    } else {
        el.greeting.textContent = "Smart Mirror";
        el.quote.textContent = s.privacy ? "Privacy mode. The camera is off." : "Waiting quietly.";
    }

    let hint = "";
    if (s.enrolling) hint = "Hold still and look at me. I'm learning your face.";
    else if (!s.present) hint = s.privacy ? "Camera is off. Press / to type." : "Step in front of the mirror, or press / to talk";
    el.presenceHint.textContent = hint;
    setHidden(el.presenceHint, !hint);
    el.presenceHint.classList.toggle("is-enrolling", Boolean(s.enrolling));
}

function renderCaptions(s) {
    const cap = s.caption || {};
    const fresh = cap.at && serverNow() - cap.at < CAPTION_HOLD_MS;
    const nudge = s.nudge;
    const sameAsNudge = nudge && nudge.text && nudge.text === cap.mirror && serverNow() - nudge.at < NUDGE_HOLD_MS;
    el.captionUser.textContent = fresh && cap.user ? cap.user : "";
    el.captionMirror.textContent = fresh && cap.mirror && !sameAsNudge ? cap.mirror : "";
}

const NUDGE_LABELS = {
    briefing: "Briefing", reminder: "Reminder", timer: "Timer", hydration: "Water", break: "Break",
    posture: "Posture", mood: "Check-in", winddown: "Wind-down", recap: "Today", weekly: "Weekly review",
    streak: "Streak", calendar: "Up next", guest: "Hello", welcome: "Welcome back", habit: "Habit",
};

function renderNudge(s) {
    const n = s.nudge;
    const show = Boolean(n && n.text && serverNow() - n.at < NUDGE_HOLD_MS);
    setHidden(el.nudge, !show);
    if (!show) return;
    el.nudgeKind.textContent = NUDGE_LABELS[n.kind] || titleCase(n.kind || "Mirror");
    el.nudgeText.textContent = n.text;
    el.nudge.dataset.kind = n.kind || "";
}

function renderAssistant(s) {
    const voice = s.voice_state || "idle";
    let label = "";
    if (voice === "listening") label = "Listening";
    else if (voice === "thinking") label = "Thinking";
    else if (voice === "speaking") label = "Speaking";
    else if (s.conversation_open && !s.demo) label = "Listening for a follow-up";
    el.assistantState.textContent = label;
    el.orb.dataset.state = voice;
}

/* ------------------------------------------------------------- left stack */

function renderCalendar(events) {
    if (!changed("calendar", events)) return;
    setHidden(el.calendarPanel, events.length === 0);
    el.calendar.innerHTML = events.slice(0, 4).map((event) => `
        <div class="event-row">
            <span>${escapeHtml(event.time)}</span>
            <p>${escapeHtml(event.title)}</p>
        </div>`).join("");
}

function dueLabel(iso) {
    if (!iso) return "";
    const due = new Date(iso);
    if (Number.isNaN(due.getTime())) return "";
    const now = new Date();
    const time = due.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (due.toDateString() === now.toDateString()) return due < now ? `was due ${time}` : time;
    const tomorrow = new Date(now);
    tomorrow.setDate(now.getDate() + 1);
    if (due.toDateString() === tomorrow.toDateString()) return `tomorrow ${time}`;
    return due.toLocaleDateString([], { day: "numeric", month: "short" });
}

function renderTasks(tasks) {
    if (!changed("tasks", tasks)) return;
    setHidden(el.tasksPanel, tasks.length === 0);
    el.tasks.innerHTML = tasks.slice(0, 5).map((task) => `
        <div class="task-item">
            <span>${escapeHtml(task.text)}<small class="task-due">${escapeHtml(dueLabel(task.due_at))}</small></span>
            <button type="button" class="task-check" data-task="${Number(task.id)}" aria-label="Mark done"></button>
        </div>`).join("");
}

function renderTimers(timers) {
    setHidden(el.timersPanel, timers.length === 0);
    if (changed("timers", timers.map((t) => [t.id, t.label, t.fired, t.ends_at_ms]))) {
        el.timers.innerHTML = timers.map((t) => `
            <div class="timer-row${t.fired ? " is-fired" : ""}" data-id="${Number(t.id)}" data-ends="${Number(t.ends_at_ms)}" data-total="${Number(t.total)}">
                <span class="timer-label">${escapeHtml(t.kind === "focus" ? "Focus session" : t.label)}</span>
                <span class="timer-time">--:--</span>
                <button type="button" class="timer-cancel" data-timer="${escapeHtml(t.label)}" aria-label="Cancel timer">×</button>
                <div class="bar timer-bar"><i></i></div>
            </div>`).join("");
    }
    tickTimers();
}

function tickTimers() {
    for (const row of el.timers.querySelectorAll(".timer-row")) {
        const ends = Number(row.dataset.ends);
        const total = Math.max(1, Number(row.dataset.total));
        const left = Math.max(0, (ends - serverNow()) / 1000);
        row.querySelector(".timer-time").textContent = row.classList.contains("is-fired") || left <= 0 ? "done" : fmtClock(left);
        row.querySelector(".timer-bar i").style.width = `${Math.min(100, (1 - left / total) * 100)}%`;
    }
}

function renderHabits(habits) {
    if (!changed("habits", habits)) return;
    setHidden(el.habitsPanel, habits.length === 0);
    el.habits.innerHTML = habits.map((h) => {
        const progress = h.target > 1 ? `${fmtNumber(h.done)}/${fmtNumber(h.target)} ${escapeHtml(h.unit || "")}` : "";
        const streak = h.streak >= 2 ? `${h.streak}-day streak` : "";
        return `
        <button type="button" class="habit-row${h.complete ? " is-done" : ""}" data-habit="${Number(h.id)}" ${h.complete ? "disabled" : ""}>
            <span class="habit-check" aria-hidden="true"></span>
            <span class="habit-name">${escapeHtml(h.name)}</span>
            <span class="habit-meta">${escapeHtml([progress, streak].filter(Boolean).join(" · "))}</span>
        </button>`;
    }).join("");
}

/* ------------------------------------------------------------ right stack */

function renderWorkout(w, demo) {
    const active = Boolean(w.active);
    setHidden(el.workoutPanel, !active);
    el.workoutPanel.classList.toggle("active", active);
    if (!active) return;
    el.workoutExercise.textContent = w.title || titleCase(w.exercise || "curls");
    const plank = w.exercise === "plank";
    el.workoutReps.textContent = plank ? fmtClock(w.hold || 0) : String(w.reps ?? 0);
    el.workoutRepsLabel.textContent = plank ? "held" : "reps";
    el.workoutTimer.textContent = w.timer || "00:00";
    let feedback = w.feedback || "";
    if (!feedback && w.tracking === false && !demo) feedback = "Rep counting is off (needs MediaPipe pose). Timing the session.";
    else if (!feedback && w.stage && w.stage !== "ready") feedback = titleCase(w.stage);
    el.workoutFeedback.textContent = feedback;
}

const STATUS_TEXT = {
    on_track: "on track", ahead: "ahead", behind: "behind", stalled: "plateau",
    just_started: "new", done: "reached", no_data: "no data", manual: "goal",
};

function setRing(circle, fraction, circumference) {
    const f = Math.max(0, Math.min(1, Number(fraction) || 0));
    circle.style.strokeDasharray = `${circumference}`;
    circle.style.strokeDashoffset = `${circumference * (1 - f)}`;
}

function renderGoal(goal) {
    setHidden(el.goalPanel, !goal);
    if (!goal || !changed("goal", goal)) return;
    el.goalTitle.textContent = goal.title || "Goal";
    el.goalStatus.textContent = STATUS_TEXT[goal.status] || "";
    el.goalStatus.dataset.status = goal.status || "";
    const pct = goal.percent;
    el.goalPercent.textContent = pct == null ? "–" : String(Math.round(pct));
    setRing(el.goalRingFill, pct == null ? 0 : pct / 100, RING_GOAL);
    el.goalCurrent.textContent = goal.current == null ? "--" : String(goal.current);
    el.goalUnit.textContent = goal.unit || "";
    el.goalTarget.textContent = goal.target == null ? "--" : `${goal.target} ${goal.unit || ""}`.trim();
    el.goalRate.textContent = goal.rate == null ? "" : `${fmtSigned(goal.rate, 2)} ${goal.unit || ""}/week`;
    el.goalHeadline.textContent = goal.headline || "";
    drawSpark(el.goalSpark, goal.series || [], goal.target, 300, 70);
}

/* Draws the weight line (raw readings faint, smoothed trend bright, target dashed). */
function drawSpark(svg, series, target, width, height, includeTarget = false) {
    const points = series.filter((p) => p.value != null);
    if (points.length < 2) {
        svg.innerHTML = "";
        return false;
    }
    const values = points.flatMap((p) => [p.value, p.trend ?? p.value]);
    let lo = Math.min(...values);
    let hi = Math.max(...values);
    const span = Math.max(hi - lo, 0.5);
    const showTarget = target != null && (includeTarget || (target >= lo - span * 0.6 && target <= hi + span * 0.6));
    if (showTarget) {
        lo = Math.min(lo, target);
        hi = Math.max(hi, target);
    }
    if (hi - lo < 0.5) {
        const mid = (hi + lo) / 2;
        lo = mid - 0.25;
        hi = mid + 0.25;
    }
    const pad = 4;
    const x = (i) => pad + (i * (width - pad * 2)) / (points.length - 1);
    const y = (v) => pad + ((hi - v) * (height - pad * 2)) / (hi - lo);
    const line = (key) => points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[key] ?? p.value).toFixed(1)}`).join(" ");
    const trend = line("trend");
    const area = `${trend} L${x(points.length - 1).toFixed(1)},${height} L${x(0).toFixed(1)},${height} Z`;
    const last = points[points.length - 1];
    const gid = `${svg.id || "spark"}-fill`;
    svg.innerHTML = `
        <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgba(168,215,255,0.28)"/><stop offset="100%" stop-color="rgba(168,215,255,0)"/>
        </linearGradient></defs>
        <path class="spark-area" d="${area}" fill="url(#${gid})"/>
        ${showTarget ? `<line class="spark-target" x1="0" x2="${width}" y1="${y(target).toFixed(1)}" y2="${y(target).toFixed(1)}"/>` : ""}
        <path class="spark-raw" d="${line("value")}"/>
        <path class="spark-trend" d="${trend}"/>
        <circle class="spark-dot" cx="${x(points.length - 1).toFixed(1)}" cy="${y(last.trend ?? last.value).toFixed(1)}" r="2.6"/>`;
    return showTarget;
}

function renderToday(today) {
    setHidden(el.todayPanel, !today);
    if (!today || !changed("today", today)) return;
    const cal = today.calories || {};
    const water = today.water || {};
    el.calText.textContent = `${fmtNumber(cal.eaten)} / ${fmtNumber(cal.target)} kcal`;
    el.calBar.style.width = `${Math.min(100, ((cal.eaten || 0) / Math.max(1, cal.target || 1)) * 100)}%`;
    el.calBar.parentElement.classList.toggle("is-over", (cal.eaten || 0) > (cal.target || Infinity));
    el.proteinText.textContent = `${fmtNumber(cal.protein)} / ${fmtNumber(cal.protein_target)} g`;
    el.proteinBar.style.width = `${Math.min(100, ((cal.protein || 0) / Math.max(1, cal.protein_target || 1)) * 100)}%`;
    el.waterText.textContent = `${fmtNumber(water.ml)} / ${fmtNumber(water.target)} ml`;

    const glasses = Math.max(1, Math.min(16, Math.round((water.target || 2000) / 250)));
    const full = Math.floor((water.ml || 0) / 250);
    el.waterGlasses.innerHTML = Array.from({ length: glasses }, (_, i) =>
        `<button type="button" class="glass${i < full ? " is-full" : ""}" data-water="250" aria-label="Log a glass of water"></button>`).join("");

    const meals = (today.meals || []).length;
    const bits = [`${meals} meal${meals === 1 ? "" : "s"} logged`, `${today.workouts || 0} workout${today.workouts === 1 ? "" : "s"}`];
    if (!cal.estimated) bits.push("default targets");
    el.todayFoot.textContent = bits.join(" · ");
}

function renderMemory(facts) {
    setHidden(el.memoryPanel, facts.length === 0);
    if (!facts.length) return;
    if (changed("facts", facts)) memoryIndex = 0;
    const now = Date.now();
    if (now - memoryRotatedAt > 9000 || !el.memoryFact.textContent) {
        memoryRotatedAt = now;
        el.memoryFact.textContent = facts[memoryIndex % facts.length];
        memoryIndex += 1;
    }
}

function renderInsight(insight) {
    setHidden(el.insightPanel, !insight);
    el.insight.textContent = insight;
}

function renderHealth(health, present) {
    const show = Boolean(health && health.items && (!present || health.overall === "blocked"));
    setHidden(el.systemPanel, !show);
    if (!show || !changed("health", health.items)) return;
    el.healthSummary.textContent = health.summary || "";
    // Static checks and live runtime checks can share a label ("Voice output"); keep the live one.
    const byLabel = new Map();
    for (const item of health.items) byLabel.set(item.label, item);
    el.healthItems.innerHTML = [...byLabel.values()].slice(0, 10).map((item) => `
        <div class="system-row">
            <span class="status-dot status-${escapeHtml(item.status)}"></span>
            <span>${escapeHtml(item.label)}</span>
            <em>${escapeHtml(item.detail)}</em>
        </div>`).join("");
}

function renderSettings(settings, visible) {
    setHidden(el.settingsPanel, !visible || !settings);
    if (!visible || !settings) return;
    const labels = { camera: "Camera index", city: "Weather city", ai: "AI router", fps: "Target FPS", preview: "Camera preview", settings_file: "Settings file" };
    el.settingsItems.innerHTML = Object.entries(settings).map(([key, value]) => `
        <div class="system-row"><span></span><span>${escapeHtml(labels[key] || titleCase(key))}</span><em>${escapeHtml(value)}</em></div>`).join("");
}

function renderDiagnostics(d, visible) {
    setHidden(el.diagnosticsPanel, !visible || !d);
    if (!visible || !d) return;
    const rows = {
        "Camera FPS": d.fps,
        "Frame loop": `${d.loop_ms} ms`,
        "CPU pressure": d.cpu_pressure,
        "Background tasks": d.background_tasks,
        "Screens connected": d.websocket_clients,
        "Last AI call": d.ai_provider ? `${d.ai_provider} · ${d.ai_latency_ms} ms` : "none yet",
    };
    el.diagnosticsItems.innerHTML = Object.entries(rows).map(([key, value]) => `
        <div class="system-row"><span></span><span>${escapeHtml(key)}</span><em>${escapeHtml(value)}</em></div>`).join("");
}

/* ------------------------------------------------------------ bottom row */

function renderSpotify(track) {
    setHidden(el.spotify, !track);
    if (!changed("track", track)) return;
    if (!track) {
        el.ambientArt.style.backgroundImage = "none";
        return;
    }
    el.spotifyTitle.textContent = track.title || "";
    el.spotifyArtist.textContent = track.stale ? `${track.artist || ""} · cached` : track.artist || "";
    if (track.image) {
        el.spotifyArt.src = track.image;
        el.spotifyArt.classList.remove("is-empty");
        el.ambientArt.style.backgroundImage = `url("${track.image}")`;
    } else {
        el.spotifyArt.removeAttribute("src");
        el.spotifyArt.classList.add("is-empty");
        el.ambientArt.style.backgroundImage = "none";
    }
}

function renderNews(headlines) {
    setHidden(el.newsStrip, headlines.length === 0);
    if (!changed("news", headlines) || headlines.length === 0) return;
    const items = headlines.map((h) => `<span class="ticker-item">${escapeHtml(h)}</span><span class="ticker-dot"></span>`).join("");
    el.newsTicker.innerHTML = items + items;  // doubled so the scroll loops seamlessly
    el.newsTicker.style.animationDuration = `${Math.min(140, Math.max(40, headlines.length * 14))}s`;
}

/* ---------------------------------------------------------------- overlays */

function renderFocus(s) {
    const focus = s.focus || {};
    const panel = focus.panel || "none";
    const expired = focus.until_ms && serverNow() > focus.until_ms;
    const dismissed = focusDismissedAt && focusDismissedAt === (focus.until_ms || panel);
    const show = panel !== "none" && panel !== "routine" && !expired && !dismissed && Boolean(s.present);
    setHidden(el.focus, !show);
    if (!show) return;
    if (!changed("focus", [focus, s.personal, s.caption && s.caption.mirror])) return;
    el.focusBody.innerHTML = focusHtml(panel, focus, s);
    const chart = el.focusBody.querySelector("svg.focus-chart");
    if (chart) {
        const goal = ((s.personal && s.personal.goals) || [])[0];
        const targetShown = drawSpark(chart, goal ? goal.series || [] : [], goal ? goal.target : null, 700, 190, true);
        const legend = el.focusBody.querySelector(".lg-target");
        if (legend) legend.hidden = !targetShown;
    }
}

function stat(label, value, sub = "") {
    return `<div class="focus-stat"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b>${sub ? `<small>${escapeHtml(sub)}</small>` : ""}</div>`;
}

function focusHtml(panel, focus, s) {
    const p = s.personal || {};
    const spoken = (s.caption && s.caption.mirror) || "";
    if (panel === "progress") {
        const goals = p.goals || [];
        if (!goals.length) return `<div class="focus-title">Progress</div><div class="focus-headline">No goals yet.</div><p class="focus-text">Say “I want to lose five kilos by December”.</p>`;
        const g = goals[0];
        const others = goals.slice(1).map((o) => `<li>${escapeHtml(o.title)}<small>${escapeHtml(o.headline || STATUS_TEXT[o.status] || "")}</small></li>`).join("");
        return `
            <div class="focus-title">${escapeHtml(g.title)} · ${escapeHtml(STATUS_TEXT[g.status] || "")}</div>
            <div class="focus-headline">${escapeHtml(g.headline || "")}</div>
            <div class="focus-grid">
                ${stat("Progress", g.percent == null ? "–" : `${Math.round(g.percent)}%`)}
                ${stat("Now", g.current == null ? "--" : `${g.current} ${g.unit || ""}`)}
                ${stat("Target", g.target == null ? "--" : `${g.target} ${g.unit || ""}`, g.target_date ? `by ${fmtDate(g.target_date)}` : "")}
                ${stat("Pace", g.rate == null ? "--" : `${fmtSigned(g.rate, 2)} ${g.unit || ""}/wk`, g.eta ? `on pace for ${fmtDate(g.eta)}` : "")}
            </div>
            <svg class="focus-chart" id="focus-chart" viewBox="0 0 700 190" preserveAspectRatio="none" aria-hidden="true"></svg>
            <div class="focus-legend"><span class="lg-trend">trend</span><span class="lg-raw">daily weigh-ins</span><span class="lg-target">target</span></div>
            ${others ? `<ul class="focus-list">${others}</ul>` : ""}`;
    }
    if (panel === "nutrition") {
        const t = p.today || {};
        const cal = t.calories || {};
        const water = t.water || {};
        const meals = (t.meals || []).map((m) => `<li>${escapeHtml(m.text)}<small>${fmtNumber(m.calories)} kcal · ${escapeHtml(m.type || "")}</small></li>`).join("");
        return `
            <div class="focus-title">Today's nutrition</div>
            <div class="focus-headline">${escapeHtml(spoken || `${fmtNumber(cal.eaten)} of ${fmtNumber(cal.target)} calories`)}</div>
            <div class="focus-grid">
                ${stat("Calories", fmtNumber(cal.eaten), `of ${fmtNumber(cal.target)} kcal${cal.estimated ? "" : " (default)"}`)}
                ${stat("Protein", `${fmtNumber(cal.protein)} g`, `of ${fmtNumber(cal.protein_target)} g`)}
                ${stat("Water", `${fmtNumber(water.ml)} ml`, `of ${fmtNumber(water.target)} ml`)}
                ${stat("Workouts", `${t.workouts || 0}`, t.workout_minutes ? `${t.workout_minutes} min today` : "today")}
            </div>
            ${meals ? `<ul class="focus-list">${meals}</ul>` : `<p class="focus-text">No meals logged yet. Try “I had two eggs and toast”.</p>`}`;
    }
    if (panel === "memory") {
        const facts = (p.facts || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("");
        return `
            <div class="focus-title">Memory</div>
            <div class="focus-headline">What I remember about ${escapeHtml(firstName(s.name) || "you")}</div>
            ${facts ? `<ul class="focus-list">${facts}</ul>` : `<p class="focus-text">Nothing yet. Tell me about yourself.</p>`}
            <p class="focus-note">Say “forget that …” to remove anything.</p>`;
    }
    if (panel === "week") {
        const w = focus.week || {};
        const units = p.units === "imperial" ? "lb" : "kg";
        const change = w.weight_change_kg == null ? null : units === "lb" ? w.weight_change_kg / 0.45359237 : w.weight_change_kg;
        const changeText = change == null ? "--" : Math.abs(change) < 0.05 ? "flat" : `${fmtSigned(change)} ${units}`;
        return `
            <div class="focus-title">Your week</div>
            <div class="focus-headline">${escapeHtml(spoken || "The last seven days")}</div>
            <div class="focus-grid">
                ${stat("Weight", changeText)}
                ${stat("Avg calories", w.avg_calories == null ? "--" : fmtNumber(w.avg_calories), w.target_calories ? `target ${fmtNumber(w.target_calories)}` : "")}
                ${stat("Food logged", `${w.days_logged_food || 0}`, "of 7 days")}
                ${stat("Workouts", `${w.workouts || 0}`)}
                ${stat("Water", w.water_pct == null ? "--" : `${Math.round(w.water_pct)}%`, "of target")}
            </div>`;
    }
    if (panel === "habits") {
        const habits = (p.habits || []).map((h) => `<li>${escapeHtml(h.name)}<small>${h.complete ? "done today" : "not yet"}${h.streak >= 2 ? ` · ${h.streak}-day streak` : ""}</small></li>`).join("");
        return `
            <div class="focus-title">Habits</div>
            <div class="focus-headline">${escapeHtml(spoken || "Your daily habits")}</div>
            ${habits ? `<ul class="focus-list">${habits}</ul>` : `<p class="focus-text">Say “track vitamins daily”.</p>`}`;
    }
    // notes and anything else: show what the mirror just said
    return `
        <div class="focus-title">${escapeHtml(titleCase(panel))}</div>
        <div class="focus-headline">${escapeHtml(spoken || "")}</div>`;
}

function renderRoutine(r) {
    setHidden(el.routine, !r.active);
    if (!r.active) return;
    if (changed("routine", [r.key, r.step, r.text])) {
        el.routineTitle.textContent = r.title || "Routine";
        el.routineText.textContent = r.text || "";
        el.routineSteps.innerHTML = Array.from({ length: r.steps || 0 }, (_, i) =>
            `<i class="${i + 1 < r.step ? "is-done" : i + 1 === r.step ? "is-current" : ""}"></i>`).join("");
    }
    tickRoutine();
}

function tickRoutine() {
    const r = state && state.routine;
    if (!r || !r.active) return;
    if (r.duration && r.ends_at_ms) {
        const left = Math.max(0, (r.ends_at_ms - serverNow()) / 1000);
        el.routineCount.textContent = String(Math.ceil(left));
        setRing(el.routineRingFill, 1 - left / r.duration, RING_ROUTINE);
    } else {
        el.routineCount.textContent = `${r.step}/${r.steps}`;
        setRing(el.routineRingFill, (r.step - 1) / Math.max(1, r.steps), RING_ROUTINE);
    }
}

function renderBreathing(b) {
    setHidden(el.breathing, !b.active);
    if (!b.active) return;
    el.breathTitle.textContent = `${b.title || "Breathe"} · tap to end`;
    tickBreathing();
}

/* Where are we in the inhale / hold / exhale cycle? Computed from the start time, so it
 * stays in sync with the backend even if the page reloads mid-session. */
function tickBreathing() {
    const b = state && state.breathing;
    if (!b || !b.active || !b.phases || !b.phases.length) return;
    const cycle = b.phases.reduce((sum, ph) => sum + ph.seconds, 0);
    let t = ((serverNow() - b.started_at_ms) / 1000) % cycle;
    let index = 0;
    while (index < b.phases.length - 1 && t >= b.phases[index].seconds) {
        t -= b.phases[index].seconds;
        index += 1;
    }
    const phase = b.phases[index];
    const p = Math.min(1, t / phase.seconds);
    const eased = 0.5 - Math.cos(Math.PI * p) / 2;
    const small = 0.42;
    let scale = 1;
    if (phase.label === "Inhale") scale = small + (1 - small) * eased;
    else if (phase.label === "Exhale") scale = 1 - (1 - small) * eased;
    else scale = index > 0 && b.phases[index - 1].label === "Exhale" ? small : 1;
    el.breathFill.style.transform = `scale(${scale.toFixed(3)})`;
    el.breathLabel.textContent = `${phase.label} ${Math.max(1, Math.ceil(phase.seconds - t))}`;
}

function renderAlarm(s) {
    setHidden(el.alarm, !s.alarm);
}

function renderCelebrate(c) {
    if (!c || !c.at || c.at === lastCelebrateAt) return;
    lastCelebrateAt = c.at;
    if (serverNow() - c.at > CELEBRATE_HOLD_MS) return;
    burstConfetti(c.label || "Nice work");
}

function burstConfetti(label) {
    const colors = ["#a8d7ff", "#ffe0b8", "#c4ecd5", "#ffffff", "#f6b8c8"];
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 90; i += 1) {
        const bit = document.createElement("i");
        bit.className = "confetti";
        bit.style.left = `${Math.random() * 100}vw`;
        bit.style.background = colors[i % colors.length];
        bit.style.setProperty("--dx", `${(Math.random() - 0.5) * 30}vw`);
        bit.style.setProperty("--rot", `${Math.random() * 900 - 450}deg`);
        bit.style.setProperty("--dur", `${2.4 + Math.random() * 2.2}s`);
        bit.style.animationDelay = `${Math.random() * 0.5}s`;
        frag.appendChild(bit);
    }
    const text = document.createElement("div");
    text.className = "celebrate-label";
    text.textContent = label;
    frag.appendChild(text);
    el.celebrate.innerHTML = "";
    el.celebrate.appendChild(frag);
    window.setTimeout(() => { el.celebrate.innerHTML = ""; }, CELEBRATE_HOLD_MS);
}

/* Hide the least important panels when a stack is taller than the screen allows. */
const PRIORITY = {
    left: ["timers-panel", "tasks-container", "calendar-panel", "habits-panel"],
    right: ["workout-container", "settings-panel", "diagnostics-panel", "system-panel", "goal-panel", "today-panel", "memory-panel", "insight-panel"],
};

function fitStacks() {
    fitStack(el.leftStack, PRIORITY.left);
    fitStack(el.rightStack, PRIORITY.right);
}

function fitStack(stack, order) {
    const panels = order.map($).filter(Boolean);
    panels.forEach((panel) => panel.classList.remove("is-squeezed"));
    const gap = parseFloat(getComputedStyle(stack).rowGap) || 0;
    const used = () => {
        const shown = [...stack.children].filter((c) => !c.classList.contains("is-hidden") && !c.classList.contains("is-squeezed"));
        return shown.reduce((sum, c) => sum + c.offsetHeight, 0) + gap * Math.max(0, shown.length - 1);
    };
    for (let i = panels.length - 1; i >= 0 && used() > stack.clientHeight; i -= 1) {
        if (!panels[i].classList.contains("is-hidden")) panels[i].classList.add("is-squeezed");
    }
}

window.addEventListener("resize", () => safe(fitStacks));

/* ----------------------------------------------------------- typed command */

const SUGGESTIONS = {
    guest: ["My name is …", "What can you do?", "What's the weather?", "Tell me a joke"],
    routine: ["next", "stop the routine"],
    breathing: ["I feel better", "stop breathing"],
    workout: ["stop workout", "how many workouts this week"],
    present: [
        "How am I doing?", "I drank two glasses of water", "I had dal and two rotis for lunch",
        "Set a timer for 2 minutes", "Remind me to call mom at 6 pm", "Brush my teeth",
        "Start a breathing exercise", "Start squats", "Show my week", "What do you know about me?",
        "Show diagnostics", "Brief me",
    ],
};

function renderSuggestions(s) {
    let key = "present";
    if (s.routine && s.routine.active) key = "routine";
    else if (s.breathing && s.breathing.active) key = "breathing";
    else if (s.workout && s.workout.active) key = "workout";
    else if (isGuest(s.name)) key = "guest";
    if (!changed("suggestions", key)) return;
    el.suggestions.innerHTML = SUGGESTIONS[key].map((text) =>
        `<button type="button" data-say="${escapeHtml(text)}">${escapeHtml(text)}</button>`).join("");
}

function openCommand() {
    setHidden(el.command, false);
    el.body.classList.add("command-open");
    el.commandInput.focus();
}

function closeCommand() {
    setHidden(el.command, true);
    el.body.classList.remove("command-open");
    el.commandInput.blur();
}

el.command.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = el.commandInput.value;
    if (sendSay(text)) el.commandInput.value = "";
});

el.suggestions.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-say]");
    if (!button) return;
    const text = button.dataset.say;
    if (text.includes("…")) {
        el.commandInput.value = text.replace("…", "").trimEnd() + " ";
        el.commandInput.focus();
        return;
    }
    sendSay(text);
});

el.orb.addEventListener("click", () => (el.command.classList.contains("is-hidden") ? openCommand() : closeCommand()));
el.presenceHint.addEventListener("click", openCommand);

document.addEventListener("keydown", (event) => {
    const typing = document.activeElement === el.commandInput;
    if (event.key === "/" && !typing) {
        event.preventDefault();
        openCommand();
    } else if (event.key === "Escape") {
        if (!el.command.classList.contains("is-hidden")) closeCommand();
        else if (!el.focus.classList.contains("is-hidden")) dismissFocus();
    }
});

/* --------------------------------------------------------- click handlers */

function dismissFocus() {
    const focus = (state && state.focus) || {};
    focusDismissedAt = focus.until_ms || focus.panel || 1;
    setHidden(el.focus, true);
    sendAction("dismiss_focus");
}

el.focusClose.addEventListener("click", dismissFocus);
el.focus.addEventListener("click", (event) => {
    if (event.target === el.focus) dismissFocus();
});

el.breathing.addEventListener("click", () => {
    setHidden(el.breathing, true);
    sendAction("breathing_stop");
});

el.nudge.addEventListener("click", () => {
    setHidden(el.nudge, true);
    sendAction("dismiss_nudge");
});

document.addEventListener("click", (event) => {
    const target = event.target;
    const action = target.closest("[data-action]");
    if (action) {
        sendAction(action.dataset.action);
        return;
    }
    const task = target.closest("[data-task]");
    if (task) {
        task.closest(".task-item").classList.add("is-done");
        sendAction("complete_task", { id: Number(task.dataset.task) });
        return;
    }
    const habit = target.closest("[data-habit]");
    if (habit && !habit.disabled) {
        habit.classList.add("is-done");
        sendAction("habit_done", { id: Number(habit.dataset.habit) });
        return;
    }
    const glass = target.closest("[data-water]");
    if (glass) {
        sendAction("log_water", { ml: Number(glass.dataset.water) });
        return;
    }
    const timer = target.closest("[data-timer]");
    if (timer) sendAction("cancel_timer", { label: timer.dataset.timer });
});

/* ------------------------------------------------------------------ clock */

function updateTime() {
    const now = new Date();
    el.time.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    el.date.textContent = now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
}

/* Fast local ticker: countdowns, breathing animation, and expiring captions/nudges.
 * The backend only pushes when state changes, so time-based visuals are animated here. */
function tick() {
    if (!state) return;
    safe(tickTimers);
    safe(tickRoutine);
    safe(tickBreathing);
    safe(renderCaptions, state);
    safe(renderNudge, state);
    safe(renderMemory, state.present ? (state.personal && state.personal.facts) || [] : []);
    if (state.focus && state.focus.until_ms && serverNow() > state.focus.until_ms) setHidden(el.focus, true);
}

updateTime();
window.setInterval(updateTime, 1000);
window.setInterval(tick, 200);
connectWebSocket();
