const dom = {
    body: document.body,
    screen: document.getElementById("screen"),
    ambientArt: document.getElementById("ambient-art"),
    time: document.getElementById("time"),
    date: document.getElementById("date"),
    connection: document.getElementById("connection-status"),
    weatherTemp: document.getElementById("weather-temp"),
    weatherExtra: document.getElementById("weather-extra"),
    weatherDesc: document.getElementById("weather-desc"),
    weatherCity: document.getElementById("weather-city"),
    calendarPanel: document.getElementById("calendar-panel"),
    calendar: document.getElementById("calendar"),
    newsPanel: document.getElementById("news-panel"),
    newsTicker: document.getElementById("news-ticker"),
    insightPanel: document.getElementById("insight-panel"),
    insight: document.getElementById("insight"),
    summaryPanel: document.getElementById("summary-panel"),
    weeklySummary: document.getElementById("weekly-summary"),
    systemPanel: document.getElementById("system-panel"),
    healthSummary: document.getElementById("health-summary"),
    healthItems: document.getElementById("health-items"),
    settingsPanel: document.getElementById("settings-panel"),
    settingsItems: document.getElementById("settings-items"),
    diagnosticsPanel: document.getElementById("diagnostics-panel"),
    diagnosticsItems: document.getElementById("diagnostics-items"),
    tasksContainer: document.getElementById("tasks-container"),
    tasks: document.getElementById("tasks"),
    workoutContainer: document.getElementById("workout-container"),
    workoutExercise: document.getElementById("workout-exercise"),
    workoutReps: document.getElementById("workout-reps"),
    workoutTimer: document.getElementById("workout-timer"),
    spotifyContainer: document.getElementById("spotify-container"),
    spotifyArt: document.getElementById("spotify-art"),
    spotifyTitle: document.getElementById("spotify-title"),
    spotifyArtist: document.getElementById("spotify-artist"),
    assistant: document.getElementById("assistant-overlay"),
    assistantState: document.getElementById("assistant-state"),
    greeting: document.getElementById("greeting"),
    quote: document.getElementById("quote")
};

let ws = null;
let reconnectDelay = 1000;
let latestPayload = null;
let renderQueued = false;
let present = false;
let cache = {};

const quietLines = [
    "You are on track.",
    "Keep it simple today.",
    "Steady pace."
];

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function titleCase(value) {
    return String(value || "")
        .replaceAll("_", " ")
        .replace(/\b\w/g, char => char.toUpperCase());
}

function updateTime() {
    const now = new Date();
    dom.time.textContent = now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });
    dom.date.textContent = now.toLocaleDateString([], {
        weekday: "long",
        month: "long",
        day: "numeric"
    });
}

function greetingFor(name) {
    const hour = new Date().getHours();
    const displayName = name && name !== "Unknown" ? name : "there";
    if (hour < 12) return `Good morning, ${displayName}`;
    if (hour < 17) return `Good afternoon, ${displayName}`;
    if (hour < 21) return `Good evening, ${displayName}`;
    return `Rest easy, ${displayName}`;
}

function setHidden(element, hidden) {
    element.classList.toggle("is-hidden", hidden);
}

function setConnectionStatus(label) {
    dom.connection.textContent = label;
    dom.connection.dataset.status = label;
    if (label !== "online") {
        dom.screen.classList.add("active");
        dom.greeting.textContent = "Smart Mirror";
        dom.quote.textContent = label === "reconnecting" ? "Reconnecting quietly." : "Starting up.";
    }
}

function queueRender(payload) {
    latestPayload = payload;
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
        renderQueued = false;
        render(latestPayload);
    });
}

function render(data) {
    if (!data) return;

    const context = data.context || {};
    const scene = context.scene || {};
    const workoutActive = Boolean(data.workout && data.workout.active);

    dom.body.className = [
        `mode-${context.day_part || "night"}`,
        `tone-${context.tone || "focused"}`,
        scene.name ? `scene-${scene.name}` : "",
        workoutActive ? "workout-active" : "",
        data.present ? "presence-active" : "system-active"
    ].filter(Boolean).join(" ");

    renderWeather(data.weather);
    renderHealth(data.health, data.present);
    renderSettings(data.settings, data.show_settings);
    renderDiagnostics(data.diagnostics, data.show_diagnostics);
    renderAssistantState(data);

    if (data.present) {
        showMirror(data);
    } else {
        showIdle(data);
    }
}

function showMirror(data) {
    const context = data.context || {};
    present = true;
    dom.screen.classList.add("active");

    dom.greeting.textContent = greetingFor(data.name);
    dom.quote.textContent =
        context.ambient_message ||
        (context.scene && context.scene.message) ||
        quietLines[Math.floor(Math.random() * quietLines.length)];

    renderCalendar(data.events || []);
    renderNews(data.news || []);
    renderInsight(data.insight || "");
    renderWeeklySummary(data.weekly_summary || {});
    renderTasks(data.tasks || []);
    renderWorkout(data.workout || {});
    renderSpotify(data.track || null);
}

function showIdle(data) {
    present = false;
    dom.screen.classList.add("active");
    dom.greeting.textContent = "Smart Mirror";
    dom.quote.textContent = data.health?.summary || "Waiting quietly.";

    renderCalendar([]);
    renderNews([]);
    renderInsight("");
    renderWeeklySummary({});
    renderTasks([]);
    renderWorkout({ active: false });
    renderSpotify(null);
}

function renderAssistantState(data) {
    const voiceState = data.voice_state || data.status || "idle";
    const active = ["listening", "thinking", "speaking"].includes(voiceState);
    dom.assistant.classList.toggle("active", active);
    dom.assistantState.textContent = active ? titleCase(voiceState) : "";
}

function renderWeather(weather) {
    const key = JSON.stringify(weather || {});
    if (cache.weather === key) return;
    cache.weather = key;

    if (!weather) {
        dom.weatherTemp.textContent = "--";
        dom.weatherExtra.textContent = "-- / --";
        dom.weatherDesc.textContent = "weather offline";
        return;
    }

    dom.weatherTemp.textContent =
        weather.temperature === null || weather.temperature === undefined
            ? "--"
            : `${weather.temperature}\u00b0C`;
    dom.weatherExtra.textContent =
        weather.high !== undefined && weather.low !== undefined && weather.high !== null && weather.low !== null
            ? `H ${weather.high}\u00b0 / L ${weather.low}\u00b0`
            : "-- / --";
    dom.weatherDesc.textContent = weather.stale ? "cached weather" : weather.description || "weather offline";
    dom.weatherCity.textContent = weather.city || "";
}

function renderCalendar(events) {
    const key = JSON.stringify(events);
    if (cache.calendar === key) return;
    cache.calendar = key;

    setHidden(dom.calendarPanel, events.length === 0);
    dom.calendar.innerHTML = events.map(event => `
        <div class="event-row">
            <span>${escapeHtml(event.time)}</span>
            <p>${escapeHtml(event.title)}</p>
        </div>
    `).join("");
}

function renderNews(headlines) {
    const key = JSON.stringify(headlines);
    if (cache.news === key) return;
    cache.news = key;

    setHidden(dom.newsPanel, headlines.length === 0);
    if (headlines.length === 0) {
        dom.newsTicker.innerHTML = "";
        return;
    }

    const buildItems = () => headlines.map(headline =>
        `<span class="ticker-item">${escapeHtml(headline)}</span><span class="ticker-dot"></span>`
    ).join("");

    dom.newsTicker.innerHTML = buildItems() + buildItems();
    dom.newsTicker.style.animationDuration = `${Math.min(90, Math.max(25, headlines.length * 6))}s`;
}

function renderInsight(insight) {
    setHidden(dom.insightPanel, !insight);
    dom.insight.textContent = insight;
}

function renderWeeklySummary(summary) {
    const headline = summary.headline || "";
    setHidden(dom.summaryPanel, !headline || headline.includes("Not enough"));
    dom.weeklySummary.innerHTML = headline ? `
        <div class="summary-headline">${escapeHtml(headline)}</div>
        <div class="summary-metrics">
            <span>${summary.sessions || 0} sessions</span>
            <span>${Math.round((summary.posture_score ?? 1) * 100)}% posture</span>
        </div>
    ` : "";
}

function renderTasks(tasks) {
    const key = JSON.stringify(tasks);
    if (cache.tasks === key) return;
    cache.tasks = key;

    setHidden(dom.tasksContainer, tasks.length === 0);
    dom.tasks.innerHTML = tasks.map(task => `
        <div class="task-item">
            <span>${escapeHtml(task.text)}</span>
            <i></i>
        </div>
    `).join("");
}

function renderWorkout(workout) {
    const active = Boolean(workout.active);
    setHidden(dom.workoutContainer, !active);
    dom.workoutContainer.classList.toggle("active", active);
    if (!active) return;

    dom.workoutExercise.textContent = titleCase(workout.exercise || "curls");
    dom.workoutReps.textContent = workout.reps ?? 0;
    dom.workoutTimer.textContent = workout.timer || "00:00";
}

function renderSpotify(track) {
    setHidden(dom.spotifyContainer, !track);
    if (!track) {
        dom.ambientArt.style.backgroundImage = "none";
        return;
    }

    dom.spotifyTitle.textContent = track.title || "";
    dom.spotifyArtist.textContent = track.stale ? `${track.artist || ""} · cached` : track.artist || "";
    if (track.image) {
        dom.spotifyArt.src = track.image;
        dom.spotifyArt.classList.remove("is-empty");
        dom.spotifyArt.style.background = "";
        dom.ambientArt.style.backgroundImage = `url("${track.image}")`;
    } else {
        dom.spotifyArt.removeAttribute("src");
        dom.spotifyArt.classList.remove("is-empty");
        dom.spotifyArt.style.background = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)";
        dom.ambientArt.style.backgroundImage = "none";
    }
}

function renderHealth(health, isPresent) {
    if (!health || !health.items) {
        setHidden(dom.systemPanel, true);
        return;
    }

    const visible = !isPresent || health.overall !== "ready";
    setHidden(dom.systemPanel, !visible);
    dom.healthSummary.textContent = health.summary || "";
    dom.healthItems.innerHTML = health.items.slice(0, 8).map(item => `
        <div class="system-row">
            <span class="status-dot status-${escapeHtml(item.status)}"></span>
            <span>${escapeHtml(item.label)}</span>
            <em>${escapeHtml(item.detail)}</em>
        </div>
    `).join("");
}

function renderSettings(settings, visible) {
    setHidden(dom.settingsPanel, !visible || !settings);
    if (!settings) return;
    dom.settingsItems.innerHTML = Object.entries(settings).map(([key, value]) => `
        <div class="system-row">
            <span>${escapeHtml(titleCase(key))}</span>
            <em>${escapeHtml(value)}</em>
        </div>
    `).join("");
}

function renderDiagnostics(diagnostics, visible) {
    setHidden(dom.diagnosticsPanel, !visible || !diagnostics);
    if (!diagnostics) return;
    const rows = {
        fps: diagnostics.fps,
        loop: `${diagnostics.loop_ms} ms`,
        pressure: diagnostics.cpu_pressure,
        clients: diagnostics.websocket_clients,
        ai: diagnostics.ai_provider ? `${diagnostics.ai_provider} ${diagnostics.ai_latency_ms} ms` : "idle"
    };
    dom.diagnosticsItems.innerHTML = Object.entries(rows).map(([key, value]) => `
        <div class="system-row">
            <span>${escapeHtml(titleCase(key))}</span>
            <em>${escapeHtml(value)}</em>
        </div>
    `).join("");
}

function connectWebSocket() {
    ws = new WebSocket("ws://localhost:8765");
    setConnectionStatus("connecting");

    ws.onopen = () => {
        reconnectDelay = 1000;
        setConnectionStatus("online");
    };

    ws.onmessage = (event) => {
        try {
            queueRender(JSON.parse(event.data));
        } catch (error) {
            console.error("Bad websocket payload:", error);
        }
    };

    ws.onerror = () => {
        setConnectionStatus("offline");
    };

    ws.onclose = () => {
        present = false;
        setConnectionStatus("reconnecting");
        const delay = reconnectDelay;
        reconnectDelay = Math.min(reconnectDelay * 1.5, 8000);
        window.setTimeout(connectWebSocket, delay);
    };
}

updateTime();
window.setInterval(updateTime, 1000);
connectWebSocket();
