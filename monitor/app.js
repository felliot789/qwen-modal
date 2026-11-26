"use strict";

const $ = (id) => document.getElementById(id);
const charts = ["cpu", "gpu", "ram"].map((name) => $(name + "-chart"));
const labels = { idle: "Idle", loading: "Loading", ready: "Ready", stopping: "Stopping", failed: "Failed" };
const busyStates = new Set(["loading", "ready", "stopping"]);
let history = [];
let models = [];
let telemetry;
let runner;
let connected = false;
let pending = false;
let revision = 0;

function put(id, text) {
  const node = $(id);
  if (node.textContent !== text) node.textContent = text;
}

function reading(value, unit, precision = 0) {
  return Number.isFinite(value) ? value.toFixed(precision) + unit : "–";
}

function gib(bytes, precision = 1) {
  return reading(bytes == null ? NaN : bytes / 1024 ** 3, " GiB", precision);
}

function memory(used, total) {
  return used == null || !total ? "–" : (used / 1024 ** 3).toFixed(1) + " / " + gib(total);
}

function status(id, state, text) {
  $(id).dataset.state = state;
  put(id, text);
}

function draw(canvas, index) {
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const css = getComputedStyle(canvas);
  const y = (value) => 4 + (height - 8) * (1 - value / 100);
  ctx.strokeStyle = css.getPropertyValue("--line").trim();
  ctx.beginPath();
  for (const tick of [0, 50, 100]) {
    ctx.moveTo(0, y(tick));
    ctx.lineTo(width, y(tick));
  }
  ctx.stroke();

  ctx.strokeStyle = ctx.fillStyle = css.getPropertyValue("--trace").trim();
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let last;
  for (const sample of history) {
    const value = sample[index + 1];
    if (!Number.isFinite(value)) {
      last = undefined;
      continue;
    }
    const x = 3 + (width - 6) * (1 - sample[0] / 120);
    if (last) ctx.lineTo(x, y(value));
    else ctx.moveTo(x, y(value));
    last = [x, y(value)];
  }
  ctx.stroke();
  if (last) {
    ctx.beginPath();
    ctx.arc(last[0], last[1], 2.7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function row(label, value) {
  const item = document.createElement("div");
  const name = document.createElement("dt");
  const data = document.createElement("dd");
  name.textContent = label;
  data.textContent = value;
  item.append(name, data);
  return item;
}

function listItem(text) {
  const li = document.createElement("li");
  li.textContent = text;
  return li;
}

function hardware(data) {
  const { cpu, gpu, ram, board_fans } = data;
  put("host", "Host · " + data.host);
  put("cpu-name", cpu.name);
  put("cpu-usage", reading(cpu.usage, "%"));
  put("cpu-temp", reading(cpu.temperature, " °C"));
  put("cpu-clock", reading(cpu.clock == null ? NaN : cpu.clock / 1000, " GHz", 2));
  put("cpu-load", reading(cpu.load[0], "", 2));
  if (!$("threads").max) $("threads").value = Math.min(4, cpu.logical_count);
  $("threads").max = cpu.logical_count;
  $("cpu-cores").replaceChildren(...cpu.core_temps.map(([name, value]) => row(name, reading(value, " °C"))));
  $("cpu-threads").replaceChildren(...cpu.threads.map(([i, load]) => listItem(i + " · " + reading(load, "%"))));

  put("gpu-name", gpu.name);
  put("gpu-usage", reading(gpu.usage, "%"));
  const gpuTemps = [reading(gpu.temperature, " °C"), reading(gpu.hotspot, " °C")];
  put("gpu-temp", gpuTemps.every((text) => text === "–") ? "–" : gpuTemps.join(" / "));
  put("vram", memory(gpu.vram_used, gpu.vram_total));
  put("gpu-fan", reading(gpu.fan, " RPM"));
  put("gpu-clock", reading(gpu.clock, " MHz"));
  put("gpu-mem-temp", reading(gpu.memory_temp, " °C"));
  put("gpu-mem-load", reading(gpu.memory_load, "%"));

  put("ram-usage", reading(ram.usage, "%"));
  put("ram-size", memory(ram.used, ram.total));
  put("ram-free", gib(ram.available));
  put("ram-swap", ram.swap_total ? memory(ram.swap_used, ram.swap_total) : "No swap");
  put("ram-temp", ram.temperature == null ? "Unavailable" : reading(ram.temperature, " °C"));

  for (const [id, watts] of [["cpu", cpu.watts], ["gpu", gpu.watts], ["ram", ram.watts]]) {
    put(id + "-watts", watts == null ? "Unavailable" : reading(watts, " W", 1));
  }
  put("cpu-watts-note", cpu.power_restricted ? "Sensor permission required" : "");
  put("ram-watts-note", ram.power_restricted ? "Sensor permission required" : "");
  const fans = Object.entries(board_fans).map(([name, value]) => name + " " + reading(value, " RPM"));
  $("board-fans").hidden = fans.length === 0;
  put("board-fans", "Motherboard fans · " + fans.join(" · "));

  history = data.history;
  charts.forEach(draw);
}

function updateControls() {
  const model = models.find((entry) => entry.id === $("model").value);
  const busy = busyStates.has(runner?.state);
  const disabled = pending || busy || !connected;
  $("model").disabled = disabled || models.length === 0;
  for (const id of ["context", "threads", "layers"]) $(id).disabled = disabled;
  $("start").disabled = disabled || !model || !runner?.binary;
  $("stop").disabled = pending || !connected || !["loading", "ready"].includes(runner?.state);

  let note = "";
  if (model && telemetry && !busy) {
    const { ram, gpu } = telemetry;
    const offload = $("layers").valueAsNumber > 0 && Number.isFinite(gpu.vram_total);
    if (model.bytes > ram.available) {
      note = gib(model.bytes) + " model weights, " + gib(ram.available) +
        " RAM available. Loading may use swap or fail.";
    } else if (offload) {
      note = gib(gpu.vram_total - gpu.vram_used) + " VRAM free. GPU offload also needs space for the context cache.";
    }
  }
  $("memory-note").hidden = !note;
  put("memory-note", note);
}

function showRunner(data) {
  runner = data;
  status("runner-state", data.binary ? data.state : "failed", data.binary ? labels[data.state] : "llama-server missing");
  put("active-model", data.model || "No model started here.");
  const meta = [];
  if (data.pid) meta.push("PID " + data.pid);
  if (data.state === "ready") meta.push(data.url + "/v1");
  if (data.state === "failed") meta.push("Exited with code " + data.exit_code);
  if (data.state === "idle" && data.model) meta.push("Stopped");
  if (!data.binary) meta.push("Set the executable with --llama-server.");
  $("runtime-meta").hidden = meta.length === 0;
  put("runtime-meta", meta.join(" · "));
  $("open-runtime").hidden = data.state !== "ready";
  $("open-runtime").href = data.url;
  const log = $("log-content");
  const follow = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  put("log-content", data.logs || "Output appears after you start a model.");
  if (follow) log.scrollTop = log.scrollHeight;

  const processes = (telemetry?.processes ?? []).filter(({ pid }) => pid !== data.pid);
  $("external").hidden = processes.length === 0;
  $("external-list").replaceChildren(...processes.map(({ model, pid, rss }) => {
    const resident = rss == null ? "" : " · " + gib(rss) + " resident RAM";
    return listItem((model || "llama-server") + " · PID " + pid + resident);
  }));
  updateControls();
}

function showChatApi(api) {
  $("chat-api").hidden = $("chat-api-nav").hidden = !api;
  if (!api) return;
  const state = api.model ? "ready" : api.checked ? "failed" : "loading";
  status("chat-api-state", state, { ready: "Ready", failed: "Unavailable", loading: "Checking" }[state]);
  put("chat-api-route", "POST " + new URL(api.base_url).pathname + "/chat/completions");
  for (const [id, value] of [["chat-api-url", api.base_url], ["chat-api-model", api.model || "–"]]) {
    if ($(id).value !== value) $(id).value = value;
  }
}

async function request(url, body) {
  const options = { signal: AbortSignal.timeout(6000) };
  if (body != null) {
    options.method = "POST";
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}

function showError(error) {
  $("form-error").hidden = !error;
  put("form-error", error?.message || "");
}

async function loadModels() {
  $("refresh").disabled = true;
  try {
    const selected = $("model").value;
    models = await request("/api/models");
    const prompt = models.length ? "Choose a model" : "No GGUF models found";
    const entries = models.map(({ id, name, folder, bytes }) => {
      const description = folder === "." ? name : folder + " / " + name;
      const size = bytes < 1024 ** 3 ? reading(bytes / 1024 ** 2, " MiB", 1) : gib(bytes, 2);
      return new Option(description + " · " + size, id);
    });
    $("model").replaceChildren(new Option(prompt, ""), ...entries);
    if (models.some(({ id }) => id === selected)) $("model").value = selected;
    showError(null);
  } catch (error) {
    showError(error);
  } finally {
    $("refresh").disabled = false;
    updateControls();
  }
}

async function poll() {
  const current = revision;
  try {
    const data = await request("/api/live");
    const fresh = Date.now() - data.updated * 1000 < 8000;
    const reconnected = fresh && !connected;
    connected = fresh;
    status("connection", fresh ? "ready" : "failed", fresh ? "Live" : "Reading delayed");
    telemetry = data;
    hardware(data);
    showChatApi(data.chat_api);
    if (current === revision && !pending) showRunner(data.runner);
    if (reconnected && models.length === 0 && !$("refresh").disabled) loadModels();
  } catch {
    connected = false;
    status("connection", "failed", "Disconnected");
    status("chat-api-state", "failed", "Status unavailable");
    updateControls();
  } finally {
    setTimeout(poll, 2000);
  }
}

async function action(url, body) {
  if (pending) return;
  revision++;
  pending = true;
  updateControls();
  showError(null);
  try {
    showRunner(await request(url, body));
  } catch (error) {
    showError(error);
  } finally {
    revision++;
    pending = false;
    updateControls();
  }
}

$("launch").addEventListener("submit", (event) => {
  event.preventDefault();
  action("/api/start", {
    model: $("model").value,
    context: $("context").valueAsNumber,
    threads: $("threads").valueAsNumber,
    layers: $("layers").valueAsNumber,
  });
});
$("stop").addEventListener("click", () => action("/api/stop", {}));
$("refresh").addEventListener("click", loadModels);
$("model").addEventListener("change", updateControls);
$("layers").addEventListener("input", updateControls);
for (const id of ["chat-api-url", "chat-api-model"]) {
  $(id).addEventListener("focus", (event) => event.currentTarget.select());
}
const resize = new ResizeObserver(() => charts.forEach(draw));
for (const chart of charts) resize.observe(chart);
loadModels();
poll();
