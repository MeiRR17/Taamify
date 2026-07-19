/* Taamify frontend — vanilla JS, no external dependencies.
   Records raw PCM via Web Audio and uploads standard WAV (no ffmpeg needed). */

"use strict";

const $ = (id) => document.getElementById(id);

// ─────────────────────────────── tabs ───────────────────────────────
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

// ─────────────────────────────── header info ────────────────────────
fetch("/api/info").then((r) => r.json()).then((d) => {
  const acc = d.test_accuracy ? ` | test accuracy ${(d.test_accuracy * 100).toFixed(1)}%` : "";
  $("model-info").textContent =
    `${d.model} | ${d.classes.length} classes${acc} | device: ${d.device}`;
});

// ─────────────────────────────── practice data ──────────────────────
let catalogue = {};
let currentClip = null;

fetch("/api/teacher").then((r) => r.json()).then((data) => {
  catalogue = data;
  const taamSel = $("taam-select");
  Object.keys(data).sort().forEach((t) => {
    const o = document.createElement("option");
    o.value = t; o.textContent = t.replace(/_/g, " ");
    taamSel.appendChild(o);
  });
  taamSel.addEventListener("change", fillWords);
  $("word-select").addEventListener("change", showClip);
  fillWords();
});

function fillWords() {
  const words = catalogue[$("taam-select").value] || [];
  const sel = $("word-select");
  sel.innerHTML = "";
  words.forEach((w, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${w.word}  (${w.verse_ref})`;
    sel.appendChild(o);
  });
  showClip();
}

function showClip() {
  const words = catalogue[$("taam-select").value] || [];
  currentClip = words[Number($("word-select").value) || 0];
  if (!currentClip) return;
  $("teacher-word").textContent = currentClip.word;
  $("teacher-ref").textContent =
    `${currentClip.verse_ref} | ${currentClip.duration}s`;
  $("teacher-audio").src = `/api/teacher/audio/${currentClip.clip_id}`;
  drawCurves(currentClip.curve, null, 0);
  $("practice-result").classList.add("hidden");
}

// ─────────────────────────────── curve canvas ───────────────────────
function drawCurves(teacher, student, score) {
  const cv = $("curve-canvas"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const all = student ? teacher.concat(student) : teacher;
  const lo = Math.min(...all) - 1, hi = Math.max(...all) + 1;
  const X = (i, n) => 40 + (i / (n - 1)) * (cv.width - 60);
  const Y = (v) => cv.height - 24 - ((v - lo) / (hi - lo)) * (cv.height - 44);

  ctx.strokeStyle = "#94a3b8"; ctx.lineWidth = 1;
  ctx.strokeRect(40, 12, cv.width - 60, cv.height - 36);
  ctx.fillStyle = "#64748b"; ctx.font = "11px sans-serif";
  ctx.fillText("semitones (key-normalized)", 46, 24);
  ctx.fillText("word progress", cv.width / 2 - 30, cv.height - 6);

  // teacher tolerance band
  ctx.beginPath();
  teacher.forEach((v, i) => { const x = X(i, teacher.length); i ? ctx.lineTo(x, Y(v + 1)) : ctx.moveTo(x, Y(v + 1)); });
  for (let i = teacher.length - 1; i >= 0; i--) ctx.lineTo(X(i, teacher.length), Y(teacher[i] - 1));
  ctx.closePath(); ctx.fillStyle = "rgba(36, 86, 166, 0.15)"; ctx.fill();

  // teacher line
  ctx.beginPath(); ctx.strokeStyle = "#2456a6"; ctx.lineWidth = 2.2;
  teacher.forEach((v, i) => { const x = X(i, teacher.length); i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); });
  ctx.stroke();
  ctx.fillStyle = "#2456a6"; ctx.fillText("teacher", cv.width - 110, 24);

  if (student) {
    const col = score >= 60 ? "#1e7e40" : "#b03434";
    ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 2.2;
    student.forEach((v, i) => { const x = X(i, student.length); i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); });
    ctx.stroke();
    ctx.fillStyle = col; ctx.fillText("you", cv.width - 110, 38);
  }
}

// ─────────────────────────────── recording (raw PCM -> WAV) ─────────
async function recordWav(seconds, onTick) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ac = new AudioContext();
  const src = ac.createMediaStreamSource(stream);
  const proc = ac.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  src.connect(proc); proc.connect(ac.destination);
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));

  for (let s = seconds; s > 0; s--) { onTick(s); await new Promise((r) => setTimeout(r, 1000)); }

  proc.disconnect(); src.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const sampleRate = ac.sampleRate;
  await ac.close();

  const n = chunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Float32Array(n);
  let off = 0; chunks.forEach((c) => { pcm.set(c, off); off += c.length; });
  return encodeWav(pcm, sampleRate);
}

function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const wr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wr(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); wr(8, "WAVE");
  wr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); wr(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ─────────────────────────────── practice flow ──────────────────────
$("record-btn").addEventListener("click", async () => {
  if (!currentClip) return;
  const btn = $("record-btn"), status = $("record-status");
  btn.disabled = true; btn.classList.add("recording");
  try {
    const secs = Math.max(2, Math.ceil(currentClip.duration) + 1);
    const wav = await recordWav(secs, (s) => { status.textContent = `recording… ${s}s`; });
    status.textContent = "scoring…";
    const fd = new FormData();
    fd.append("audio", wav, "attempt.wav");
    const res = await fetch(`/api/practice/${currentClip.clip_id}`, { method: "POST", body: fd });
    const d = await res.json();
    if (d.error) { status.textContent = d.error; return; }
    status.textContent = "";
    drawCurves(d.teacher_curve, d.student_curve, d.score);
    $("practice-result").classList.remove("hidden");
    $("score-combined").textContent = `${Math.round(d.score)}/100`;
    const top = d.cnn_top3[0] || {};
    $("score-cnn").textContent = top.taam === d.target
      ? `${top.taam.replace(/_/g, " ")} ✓`.replace(" ✓", " (correct)")
      : `heard: ${String(top.taam || "?").replace(/_/g, " ")}`;
    $("score-dtw").textContent = `${Math.round(d.dtw_component)}`;
    const v = $("verdict");
    v.className = "verdict";
    if (d.score >= 70) { v.classList.add("good"); v.textContent =
      "Excellent. The network recognizes the target mark and your melody tracks the teacher."; }
    else if (d.score >= 45 || top.taam === d.target) { v.classList.add("warn"); v.textContent =
      "Close. Listen to the teacher again and mind the rises and falls."; }
    else { v.classList.add("bad"); v.textContent =
      "The melody differs from the target. Play the teacher clip and try again."; }
  } catch (err) {
    status.textContent = `microphone error: ${err.message}`;
  } finally {
    btn.disabled = false; btn.classList.remove("recording");
  }
});

// ─────────────────────────────── analyze flow ───────────────────────
$("analyze-btn").addEventListener("click", async () => {
  const f = $("analyze-file").files[0];
  const status = $("analyze-status");
  if (!f) { status.textContent = "choose a WAV file first"; return; }
  $("analyze-btn").disabled = true;
  status.textContent = "transcribing and classifying… (first run downloads the transcription model)";
  try {
    const fd = new FormData();
    fd.append("audio", f, f.name);
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const d = await res.json();
    const tbody = document.querySelector("#analyze-table tbody");
    tbody.innerHTML = "";
    (d.words || []).forEach((w, i) => {
      const alt = w.predictions.slice(1)
        .map((p) => `${p.taam.replace(/_/g, " ")} ${(p.prob * 100).toFixed(0)}%`).join(", ");
      const best = w.predictions[0];
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${i + 1}</td><td class="hebrew">${w.word}</td>` +
        `<td>${w.start}–${w.end}s</td>` +
        `<td>${best.taam.replace(/_/g, " ")}</td>` +
        `<td><span class="prob-bar" style="width:${Math.round(best.prob * 70)}px"></span>` +
        `${(best.prob * 100).toFixed(0)}%</td><td class="muted">${alt}</td>`;
      tbody.appendChild(tr);
    });
    $("analyze-table").classList.remove("hidden");
    status.textContent = `${(d.words || []).length} words analyzed`;
  } catch (err) {
    status.textContent = `error: ${err.message}`;
  } finally {
    $("analyze-btn").disabled = false;
  }
});
