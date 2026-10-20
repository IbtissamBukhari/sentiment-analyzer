// The page is served by the FastAPI app, so API calls use relative URLs.
const LABELS = ['negative', 'neutral', 'positive'];

const $ = (id) => document.getElementById(id);
const textInput = $('textInput');

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function setStatus(message, isError = false) {
  const el = $('status');
  el.textContent = message;
  el.classList.toggle('error', isError);
}

function setBusy(busy) {
  $('predictBtn').disabled = busy;
  $('explainBtn').disabled = busy;
}

async function apiError(response) {
  const body = await response.json().catch(() => ({}));
  if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join('; ');
  return body.detail || `Request failed (${response.status})`;
}

function readText() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus('Enter some text first.', true);
    textInput.focus();
    return null;
  }
  return text;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await apiError(response));
  return response.json();
}

// ---------- Prediction ----------
function renderPrediction(data) {
  const bars = LABELS.map((label) => {
    const pct = (data.scores[label] * 100).toFixed(1);
    return `
      <div class="score-row">
        <span class="score-label">${label}</span>
        <div class="score-track"><div class="score-fill ${label}" style="width:${pct}%"></div></div>
        <span class="score-val">${pct}%</span>
      </div>`;
  }).join('');

  $('predictionResult').className = '';
  $('predictionResult').innerHTML = `
    <div class="verdict ${data.label}">${data.label}</div>
    <p class="muted">Confidence ${(data.confidence * 100).toFixed(1)}%</p>
    <div class="scores">${bars}</div>`;
}

async function predictSentiment() {
  const text = readText();
  if (!text) return;
  setBusy(true);
  setStatus('Analyzing...');
  try {
    renderPrediction(await postJson('/predict', { text }));
    setStatus('');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

// ---------- Explanation ----------
function highlightText(text, impacts) {
  const max = Math.max(...Object.values(impacts).map(Math.abs), 1e-6);
  // LIME splits on non-word characters, so split the same way and keep separators
  return text.split(/(\W+)/).map((token) => {
    const impact = impacts[token];
    if (impact === undefined) return escapeHtml(token);
    const strength = (0.15 + 0.6 * Math.abs(impact) / max).toFixed(2);
    const kind = impact >= 0 ? 'toward' : 'against';
    return `<mark class="${kind}" style="--a:${strength}" title="${impact.toFixed(3)}">${escapeHtml(token)}</mark>`;
  }).join('');
}

function renderExplanation(text, data) {
  const impacts = Object.fromEntries(data.explanation.map((item) => [item.word, item.impact]));
  const max = Math.max(...data.explanation.map((item) => Math.abs(item.impact)), 1e-6);

  const rows = data.explanation.map((item) => {
    const width = (Math.abs(item.impact) / max) * 100;
    const kind = item.impact >= 0 ? 'toward' : 'against';
    const sign = item.impact >= 0 ? '+' : '';
    return `
      <div class="word-row">
        <span class="word">${escapeHtml(item.word)}</span>
        <div class="word-track"><div class="word-fill ${kind}" style="width:${width}%"></div></div>
        <span class="word-val">${sign}${item.impact.toFixed(3)}</span>
      </div>`;
  }).join('');

  $('explanationResult').className = '';
  $('explanationResult').innerHTML = `
    <p class="muted">Prediction: <strong class="${data.prediction}-text">${data.prediction}</strong> (${(data.confidence * 100).toFixed(1)}%)</p>
    <div class="legend">
      <span><i class="swatch toward"></i>pushes toward ${data.prediction}</span>
      <span><i class="swatch against"></i>pushes away</span>
    </div>
    <p class="highlighted">${highlightText(text, impacts)}</p>
    <div class="words">${rows}</div>`;
}

async function explainSentiment() {
  const text = readText();
  if (!text) return;
  setBusy(true);
  setStatus('Explaining... this takes 10-30 seconds.');
  try {
    renderExplanation(text, await postJson('/explain', { text }));
    setStatus('');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

// ---------- CSV ----------
async function uploadCSV() {
  const file = $('csvFile').files[0];
  const out = $('csvResult');
  out.classList.remove('error');
  if (!file) {
    out.textContent = 'Choose a CSV file first.';
    out.classList.add('error');
    return;
  }
  $('csvBtn').disabled = true;
  out.textContent = 'Analyzing file...';
  try {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/predict_csv', { method: 'POST', body: form });
    if (!response.ok) throw new Error(await apiError(response));

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = file.name.replace(/\.csv$/i, '') + '_sentiment.csv';
    link.click();
    URL.revokeObjectURL(url);
    out.textContent = 'Done. Your file with predictions has been downloaded.';
  } catch (err) {
    out.textContent = err.message;
    out.classList.add('error');
  } finally {
    $('csvBtn').disabled = false;
  }
}

// ---------- Setup ----------
function updateCount() {
  $('charCount').textContent = `${textInput.value.length} / 5000`;
}

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    textInput.value = chip.dataset.example;
    updateCount();
    setStatus('');
  });
});

textInput.addEventListener('input', updateCount);
textInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) predictSentiment();
});
$('predictBtn').addEventListener('click', predictSentiment);
$('explainBtn').addEventListener('click', explainSentiment);
$('csvBtn').addEventListener('click', uploadCSV);

fetch('/model-info')
  .then((r) => (r.ok ? r.json() : null))
  .then((info) => { if (info) $('maxRows').textContent = info.max_csv_rows; })
  .catch(() => {});
