/**
 * Traffic Sign Recognizer — script.js
 * Handles all API communication, UI updates, drag-and-drop, and history.
 */

'use strict';

/* ============================================================
   CONFIG
   ============================================================ */
const API = {
  MODEL_INFO: '/api/model-info',
  PREDICT:    '/api/predict',
  HISTORY:    '/api/history',
  HEALTH:     '/api/health',
};

const MODEL_REFRESH_INTERVAL_MS = 30_000;
let modelRefreshTimer = null;
let selectedFile = null;
let toastTimer = null;

/* ============================================================
   DOM REFERENCES
   ============================================================ */
const $ = id => document.getElementById(id);

const els = {
  loadingOverlay:    $('loading-overlay'),
  toast:             $('toast'),

  // Model status
  modelStatusBadge:  $('model-status-badge'),
  modelAccuracy:     $('model-accuracy'),
  modelClasses:      $('model-classes'),
  modelWarning:      $('model-warning'),
  refreshModelBtn:   $('refresh-model-btn'),

  // Upload
  dropZone:          $('drop-zone'),
  fileInput:         $('file-input'),
  dropZoneContent:   $('drop-zone-content'),
  previewContainer:  $('preview-container'),
  imagePreview:      $('image-preview'),
  clearPreviewBtn:   $('clear-preview-btn'),
  predictBtn:        $('predict-btn'),
  btnText:           document.querySelector('#predict-btn .btn-text'),
  btnSpinner:        document.querySelector('#predict-btn .btn-spinner'),

  // Results
  resultPlaceholder: $('result-placeholder'),
  resultContent:     $('result-content'),
  resultClassName:   $('result-class-name'),
  confidenceValue:   $('confidence-value'),
  confidenceBarFill: $('confidence-bar-fill'),
  confidenceBarTrack:$('confidence-bar-track'),
  resultTimestamp:   $('result-timestamp'),

  // History
  historyEmpty:      $('history-empty'),
  historyTableWrap:  $('history-table-wrap'),
  historyTbody:      $('history-tbody'),
  clearHistoryBtn:   $('clear-history-btn'),
};

/* ============================================================
   TOAST NOTIFICATION
   ============================================================ */
/**
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 * @param {number} duration  ms
 */
function showToast(message, type = 'info', duration = 4000) {
  const el = els.toast;
  el.textContent = message;
  el.className = `toast toast-${type} show`;

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove('show');
  }, duration);
}

/* ============================================================
   LOADING OVERLAY
   ============================================================ */
function showLoading() { els.loadingOverlay.classList.add('active'); }
function hideLoading() { els.loadingOverlay.classList.remove('active'); }

/* ============================================================
   TIMESTAMP FORMATTING
   ============================================================ */
/**
 * Formats an ISO timestamp string into a human-friendly local time.
 * @param {string} isoString
 * @returns {string}
 */
function formatTimestamp(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      year:   'numeric',
      month:  'short',
      day:    'numeric',
      hour:   '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return isoString;
  }
}

/* ============================================================
   MODEL INFO
   ============================================================ */
async function fetchModelInfo() {
  try {
    const res = await fetch(API.MODEL_INFO);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderModelInfo(data);
  } catch (err) {
    console.warn('Could not fetch model info:', err);
    renderModelInfoError();
  }
}

/**
 * @param {{ accuracy: number, model_loaded: boolean, num_classes: number }} data
 */
function renderModelInfo(data) {
  const { accuracy, model_loaded, num_classes } = data;

  // Accuracy
  if (typeof accuracy === 'number') {
    els.modelAccuracy.textContent = `${(accuracy * 100).toFixed(2)}%`;
  } else {
    els.modelAccuracy.textContent = '—';
  }

  // Num classes
  els.modelClasses.textContent = typeof num_classes === 'number' ? num_classes : '—';

  // Status badge
  if (model_loaded) {
    els.modelStatusBadge.textContent = 'Loaded';
    els.modelStatusBadge.className   = 'status-badge badge-loaded';
    els.modelWarning.classList.add('hidden');
    els.predictBtn.disabled = selectedFile === null;
  } else {
    els.modelStatusBadge.textContent = 'Not Loaded';
    els.modelStatusBadge.className   = 'status-badge badge-not-loaded';
    els.modelWarning.classList.remove('hidden');
    els.predictBtn.disabled = true;
  }
}

function renderModelInfoError() {
  els.modelStatusBadge.textContent = 'Unavailable';
  els.modelStatusBadge.className   = 'status-badge badge-loading';
  els.modelAccuracy.textContent    = '—';
  els.modelClasses.textContent     = '—';
}

function startModelAutoRefresh() {
  clearInterval(modelRefreshTimer);
  modelRefreshTimer = setInterval(fetchModelInfo, MODEL_REFRESH_INTERVAL_MS);
}

/* ============================================================
   FILE SELECTION & DRAG-AND-DROP
   ============================================================ */
function handleFileSelected(file) {
  if (!file) return;

  if (!file.type.startsWith('image/')) {
    showToast('Please select an image file (JPG, PNG, BMP, PPM).', 'error');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showToast('File size must be under 10 MB.', 'error');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    els.imagePreview.src = e.target.result;
    els.dropZoneContent.classList.add('hidden');
    els.previewContainer.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  // Enable predict only if model is loaded (badge says "Loaded")
  if (!els.predictBtn.disabled || els.modelStatusBadge.classList.contains('badge-loaded')) {
    els.predictBtn.disabled = false;
  }
  // Re-check model status to decide
  fetch(API.MODEL_INFO)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data) {
        els.predictBtn.disabled = !data.model_loaded;
      }
    })
    .catch(() => {});
}

function clearPreview() {
  selectedFile = null;
  els.fileInput.value = '';
  els.imagePreview.src = '';
  els.previewContainer.classList.add('hidden');
  els.dropZoneContent.classList.remove('hidden');
  els.predictBtn.disabled = true;
}

function setupDragAndDrop() {
  const zone = els.dropZone;

  ['dragenter', 'dragover'].forEach(evt => {
    zone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('drag-over');
    });
  });

  ['dragleave', 'dragend'].forEach(evt => {
    zone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('drag-over');
    });
  });

  zone.addEventListener('drop', e => {
    e.preventDefault();
    e.stopPropagation();
    zone.classList.remove('drag-over');

    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileSelected(file);
  });

  // Keyboard activation
  zone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      els.fileInput.click();
    }
  });

  els.fileInput.addEventListener('change', () => {
    const file = els.fileInput.files?.[0];
    if (file) handleFileSelected(file);
  });

  els.clearPreviewBtn.addEventListener('click', e => {
    e.stopPropagation();
    clearPreview();
  });
}

/* ============================================================
   PREDICT
   ============================================================ */
function setPredictLoading(loading) {
  if (loading) {
    els.btnText.textContent = 'Predicting…';
    els.btnSpinner.classList.remove('hidden');
    els.predictBtn.disabled = true;
  } else {
    els.btnText.textContent = 'Predict Sign';
    els.btnSpinner.classList.add('hidden');
    els.predictBtn.disabled = false;
  }
}

async function predict() {
  if (!selectedFile) {
    showToast('Please select an image first.', 'error');
    return;
  }

  setPredictLoading(true);
  showLoading();

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const res = await fetch(API.PREDICT, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errMsg = `Server error: HTTP ${res.status}`;
      try {
        const errData = await res.json();
        if (errData.error) errMsg = errData.error;
      } catch {}
      throw new Error(errMsg);
    }

    const data = await res.json();
    renderPredictResult(data);
    showToast(`Prediction: ${data.class_name} (${data.confidence.toFixed(1)}%)`, 'success');
    await fetchHistory();

  } catch (err) {
    console.error('Predict error:', err);
    showToast(`Prediction failed: ${err.message}`, 'error');
  } finally {
    hideLoading();
    setPredictLoading(false);
  }
}

/**
 * @param {{ class_id: number, class_name: string, confidence: number, timestamp: string }} data
 */
function renderPredictResult(data) {
  const { class_name, confidence, timestamp } = data;

  els.resultPlaceholder.classList.add('hidden');
  els.resultContent.classList.remove('hidden');

  els.resultClassName.textContent = class_name ?? '—';
  els.resultTimestamp.textContent = formatTimestamp(timestamp);

  const pct = typeof confidence === 'number' ? Math.min(100, Math.max(0, confidence)) : 0;
  els.confidenceValue.textContent = `${pct.toFixed(1)}%`;

  // Animate confidence bar
  els.confidenceBarFill.style.width = '0%';
  els.confidenceBarTrack.setAttribute('aria-valuenow', pct);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      els.confidenceBarFill.style.width = `${pct}%`;
    });
  });
}

/* ============================================================
   HISTORY
   ============================================================ */
async function fetchHistory() {
  try {
    const res = await fetch(API.HISTORY);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderHistory(data);
  } catch (err) {
    console.warn('Could not fetch history:', err);
  }
}

/**
 * @param {Array<{ id: number, timestamp: string, class_name: string, confidence: number, image_base64: string }>} records
 */
function renderHistory(records) {
  const tbody = els.historyTbody;
  tbody.innerHTML = '';

  if (!Array.isArray(records) || records.length === 0) {
    els.historyTableWrap.classList.add('hidden');
    els.historyEmpty.classList.remove('hidden');
    return;
  }

  els.historyEmpty.classList.add('hidden');
  els.historyTableWrap.classList.remove('hidden');

  // Backend already returns newest-first (ORDER BY id DESC); just limit to 20
  const rows = records.slice(0, 20);

  rows.forEach((rec, idx) => {
    const tr = document.createElement('tr');

    // Index
    const tdId = document.createElement('td');
    tdId.className = 'history-id-cell';
    tdId.textContent = idx + 1;
    tr.appendChild(tdId);

    // Thumbnail
    const tdThumb = document.createElement('td');
    if (rec.image_base64) {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${rec.image_base64}`;
      img.alt = `Thumbnail for ${rec.class_name}`;
      img.className = 'history-thumbnail';
      img.loading = 'lazy';
      tdThumb.appendChild(img);
    } else {
      const placeholder = document.createElement('div');
      placeholder.className = 'history-no-thumb';
      placeholder.textContent = 'N/A';
      tdThumb.appendChild(placeholder);
    }
    tr.appendChild(tdThumb);

    // Class name
    const tdName = document.createElement('td');
    tdName.textContent = rec.class_name ?? '—';
    tdName.style.fontWeight = '500';
    tdName.style.color = 'var(--text-primary)';
    tr.appendChild(tdName);

    // Confidence
    const tdConf = document.createElement('td');
    const pill = document.createElement('span');
    pill.className = 'history-conf-pill';
    const conf = typeof rec.confidence === 'number' ? rec.confidence : 0;
    pill.textContent = `${conf.toFixed(1)}%`;
    tdConf.appendChild(pill);
    tr.appendChild(tdConf);

    // Timestamp
    const tdTime = document.createElement('td');
    tdTime.textContent = formatTimestamp(rec.timestamp);
    tdTime.style.whiteSpace = 'nowrap';
    tr.appendChild(tdTime);

    tbody.appendChild(tr);
  });
}

async function clearHistory() {
  if (!confirm('Clear all prediction history? This cannot be undone.')) return;

  showLoading();
  try {
    const res = await fetch(API.HISTORY, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await fetchHistory();
    showToast('History cleared.', 'success');
  } catch (err) {
    showToast(`Failed to clear history: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

/* ============================================================
   EVENT LISTENERS
   ============================================================ */
function attachEventListeners() {
  // Predict button
  els.predictBtn.addEventListener('click', predict);

  // Clear history
  els.clearHistoryBtn.addEventListener('click', clearHistory);

  // Manual refresh model
  els.refreshModelBtn.addEventListener('click', () => {
    fetchModelInfo();
    showToast('Refreshing model info…', 'info', 2000);
  });
}

/* ============================================================
   INIT
   ============================================================ */
async function init() {
  setupDragAndDrop();
  attachEventListeners();

  // Initial data fetch
  await Promise.allSettled([
    fetchModelInfo(),
    fetchHistory(),
  ]);

  // Start auto-refresh
  startModelAutoRefresh();
}

// Bootstrap
document.addEventListener('DOMContentLoaded', init);
