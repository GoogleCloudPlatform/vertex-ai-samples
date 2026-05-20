/**
 * @fileoverview Front-end controller for the LiveAPI Recording Viewer.
 *
 * Responsibilities:
 *   - Load a recording from a server path or browser file upload.
 *   - Render per-agent timelines with color-coded message bars.
 *   - Fisheye magnification on hover for dense timeline regions.
 *   - Pinnable tooltip with proto detail + per-chunk audio playback.
 *   - Per-agent zoom controls (independent horizontal scale).
 *
 * The companion server must expose:
 *   GET  /api/agents                       — JSON with per-agent message
 *                                            summaries.
 *   GET  /api/audio/<idx>.wav              — 24 kHz stereo WAV per agent.
 *   GET  /api/recordings                   — JSON list of recordings the
 *                                            backend has saved to its
 *                                            recordings directory. The viewer
 *                                            reads only from this directory;
 *                                            the user never types a path.
 *   GET  /api/recordings/download?name=... — Streams the named .pb recording
 *                                            for the user to download.
 *   POST /api/load                         — Body: {"name": "..."} naming an
 *                                            entry returned by
 *                                            /api/recordings. Returns
 *                                            agents JSON.
 *   POST /api/upload                       — Multipart file upload; returns
 *                                            agents JSON.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ===========================================================================
  // DOM lookup
  // ===========================================================================
  const $ = (id) => document.getElementById(id);

  const inputPathLabel = $('input-path');
  const recordingsList = $('recordings-list');
  const refreshBtn = $('refresh-btn');
  const uploadBtn = $('upload-btn');
  const uploadInput = $('upload-input');
  const statusEl = $('status');
  const globalModeToggle = $('global-mode-toggle');
  const root = $('root');
  const tooltip = $('tooltip');

  // Tracks which recording (if any) is currently rendered in the main
  // pane. Used to mark the active row in the side panel.
  let currentRecordingName = '';
  // Cache of the most recent /api/recordings response so we can re-render
  // the side panel (e.g. when the active row changes) without re-fetching.
  let recordingsCache = [];

  // ===========================================================================
  // Constants
  // ===========================================================================
  const LABEL_W = 90;     // Matches .row-label width in CSS.
  const RIGHT_PAD = 60;   // Empty space after the last bar.

  const ZOOM_STEP = Math.SQRT2;
  const ZOOM_MIN = 0.125;
  const ZOOM_MAX = 64;
  const agentZoom = new Map();

  // View mode state: 'playback' or 'message'.
  // - Playback: bars show reconstructed [start_ms, end_ms] windows.
  // - Message: every message is a fixed-width bar at wire_ms.
  let globalMode = 'playback';
  const agentModeOverride = new Map(); // agent.index -> 'playback'|'message'
  const MESSAGE_BAR_W = 4; // Fixed bar width (px) in message mode.

  /** Returns the effective mode for an agent (override or global). */
  function getAgentMode(agent) {
    return agentModeOverride.get(agent.index) || globalMode;
  }

  /**
   * Computes the effective total_ms for an agent based on view mode.
   * Playback mode: max(end_ms). Message mode: max(wire_ms).
   */
  function getAgentTotalMs(agent) {
    const msgs = agent.messages;
    if (!msgs.length) return 100;
    const mode = getAgentMode(agent);
    let maxMs = 0;
    for (const m of msgs) {
      const t = mode === 'message' ? (m.wire_ms || 0) : m.end_ms;
      if (t > maxMs) maxMs = t;
    }
    return Math.max(maxMs, 100);
  }

  // Fisheye parameters.
  const FISHEYE_RADIUS_PX = 140;
  const FISHEYE_K = 2.4;
  const FISHEYE_GAP_PX = 1;
  const FISHEYE_MAX_SCALE = 2.0;
  const FISHEYE_SIGMA_PX = 70;
  const FISHEYE_LANE_PUSH = 2.5;

  // Audio playback state.
  let _audioCtx = null;
  let _activeChunkSource = null;
  const _decodedBuffers = new Map();
  const _decodingPromises = new Map();

  // Tooltip state.
  let tooltipPinned = false;

  // ===========================================================================
  // Utilities
  // ===========================================================================
  function fmtMs(ms) {
    if (ms < 1000) return ms.toFixed(1) + ' ms';
    const s = ms / 1000;
    if (s < 60) return s.toFixed(2) + ' s';
    const m = Math.floor(s / 60);
    const r = (s - m * 60).toFixed(2);
    return `${m}m ${r}s`;
  }

  function fmtBytes(n) {
    if (!n) return '';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KiB';
    return (n / 1024 / 1024).toFixed(2) + ' MiB';
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
    }[c]));
  }

  function getAgentZoom(agent) {
    return agentZoom.get(agent.index) || 1;
  }

  // ===========================================================================
  // Status indicator
  // ===========================================================================
  function setStatus(msg, kind) {
    statusEl.textContent = msg || '';
    statusEl.className = 'status-badge ' + (kind || '');
  }

  // ===========================================================================
  // Legend
  // ===========================================================================
  function buildLegend() {
    const wrap = document.createElement('div');
    wrap.className = 'legend';
    const items = [
      ['var(--msg-client-audio)', 'audio (client)'],
      ['var(--msg-server-audio)', 'audio (server)'],
      ['var(--msg-text)', 'text'],
      ['var(--msg-image)', 'image'],
      ['var(--msg-video)', 'video'],
      ['var(--msg-tool)', 'tool'],
      ['var(--msg-setup)', 'setup'],
      ['var(--msg-interrupt)', 'interrupt'],
      ['var(--border-strong)', 'dropped (interrupted)'],
    ];
    for (const [c, label] of items) {
      const e = document.createElement('span');
      e.innerHTML =
          `<span class="swatch" style="background:${c}"></span>${label}`;
      wrap.appendChild(e);
    }
    return wrap;
  }

  // ===========================================================================
  // Fisheye magnification (macOS-dock-style on timeline rows)
  // ===========================================================================
  function fisheyeMap(x, cx) {
    const d = x - cx;
    const r = FISHEYE_RADIUS_PX;
    if (d <= -r || d >= r) return x;
    const t = d / r;
    const sign = t < 0 ? -1 : 1;
    const tNew = sign * (1 - Math.pow(1 - Math.abs(t), FISHEYE_K));
    return cx + tNew * r;
  }

  function fisheyeMag(distancePx) {
    if (Math.abs(distancePx) >= FISHEYE_RADIUS_PX) return 1;
    const g = Math.exp(
        -(distancePx * distancePx) / (FISHEYE_SIGMA_PX * FISHEYE_SIGMA_PX));
    return 1 + (FISHEYE_MAX_SCALE - 1) * g;
  }

  function fisheyeMagForRange(left, right, cx) {
    let dist;
    if (cx < left) dist = left - cx;
    else if (cx > right) dist = cx - right;
    else dist = 0;
    return fisheyeMag(dist);
  }

  function setupFisheye(track) {
    const bars = [];
    const clusters = new Map();
    for (const el of track.querySelectorAll('.msg')) {
      const left = parseFloat(el.style.left) || 0;
      const isInstant = el.classList.contains('instant');
      const width = isInstant ? 2 : (parseFloat(el.style.width) || 0);
      const laneDy = parseFloat(el.dataset.laneDy || '0') || 0;
      const bandLocked = el.dataset.bandLocked === '1';
      const bar = {el, left, width, instant: isInstant, laneDy, bandLocked};
      if (bandLocked) {
        bar.restTop = parseFloat(el.style.top) || 0;
        bar.restH = parseFloat(el.style.height) || 0;
        bar.bandIndex = parseInt(el.dataset.bandIndex || '0', 10);
        bar.bandCount = parseInt(el.dataset.bandCount || '1', 10);
        const cid = el.dataset.clusterId || '';
        if (!clusters.has(cid)) clusters.set(cid, []);
        clusters.get(cid).push(bar);
      }
      bars.push(bar);
    }
    if (!bars.length) return;
    for (const group of clusters.values()) {
      group.sort((a, b) => a.bandIndex - b.bandIndex);
    }

    const cursor = document.createElement('div');
    cursor.className = 'cursor-line';
    track.appendChild(cursor);

    let raf = null;
    let lastCx = null;
    let lastCy = null;

    function paint() {
      raf = null;
      const cx = lastCx;
      if (cx == null) return;
      cursor.style.display = 'block';
      cursor.style.left = (cx - 0.5) + 'px';
      for (const b of bars) {
        const newLeft = fisheyeMap(b.left, cx);
        let mag;
        if (b.instant) {
          b.el.style.left = newLeft + 'px';
          mag = fisheyeMag(newLeft + b.width / 2 - cx);
        } else {
          const newRight = fisheyeMap(b.left + b.width, cx);
          b.el.style.left = newLeft + 'px';
          const w = Math.max(1, newRight - newLeft - FISHEYE_GAP_PX);
          b.el.style.width = w + 'px';
          mag = fisheyeMagForRange(newLeft, newLeft + w, cx);
        }
        const laneSpread = 1 + (mag - 1) * FISHEYE_LANE_PUSH;
        const dy = b.laneDy * laneSpread;
        const parts = [];
        if (dy) parts.push(`translateY(${dy.toFixed(2)}px)`);
        if (b.instant && mag !== 1) parts.push(`scaleX(${mag.toFixed(3)})`);
        b.el.style.transform = parts.join(' ');
      }

      // Band-locked cluster redistribution.
      if (clusters.size && lastCy != null) {
        const ROW_H = 48;
        const Y_SIGMA = 6;
        const WEIGHT_PEAK = 6;
        for (const group of clusters.values()) {
          const n = group.length;
          if (n < 2) continue;
          const colX = group[0].left + group[0].width / 2;
          const horizGate = fisheyeMag(colX - cx) - 1;
          const horizProx =
              Math.min(1, horizGate / (FISHEYE_MAX_SCALE - 1));
          const restBandH = ROW_H / n;
          const weights = new Array(n);
          let sumW = 0;
          for (let i = 0; i < n; i++) {
            const center = (i + 0.5) * restBandH;
            const dy = lastCy - center;
            const peak = 1 + horizProx * (WEIGHT_PEAK - 1);
            const w = 1 + (peak - 1) *
                Math.exp(-(dy * dy) / (Y_SIGMA * Y_SIGMA));
            weights[i] = w;
            sumW += w;
          }
          let curTop = 0;
          for (let i = 0; i < n; i++) {
            const h = ROW_H * weights[i] / sumW;
            group[i].el.style.top = curTop.toFixed(2) + 'px';
            group[i].el.style.height = h.toFixed(2) + 'px';
            curTop += h;
          }
        }
      }
    }

    function reset() {
      cursor.style.display = 'none';
      for (const b of bars) {
        b.el.style.left = b.left + 'px';
        if (!b.instant) b.el.style.width = b.width + 'px';
        if (b.bandLocked) {
          b.el.style.top = b.restTop.toFixed(2) + 'px';
          b.el.style.height = b.restH.toFixed(2) + 'px';
        }
        b.el.style.transform = b.laneDy ? `translateY(${b.laneDy}px)` : '';
      }
      track.classList.remove('zoom');
    }

    track.addEventListener('mousemove', (e) => {
      const rect = track.getBoundingClientRect();
      lastCx = e.clientX - rect.left;
      lastCy = e.clientY - rect.top;
      track.classList.add('zoom');
      if (raf == null) raf = requestAnimationFrame(paint);
    });
    track.addEventListener('mouseleave', () => {
      if (raf != null) { cancelAnimationFrame(raf); raf = null; }
      lastCx = null;
      lastCy = null;
      reset();
    });
  }

  // ===========================================================================
  // Audio playback (Web Audio API for per-chunk playback)
  // ===========================================================================
  function getAudioCtx() {
    if (!_audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      _audioCtx = new Ctx();
    }
    return _audioCtx;
  }

  async function getAgentBuffer(agent) {
    if (_decodedBuffers.has(agent.index)) {
      return _decodedBuffers.get(agent.index);
    }
    if (_decodingPromises.has(agent.index)) {
      return _decodingPromises.get(agent.index);
    }
    const p = (async () => {
      const resp = await fetch(`/api/audio/${agent.index}.wav`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const arr = await resp.arrayBuffer();
      const ctx = getAudioCtx();
      const buf = await ctx.decodeAudioData(arr);
      _decodedBuffers.set(agent.index, buf);
      _decodingPromises.delete(agent.index);
      return buf;
    })();
    _decodingPromises.set(agent.index, p);
    return p;
  }

  async function playMessage(agent, m) {
    try {
      const ctx = getAudioCtx();
      if (ctx.state === 'suspended') await ctx.resume();
      const buffer = await getAgentBuffer(agent);
      if (_activeChunkSource) {
        try { _activeChunkSource.stop(); } catch (_) {}
        _activeChunkSource = null;
      }
      const startSec = Math.max(0, m.start_ms / 1000);
      const endSec =
          m.end_ms > m.start_ms ? m.end_ms / 1000 : startSec + 1;
      const durationSec = Math.max(0, endSec - startSec);
      if (durationSec <= 0) return;
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      src.onended = () => {
        if (_activeChunkSource === src) _activeChunkSource = null;
      };
      _activeChunkSource = src;
      src.start(0, startSec, durationSec);
    } catch (e) {
      console.error('playMessage failed', e);
    }
  }

  function attachStopAt(audio) {
    if (audio.__stopAtAttached) return;
    audio.__stopAtAttached = true;
    audio.__stopAtSec = null;
    audio.addEventListener('timeupdate', () => {
      if (audio.__stopAtSec != null &&
          audio.currentTime >= audio.__stopAtSec) {
        audio.pause();
        audio.__stopAtSec = null;
      }
    });
    audio.addEventListener('play', () => {
      if (audio.__nextPlayKeepStop) {
        audio.__nextPlayKeepStop = false;
      } else {
        audio.__stopAtSec = null;
      }
    });
  }

  function playFull(agent) {
    const audio = $('audio-' + agent.index);
    if (!audio) return;
    attachStopAt(audio);
    audio.__stopAtSec = null;
    audio.currentTime = 0;
    const p = audio.play();
    if (p && p.catch) p.catch(() => {});
  }

  // ===========================================================================
  // Tooltip
  // ===========================================================================
  function buildTooltipContent(m, pinned) {
    const parts = [];
    if (pinned) {
      parts.push(
          '<div id="tt-pin-bar">' +
          '<span class="badge" style="background:var(--accent);color:white">' +
          'PINNED</span>' +
          '<span class="hint">scroll to read · click outside to close</span>' +
          '<button id="tt-close" title="Close">&times;</button>' +
          '</div>');
    }
    parts.push(`<h3>${m.direction.toUpperCase()} · ${m.kind}</h3>`);
    const badges = [];
    badges.push(`<span class="badge">${m.modality}</span>`);
    if (m.interrupted) {
      badges.push(
          '<span class="badge" ' +
          'style="background:var(--danger);color:white">' +
          'INTERRUPTED</span>');
    }
    parts.push(`<div>${badges.join('')}</div>`);
    if (pinned && m.audio_bytes > 0 && !m.interrupted) {
      parts.push(
          '<div style="margin-top:6px">' +
          '<button id="tt-play" class="play-action">&#9654; Play audio' +
          '</button>' +
          '<span class="hint" style="margin-left:8px">' +
          `${fmtMs(m.duration_ms)} · ${fmtBytes(m.audio_bytes)} @ ` +
          `${m.audio_rate_hz} Hz</span>` +
          '</div>');
    } else if (pinned && m.audio_bytes > 0 && m.interrupted) {
      parts.push(
          '<div style="margin-top:6px;color:var(--danger);font-size:0.78rem">' +
          'Audio was interrupted; not present in the agent WAV.' +
          '</div>');
    }
    parts.push('<div class="kv" style="margin-top:6px">');
    parts.push(`<b>start</b><span>${fmtMs(m.start_ms)}</span>`);
    parts.push(`<b>end</b><span>${fmtMs(m.end_ms)}</span>`);
    parts.push(`<b>duration</b><span>${fmtMs(m.duration_ms)}</span>`);
    if (m.wire_ms != null) {
      parts.push(`<b>wire</b><span>${fmtMs(m.wire_ms)}</span>`);
    }
    if (m.audio_bytes) {
      parts.push(
          `<b>audio</b><span>${fmtBytes(m.audio_bytes)} @ ` +
          `${m.audio_rate_hz} Hz</span>`);
    }
    if (m.mime_types && m.mime_types.length) {
      parts.push(`<b>mime</b><span>${m.mime_types.join(', ')}</span>`);
    }
    if (m.text_preview) {
      parts.push(
          `<b>text</b><span>${escapeHtml(m.text_preview)}</span>`);
    }
    parts.push('</div>');
    parts.push(
        `<pre>${escapeHtml(JSON.stringify(m.detail, null, 2))}</pre>`);
    return parts.join('');
  }

  function positionTooltipNear(clientX, clientY) {
    const pad = 14;
    const prev = tooltip.style.display;
    if (prev === 'none') tooltip.style.display = 'block';
    const rect = tooltip.getBoundingClientRect();
    let x = clientX + pad;
    let y = clientY + pad;
    if (x + rect.width > window.innerWidth) x = clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight) {
      y = clientY - rect.height - pad;
    }
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }

  function showTooltip(e, agent, idx) {
    if (tooltipPinned) return;
    const m = agent.messages[idx];
    tooltip.innerHTML = buildTooltipContent(m, false);
    tooltip.style.display = 'block';
    positionTooltipNear(e.clientX, e.clientY);
  }

  function moveTooltip(e) {
    if (tooltipPinned) return;
    positionTooltipNear(e.clientX, e.clientY);
  }

  function hideTooltip() {
    if (tooltipPinned) return;
    tooltip.style.display = 'none';
  }

  function pinTooltip(e, agent, m) {
    tooltipPinned = true;
    tooltip.classList.add('pinned');
    tooltip.innerHTML = buildTooltipContent(m, true);
    tooltip.style.display = 'block';
    positionTooltipNear(e.clientX, e.clientY);
    const closeBtn = tooltip.querySelector('#tt-close');
    if (closeBtn) closeBtn.addEventListener('click', unpinTooltip);
    const playBtn = tooltip.querySelector('#tt-play');
    if (playBtn) {
      playBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        playMessage(agent, m);
      });
    }
  }

  function unpinTooltip() {
    tooltipPinned = false;
    tooltip.classList.remove('pinned');
    tooltip.style.display = 'none';
    tooltip.innerHTML = '';
  }

  document.addEventListener('mousedown', (e) => {
    if (!tooltipPinned) return;
    if (tooltip.contains(e.target)) return;
    if (e.target.closest && e.target.closest('.msg')) return;
    unpinTooltip();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && tooltipPinned) unpinTooltip();
  });

  // ===========================================================================
  // Ruler
  // ===========================================================================
  function renderRuler(totalMs, pxPerMs) {
    const ruler = document.createElement('div');
    ruler.className = 'ruler';
    ruler.style.width = (totalMs * pxPerMs + LABEL_W + RIGHT_PAD) + 'px';
    const targetTicks = 10;
    const rawStep = totalMs / targetTicks;
    const niceSteps = [
      50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000,
      120000, 300000, 600000,
    ];
    let step = niceSteps[niceSteps.length - 1];
    for (const s of niceSteps) {
      if (s >= rawStep) { step = s; break; }
    }
    for (let t = 0; t <= totalMs; t += step) {
      const tick = document.createElement('span');
      tick.className = 'tick';
      tick.style.left = (LABEL_W + t * pxPerMs) + 'px';
      tick.textContent = fmtMs(t);
      ruler.appendChild(tick);
    }
    return ruler;
  }

  // ===========================================================================
  // Timeline rendering (supports playback and message modes)
  // ===========================================================================

  /**
   * Lane-packs a connected component of bars into the minimum number of
   * vertical lanes such that no two bars in the same lane overlap on x.
   *
   * This is interval-graph coloring via a first-fit sweep: sort by `left`,
   * then for each bar pick the lowest-indexed lane whose previously-assigned
   * bars all end before the current bar's `left`. The result is the chromatic
   * number of the interval overlap graph, which equals the maximum number of
   * bars overlapping at any single x — NOT the size of the transitive group.
   *
   * Example: A=[0,10], B=[8,18], C=[16,26] form a chain (A-B overlap, B-C
   * overlap, A-C disjoint). Naive transitive grouping splits 3 bars into 3
   * bands. Lane packing assigns A->lane 0, B->lane 1, C->lane 0 — only 2
   * lanes are needed.
   *
   * @param items Array of {left, right} bar descriptors. Mutated in place
   *   to add `lane` (int) and `numLanes` (int) fields.
   * @return The total number of lanes used (== max simultaneous overlap).
   */
  function packLanes(items) {
    if (!items.length) return 0;
    const sorted = [...items].sort((a, b) => a.left - b.left);
    // laneEnds[i] = right edge of the last bar placed in lane i.
    const laneEnds = [];
    for (const it of sorted) {
      let assigned = -1;
      for (let i = 0; i < laneEnds.length; i++) {
        if (laneEnds[i] <= it.left) {
          assigned = i;
          break;
        }
      }
      if (assigned === -1) {
        assigned = laneEnds.length;
        laneEnds.push(it.right);
      } else {
        laneEnds[assigned] = it.right;
      }
      it.lane = assigned;
    }
    const numLanes = laneEnds.length;
    for (const it of sorted) it.numLanes = numLanes;
    return numLanes;
  }

  /**
   * Splits a list of bars (each {left, right}) into connected components
   * under the relation "x-ranges intersect". Two bars with disjoint
   * intervals end up in different components even if both overlap a third
   * bar — this is intentional; lane packing happens per-component.
   *
   * @return Array of components; each is an array of input items.
   */
  function findOverlapComponents(items) {
    if (!items.length) return [];
    const sorted = [...items].sort((a, b) => a.left - b.left);
    const components = [];
    let cur = [sorted[0]];
    let curRight = sorted[0].right;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].left < curRight) {
        cur.push(sorted[i]);
        curRight = Math.max(curRight, sorted[i].right);
      } else {
        components.push(cur);
        cur = [sorted[i]];
        curRight = sorted[i].right;
      }
    }
    components.push(cur);
    return components;
  }

  /**
   * Reorders lane indices into a "zigzag" sequence so adjacent lanes (in
   * lane-index order) end up on opposite sides of the row instead of all
   * stacking downward. For each lane i the returned slot is computed as:
   *
   *   slot(i) = i % 2 === 0 ? i / 2 : numLanes - 1 - (i - 1) / 2
   *
   * Examples:
   *   numLanes=2 -> [0, 1]               (top, bottom)
   *   numLanes=3 -> [0, 2, 1]            (top, bottom, middle)
   *   numLanes=4 -> [0, 3, 1, 2]         (top, bottom, upper-mid, lower-mid)
   *
   * Returns a permutation array `perm` such that bar with original lane
   * `i` should render at vertical slot `perm[i]`.
   */
  function zigzagPermutation(numLanes) {
    if (numLanes <= 1) return [0];
    const perm = new Array(numLanes);
    for (let i = 0; i < numLanes; i++) {
      perm[i] = (i % 2 === 0)
          ? (i >> 1)
          : numLanes - 1 - ((i - 1) >> 1);
    }
    return perm;
  }

  /**
   * Builds the cluster band layout for a list of bar objects that should
   * be treated as instant-like (fixed-width). Used by both playback mode
   * (for true instants) and message mode (for all bars).
   *
   * Same as applyOverlapBands below, but bars are inflated to width
   * CLUSTER_PX so close-but-not-touching instants are still de-stacked.
   */
  function applyClusterBands(barList, track) {
    const CLUSTER_PX = 6;
    const ROW_H = 48;
    const MIN_BAND_H = 4;
    if (!barList.length) return;
    // Treat each instant as occupying [left, left + CLUSTER_PX] so close
    // instants are considered overlapping.
    const items = barList.map(b => ({
      el: b.el,
      left: b.left,
      right: b.left + CLUSTER_PX,
    }));
    const components = findOverlapComponents(items);
    for (const comp of components) {
      if (comp.length === 1) {
        const it = comp[0];
        it.el.dataset.laneDy = '0';
        it.el.style.zIndex = '10';
        track.appendChild(it.el);
        continue;
      }
      const numLanes = packLanes(comp);
      const idealH = ROW_H / numLanes;
      // Honor MIN_BAND_H only while the lanes still fit within ROW_H.
      const bandH = numLanes * MIN_BAND_H <= ROW_H
          ? Math.max(MIN_BAND_H, idealH) : idealH;
      const perm = zigzagPermutation(numLanes);
      const clusterId = 'inst_' + comp[0].left.toFixed(2);
      for (const it of comp) {
        const slot = perm[it.lane];
        it.el.style.top = (slot * bandH).toFixed(2) + 'px';
        it.el.style.height = bandH.toFixed(2) + 'px';
        it.el.style.zIndex = '10';
        it.el.dataset.laneDy = '0';
        it.el.dataset.bandLocked = '1';
        it.el.dataset.clusterId = clusterId;
        it.el.dataset.bandIndex = String(slot);
        it.el.dataset.bandCount = String(numLanes);
        track.appendChild(it.el);
      }
    }
  }

  /**
   * Detects overlapping duration bars (bars whose [left, left+width] ranges
   * intersect) and splits them into exclusive vertical lanes within the row
   * height. Lanes are assigned by interval-graph coloring (see packLanes),
   * so a chain like A-B-C where A and C are disjoint uses 2 lanes, not 3.
   * Lanes are then reordered into a zigzag pattern around the row midline.
   *
   * Tagged with data-band-* attributes so the fisheye cluster redistribution
   * code picks them up automatically.
   */
  function applyOverlapBands(durationBars) {
    if (durationBars.length < 2) return;
    const ROW_H = 48;
    const MIN_BAND_H = 4;
    const items = durationBars.map(b => ({
      el: b.el,
      left: b.left,
      right: b.left + b.width,
    }));
    const components = findOverlapComponents(items);
    for (const comp of components) {
      if (comp.length < 2) continue;
      const numLanes = packLanes(comp);
      if (numLanes < 2) continue;  // No actual overlap inside this component.
      const idealH = ROW_H / numLanes;
      // Honor MIN_BAND_H only while the lanes still fit within ROW_H.
      const bandH = numLanes * MIN_BAND_H <= ROW_H
          ? Math.max(MIN_BAND_H, idealH) : idealH;
      const perm = zigzagPermutation(numLanes);
      const clusterId = 'dur_' + comp[0].left.toFixed(2);
      for (const it of comp) {
        const slot = perm[it.lane];
        it.el.style.top = (slot * bandH).toFixed(2) + 'px';
        it.el.style.height = bandH.toFixed(2) + 'px';
        it.el.dataset.laneDy = '0';
        it.el.dataset.bandLocked = '1';
        it.el.dataset.clusterId = clusterId;
        it.el.dataset.bandIndex = String(slot);
        it.el.dataset.bandCount = String(numLanes);
      }
    }
  }

  function renderTimeline(agent) {
    const mode = getAgentMode(agent);
    const wrap = document.createElement('div');
    wrap.className = 'timeline-wrap';

    const totalMs = getAgentTotalMs(agent);
    const minWidth = Math.max(window.innerWidth - 80, 1200);
    const basePxPerMs = Math.max(minWidth / totalMs, 0.001);
    const pxPerMs = basePxPerMs * getAgentZoom(agent);

    const tl = document.createElement('div');
    tl.className = 'timeline';
    tl.style.width = (totalMs * pxPerMs + LABEL_W + RIGHT_PAD) + 'px';
    tl.appendChild(renderRuler(totalMs, pxPerMs));

    const directions = [
      {key: 'client', label: 'CLIENT →'},
      {key: 'server', label: 'SERVER ←'},
    ];
    for (const dir of directions) {
      const row = document.createElement('div');
      row.className = 'row';
      const lbl = document.createElement('span');
      lbl.className = 'row-label';
      lbl.textContent = dir.label;
      row.appendChild(lbl);
      const track = document.createElement('div');
      track.className = 'row-track';
      row.appendChild(track);

      if (mode === 'message') {
        // ---- Message mode: every message is a fixed-width bar at wire_ms.
        const allBars = [];
        for (let i = 0; i < agent.messages.length; i++) {
          const m = agent.messages[i];
          if (m.direction !== dir.key) continue;
          const el = document.createElement('div');
          const cls = ['msg', 'instant', m.modality];
          if (m.direction === 'server' && m.modality === 'audio') {
            cls.push('server');
          }
          if (m.interrupted) cls.push('dropped');
          const left = (m.wire_ms || 0) * pxPerMs;
          el.style.left = left + 'px';
          el.style.width = MESSAGE_BAR_W + 'px';
          el.title = `${m.kind} @ wire ${(m.wire_ms || 0).toFixed(1)} ms`;
          el.className = cls.join(' ');
          el.dataset.idx = i;
          el.addEventListener('mouseenter', (e) => showTooltip(e, agent, i));
          el.addEventListener('mousemove', moveTooltip);
          el.addEventListener('mouseleave', hideTooltip);
          el.addEventListener('click', (ev) => {
            ev.stopPropagation();
            pinTooltip(ev, agent, m);
          });
          allBars.push({el, left});
        }
        applyClusterBands(allBars, track);
      } else {
        // ---- Playback mode: duration bars + instant markers.
        const durationBars = [];
        const instants = [];
        for (let i = 0; i < agent.messages.length; i++) {
          const m = agent.messages[i];
          if (m.direction !== dir.key) continue;

          const el = document.createElement('div');
          const cls = ['msg', m.modality];
          if (m.direction === 'server' && m.modality === 'audio') {
            cls.push('server');
          }
          if (m.interrupted) cls.push('dropped');
          const durationMs = m.end_ms - m.start_ms;
          const isInstant = durationMs <= 0;
          const left = m.start_ms * pxPerMs;
          el.style.left = left + 'px';
          let widthPx = 2;
          if (isInstant) {
            cls.push('instant');
            el.title = `${m.kind} @ ${m.start_ms.toFixed(1)} ms`;
          } else {
            widthPx = Math.max(2, durationMs * pxPerMs);
            el.style.width = widthPx + 'px';
            if (widthPx < 30) cls.push('tiny');
            else el.textContent = m.modality;
          }
          el.className = cls.join(' ');
          el.dataset.idx = i;
          el.addEventListener('mouseenter', (e) =>
              showTooltip(e, agent, i));
          el.addEventListener('mousemove', moveTooltip);
          el.addEventListener('mouseleave', hideTooltip);
          el.addEventListener('click', (ev) => {
            ev.stopPropagation();
            pinTooltip(ev, agent, m);
          });
          if (isInstant) instants.push({el, left});
          else durationBars.push({el, left, width: widthPx});
        }
        // Longest first so short bars paint on top.
        durationBars.sort((a, b) => b.width - a.width);
        for (const b of durationBars) track.appendChild(b.el);
        // Split overlapping duration bars into exclusive vertical bands.
        applyOverlapBands(durationBars);
        applyClusterBands(instants, track);
      }
      setupFisheye(track);
      tl.appendChild(row);
    }
    wrap.appendChild(tl);
    return wrap;
  }

  // ===========================================================================
  // Zoom controls (per-agent)
  // ===========================================================================
  function buildAgentZoomControls(agent) {
    const wrap = document.createElement('div');
    wrap.className = 'zoom-ctrl';
    wrap.title = 'Zoom timeline (- / +). Reset returns to fit-to-width.';

    const out = document.createElement('button');
    out.type = 'button';
    out.textContent = '\u2212'; // minus sign
    const lbl = document.createElement('span');
    lbl.className = 'zoom-level';
    lbl.id = 'zoom-level-' + agent.index;
    const inn = document.createElement('button');
    inn.type = 'button';
    inn.textContent = '+';
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.textContent = 'Reset';
    reset.style.fontSize = '11px';
    reset.style.minWidth = '0';

    out.addEventListener('click', () =>
        setAgentZoom(agent, getAgentZoom(agent) / ZOOM_STEP));
    inn.addEventListener('click', () =>
        setAgentZoom(agent, getAgentZoom(agent) * ZOOM_STEP));
    reset.addEventListener('click', () => setAgentZoom(agent, 1));

    wrap.appendChild(out);
    wrap.appendChild(lbl);
    wrap.appendChild(inn);
    wrap.appendChild(reset);
    return wrap;
  }

  function updateAgentZoomLabel(agent) {
    const el = $('zoom-level-' + agent.index);
    if (!el) return;
    const pct = getAgentZoom(agent) * 100;
    el.textContent = (pct >= 10 && Number.isInteger(pct))
        ? pct.toFixed(0) + '%'
        : pct.toFixed(pct < 10 ? 1 : 0) + '%';
  }

  function setAgentZoom(agent, newZoom) {
    const clamped = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, newZoom));
    const oldZoom = getAgentZoom(agent);
    if (clamped === oldZoom) {
      updateAgentZoomLabel(agent);
      return;
    }
    const block = $('agent-block-' + agent.index);
    if (!block) {
      agentZoom.set(agent.index, clamped);
      return;
    }
    const oldWrap = block.querySelector('.timeline-wrap');
    if (!oldWrap) {
      agentZoom.set(agent.index, clamped);
      return;
    }
    // Preserve scroll center across zoom.
    const totalMs = getAgentTotalMs(agent);
    const minWidth = Math.max(window.innerWidth - 80, 1200);
    const basePxPerMs = Math.max(minWidth / totalMs, 0.001);
    const oldPxPerMs = basePxPerMs * oldZoom;
    const newPxPerMs = basePxPerMs * clamped;
    const viewportW = oldWrap.clientWidth;
    const centerPx = oldWrap.scrollLeft + viewportW / 2;
    let centerMs = (centerPx - LABEL_W) / oldPxPerMs;
    if (!Number.isFinite(centerMs) || centerMs < 0) centerMs = 0;
    if (centerMs > totalMs) centerMs = totalMs;

    agentZoom.set(agent.index, clamped);
    updateAgentZoomLabel(agent);

    const newWrap = renderTimeline(agent);
    oldWrap.replaceWith(newWrap);

    let target = centerMs * newPxPerMs + LABEL_W - viewportW / 2;
    const maxScroll =
        Math.max(0, newWrap.scrollWidth - newWrap.clientWidth);
    if (target < 0) target = 0;
    if (target > maxScroll) target = maxScroll;
    newWrap.scrollLeft = target;
  }

  // ===========================================================================
  // Per-agent mode toggle
  // ===========================================================================
  function buildAgentModeToggle(agent) {
    const wrap = document.createElement('div');
    wrap.className = 'segmented segmented-sm';
    wrap.id = 'agent-mode-' + agent.index;
    const mode = getAgentMode(agent);
    for (const val of ['playback', 'message']) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className =
          'segmented-option' + (val === mode ? ' is-active' : '');
      btn.dataset.value = val;
      btn.textContent = val.charAt(0).toUpperCase() + val.slice(1);
      btn.addEventListener('click', () => {
        agentModeOverride.set(agent.index, val);
        // Update toggle active state.
        wrap.querySelectorAll('.segmented-option').forEach((o) => {
          o.classList.toggle(
              'is-active', o.dataset.value === val);
        });
        rerenderAgentTimeline(agent);
      });
      wrap.appendChild(btn);
    }
    return wrap;
  }

  /** Re-renders just the timeline portion of an agent block. */
  function rerenderAgentTimeline(agent) {
    const block = $('agent-block-' + agent.index);
    if (!block) return;
    const oldWrap = block.querySelector('.timeline-wrap');
    if (!oldWrap) return;
    const newWrap = renderTimeline(agent);
    oldWrap.replaceWith(newWrap);
  }

  // ===========================================================================
  // Agent block rendering
  // ===========================================================================
  function renderAgent(agent) {
    const block = document.createElement('div');
    block.className = 'agent';
    block.id = 'agent-block-' + agent.index;

    const header = document.createElement('div');
    header.className = 'agent-header';
    const title = document.createElement('h2');
    title.textContent = agent.agent_name;
    header.appendChild(title);
    const total = document.createElement('span');
    total.className = 'total';
    total.textContent =
        `${agent.messages.length} messages, ${fmtMs(agent.total_ms)} total`;
    header.appendChild(total);

    const player = document.createElement('div');
    player.className = 'player';
    const playBtn = document.createElement('button');
    playBtn.className = 'btn btn-primary btn-sm';
    playBtn.textContent = '\u25B6 Play full timeline';
    const audio = document.createElement('audio');
    audio.id = 'audio-' + agent.index;
    audio.controls = true;
    audio.preload = 'none';
    audio.src = `/api/audio/${agent.index}.wav`;
    playBtn.addEventListener('click', () => playFull(agent));
    player.appendChild(playBtn);
    player.appendChild(audio);
    header.appendChild(player);

    header.appendChild(buildAgentModeToggle(agent));
    header.appendChild(buildAgentZoomControls(agent));
    block.appendChild(header);
    block.appendChild(buildLegend());
    block.appendChild(renderTimeline(agent));
    updateAgentZoomLabel(agent);
    return block;
  }

  // ===========================================================================
  // Top-level render
  // ===========================================================================
  function render(data) {
    _lastAgentsData = data;
    inputPathLabel.textContent = data.input_path || '';
    // Update the side-panel active-row highlight. Uploaded recordings
    // (input_path starts with '<') have no matching row.
    if (data.input_path && !data.input_path.startsWith('<')) {
      currentRecordingName = data.recording_name || data.input_path;
    } else {
      currentRecordingName = '';
    }
    renderRecordingsPanel();
    agentZoom.clear();
    agentModeOverride.clear();
    _decodedBuffers.clear();
    _decodingPromises.clear();
    root.innerHTML = '';
    if (!data.agents || !data.agents.length) {
      root.innerHTML =
          '<div class="empty-state">' +
          '<div class="empty-icon">📼</div>' +
          '<div class="empty-title">No recording loaded</div>' +
          '<div class="empty-hint">' +
          'Pick a saved recording from the left panel, or upload a' +
          ' .pb file to visualize a session.' +
          '</div></div>';
      return;
    }
    for (const a of data.agents) root.appendChild(renderAgent(a));
  }

  // ===========================================================================
  // API helpers
  // ===========================================================================
  async function fetchJsonOrThrow(resp) {
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  /**
   * Builds one row of the recordings side panel.
   *
   * Each row has:
   *   - A clickable label (recording name + size + saved-at) that
   *     loads the recording into the main pane.
   *   - A "Download" button that streams the raw .pb to the user via
   *     GET /api/recordings/download?name=...
   *
   * The row corresponding to the currently loaded recording is
   * marked with the `.is-active` class.
   */
  function buildRecordingRow(rec) {
    const li = document.createElement('li');
    li.className = 'recording-row';
    if (rec.name === currentRecordingName) li.classList.add('is-active');
    li.dataset.name = rec.name;
    li.setAttribute('role', 'option');

    const loadBtn = document.createElement('button');
    loadBtn.type = 'button';
    loadBtn.className = 'recording-load';
    loadBtn.title = 'Load ' + rec.name;
    const nameEl = document.createElement('div');
    nameEl.className = 'recording-name mono';
    nameEl.textContent = rec.name;
    const metaEl = document.createElement('div');
    metaEl.className = 'recording-meta';
    const metaParts = [];
    if (rec.size_bytes) metaParts.push(fmtBytes(rec.size_bytes));
    if (rec.saved_at) metaParts.push(rec.saved_at);
    metaEl.textContent = metaParts.join(' · ');
    loadBtn.appendChild(nameEl);
    if (metaParts.length) loadBtn.appendChild(metaEl);
    loadBtn.addEventListener('click', () => loadByName(rec.name));

    const dlBtn = document.createElement('a');
    dlBtn.className = 'btn btn-ghost btn-sm recording-download';
    dlBtn.href =
        '/api/recordings/download?name=' + encodeURIComponent(rec.name);
    dlBtn.setAttribute('download', rec.name);
    dlBtn.title = 'Download ' + rec.name;
    dlBtn.textContent = '↓';
    // Stop the row's load-on-click from firing when the user clicks
    // the download button.
    dlBtn.addEventListener('click', (e) => e.stopPropagation());

    li.appendChild(loadBtn);
    li.appendChild(dlBtn);
    return li;
  }

  /** Re-renders the side panel from the cached /api/recordings response. */
  function renderRecordingsPanel() {
    if (!recordingsList) return;
    recordingsList.innerHTML = '';
    if (!recordingsCache.length) {
      const empty = document.createElement('li');
      empty.className = 'recordings-empty';
      empty.textContent = '(no recordings saved yet)';
      recordingsList.appendChild(empty);
      return;
    }
    for (const r of recordingsCache) {
      recordingsList.appendChild(buildRecordingRow(r));
    }
  }

  /**
   * Fetches /api/recordings, updates the cache + side panel. The
   * backend serves only recordings from its own recordings directory;
   * the user never types or sees a filesystem path.
   */
  async function loadRecordingsList() {
    if (!recordingsList) return;
    try {
      const data = await fetchJsonOrThrow(
          await fetch('/api/recordings'));
      recordingsCache = data.recordings || [];
      renderRecordingsPanel();
    } catch (e) {
      recordingsCache = [];
      renderRecordingsPanel();
      setStatus(
          'Failed to list recordings: ' + e.message, 'error');
      console.error(e);
    }
  }

  async function loadInitial() {
    const requested = getRequestedRecordingFromHash();
    await loadRecordingsList();
    if (requested
        && recordingsCache.some((r) => r.name === requested)) {
      await loadByName(requested);
      return;
    }
    try {
      const data = await fetchJsonOrThrow(await fetch('/api/agents'));
      render(data);
    } catch (e) {
      setStatus('Failed to load: ' + e.message, 'error');
      console.error(e);
    }
  }

  /**
   * Loads the named recording from the backend's recordings directory
   * into the main pane.
   */
  async function loadByName(name) {
    if (!name) {
      setStatus('Pick a recording first.', 'error');
      return;
    }
    setStatus('Loading ' + name + ' …');
    try {
      const resp = await fetch('/api/load', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name}),
      });
      const data = await fetchJsonOrThrow(resp);
      // Ensure the active-row highlight tracks the just-loaded file
      // even if the server's response omitted `recording_name`.
      if (!data.recording_name) data.recording_name = name;
      render(data);
      setStatus(
          'Loaded ' + (data.agents?.length || 0) + ' agent(s).', 'ok');
    } catch (e) {
      setStatus('Load failed: ' + e.message, 'error');
      console.error(e);
    }
  }

  /**
   * If the shell handed us a `recording=<name>` query in the URL
   * hash (e.g. `#/viewer?recording=session_20240101.pb` produced by
   * the chat UI's "Open Recording viewer" action), preselect and
   * load that recording on initial render.
   */
  function getRequestedRecordingFromHash() {
    const hash = window.location.hash || '';
    const qIdx = hash.indexOf('?');
    if (qIdx === -1) return '';
    const params = new URLSearchParams(hash.slice(qIdx + 1));
    return params.get('recording') || '';
  }

  async function uploadFile(file) {
    uploadBtn.disabled = true;
    setStatus('Uploading ' + file.name + ' (' + fmtBytes(file.size) + ') …');
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      const resp = await fetch('/api/upload', {method: 'POST', body: fd});
      const data = await fetchJsonOrThrow(resp);
      render(data);
      setStatus(
          'Loaded ' + (data.agents?.length || 0) +
          ' agent(s) from upload.',
          'ok');
    } catch (e) {
      setStatus('Upload failed: ' + e.message, 'error');
      console.error(e);
    } finally {
      uploadBtn.disabled = false;
    }
  }

  // ===========================================================================
  // Global mode toggle
  // ===========================================================================
  let _lastAgentsData = null; // cached for re-render on mode switch.

  globalModeToggle.querySelectorAll('.segmented-option').forEach((btn) => {
    btn.addEventListener('click', () => {
      const newMode = btn.dataset.value;
      if (newMode === globalMode) return;
      globalMode = newMode;
      // Update global toggle active state.
      globalModeToggle.querySelectorAll('.segmented-option').forEach((o) => {
        o.classList.toggle('is-active', o.dataset.value === newMode);
      });
      // Clear per-agent overrides so all agents follow the new global.
      agentModeOverride.clear();
      // Re-render all agents.
      if (_lastAgentsData) render(_lastAgentsData);
    });
  });

  // ===========================================================================
  // Event wiring
  // ===========================================================================
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => loadRecordingsList());
  }
  if (uploadBtn) {
    uploadBtn.addEventListener('click', (e) => {
      e.preventDefault();
      uploadInput.click();
    });
  }
  if (uploadInput) {
    uploadInput.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) uploadFile(f);
      e.target.value = '';
    });
  }

  // When the parent shell re-routes to `#/viewer?recording=<name>`,
  // honor the new query and auto-load it.
  window.addEventListener('hashchange', () => {
    const requested = getRequestedRecordingFromHash();
    if (requested && requested !== currentRecordingName) {
      loadByName(requested);
    }
  });

  loadInitial();
});
