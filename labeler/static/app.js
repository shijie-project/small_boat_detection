/* The review GUI: a main canvas on the left, and on the right one magnified crop per box.
   The boxes are 7 px across at the median, so the crops are where the actual work happens --
   they are editable in place, and the canvas follows whatever the grid selects.

   With a predictions file loaded, a model's detections are shown too, read-only and dashed
   (ship-pred). Their ids are negative, so an id alone says which of the two a grid item is.
   Each image pairs detections with annotations greedily: detections in falling score order, each
   to the still-unpaired annotation it overlaps most, if that IoU reaches the pairing threshold. */

const $ = (s) => document.querySelector(s);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const S = {
  images: [], imgById: new Map(),
  anns: [], annById: new Map(),
  status: {},                    // annotation id -> 'ok' | 'flag'
  edited: new Set(), deleted: [],
  curImage: null, selId: null,
  scope: 'image', filter: 'all', sort: 'default', cell: 120,
  nextId: 1, categoryId: 1,
  undo: [], redo: [], dirty: false, saveTimer: null,
  // predictions (read-only) and their pairing with the annotations
  hasPred: false, preds: [], predById: new Map(), predsByImage: new Map(),
  showPred: true, scoreThr: 0.5, pairMin: 0.1,
  gtPair: new Map(),             // annotation id -> { pid, iou }
  predPair: new Map(),           // prediction id -> { gid, iou }
};

/* Header controls keep the keyboard focus after use, and the shortcuts ignore keys typed into a
   control; a crop swallows its pointerdown (to drag the box), so the focus never leaves by itself.
   Every click on a crop or the canvas hands the keyboard back to the page. */
function releaseFocus() {
  const el = document.activeElement;
  if (el && el !== document.body && el.matches('input, select, textarea, button')) el.blur();
}

const TP_IOU = 0.5;              // the TP / FP / FN in the header count pairs at IoU >= 0.5
const isPred = (id) => id < 0;
const itemOf = (id) => (isPred(id) ? S.predById.get(id) : S.annById.get(id));

const view = { scale: 1, ox: 0, oy: 0 };   // canvas top-left in image coordinates
const imgCache = new Map();
const canvas = $('#canvas');
const ctx = canvas.getContext('2d');

/* ---------------------------------------------------------------- state */

async function boot(st) {
  st = st || await (await fetch('/api/state')).json();
  S.paths = st.paths;
  S.images = st.images;
  S.images.forEach((im) => S.imgById.set(im.id, im));
  S.anns = st.annotations;
  S.anns.forEach((a) => S.annById.set(a.id, a));
  S.status = st.review.status || {};
  S.deleted = st.review.deleted || [];
  S.edited = new Set(st.review.edited || []);
  // past the deleted and edited ids too: a box drawn after a reload must not take the id of one
  // deleted before it, or the review record would name two different boxes by one id
  S.nextId = 1 + Math.max(0, ...S.anns.map((a) => a.id), ...S.deleted.map((d) => d.id), ...S.edited);
  S.categoryId = st.categories[0]?.id ?? 1;
  S.undo.length = S.redo.length = 0;
  S.dirty = false;
  loadPredictions(st);
  imgCache.clear();
  document.title = `ship box review — ${base(st.paths.coco)}`;
  $('#source').textContent = `${base(st.paths.coco)}${S.hasPred ? ' + ' + base(st.paths.pred) : ''}  ←  ${base(st.paths.images)}\\`;
  $('#source').title =
    `Annotations: ${st.paths.coco}\nImages: ${st.paths.images}\nWrites to: ${st.paths.out}\n` +
    (S.hasPred ? `Predictions: ${st.paths.pred} (${st.pred.n_kept} of ${st.pred.n_read}, score ≥ ${st.pred.min_score})\n` : '') +
    `Browsable from: ${st.paths.root}`;
  const warn = st.missing_images?.length
    ? `⚠ ${st.missing_images[0]} and others are not in the image folder`
    : st.pred?.n_unknown_image
      ? `⚠ ${st.pred.n_unknown_image} predictions have an image_id not in the annotation file (not loaded)`
      : '';
  setSaveState(warn || '—', warn ? 'error' : '');
  renderImageList();
  selectImage(S.images.find((im) => im.n_boxes > 0)?.id ?? S.images[0].id);
}

const base = (p) => String(p).split(/[\\/]/).pop();
const plural = (n, one, many = one + 's') => `${n} ${n === 1 ? one : many}`;

function annsOf(imageId) {
  return S.anns.filter((a) => a.image_id === imageId);
}

/* ----------------------------------------------------------- predictions */

function loadPredictions(st) {
  S.preds = st.predictions || [];
  S.predById = new Map(S.preds.map((p) => [p.id, p]));
  S.predsByImage = new Map();
  for (const p of S.preds) {
    if (!S.predsByImage.has(p.image_id)) S.predsByImage.set(p.image_id, []);
    S.predsByImage.get(p.image_id).push(p);
  }
  S.hasPred = Boolean(st.pred?.path);
  $('#predctl').hidden = !S.hasPred;
  for (const el of document.querySelectorAll('.predonly, .predonly option')) {
    el.hidden = !S.hasPred;
    el.disabled = !S.hasPred;
  }
  if (!S.hasPred) {                       // a filter or sort that needs predictions falls back
    if (!['all', 'unreviewed', 'ok', 'flag', 'edited'].includes(S.filter)) S.filter = $('#filter').value = 'all';
    if (!['default', 'size-asc', 'size-desc'].includes(S.sort)) S.sort = $('#sort').value = 'default';
  }
  computeAllMatches();
}

/* the detections of an image that are drawn and paired: those at or above the score threshold */
function predsOf(imageId) {
  return (S.predsByImage.get(imageId) || []).filter((p) => p.score >= S.scoreThr);
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter > 0 ? inter / (a[2] * a[3] + b[2] * b[3] - inter) : 0;
}

function matchImage(imageId) {
  const gts = annsOf(imageId);
  for (const g of gts) S.gtPair.delete(g.id);
  for (const p of S.predsByImage.get(imageId) || []) S.predPair.delete(p.id);
  if (!S.hasPred) return;
  const used = new Set();
  for (const p of predsOf(imageId).sort((a, b) => b.score - a.score)) {
    let best = null, bestIou = 0;
    for (const g of gts) {
      if (used.has(g.id)) continue;
      const v = iou(p.bbox, g.bbox);
      if (v > bestIou) { bestIou = v; best = g; }
    }
    if (best && bestIou >= S.pairMin) {
      used.add(best.id);
      S.gtPair.set(best.id, { pid: p.id, iou: bestIou });
      S.predPair.set(p.id, { gid: best.id, iou: bestIou });
    }
  }
}

function computeAllMatches() {
  S.gtPair.clear();
  S.predPair.clear();
  for (const im of S.images) matchImage(im.id);
}

/* the partner of a grid item: an annotation's paired detection, or a detection's paired annotation */
function partnerOf(id) {
  if (isPred(id)) { const m = S.predPair.get(id); return m && { item: S.annById.get(m.gid), iou: m.iou }; }
  const m = S.gtPair.get(id);
  return m && { item: S.predById.get(m.pid), iou: m.iou };
}

function updatePredStats() {
  if (!S.hasPred) return;
  const nPred = S.preds.filter((p) => p.score >= S.scoreThr).length;
  let tp = 0;
  for (const a of S.anns) if ((S.gtPair.get(a.id)?.iou ?? 0) >= TP_IOU) tp++;
  const fp = nPred - tp, fn = S.anns.length - tp;
  const pr = nPred ? tp / nPred : 0, rc = S.anns.length ? tp / S.anns.length : 0;
  $('#predstats').innerHTML =
    `Predictions <b>${nPred}</b> · IoU≥${TP_IOU}: TP <b>${tp}</b> FP <b>${fp}</b> FN <b>${fn}</b> · ` +
    `P <b>${pr.toFixed(3)}</b> R <b>${rc.toFixed(3)}</b>`;
}

function acceptPred(id) {
  const p = S.predById.get(id);
  if (!p) return;
  pushUndo();
  const a = {
    id: S.nextId++, image_id: p.image_id, category_id: S.categoryId, bbox: [...p.bbox],
    area: Math.round(p.bbox[2] * p.bbox[3] * 100) / 100, iscrowd: 0, ignore: 0, segmentation: [],
  };
  S.anns.push(a);
  S.annById.set(a.id, a);
  S.edited.add(a.id);
  S.status[a.id] = 'ok';                 // a detection you accepted is a box you looked at
  matchImage(p.image_id);
  S.selId = a.id;
  renderAll();
  markDirty();
}

function statusOf(id) {
  return S.status[id] || 'unreviewed';
}

function snapshot() {
  return JSON.stringify({ a: S.anns, s: S.status, d: S.deleted, e: [...S.edited] });
}

function pushUndo() {
  S.undo.push(snapshot());
  if (S.undo.length > 60) S.undo.shift();
  S.redo.length = 0;
}

function restore(json) {
  const o = JSON.parse(json);
  S.anns = o.a;
  S.annById = new Map(S.anns.map((a) => [a.id, a]));
  S.status = o.s;
  S.deleted = o.d;
  S.edited = new Set(o.e);
  if (!itemOf(S.selId)) S.selId = null;
  computeAllMatches();
  renderAll();
  markDirty();
}

function undo() { if (S.undo.length) { S.redo.push(snapshot()); restore(S.undo.pop()); } }
function redo() { if (S.redo.length) { S.undo.push(snapshot()); restore(S.redo.pop()); } }

function markDirty() {
  S.dirty = true;
  setSaveState('Unsaved', 'dirty');
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, 2000);
}

function setSaveState(text, cls = '') {
  const el = $('#savestate');
  el.textContent = text;
  el.className = 'savestate ' + cls;
}

async function save() {
  clearTimeout(S.saveTimer);
  try {
    const body = {
      annotations: S.anns,
      review: { status: S.status, deleted: S.deleted, edited: [...S.edited] },
    };
    const r = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    S.dirty = false;
    setSaveState('Saved ' + new Date().toLocaleTimeString());
  } catch (e) {
    setSaveState('Save failed: ' + e.message, 'error');
  }
}

/* ------------------------------------------------------------ image list */

function renderImageList() {
  const q = $('#imagesearch').value.trim().toLowerCase();
  const host = $('#images');
  host.innerHTML = '';
  const frag = document.createDocumentFragment();
  for (const im of S.images) {
    if (q && !im.file_name.toLowerCase().includes(q)) continue;
    const boxes = annsOf(im.id);
    const done = boxes.filter((a) => statusOf(a.id) !== 'unreviewed').length;
    const flagged = boxes.some((a) => statusOf(a.id) === 'flag');
    const row = document.createElement('div');
    row.className = 'imgrow' + (im.id === S.curImage ? ' active' : '');
    row.dataset.id = im.id;
    const cls = flagged ? 'flag' : !boxes.length ? 'none' : done === boxes.length ? 'done' : done ? 'part' : 'none';
    let pm = '';
    if (S.hasPred) {                      // unpaired annotations (missed) and detections (false alarms)
      const fn = boxes.filter((a) => !S.gtPair.has(a.id)).length;
      const fp = predsOf(im.id).filter((p) => !S.predPair.has(p.id)).length;
      if (fn || fp) pm = `<span class="pm" title="annotations without a prediction / predictions without an annotation">FN ${fn} FP ${fp}</span>`;
    }
    row.innerHTML = `<span class="dot ${cls}"></span>
      <span class="name">${im.file_name}</span>${pm}
      <span class="count">${done}/${boxes.length}</span>`;
    row.onclick = () => selectImage(im.id);
    frag.appendChild(row);
  }
  host.appendChild(frag);
}

function selectImage(id) {
  S.curImage = id;
  S.selId = null;
  loadImage(S.imgById.get(id).file_name).then(() => { fitView(); draw(); });
  renderImageList();
  renderGrid();
  const row = $(`#images .imgrow[data-id="${id}"]`);
  if (row) row.scrollIntoView({ block: 'nearest' });
}

function loadImage(name) {
  if (imgCache.has(name)) return Promise.resolve(imgCache.get(name));
  return new Promise((res) => {
    const im = new Image();
    im.onload = () => { imgCache.set(name, im); res(im); };
    im.onerror = () => res(null);
    im.src = '/images/' + encodeURIComponent(name);
  });
}

/* --------------------------------------------------------- main canvas */

function fitView() {
  const im = S.imgById.get(S.curImage);
  const r = canvas.getBoundingClientRect();
  view.scale = Math.min(r.width / im.width, r.height / im.height);
  view.ox = (im.width - r.width / view.scale) / 2;
  view.oy = (im.height - r.height / view.scale) / 2;
}

function centerOn(a, zoomTo = true) {
  const r = canvas.getBoundingClientRect();
  if (zoomTo) {
    const side = Math.max(a.bbox[2], a.bbox[3], 1);
    view.scale = clamp(180 / side, 1, 40);      // a 7 px box lands at ~180 px on screen
  }
  view.ox = a.bbox[0] + a.bbox[2] / 2 - r.width / view.scale / 2;
  view.oy = a.bbox[1] + a.bbox[3] / 2 - r.height / view.scale / 2;
  draw();
}

const toScreen = (x, y) => [(x - view.ox) * view.scale, (y - view.oy) * view.scale];
const toWorld = (sx, sy) => [sx / view.scale + view.ox, sy / view.scale + view.oy];

function draw() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);

  const meta = S.imgById.get(S.curImage);
  const img = imgCache.get(meta.file_name);
  if (img) {
    ctx.imageSmoothingEnabled = view.scale < 2;
    const [sx, sy] = toScreen(0, 0);
    ctx.drawImage(img, sx, sy, meta.width * view.scale, meta.height * view.scale);
  }

  ctx.lineWidth = 1.5;
  for (const a of annsOf(S.curImage)) {
    const [x, y] = toScreen(a.bbox[0], a.bbox[1]);
    const w = a.bbox[2] * view.scale;
    const h = a.bbox[3] * view.scale;
    const sel = a.id === S.selId;
    ctx.strokeStyle = sel ? '#ffffff' : { ok: '#3ddc84', flag: '#ff5f57', unreviewed: '#f5b93b' }[statusOf(a.id)];
    ctx.strokeRect(x, y, w, h);
    if (sel) {
      ctx.fillStyle = '#ffffff';
      for (const [hx, hy] of corners(x, y, w, h)) ctx.fillRect(hx - 3, hy - 3, 6, 6);
    } else if (w < 6 && h < 6) {
      ctx.strokeRect(x - 4, y - 4, w + 8, h + 8);   // a halo, so a 3 px box is findable at all
    }
  }
  if (S.hasPred && S.showPred) {
    ctx.setLineDash([4, 3]);
    ctx.font = '11px "Segoe UI", sans-serif';
    for (const p of predsOf(S.curImage)) {
      const [x, y] = toScreen(p.bbox[0], p.bbox[1]);
      const w = p.bbox[2] * view.scale, h = p.bbox[3] * view.scale;
      ctx.strokeStyle = p.id === S.selId ? '#ffffff' : '#35c7ff';
      ctx.strokeRect(x, y, w, h);
      if (w < 6 && h < 6) ctx.strokeRect(x - 5, y - 5, w + 10, h + 10);
      if (view.scale >= 3 || p.id === S.selId) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(`ship-pred ${p.score.toFixed(2)}`, x, y - 3);
      }
    }
    ctx.setLineDash([]);
  }
  if (drag.mode === 'new' && drag.rect) {
    const [x, y] = toScreen(drag.rect[0], drag.rect[1]);
    ctx.strokeStyle = newBoxProblem(drag.rect) === 'thin' ? '#ff5f57' : '#4aa3ff';   // red: will not be kept
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(x, y, drag.rect[2] * view.scale, drag.rect[3] * view.scale);
    ctx.setLineDash([]);
  }
  const sel = itemOf(S.selId);
  const pair = sel && partnerOf(sel.id);
  $('#stageinfo').textContent = [
    `${meta.file_name}`,
    plural(annsOf(S.curImage).length, 'box', 'boxes'),
    S.hasPred ? plural(predsOf(S.curImage).length, 'prediction') : '',
    `${Math.round(view.scale * 100)}%`,
    sel ? `selected ${isPred(sel.id) ? `prediction ${sel.score.toFixed(3)}` : `#${sel.id}`} ${sel.bbox.map((v) => Math.round(v)).join(', ')}` : '',
    sel && S.hasPred ? (pair ? `IoU ${pair.iou.toFixed(2)}` : isPred(sel.id) ? 'no annotation' : 'no prediction') : '',
  ].filter(Boolean).join('  ·  ');
}

const corners = (x, y, w, h) => [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];

/* The selected box's handles on the canvas: a corner (0-3, as corners() lists them) within 8 px,
   or a side ('l', 'r', 't', 'b') within 5 px of it. The sides are only offered on a box at least
   24 px across on screen: on a smaller one they would leave nothing in the middle to move it by. */
const HANDLE_CURSOR = {
  0: 'nwse-resize', 3: 'nwse-resize', 1: 'nesw-resize', 2: 'nesw-resize',
  l: 'ew-resize', r: 'ew-resize', t: 'ns-resize', b: 'ns-resize',
};

function handleAt(sx, sy) {
  const sel = S.annById.get(S.selId);
  if (!sel || sel.image_id !== S.curImage) return null;
  const [x, y] = toScreen(sel.bbox[0], sel.bbox[1]);
  const w = sel.bbox[2] * view.scale, h = sel.bbox[3] * view.scale;
  const cs = corners(x, y, w, h);
  for (let i = 0; i < 4; i++) if (Math.hypot(cs[i][0] - sx, cs[i][1] - sy) < 8) return i;
  if (w < 24 || h < 24) return null;
  const inX = sx > x && sx < x + w, inY = sy > y && sy < y + h;
  if (inY && Math.abs(sx - x) < 5) return 'l';
  if (inY && Math.abs(sx - x - w) < 5) return 'r';
  if (inX && Math.abs(sy - y) < 5) return 't';
  if (inX && Math.abs(sy - y - h) < 5) return 'b';
  return null;
}

/* What a press here would do, shown before it is done: the canvas cursor is the only sign that
   a box on it can be grabbed at all. */
function hoverCursor(sx, sy) {
  const h = handleAt(sx, sy);
  if (h !== null) return HANDLE_CURSOR[h];
  const [wx, wy] = toWorld(sx, sy);
  if (hitTest(wx, wy)) return 'move';
  if (S.hasPred && S.showPred && hitTest(wx, wy, predsOf(S.curImage))) return 'pointer';
  return 'crosshair';
}

/* dragging on the canvas: pan, move, resize, or draw a new box */
const drag = { mode: null, id: null, corner: null, start: null, orig: null, rect: null };

/* A rectangle dragged on empty space becomes a box only if it could be a ship: at least 3 screen
   px a side, so a click that slips is not a box, and at most 12 times as long as it is wide. The
   tier_b boxes stay under 5.5 to 1; a long sliver is a swipe across the image (a pan tried with
   the left button), which is how 116.8 x 1.4 and 304.6 x 2.4 px boxes got drawn on image 0. */
const NEW_MIN_PX = 3, NEW_MAX_RATIO = 12;

function newBoxProblem([, , w, h]) {
  if (w * view.scale < NEW_MIN_PX || h * view.scale < NEW_MIN_PX) return 'small';
  if (Math.max(w, h) / Math.min(w, h) > NEW_MAX_RATIO) return 'thin';
  return null;
}

/* The hint in the canvas corner says for a moment why nothing was drawn. */
function flashHint(msg) {
  const el = $('#stagehint');
  el.dataset.text ??= el.textContent;
  el.textContent = msg;
  el.classList.add('warn');
  clearTimeout(flashHint.timer);
  flashHint.timer = setTimeout(() => { el.textContent = el.dataset.text; el.classList.remove('warn'); }, 3000);
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('pointerdown', (e) => {
  releaseFocus();
  canvas.setPointerCapture(e.pointerId);
  const r = canvas.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [wx, wy] = toWorld(sx, sy);
  drag.start = [sx, sy, wx, wy];

  if (e.button === 2 || e.button === 1 || e.altKey) {
    drag.mode = 'pan';
    drag.orig = [view.ox, view.oy];
    return;
  }
  const handle = handleAt(sx, sy);                // a corner or side of the selected box?
  if (handle !== null) {
    const sel = S.annById.get(S.selId);
    drag.mode = 'resize'; drag.id = sel.id; drag.corner = handle; drag.orig = [...sel.bbox];
    pushUndo();
    return;
  }
  const hit = hitTest(wx, wy);
  const predHit = !hit && S.hasPred && S.showPred && hitTest(wx, wy, predsOf(S.curImage));
  if (hit) {
    selectAnn(hit.id, false);
    drag.mode = 'move'; drag.id = hit.id; drag.orig = [...hit.bbox];
    pushUndo();
  } else if (predHit) {                    // a detection is read-only: selecting it is all a click does
    selectAnn(predHit.id, false);
    drag.mode = null;
  } else {
    drag.mode = 'new'; drag.rect = [wx, wy, 0, 0];
  }
});

canvas.addEventListener('pointermove', (e) => {
  const r = canvas.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (!drag.mode) { canvas.style.cursor = hoverCursor(sx, sy); return; }
  const [s0x, s0y, w0x, w0y] = drag.start;
  const dxw = (sx - s0x) / view.scale, dyw = (sy - s0y) / view.scale;

  if (drag.mode === 'pan') {
    view.ox = drag.orig[0] - dxw;
    view.oy = drag.orig[1] - dyw;
  } else if (drag.mode === 'move') {
    const a = S.annById.get(drag.id);
    a.bbox[0] = drag.orig[0] + dxw;
    a.bbox[1] = drag.orig[1] + dyw;
    touched(a);
  } else if (drag.mode === 'resize') {
    const a = S.annById.get(drag.id);
    resizeHandle(a, drag.orig, drag.corner, dxw, dyw);
    touched(a);
  } else if (drag.mode === 'new') {
    const [wx, wy] = toWorld(sx, sy);
    drag.rect = [Math.min(w0x, wx), Math.min(w0y, wy), Math.abs(wx - w0x), Math.abs(wy - w0y)];
  }
  draw();
});

canvas.addEventListener('pointerup', () => {
  const problem = drag.mode === 'new' && drag.rect ? newBoxProblem(drag.rect) : 'none';
  if (problem === 'thin') {
    const [w, h] = drag.rect.slice(2).map((v) => v.toFixed(1));
    flashHint(`Not drawn: ${w} × ${h} px is too thin for a ship (over ${NEW_MAX_RATIO}:1). Pan with right-drag or Alt+drag`);
  }
  if (problem === null) {
    pushUndo();
    const a = {
      id: S.nextId++, image_id: S.curImage, category_id: S.categoryId,
      bbox: drag.rect.map((v) => Math.round(v * 10) / 10),
      area: 0, iscrowd: 0, ignore: 0, segmentation: [],
    };
    S.anns.push(a);
    S.annById.set(a.id, a);
    S.edited.add(a.id);
    S.status[a.id] = 'ok';               // a box you just drew is a box you just looked at
    S.selId = a.id;
    matchImage(a.image_id);
    renderGrid();
    renderImageList();
    markDirty();
  } else if (drag.mode === 'move' || drag.mode === 'resize') {
    renderGrid();
    if (S.hasPred) renderImageList();
  }
  drag.mode = null; drag.rect = null;
  draw();
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [wx, wy] = toWorld(sx, sy);
  view.scale = clamp(view.scale * (e.deltaY < 0 ? 1.2 : 1 / 1.2), 0.05, 80);
  view.ox = wx - sx / view.scale;         // keep the point under the cursor fixed
  view.oy = wy - sy / view.scale;
  draw();
}, { passive: false });

function hitTest(wx, wy, boxes = annsOf(S.curImage)) {
  const pad = 4 / view.scale;             // tiny boxes need a hit area larger than themselves
  const hits = boxes.filter((a) => {
    const [x, y, w, h] = a.bbox;
    return wx >= x - pad && wx <= x + w + pad && wy >= y - pad && wy <= y + h + pad;
  });
  return hits.sort((a, b) => a.bbox[2] * a.bbox[3] - b.bbox[2] * b.bbox[3])[0];   // smallest first
}

/* A corner (0-3) moves two sides, a side ('l', 'r', 't', 'b') one. */
function resizeHandle(a, orig, handle, dx, dy) {
  let [x, y, w, h] = orig;
  if (handle === 0 || handle === 2 || handle === 'l') { x += dx; w -= dx; }
  if (handle === 1 || handle === 3 || handle === 'r') w += dx;
  if (handle === 0 || handle === 1 || handle === 't') { y += dy; h -= dy; }
  if (handle === 2 || handle === 3 || handle === 'b') h += dy;
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  a.bbox = [x, y, Math.max(1, w), Math.max(1, h)];
}

function touched(a) {
  a.bbox = a.bbox.map((v) => Math.round(v * 10) / 10);
  a.area = Math.round(a.bbox[2] * a.bbox[3] * 100) / 100;
  S.edited.add(a.id);
  if (S.hasPred) { matchImage(a.image_id); updatePredStats(); }   // the pairing follows the edited box
  updateCell(a.id);
  markDirty();
}

/* ------------------------------------------------------------- the grid */

function gridItems() {
  let items = S.scope === 'image' ? annsOf(S.curImage) : S.anns.slice();
  const preds = () => (S.scope === 'image' ? predsOf(S.curImage) : S.preds.filter((p) => p.score >= S.scoreThr));
  if (S.filter === 'unreviewed') items = items.filter((a) => statusOf(a.id) === 'unreviewed');
  else if (S.filter === 'ok') items = items.filter((a) => statusOf(a.id) === 'ok');
  else if (S.filter === 'flag') items = items.filter((a) => statusOf(a.id) === 'flag');
  else if (S.filter === 'edited') items = items.filter((a) => S.edited.has(a.id));
  else if (S.filter === 'fn') items = items.filter((a) => !S.gtPair.has(a.id));
  else if (S.filter === 'fp') items = preds().filter((p) => !S.predPair.has(p.id));
  else if (S.filter === 'preds') items = preds();
  else if (S.filter.startsWith('iou:')) {       // pairs, shown as their annotation with the detection over it
    const [lo, hi] = S.filter.slice(4).split(':').map(Number);
    items = items.filter((a) => { const m = S.gtPair.get(a.id); return m && m.iou >= lo && m.iou < hi; });
  }
  const size = (a) => Math.max(a.bbox[2], a.bbox[3]);
  const pairIou = (a) => partnerOf(a.id)?.iou ?? 0;
  const score = (a) => (isPred(a.id) ? a.score : partnerOf(a.id)?.item?.score ?? -1);
  if (S.sort === 'size-asc') items.sort((a, b) => size(a) - size(b));
  if (S.sort === 'size-desc') items.sort((a, b) => size(b) - size(a));
  if (S.sort === 'iou-asc') items.sort((a, b) => pairIou(a) - pairIou(b));
  if (S.sort === 'score-desc') items.sort((a, b) => score(b) - score(a));
  if (S.sort === 'score-asc') items.sort((a, b) => score(a) - score(b));
  return items;
}

let observer = null;

function renderGrid() {
  const host = $('#grid');
  host.style.setProperty('--cell', S.cell + 'px');
  host.innerHTML = '';
  if (observer) observer.disconnect();
  // only what is on screen keeps a decoded bitmap: 1688 crops over 914 images would not fit
  observer = new IntersectionObserver((entries) => {
    for (const en of entries) {
      const img = en.target.querySelector('img');
      if (en.isIntersecting) {
        if (!img.src) img.src = '/images/' + encodeURIComponent(en.target.dataset.file);
        layoutCell(en.target);
      } else if (img.src) {
        img.removeAttribute('src');
      }
    }
  }, { root: host, rootMargin: '500px' });

  const items = gridItems();
  const frag = document.createDocumentFragment();
  for (const a of items) frag.appendChild(makeCell(a));
  host.appendChild(frag);
  host.querySelectorAll('.cell').forEach((c) => observer.observe(c));
  updateHead(items);
  updateProgress();
}

function makeCell(a) {
  const meta = S.imgById.get(a.image_id);
  const el = document.createElement('div');
  el.className = 'cell' + (isPred(a.id) ? ' pred' : '');
  el.dataset.id = a.id;
  el.dataset.file = meta.file_name;
  el.innerHTML = `<img alt="" draggable="false" style="width:${meta.width}px;height:${meta.height}px">
    <div class="pbox" hidden></div>
    <div class="box"><i class="hd nw"></i><i class="hd ne"></i><i class="hd sw"></i><i class="hd se"></i></div>
    <div class="tag"></div><div class="tag zoomtag"></div>
    ${isPred(a.id) ? '' : '<button class="del" title="Delete this box (D / Delete)">×</button>'}`;
  wireCell(el);
  return el;
}

function placeBox(box, bbox, c, ax, ay, z) {
  box.style.left = `${c / 2 + (bbox[0] - ax) * z}px`;
  box.style.top = `${c / 2 + (bbox[1] - ay) * z}px`;
  box.style.width = `${bbox[2] * z}px`;
  box.style.height = `${bbox[3] * z}px`;
}

/* The crop: the image is scaled by z and translated so the box centre sits in the middle of the
   cell. The anchor stays put while the box is dragged, so a shifted box reads as shifted. */
function layoutCell(el) {
  const a = itemOf(Number(el.dataset.id));
  if (!a) return;
  const pred = isPred(a.id);
  const pair = S.hasPred ? partnerOf(a.id) : null;
  const c = el.clientWidth || S.cell;
  // the crop is framed on the item, and wide enough to show its partner as well
  const frame = pair?.item ? union(a.bbox, pair.item.bbox) : a.bbox;
  const z = clamp((0.45 * c) / Math.max(frame[2], frame[3], 1), 1, 24);
  if (!el.dataset.ax) {
    el.dataset.ax = frame[0] + frame[2] / 2;
    el.dataset.ay = frame[1] + frame[3] / 2;
    el.dataset.z = z;
  }
  const ax = Number(el.dataset.ax), ay = Number(el.dataset.ay), zz = Number(el.dataset.z);
  const img = el.querySelector('img');
  img.style.transform = `translate(${c / 2 - ax * zz}px, ${c / 2 - ay * zz}px) scale(${zz})`;
  placeBox(el.querySelector('.box'), a.bbox, c, ax, ay, zz);
  const pbox = el.querySelector('.pbox');
  pbox.hidden = !(pair?.item && S.showPred);
  if (!pbox.hidden) placeBox(pbox, pair.item.bbox, c, ax, ay, zz);
  el.classList.toggle('sel', a.id === S.selId);
  el.classList.toggle('ok', !pred && statusOf(a.id) === 'ok');
  el.classList.toggle('flag', !pred && statusOf(a.id) === 'flag');
  const wh = `${Math.round(a.bbox[2])}×${Math.round(a.bbox[3])}`;
  const vs = !S.hasPred ? '' : pair ? ` · <span class="iou">IoU ${pair.iou.toFixed(2)}</span>` : pred ? ' · FP' : ' · FN';
  el.querySelector('.tag').innerHTML = pred
    ? `<span class="iou">pred ${a.score.toFixed(2)}</span> · ${wh}${vs}`
    : `${wh}${S.edited.has(a.id) ? ' ✎' : ''}${vs}`;
  el.querySelector('.zoomtag').textContent = `${zz.toFixed(0)}×`;
}

function union(a, b) {
  const x0 = Math.min(a[0], b[0]), y0 = Math.min(a[1], b[1]);
  return [x0, y0, Math.max(a[0] + a[2], b[0] + b[2]) - x0, Math.max(a[1] + a[3], b[1] + b[3]) - y0];
}

function updateCell(id) {
  const el = $(`#grid .cell[data-id="${id}"]`);
  if (el) layoutCell(el);
}

function wireCell(el) {
  const id = Number(el.dataset.id);
  const box = el.querySelector('.box');
  let d = null;
  el.addEventListener('pointerdown', releaseFocus);
  if (isPred(id)) {                        // a detection cannot be edited, only looked at (or accepted with A)
    el.addEventListener('click', () => selectAnn(id, true));
    el.addEventListener('dblclick', () => { selectAnn(id, true); centerOn(itemOf(id), true); });
    return;
  }

  const down = (e, corner) => {
    e.preventDefault();
    e.stopPropagation();                  // (so the crop's own releaseFocus does not run: do it here)
    releaseFocus();
    el.setPointerCapture(e.pointerId);
    selectAnn(id, true);
    pushUndo();
    d = { x: e.clientX, y: e.clientY, orig: [...S.annById.get(id).bbox], corner, z: Number(el.dataset.z) || 1 };
  };
  box.addEventListener('pointerdown', (e) => down(e, null));
  const del = el.querySelector('.del');
  del.addEventListener('pointerdown', (e) => e.stopPropagation());
  del.addEventListener('click', (e) => { e.stopPropagation(); deleteAnn(id); });
  el.querySelectorAll('.hd').forEach((h, i) => h.addEventListener('pointerdown', (e) => down(e, i)));

  el.addEventListener('pointermove', (e) => {
    if (!d) return;
    const a = S.annById.get(id);
    const dx = (e.clientX - d.x) / d.z, dy = (e.clientY - d.y) / d.z;
    if (d.corner === null) a.bbox = [d.orig[0] + dx, d.orig[1] + dy, d.orig[2], d.orig[3]];
    else resizeHandle(a, d.orig, d.corner, dx, dy);
    touched(a);
    draw();
  });
  el.addEventListener('pointerup', () => { d = null; });
  el.addEventListener('click', () => selectAnn(id, true));
  el.addEventListener('dblclick', () => { selectAnn(id, true); centerOn(S.annById.get(id), true); });
}

function updateHead(items) {
  const gts = items.filter((a) => !isPred(a.id));
  const done = gts.filter((a) => statusOf(a.id) !== 'unreviewed').length;
  const where = S.scope === 'image' ? 'This image' : 'All images';
  $('#gridhead').textContent = gts.length === items.length
    ? `${where}: ${plural(items.length, 'box', 'boxes')}, ${done} reviewed`
    : `${where}: ${plural(items.length - gts.length, 'prediction')}`;
}

function updateProgress() {
  const total = S.anns.length;
  const done = S.anns.filter((a) => statusOf(a.id) !== 'unreviewed').length;
  const flag = S.anns.filter((a) => statusOf(a.id) === 'flag').length;
  $('#progress').innerHTML =
    `Reviewed <b>${done}</b>/${total} · flagged <b>${flag}</b> · edited <b>${S.edited.size}</b> · deleted <b>${S.deleted.length}</b>`;
  updatePredStats();
}

/* ------------------------------------------------------------- commands */

function selectAnn(id, fromGrid) {
  const a = itemOf(id);
  if (!a) return;
  if (a.image_id !== S.curImage) {
    S.curImage = a.image_id;
    S.selId = id;
    loadImage(S.imgById.get(a.image_id).file_name).then(() => { centerOn(a, true); });
    renderImageList();
    if (S.scope === 'image') renderGrid();
  } else {
    S.selId = id;
    if (fromGrid) centerOn(a, view.scale < 4);
  }
  S.selId = id;
  document.querySelectorAll('#grid .cell').forEach((el) =>
    el.classList.toggle('sel', Number(el.dataset.id) === id));
  const cell = $(`#grid .cell[data-id="${id}"]`);
  if (cell && !fromGrid) cell.scrollIntoView({ block: 'nearest' });
  draw();
}

function setStatus(id, s) {
  if (!S.annById.has(id)) return;
  pushUndo();
  S.status[id] = s;
  updateCell(id);
  updateProgress();
  updateHead(gridItems());
  renderImageList();
  markDirty();
  draw();
}

function deleteAnn(id) {
  const a = S.annById.get(id);
  if (!a) return;
  pushUndo();
  S.anns = S.anns.filter((x) => x.id !== id);
  S.annById.delete(id);
  S.deleted.push({ id, image_id: a.image_id, bbox: a.bbox });
  delete S.status[id];
  S.gtPair.delete(id);
  matchImage(a.image_id);                // its detection may now pair with a neighbour, or with nothing
  S.selId = null;
  renderGrid();
  renderImageList();
  markDirty();
  draw();
}

function step(delta) {
  const items = gridItems();
  if (!items.length) return;
  const i = items.findIndex((a) => a.id === S.selId);
  const next = items[clamp(i + delta, 0, items.length - 1)] || items[0];
  selectAnn(next.id, true);
}

function nextUnreviewed() {
  const items = gridItems();
  const i = items.findIndex((a) => a.id === S.selId);
  const rest = items.slice(i + 1).concat(items.slice(0, Math.max(i, 0)));
  const n = rest.find((a) => !isPred(a.id) && statusOf(a.id) === 'unreviewed');
  if (n) selectAnn(n.id, true);
}

function stepImage(delta) {
  const i = S.images.findIndex((im) => im.id === S.curImage);
  const next = S.images[clamp(i + delta, 0, S.images.length - 1)];
  if (next) selectImage(next.id);
}

function nudge(dx, dy, resize) {
  const a = S.annById.get(S.selId);
  if (!a) return;
  pushUndo();
  if (resize) {
    a.bbox[2] = Math.max(1, a.bbox[2] + dx);
    a.bbox[3] = Math.max(1, a.bbox[3] + dy);
  } else {
    a.bbox[0] += dx;
    a.bbox[1] += dy;
  }
  touched(a);
  draw();
}

function renderAll() {
  renderImageList();
  renderGrid();
  draw();
}

/* ------------------------------------------------------- source picker */

const P = { cwd: null, coco: null, images: null, out: null, pred: '', outEdited: false };
const pickTarget = () => document.querySelector('input[name="target"]:checked').value;

const dirOf = (p) => String(p).replace(/[\\/][^\\/]*$/, '');
const autoOut = (coco) => (coco ? coco.replace(/\.json$/i, '') + '_reviewed.json' : '');

async function openPicker() {
  P.coco = S.paths.coco;
  P.images = S.paths.images;
  P.out = S.paths.out;
  P.pred = S.paths.pred || '';
  P.outEdited = S.paths.out !== autoOut(S.paths.coco);
  $('#picker').hidden = false;
  pickMsg('Type or paste the paths, or pick them in the browser below');
  showPicked();
  await browse(dirOf(S.paths.coco));            // start in the annotations folder
}

/* The three paths are inputs: browsing fills them in, and typing or pasting a path works just as
   well. ``Write to`` follows the annotation file's name until it is edited by hand. */
function showPicked() {
  $('#pickcoco').value = P.coco || '';
  $('#pickimages').value = P.images || '';
  $('#pickout').value = P.outEdited ? (P.out || '') : autoOut(P.coco);
  $('#pickpred').value = P.pred || '';
  for (const el of document.querySelectorAll('.picked input')) el.classList.remove('bad');
}

function readPicked() {
  P.coco = $('#pickcoco').value.trim();
  P.images = $('#pickimages').value.trim();
  P.pred = $('#pickpred').value.trim();
  const out = $('#pickout').value.trim();
  P.out = out || autoOut(P.coco);
  P.outEdited = !!out && out !== autoOut(P.coco);
}

function pickMsg(msg, cls = '') {
  const el = $('#pickmsg');
  el.textContent = String(msg).slice(0, 220);
  el.className = cls;
}

async function browse(path) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(path));
  if (!r.ok) { pickMsg(detail(await r.text()), 'error'); return; }
  const d = await r.json();
  P.cwd = d.path;
  renderCrumbs(d);
  const host = $('#browser');
  host.innerHTML = '';
  const frag = document.createDocumentFragment();
  if (d.parent) frag.appendChild(row('📁', '..', () => browse(d.parent)));
  for (const dir of d.dirs) {
    const el = row('📁', dir.name, () => browse(dir.path));
    el.classList.toggle('chosen', dir.path === P.images);
    frag.appendChild(el);
  }
  for (const f of d.files) {
    const pick = () => {                  // the radio above says which of the two a clicked json fills
      readPicked();
      if (pickTarget() === 'pred') P.pred = f.path; else P.coco = f.path;
      showPicked();
      browse(P.cwd);
    };
    const el = row(f.path === P.pred ? '🔷' : '📄', f.name, pick, `${(f.size / 1024).toFixed(0)} KB`);
    el.classList.toggle('chosen', f.path === P.coco || f.path === P.pred);
    frag.appendChild(el);
  }
  host.appendChild(frag);
}

function row(ico, name, onclick, size = '') {
  const el = document.createElement('div');
  el.className = 'brow';
  el.innerHTML = `<span class="ico">${ico}</span><span class="nm"></span><span class="sz">${size}</span>`;
  el.querySelector('.nm').textContent = name;
  el.onclick = onclick;
  return el;
}

function renderCrumbs(d) {
  const host = $('#crumbs');
  host.innerHTML = '';
  const parts = d.path.split(/[\\/]/);
  // the folders above --root are shown for orientation only: the server refuses to list them
  const top = d.root.split(/[\\/]/).filter(Boolean).length - 1;
  parts.forEach((p, i) => {
    const full = parts.slice(0, i + 1).join('\\');
    const el = document.createElement('span');
    el.className = 'crumb';
    el.textContent = p || '\\';
    if (i < top) {
      el.classList.add('above');
      el.title = `outside --root (${d.root}); restart the server with a wider --root to browse it`;
    } else {
      el.onclick = () => browse(/:$/.test(full) ? full + '\\' : full);  // "C:" alone is C:'s cwd
    }
    host.appendChild(el);
    if (i < parts.length - 1) host.appendChild(document.createTextNode(' \\ '));
  });
  const n = document.createElement('span');
  n.className = 'n';
  // this folder holds images, so it is a candidate for the image folder itself
  n.innerHTML = d.n_images
    ? `${d.n_images} images <span class="use">use this folder for the images</span>`
    : '(no images here)';
  n.onclick = () => { if (d.n_images) { readPicked(); P.images = d.path; showPicked(); } };
  n.style.cursor = d.n_images ? 'pointer' : 'default';
  host.appendChild(n);
}

/* The server answers an error as {"detail": "..."}; anything else is shown as it came. */
function detail(text) {
  try { return JSON.parse(text).detail ?? text; } catch { return text; }
}

async function applyPicker() {
  readPicked();
  if (!P.coco || !P.images) {
    pickMsg('Both the annotation file and the image folder are needed', 'error');
    $('#pickcoco').classList.toggle('bad', !P.coco);
    $('#pickimages').classList.toggle('bad', !P.images);
    return;
  }
  if (S.dirty) await save();
  pickMsg('Opening…');
  const r = await fetch('/api/open', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ coco: P.coco, images: P.images, out: P.out, pred: P.pred || null }),
  });
  if (!r.ok) { pickMsg(detail(await r.text()), 'error'); return; }
  $('#picker').hidden = true;
  await boot(await r.json());
}

/* ------------------------------------------------------------ splitters */

/* The image list and the grid are as wide as their inner edge is dragged, and a double-click on
   the edge gives back the default. The widths are remembered by this browser, nowhere else. */
const PANE_MIN = 160, STAGE_MIN = 320;

function setPaneWidth(side, px) {
  if (px == null) $('main').style.removeProperty(`--${side}-w`);
  else $('main').style.setProperty(`--${side}-w`, `${px}px`);
  if (S.curImage != null) draw();          // the canvas takes whatever width is left
}

function rememberPaneWidth(side, px) {
  try {
    if (px == null) localStorage.removeItem(`labeler.${side}-w`);
    else localStorage.setItem(`labeler.${side}-w`, String(px));
  } catch { /* private window or blocked storage: the width just is not remembered */ }
}

for (const h of document.querySelectorAll('.splitter')) {
  const side = h.dataset.side;
  const pane = h.parentElement;
  const other = side === 'left' ? $('#gridpane') : $('#imagelist');
  h.onpointerdown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    h.setPointerCapture(e.pointerId);
    h.classList.add('drag');
    document.body.classList.add('resizing');
    const x0 = e.clientX, w0 = pane.getBoundingClientRect().width;
    let px = Math.round(w0);
    h.onpointermove = (m) => {
      const dx = side === 'left' ? m.clientX - x0 : x0 - m.clientX;
      const max = $('main').clientWidth - other.getBoundingClientRect().width - STAGE_MIN;
      px = Math.round(clamp(w0 + dx, PANE_MIN, Math.max(PANE_MIN, max)));
      setPaneWidth(side, px);
    };
    h.onpointerup = h.onpointercancel = () => {
      h.onpointermove = h.onpointerup = h.onpointercancel = null;
      h.classList.remove('drag');
      document.body.classList.remove('resizing');
      rememberPaneWidth(side, px);
    };
  };
  h.ondblclick = () => { setPaneWidth(side, null); rememberPaneWidth(side, null); };
  let saved = 0;
  try { saved = Number(localStorage.getItem(`labeler.${side}-w`)) || 0; } catch { /* as above */ }
  // a width kept from a wider window must still leave the canvas room
  if (saved) setPaneWidth(side, Math.round(clamp(saved, PANE_MIN, window.innerWidth * 0.45)));
}

/* --------------------------------------------------------------- wiring */

$('#scope').onchange = (e) => { S.scope = e.target.value; renderGrid(); e.target.blur(); };
$('#filter').onchange = (e) => { S.filter = e.target.value; renderGrid(); e.target.blur(); };
$('#sort').onchange = (e) => { S.sort = e.target.value; renderGrid(); e.target.blur(); };
$('#cellzoom').oninput = (e) => { S.cell = +e.target.value; renderGrid(); };
$('#imagesearch').oninput = renderImageList;
$('#save').onclick = save;
$('#source').onclick = openPicker;
$('#pickcancel').onclick = () => ($('#picker').hidden = true);
$('#pickopen').onclick = applyPicker;
for (const el of document.querySelectorAll('.picked input')) {
  // only the Write-to field is rewritten while typing, so the caret is never moved under the hand
  el.addEventListener('input', () => {
    readPicked();
    if (el.id === 'pickcoco' && !P.outEdited) $('#pickout').value = autoOut(P.coco);
    el.classList.remove('bad');
  });
  el.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyPicker(); });
}
for (const b of document.querySelectorAll('.goto')) {
  b.onclick = () => {
    readPicked();
    if (b.dataset.go === 'coco' && P.coco) browse(dirOf(P.coco));
    else if (b.dataset.go === 'images' && P.images) browse(P.images);
    else if (b.dataset.go === 'reset') { P.outEdited = false; showPicked(); }
    else if (b.dataset.go === 'pred') {
      document.querySelector('input[name="target"][value="pred"]').checked = true;
      browse(P.pred ? dirOf(P.pred) : P.cwd);
    } else if (b.dataset.go === 'clearpred') { P.pred = ''; showPicked(); browse(P.cwd); }
  };
}

/* predictions: shown or not, and the score and IoU thresholds that decide what is drawn and paired */
$('#showpred').onchange = (e) => { S.showPred = e.target.checked; renderGrid(); draw(); };
function setThreshold(el, key, lo) {
  const v = Number(el.value);
  if (!Number.isFinite(v)) return;
  S[key] = clamp(v, lo, 1);
  computeAllMatches();
  renderAll();
}
$('#scorethr').onchange = (e) => setThreshold(e.target, 'scoreThr', 0);
$('#pairmin').onchange = (e) => setThreshold(e.target, 'pairMin', 0.01);
$('#help').onclick = () => ($('#helpbox').hidden = false);
$('#helpclose').onclick = () => ($('#helpbox').hidden = true);
window.addEventListener('resize', draw);
window.addEventListener('beforeunload', (e) => { if (S.dirty) { save(); e.preventDefault(); } });

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (!$('#picker').hidden) {              // the picker is modal: only Escape gets through
    if (e.key === 'Escape') $('#picker').hidden = true;
    return;
  }
  const k = e.key.toLowerCase();
  const big = e.shiftKey ? 5 : 1;
  if (e.ctrlKey && k === 's') { e.preventDefault(); save(); return; }
  if (e.ctrlKey && k === 'z') { e.preventDefault(); undo(); return; }
  if (e.ctrlKey && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return; }
  if (e.key.startsWith('Arrow')) {
    e.preventDefault();
    const d = { ArrowLeft: [-big, 0], ArrowRight: [big, 0], ArrowUp: [0, -big], ArrowDown: [0, big] }[e.key];
    nudge(d[0], d[1], e.ctrlKey);
    return;
  }
  if (e.code === 'Space') { e.preventDefault(); if (S.selId) { setStatus(S.selId, 'ok'); nextUnreviewed(); } return; }
  if (k === 'a' && isPred(S.selId)) { acceptPred(S.selId); return; }
  if (k === 'x' && S.selId) setStatus(S.selId, 'flag');
  else if ((k === 'd' || e.key === 'Delete') && S.selId) deleteAnn(S.selId);
  else if (k === 'n') step(1);
  else if (k === 'p') step(-1);
  else if (k === 'j') stepImage(1);
  else if (k === 'k') stepImage(-1);
  else if (k === 'f') { fitView(); draw(); }
  else if (k === '?') $('#helpbox').hidden = false;
  else if (k === 'o') openPicker();
  else if (e.key === 'Escape') { $('#helpbox').hidden = true; $('#picker').hidden = true; }
});

boot();
