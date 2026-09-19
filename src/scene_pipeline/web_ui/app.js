const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const terminal = new Set(['succeeded', 'partial', 'failed', 'cancelled', 'interrupted']);
const flowNames = {mapping:'Explore & map', interaction:'Drawer interaction', navigate:'Goal navigation'};
const flowIcons = {mapping:'⌗', interaction:'⌁', navigate:'↗'};
let state = {scenes:[], jobs:[], settings:{}}, selectedId = null, scene = null;
let selectedFlow = 'mapping', view = 'overview', selectedRun = null, selectedJob = null;
let sceneRequest = 0, busy = false, lastJobs = '', toastTimer;

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json', 'X-Studio-Token':state.token}, body:JSON.stringify(data)
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}
function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5000);
}
function link(url, label, cls = '') {
  return url ? `<a class="${cls}" href="${escapeHTML(url)}" target="_blank" rel="noopener">${escapeHTML(label)}</a>` : '';
}
function showBanner(message) { $('banner').textContent = message; $('banner').hidden = !message; }

function renderLibrary() {
  const filter = $('search').value.toLowerCase();
  const scenes = state.scenes.filter(s => s.name.toLowerCase().includes(filter));
  $('scene-count').textContent = state.scenes.length;
  $('scene-list').innerHTML = scenes.map(s => `<button class="scene-card ${s.id === selectedId ? 'active' : ''}" data-scene="${escapeHTML(s.id)}" aria-current="${s.id === selectedId}">
    ${s.image ? `<img class="scene-thumb" src="${escapeHTML(s.image)}" alt="" loading="lazy">` : '<span class="scene-thumb"></span>'}
    <span><strong>${escapeHTML(s.name)}</strong><small><i class="mini-dot ${s.status === 'validated' ? '' : 'partial'}"></i>${s.object_count} objects · ${s.status === 'validated' ? 'Validated' : 'Review checks'}</small></span></button>`).join('') || '<p class="muted empty-line">No matching scenes.</p>';
}

async function selectScene(id, preserve = false) {
  const request = ++sceneRequest;
  try {
    const next = await api('/api/scenes/' + encodeURIComponent(id));
    if (request !== sceneRequest) return;
    selectedId = id; scene = next;
    if (!preserve) { selectedRun = null; view = 'overview'; }
    renderLibrary(); renderScene();
  } catch (error) { if (request === sceneRequest) toast(error.message); }
}

function renderScene() {
  if (!scene) return;
  $('scene-title').textContent = scene.name;
  $('scene-description').textContent = `${scene.object_count} objects. One environment. Ready to explore.`;
  $('stat-objects').textContent = scene.object_count;
  $('stat-area').textContent = scene.area == null ? '—' : `${Number(scene.area).toFixed(0)} m²`;
  $('stat-seed').textContent = scene.seed ?? '—';
  $('stat-runs').textContent = scene.runs.length;
  $('scene-badge').className = 'badge ' + (scene.status === 'validated' ? '' : 'partial');
  $('scene-badge').textContent = scene.status === 'validated' ? '✓ VALIDATED' : 'PARTIAL SCENE';
  const failed = Object.entries(scene.validation.checks || {}).filter(([,passed]) => passed === false).map(([key]) => key.replaceAll('_',' '));
  showBanner(failed.length ? `This scene needs review: ${failed.join(', ')}. You can still run a demo; each run reports its own result.` : '');
  const previousTarget = $('drawer-target').value;
  $('drawer-target').innerHTML = scene.drawers.map(id => `<option value="${escapeHTML(id)}">${escapeHTML(id)}</option>`).join('');
  const recordedTarget = scene.runs.find(r => r.flow === 'interaction' && r.report.passed)?.report.target;
  if (scene.drawers.includes(previousTarget)) $('drawer-target').value = previousTarget;
  else if (scene.drawers.includes(recordedTarget)) $('drawer-target').value = recordedTarget;
  const labels = {numerical:'Numerical stability', stability:'Settling stability', collision_clearance:'Collision clearance', robot_access:'Robot access', physical_provenance:'Physical provenance', pairwise_articulation:'Articulation checks'};
  $('check-list').innerHTML = Object.entries(labels).filter(([key]) => key in (scene.validation.checks || {})).map(([key,label]) => {
    const passed = scene.validation.checks[key];
    return `<div class="check ${passed ? '' : 'failed'}"><i>${passed ? '✓' : '!'}</i>${label}</div>`;
  }).join('') || '<p class="muted">No validation report available.</p>';
  $('artifact-links').innerHTML = link(scene.artifacts.scene,'Scene bundle ↓') + link(scene.artifacts.json,'Scene JSON ↗') + link(scene.artifacts.calls,'Agent calls ↗') + link(scene.artifacts.validation,'Validation ↗') + link(scene.artifacts.gallery,'Full gallery ↗');
  $('render-scene').disabled = false;
  renderRuns(); renderFlowControls(); renderViewer();
}

function currentRun() {
  if (!scene) return null;
  if (selectedRun) {
    const chosen = scene.runs.find(r => r.id === selectedRun);
    if (chosen && (view !== 'map' || chosen.artifacts.map)) return chosen;
  }
  return scene.runs.find(r => view === 'map' ? r.artifacts.map : r.artifacts.video) || null;
}
function renderViewer() {
  document.querySelectorAll('.tab').forEach(b => { b.classList.toggle('active',b.dataset.view === view); b.setAttribute('aria-selected',b.dataset.view === view); });
  if (!scene) return;
  const run = currentRun();
  const url = view === 'overview' ? scene.artifacts.overview || scene.artifacts.preview
    : view === 'topdown' ? scene.artifacts.topdown : view === 'map' ? run?.artifacts.map : run?.artifacts.video;
  const captions = {overview:'Compiled scene · overview', topdown:'Floor plan · object placement', map:run ? `${flowNames[run.flow] || run.flow} · observed occupancy` : 'No mapping recording yet', video:run ? `${flowNames[run.flow] || run.flow} · recorded physical states` : 'No replay recorded yet'};
  $('view-caption').textContent = captions[view];
  $('open-artifact').hidden = !url;
  if (url) $('open-artifact').href = url;
  // Do not restart a playing video during background polling.
  const key = `${view}:${url || ''}`;
  if ($('viewer').dataset.key === key) return;
  $('viewer').dataset.key = key;
  if (url && view === 'video') $('viewer').innerHTML = `<video controls playsinline preload="metadata" src="${escapeHTML(url)}" ${run?.artifacts.image ? `poster="${escapeHTML(run.artifacts.image)}"` : ''} aria-label="Recorded robot replay"></video>`;
  else if (url) $('viewer').innerHTML = `<img class="${view === 'map' ? 'map-image' : ''}" src="${escapeHTML(url)}" alt="${escapeHTML(scene.name + ' — ' + captions[view])}">`;
  else $('viewer').innerHTML = `<div class="empty-state"><span class="empty-icon">${view === 'map' ? '⌗' : '▷'}</span><h2>${view === 'map' ? 'A map is a run away' : view === 'video' ? 'See your robot in action' : 'Preview not available yet'}</h2><p>${['map','video'].includes(view) ? 'Choose a behavior and start a run. Its recorded result will appear here when it is ready.' : 'Use Render still to create an overview of this scene.'}</p></div>`;
}

function renderRuns() {
  $('run-count').textContent = scene.runs.length;
  $('run-list').innerHTML = scene.runs.slice(0,8).map(r => {
    let metric = r.flow === 'mapping' && Number.isFinite(r.report.occupancy_iou) ? ` · IoU ${r.report.occupancy_iou.toFixed(2)}` : r.flow === 'interaction' && Number.isFinite(r.report.upper_endpoint_error_m) ? ` · opening error ${(r.report.upper_endpoint_error_m * 1000).toFixed(1)} mm` : '';
    return `<div class="run-row"><span class="run-icon">${flowIcons[r.flow] || '▷'}</span><div class="run-info"><strong>${escapeHTML(flowNames[r.flow] || r.flow)}</strong><small>${Math.round(r.report.simulated_seconds || 0)}s · ${r.report.passed ? 'Passed' : 'Checks failed'}${metric}</small></div><div class="run-actions">${r.artifacts.map ? `<button class="text-button" data-run="${escapeHTML(r.id)}" data-run-view="map">Map</button>` : ''}${r.artifacts.video ? `<button class="text-button" data-run="${escapeHTML(r.id)}" data-run-view="video">▶ Replay</button>` : ''}${link(r.artifacts.report,'Report ↗')}${link(r.artifacts.data,'Data ↓')}</div></div>`;
  }).join('') || '<p class="muted empty-line">No recordings yet. Run a behavior to see its replay and measurements here.</p>';
}

function renderFlowControls() {
  document.querySelectorAll('.behavior').forEach(b => { const active = b.dataset.flow === selectedFlow; b.classList.toggle('selected',active); b.setAttribute('aria-pressed',active); b.disabled = b.dataset.flow === 'interaction' && !!scene && !scene.drawers.length; });
  $('drawer-field').hidden = selectedFlow !== 'interaction'; $('goal-field').hidden = selectedFlow !== 'navigate';
  for (const option of $('duration').options) option.disabled = selectedFlow === 'interaction' && Number(option.value) < 60;
  if (selectedFlow === 'interaction' && Number($('duration').value) < 60) $('duration').value = '60';
  $('flow-help').textContent = {mapping:'Build an occupancy map from noisy range observations.', interaction:'The Panda opens and closes the selected drawer through physical contact. Completion is measured.', navigate:'The planner resolves your goal against the scene’s object manifest and records its approach.'}[selectedFlow];
  $('run-flow').innerHTML = '<span>▶</span> ' + {mapping:'Run mapping',interaction:'Run interaction',navigate:'Run navigation'}[selectedFlow];
  $('run-flow').disabled = !scene || busy || (selectedFlow === 'interaction' && !scene.drawers.length);
}

function renderJobs() {
  $('job-count').textContent = state.jobs.filter(j => !terminal.has(j.status)).length;
  $('job-list').innerHTML = state.jobs.slice(0,10).map(j => {
    const elapsed = j.started ? Math.max(0, Math.round((j.finished || Date.now()/1000) - j.started)) : 0;
    return `<div class="job-row"><span class="job-state ${escapeHTML(j.status)}"></span><div class="job-info"><strong>${escapeHTML(j.title)} <span class="muted">· ${escapeHTML(j.status)}</span></strong><p>${escapeHTML(j.stage)}</p></div><span class="job-time">${Math.floor(elapsed/60)}m ${elapsed%60}s</span><button class="text-button" data-job="${escapeHTML(j.id)}">Logs</button>${terminal.has(j.status) ? (state.scenes.some(s => s.id === j.output) ? `<button class="text-button" data-scene="${escapeHTML(j.output)}">Open scene ↗</button>` : '') : `<button class="text-button" data-cancel="${escapeHTML(j.id)}">Cancel</button>`}</div>`;
  }).join('') || '<p class="muted empty-line">Your next experiment starts here. Create a scene or run a robot behavior.</p>';
}

async function poll() {
  try {
    const next = await api('/api/state');
    const signature = JSON.stringify(next.jobs.map(j => [j.id,j.status,j.stage]));
    const changed = signature !== lastJobs;
    state = next; lastJobs = signature;
    $('connection').textContent = 'Connected to local pipeline';
    $('provider-status').textContent = `${state.settings.openrouter ? 'OpenRouter ready' : 'OpenRouter not configured'} · ${state.settings.renderer.toUpperCase()}`;
    if (!selectedId && state.scenes.length) await selectScene(state.scenes[0].id);
    else if (selectedId && changed) await selectScene(selectedId,true);
    renderLibrary(); renderJobs();
    if ($('log-details').open && selectedJob) await loadLog(selectedJob);
  } catch { $('connection').textContent = 'Disconnected · retrying'; }
}

async function loadLog(id) {
  selectedJob = id;
  try {
    const result = await api('/api/logs/' + encodeURIComponent(id));
    const log = $('job-log'), atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 35;
    log.textContent = result.log || 'Waiting for worker output…';
    $('log-title').textContent = id;
    if (atEnd) log.scrollTop = log.scrollHeight;
  } catch (error) { toast(error.message); }
}

function generationMode() {
  const mode = $('backend').value, fal = $('appearance').value === 'fal';
  $('json-field').hidden = mode !== 'json'; $('model-field').hidden = mode === 'json';
  $('api-budget-field').hidden = mode !== 'openrouter'; $('fal-budget-field').hidden = !fal;
  $('iterations').disabled = mode === 'json';
  $('attempts-help').textContent = mode === 'json' ? 'Supplied JSON runs once without agent repairs.' : 'Total attempts, including the first. Failed checks are sent back to the agent for repair.';
  $('budget-note').textContent = mode === 'codex' ? 'Codex CLI does not report dollar costs. Generation is bounded by the attempt limit and a 20-minute timeout. fal has a separate budget.' : mode === 'json' ? 'No model calls. fal limits use list-price estimates; existing cached assets are reused.' : 'Model and fal budgets are shared across all attempts. Credential errors and exhausted budgets stop the run early.';
  $('credential-hint').textContent = mode === 'openrouter' ? state.settings.openrouter ? 'OpenRouter key loaded on the server from the environment or .env.' : 'OpenRouter key missing. Set OPENROUTER_API_KEY in .env and restart the server.' : mode === 'codex' ? 'Follows the pipeline skill using the existing Codex login. Reuses assets, runs scene tools, and repairs failed checks.' : 'Uses your declarative scene JSON without a nested model call.';
  if (fal && !state.settings.fal) $('credential-hint').textContent += ' fal key missing: set FAL_KEY or FAL_API_KEY and restart.';
}
function openDialog() { $('generation-error').hidden = true; generationMode(); $('generate-dialog').showModal(); }
function importJSON(text) {
  $('program-json').value = text;
  try { const program = JSON.parse(text); if (typeof program.prompt === 'string') $('prompt').value = program.prompt; } catch { /* validated on submit */ }
}

document.addEventListener('click', async event => {
  const sceneButton = event.target.closest('[data-scene]');
  if (sceneButton) return selectScene(sceneButton.dataset.scene);
  const tab = event.target.closest('[data-view]');
  if (tab) { view = tab.dataset.view; renderViewer(); }
  const behavior = event.target.closest('[data-flow]');
  if (behavior && !behavior.disabled) { selectedFlow = behavior.dataset.flow; renderFlowControls(); }
  const run = event.target.closest('[data-run]');
  if (run) { selectedRun = run.dataset.run; view = run.dataset.runView; renderViewer(); $('viewer').scrollIntoView({behavior:'smooth',block:'center'}); }
  const job = event.target.closest('[data-job]');
  if (job) { $('log-details').open = true; await loadLog(job.dataset.job); }
  const cancel = event.target.closest('[data-cancel]');
  if (cancel) { try { await api('/api/cancel/' + cancel.dataset.cancel,{}); toast('Cancellation requested. Existing output is retained.'); await poll(); } catch (error) { toast(error.message); } }
  const preset = event.target.closest('[data-preset]');
  if (preset) $('prompt').value = {
    office:'A small office with two study desks and chairs, three storage cabinets, two drawer units, a bookshelf, and a clear central aisle. Add everyday desk clutter.',
    dorm:'A practical two-person dorm room with two beds, two study desks and chairs, clothing storage, drawers, a shared bookshelf, and personal study items. Keep a clear circulation path.',
    shop:'A small machine shop with a lathe, milling machine, drill press, workbench, tool cabinets and drawer storage. Keep clear working areas around the machines.'
  }[preset.dataset.preset];
});
$('new-scene').addEventListener('click',openDialog); $('empty-create').addEventListener('click',openDialog);
$('close-dialog').addEventListener('click',() => $('generate-dialog').close()); $('cancel-dialog').addEventListener('click',() => $('generate-dialog').close());
$('search').addEventListener('input',renderLibrary); $('refresh').addEventListener('click',async () => { await poll(); if (selectedId) await selectScene(selectedId,true); });
$('appearance').addEventListener('change',generationMode); $('backend').addEventListener('change',() => {
  $('model').value = $('backend').value === 'openrouter' ? state.settings.default_model : '';
  $('model').placeholder = $('backend').value === 'codex' ? 'Use the CLI default model' : 'provider/model';
  generationMode();
});
$('program-json').addEventListener('change',() => importJSON($('program-json').value));
$('json-file').addEventListener('change',async () => { const file = $('json-file').files[0]; if (file) { if (file.size > 120000) return toast('Scene JSON must be under 120 KB.'); importJSON(await file.text()); } });
$('flow-form').addEventListener('submit',async event => {
  event.preventDefault(); if (!scene || busy) return; busy = true; renderFlowControls();
  try {
    const job = await api('/api/jobs',{kind:'flow',scene:selectedId,flow:selectedFlow,target:$('drawer-target').value,goal:$('goal').value,seconds:Number($('duration').value),tier:$('capture').value,seed:Number($('run-seed').value)});
    selectedJob = job.id; toast('Robot flow added to the queue.'); await poll();
  } catch (error) { toast(error.message); }
  finally { busy = false; renderFlowControls(); }
});
$('render-scene').addEventListener('click',async () => {
  if (!scene) return; $('render-scene').disabled = true;
  try { await api('/api/jobs',{kind:'render',scene:selectedId}); toast('Still render added to the queue.'); await poll(); }
  catch(error) { toast(error.message); } finally { $('render-scene').disabled = false; }
});
$('generate-form').addEventListener('submit',async event => {
  event.preventDefault(); $('submit-generation').disabled = true; $('generation-error').hidden = true;
  try {
    const data = {kind:'generate',prompt:$('prompt').value,mode:$('backend').value,model:$('model').value,appearance:$('appearance').value,seed:Number($('generation-seed').value),max_cost:Number($('api-budget').value),max_fal:Number($('fal-budget').value),iterations:Number($('iterations').value)};
    if (data.mode === 'json') data.program = JSON.parse($('program-json').value);
    const job = await api('/api/jobs',data); selectedJob = job.id;
    $('generate-dialog').close(); toast('Scene generation queued. Follow its progress below.'); await poll();
  } catch(error) { $('generation-error').textContent = error.message; $('generation-error').hidden = false; }
  finally { $('submit-generation').disabled = false; }
});
poll(); setInterval(poll,3000);
