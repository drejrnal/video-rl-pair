(() => {
  const report = JSON.parse(document.getElementById("progression-data").textContent);
  const element = id => document.getElementById(id);
  const pairSelect = element("pair"), methodSelect = element("method");
  const direction = element("direction"), query = element("query");
  const horizon = element("horizon"), step = element("step");
  let view, current = 0;
  const clock = Progression.playback(() => {
    if (current >= view.steps.length - 1) { pause(); return false; }
    current += 1;
    renderStep();
    if (current >= view.steps.length - 1) { pause(); return false; }
    return true;
  });
  function pause() { clock.stop(); element("play").textContent = "Play"; }
  function option(value, label) {
    const result = document.createElement("option");
    result.value = value; result.textContent = label; return result;
  }
  function showFrame(prefix, video, index) {
    const image = element(prefix + "-image");
    image.src = video.images[index];
    image.alt = `${prefix} retrieved frame ${index} at ${video.times[index].toFixed(3)} seconds`;
    element(prefix + "-path").textContent = video.path;
    element(prefix + "-caption").textContent = `Sample ${index} · original time ${video.times[index].toFixed(3)} s`;
  }
  function renderStep() {
    const selected = view.steps[current];
    step.value = String(current);
    element("step-value").textContent = `${current} / ${view.steps.length - 1}`;
    showFrame("original", view.reference, view.t);
    showFrame("middle", view.demonstration, selected.demonstration);
    showFrame("return", view.reference, selected.returned);
    element("previous").disabled = current === 0;
    element("next").disabled = current === view.steps.length - 1;
    element("play").disabled = view.steps.length < 2;
    const delta = current ? `${selected.delta >= 0 ? "+" : ""}${selected.delta} samples, ${selected.secondsDelta.toFixed(3)} s` : "start of chunk";
    element("jump").textContent = `Return index ${selected.returned} · change from previous: ${delta}`;
    const position = index => 20 + 560 * index / Math.max(1, view.reference.times.length - 1);
    element("query-marker").setAttribute("cx", position(view.t));
    const x = position(selected.returned);
    element("return-marker").setAttribute("d", `M ${x} 26 l -8 16 h 16 Z`);
    element("query-marker-label").textContent = `● Query t = ${view.t}`;
    element("return-marker-label").textContent = `▲ Return = ${selected.returned} / ${view.reference.times.length - 1}`;
    Array.from(element("sequence").children).forEach((row, i) => {
      row.setAttribute("aria-current", i === current ? "true" : "false");
    });
  }
  function rebuild() {
    pause(); current = 0;
    const pair = report.pairs[Number(pairSelect.value)];
    const maximum = Math.max(1000000, report.horizon);
    horizon.max = String(maximum);
    horizon.value = String(Progression.horizon(horizon.value, maximum));
    view = Progression.compose(pair, direction.value, methodSelect.value, Number(query.value), Number(horizon.value));
    query.max = String(view.reference.times.length - 1); query.value = String(view.t);
    step.max = String(view.steps.length - 1);
    element("query-value").textContent = `${view.t} / ${view.reference.times.length - 1}`;
    element("cycle").textContent = `t = ${view.t} → q = ${view.q} → t₀ = ${view.t0} · absolute timestamp cycle error: ${view.cycleSeconds.toFixed(3)} s`;
    element("chunk").textContent = `Requested H = ${horizon.value} · effective chunk length = ${view.steps.length} · D indices ${view.q}–${view.q + view.steps.length - 1}. Chunk truncates at the video end; no repeated padding.`;
    const rows = view.steps.map(item => {
      const row = document.createElement("tr");
      const values = [item.i, item.demonstration, view.demonstration.times[item.demonstration].toFixed(3), item.returned, view.reference.times[item.returned].toFixed(3), item.i ? item.delta : "—", item.i ? item.secondsDelta.toFixed(3) : "—"];
      for (const value of values) {
        const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell);
      }
      return row;
    });
    element("sequence").replaceChildren(...rows);
    renderStep();
  }
  function changePair() {
    const pair = report.pairs[Number(pairSelect.value)];
    const oldMethod = methodSelect.value;
    methodSelect.replaceChildren(...pair.matches.map(match => option(match.method, {nearest:"Nearest neighbor",dtw:"DTW",time:"Normalized time"}[match.method])));
    if (pair.matches.some(match => match.method === oldMethod)) methodSelect.value = oldMethod;
    query.value = "0";
    rebuild();
  }
  element("provenance").textContent = report.initialization_baseline
    ? "Initialization-only baseline · no trained checkpoint loaded."
    : `Checkpoint: ${report.checkpoint ?? "not recorded"}`;
  if (report.pairs.length === 0) {
    element("viewer").hidden = true; element("empty").hidden = false; return;
  }
  pairSelect.replaceChildren(...report.pairs.map((pair, index) => option(String(index), `${pair.task_id} · ${pair.pair_id}`)));
  horizon.value = String(report.horizon);
  pairSelect.addEventListener("change", changePair);
  direction.addEventListener("change", () => { query.value = "0"; rebuild(); });
  methodSelect.addEventListener("change", rebuild);
  query.addEventListener("input", rebuild);
  horizon.addEventListener("input", () => {
    pause();
    if (horizon.value !== "" && Number.isFinite(horizon.valueAsNumber)) rebuild();
  });
  horizon.addEventListener("change", rebuild);
  step.addEventListener("input", () => { pause(); current = Number(step.value); renderStep(); });
  element("previous").addEventListener("click", () => { pause(); current = Math.max(0, current - 1); renderStep(); });
  element("next").addEventListener("click", () => { pause(); current = Math.min(view.steps.length - 1, current + 1); renderStep(); });
  element("play").addEventListener("click", () => {
    if (clock.running()) { pause(); return; }
    if (current === view.steps.length - 1) { current = 0; renderStep(); }
    clock.start(); element("play").textContent = "Pause";
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  window.addEventListener("pagehide", pause);
  changePair();
})();
