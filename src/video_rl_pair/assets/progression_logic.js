/* Pure sampled-frame composition, shared by the offline page and Node tests. */
const Progression = (() => {
  function horizon(value, maximum) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.max(1, Math.min(maximum, Math.trunc(parsed))) : 1;
  }
  function compose(pair, direction, method, query, requestedHorizon) {
    const match = pair.matches.find(item => item.method === method);
    const swapped = direction === "target";
    const reference = swapped ? pair.target : pair.source;
    const demonstration = swapped ? pair.source : pair.target;
    const forward = swapped ? match.reverse : match.forward;
    const reverse = swapped ? match.forward : match.reverse;
    const t = Math.max(0, Math.min(reference.times.length - 1, Math.trunc(query)));
    const q = forward[t];
    const length = Math.min(horizon(requestedHorizon, demonstration.times.length), demonstration.times.length - q);
    const t0 = reverse[q];
    const steps = Array.from({length}, (_, i) => {
      const returned = reverse[q + i];
      const previous = i ? reverse[q + i - 1] : returned;
      return {i, demonstration: q + i, returned, delta: returned - previous,
        secondsDelta: reference.times[returned] - reference.times[previous]};
    });
    return {reference, demonstration, t, q, t0, steps,
      cycleSeconds: Math.abs(reference.times[t0] - reference.times[t])};
  }
  function playback(advance, schedule = setInterval, cancel = clearInterval) {
    let timer = null;
    function stop() {
      if (timer !== null) cancel(timer);
      timer = null;
    }
    function start() {
      stop();
      timer = schedule(() => { if (!advance()) stop(); }, 500);
    }
    return {start, stop, running: () => timer !== null};
  }
  return {horizon, compose, playback};
})();
if (typeof module !== "undefined") module.exports = Progression;
