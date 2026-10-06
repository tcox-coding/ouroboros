// Shared comparison view. Both panes use the same zoom and normalized scroll position.
(() => {
  const dialog = document.createElement('dialog');
  dialog.className = 'compare-dialog';
  dialog.innerHTML = `<div class="compare-toolbar"><strong>Compare images</strong>
    <label>Zoom <input type="range" min="1" max="6" step="0.1" value="1" aria-label="Comparison zoom"></label>
    <output>1×</output><button class="btn small" data-reset>Fit</button>
    <button class="btn small" data-close-compare>Close</button></div>
    <p class="muted">Zoom, then scroll or drag either image. Both views move together.</p>
    <div class="compare-panes">${['Original / reference', 'Candidate'].map(label =>
      `<section><h3>${label}</h3><div class="compare-viewport" tabindex="0" aria-label="${label}, scroll to pan"><div class="compare-surface"><img alt="${label}" draggable="false"></div></div></section>`).join('')}</div>`;
  document.body.append(dialog);
  const panes = [...dialog.querySelectorAll('.compare-viewport')];
  const slider = dialog.querySelector('input');
  let previousFocus, zoom = 1, syncing = false;
  function position(p) {
    return [(p.scrollLeft + p.clientWidth / 2) / Math.max(p.scrollWidth, 1),
            (p.scrollTop + p.clientHeight / 2) / Math.max(p.scrollHeight, 1)];
  }
  function move(p, x, y) {
    p.scrollLeft = x * p.scrollWidth - p.clientWidth / 2;
    p.scrollTop = y * p.scrollHeight - p.clientHeight / 2;
  }
  function resize() {
    const [x, y] = position(panes[0]);
    zoom = Number(slider.value);
    dialog.querySelector('output').textContent = `${zoom.toFixed(1)}×`;
    for (const p of panes) {
      const surface = p.firstElementChild;
      surface.style.width = `${p.clientWidth * zoom}px`;
      surface.style.height = `${p.clientHeight * zoom}px`;
      move(p, x, y);
    }
  }
  slider.addEventListener('input', resize);
  dialog.querySelector('[data-reset]').onclick = () => { slider.value = 1; resize(); };
  dialog.querySelector('[data-close-compare]').onclick = () => dialog.close();
  dialog.addEventListener('close', () => previousFocus?.focus());
  window.addEventListener('resize', () => { if (dialog.open) resize(); });
  panes.forEach(p => {
    p.addEventListener('scroll', () => {
      if (syncing) return;
      syncing = true;
      const [x, y] = position(p);
      move(panes.find(q => q !== p), x, y);
      requestAnimationFrame(() => { syncing = false; });
    });
    let drag;
    p.addEventListener('pointerdown', e => {
      if (zoom <= 1) return;
      drag = [e.clientX, e.clientY, p.scrollLeft, p.scrollTop];
      p.setPointerCapture(e.pointerId);
    });
    p.addEventListener('pointermove', e => {
      if (!drag) return;
      p.scrollLeft = drag[2] - e.clientX + drag[0];
      p.scrollTop = drag[3] - e.clientY + drag[1];
    });
    p.addEventListener('pointerup', () => { drag = null; });
    p.addEventListener('pointercancel', () => { drag = null; });
  });
  document.addEventListener('click', e => {
    const button = e.target.closest('[data-compare]');
    if (!button) return;
    previousFocus = document.activeElement;
    panes[0].querySelector('img').src = button.dataset.compare;
    panes[1].querySelector('img').src = button.dataset.candidate;
    slider.value = 1;
    dialog.showModal();
    resize();
  });
})();
