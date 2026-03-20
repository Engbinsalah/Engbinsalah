(function(){
  const countdowns = document.querySelectorAll('[data-countdown]');
  countdowns.forEach(el => {
    const end = new Date(el.getAttribute('data-countdown'));
    const tick = () => {
      const diff = end - new Date();
      if (diff <= 0) { el.textContent = 'Deal ended'; return; }
      const h = Math.floor(diff / 1000 / 60 / 60);
      const m = Math.floor((diff / 1000 / 60) % 60);
      const s = Math.floor((diff / 1000) % 60);
      el.textContent = `${h}h ${m}m ${s}s`;
      requestAnimationFrame(tick);
    };
    tick();
  });

  const carousels = document.querySelectorAll('[data-carousel]');
  carousels.forEach(carousel => {
    const next = carousel.parentElement.querySelector('[data-carousel-next]');
    const prev = carousel.parentElement.querySelector('[data-carousel-prev]');
    const scroll = dir => carousel.scrollBy({left: dir * 320, behavior: 'smooth'});
    next && next.addEventListener('click', () => scroll(1));
    prev && prev.addEventListener('click', () => scroll(-1));
  });
})();
