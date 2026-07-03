// Light progressive enhancement. Page is fully readable without it.
(function () {
  "use strict";

  // --- active section in nav ---
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a[href^='#']"));
  var map = {};
  links.forEach(function (a) {
    var id = a.getAttribute("href").slice(1);
    var el = document.getElementById(id);
    if (el) map[id] = a;
  });
  var ids = Object.keys(map);

  function onScroll() {
    var y = window.scrollY + 120;
    var current = null;
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      if (el && el.offsetTop <= y) current = id;
    });
    links.forEach(function (a) { a.classList.remove("active"); });
    if (current && map[current]) map[current].classList.add("active");
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // --- reveal on scroll ---
  var revs = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 });
    revs.forEach(function (el) { io.observe(el); });
  } else {
    revs.forEach(function (el) { el.classList.add("in"); });
  }

  // --- count-up for hero stats ---
  function animateCount(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    var dec = parseInt(el.getAttribute("data-dec") || "0", 10);
    var dur = 1100, start = null;
    function fmt(v) {
      return v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
    }
    function step(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * eased);
      if (p < 1) requestAnimationFrame(step);
      else el.textContent = fmt(target);
    }
    requestAnimationFrame(step);
  }
  var counters = document.querySelectorAll("[data-count]");
  if ("IntersectionObserver" in window) {
    var io2 = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { animateCount(e.target); io2.unobserve(e.target); }
      });
    }, { threshold: 0.6 });
    counters.forEach(function (el) { io2.observe(el); });
  } else {
    counters.forEach(function (el) { el.textContent = el.getAttribute("data-count"); });
  }
})();
