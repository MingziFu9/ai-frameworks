// 嵌入教材页面时，把自身高度告诉外层页面，让 iframe 自动调整高度
(function () {
  if (window.parent === window) return;
  function report() { window.parent.postMessage({ simHeight: Math.ceil(document.body.getBoundingClientRect().height) }, "*"); }
  new ResizeObserver(report).observe(document.body);
  window.addEventListener("load", report);
})();
