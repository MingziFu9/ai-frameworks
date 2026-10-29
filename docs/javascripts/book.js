// 教材页面交互：
// 1. 章末练习题：单选、判断题点击选项即时判断对错；多选题（quiz-multi）选好后点“提交”；
//    判断后自动展开紧跟其后的“答案与解析”；
// 2. 术语链接：鼠标悬停时显示术语表中的定义；
// 3. 交互动画：iframe 高度随内容自动调整。
(function () {
  function letterOf(li) {
    var t = li.textContent.trim();
    return t.charAt(0);
  }

  function nextDetails(el) {
    var n = el.nextElementSibling;
    while (n && n.tagName !== "DETAILS" && !n.classList.contains("quiz")) n = n.nextElementSibling;
    return n && n.tagName === "DETAILS" ? n : null;
  }

  function initTerms() {
    document.querySelectorAll("a.term[title]").forEach(function (a) {
      a.dataset.def = a.getAttribute("title");
      a.removeAttribute("title");
    });
  }

  function init() {
    initTerms();
    var quizzes = document.querySelectorAll(".quiz");
    if (!quizzes.length) return;

    var total = quizzes.length, done = 0, right = 0;
    var score = document.createElement("div");
    score.className = "quiz-score";
    function updateScore() {
      score.textContent = "客观题进度：已答 " + done + " / " + total + "，答对 " + right + " 题" +
        (done === total ? "（全部完成！错题请点“回看”复习对应小节）" : "");
    }
    updateScore();
    quizzes[0].parentNode.insertBefore(score, quizzes[0].previousElementSibling || quizzes[0]);

    function finish(q, ok, fb) {
      q.classList.add("answered");
      done += 1; if (ok) right += 1;
      fb.textContent = ok ? "回答正确！" : "再想想，正确答案已标出，看看下面的解析。";
      fb.className = "quiz-feedback " + (ok ? "ok" : "no");
      var d = nextDetails(q);
      if (d) d.open = true;
      updateScore();
    }

    quizzes.forEach(function (q) {
      var answer = (q.dataset.answer || "").replace(/\s/g, "");
      var items = q.querySelectorAll("li");
      var multi = q.classList.contains("quiz-multi");
      var fb = document.createElement("div");
      fb.className = "quiz-feedback";
      q.appendChild(fb);
      if (multi) {
        var btn = document.createElement("button");
        btn.className = "quiz-submit";
        btn.textContent = "提交答案";
        q.insertBefore(btn, fb);
        items.forEach(function (li) {
          li.addEventListener("click", function () {
            if (!q.classList.contains("answered")) li.classList.toggle("picked");
          });
        });
        btn.addEventListener("click", function () {
          if (q.classList.contains("answered")) return;
          var picked = "";
          items.forEach(function (x) { if (x.classList.contains("picked")) picked += letterOf(x); });
          if (!picked) { fb.textContent = "请至少选一项。"; fb.className = "quiz-feedback no"; return; }
          items.forEach(function (x) {
            var inAns = answer.indexOf(letterOf(x)) >= 0;
            if (inAns) x.classList.add("correct");
            else if (x.classList.contains("picked")) x.classList.add("wrong");
          });
          btn.disabled = true;
          finish(q, picked.split("").sort().join("") === answer.split("").sort().join(""), fb);
        });
        return;
      }
      items.forEach(function (li) {
        li.addEventListener("click", function () {
          if (q.classList.contains("answered")) return;
          var ok = letterOf(li) === answer;
          items.forEach(function (x) { if (letterOf(x) === answer) x.classList.add("correct"); });
          if (!ok) li.classList.add("wrong");
          finish(q, ok, fb);
        });
      });
    });
  }

  // 交互动画 iframe 自动调整高度（动画页面会通过 postMessage 报告自己的高度）
  window.addEventListener("message", function (e) {
    if (!e.data || !e.data.simHeight) return;
    document.querySelectorAll("iframe.sim-frame").forEach(function (f) {
      if (f.contentWindow === e.source) f.style.height = (e.data.simHeight + 4) + "px";
    });
  });

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
