// 把网页里的 Mermaid 图和交互动画渲染成 PNG 图片，供课件、教案和 Word / PDF 版使用。
//
// 用法（在仓库根目录）：
//     npm install                      # 第一次运行前安装依赖（mermaid、playwright）
//     npx playwright install chromium  # 第一次运行前下载浏览器
//     npm run figures                  # 或 node tools/render_figures.mjs
//
// 输出：
//     build/figures/mermaid/<代码哈希>.png    每个 Mermaid 代码块一张，文件名是代码内容的 SHA-1 前 12 位
//     build/figures/sims/<动画名>.png         每个交互动画的初始画面截图
// Python 脚本用 tools/common.py 里的 mermaid_png()、sim_png() 找到对应图片；找不到时退回文字提示。
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DOCS = join(ROOT, "docs");
const OUT = join(ROOT, "build", "figures");
const FONT = '"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","WenQuanYi Zen Hei",sans-serif';

export const mermaidKey = (code) => createHash("sha1").update(code.trim(), "utf8").digest("hex").slice(0, 12);

function mermaidBlocks() {
  const files = [];
  const walk = (d) => readdirSync(d, { withFileTypes: true }).forEach((e) => {
    const p = join(d, e.name);
    if (e.isDirectory()) walk(p); else if (e.name.endsWith(".md")) files.push(p);
  });
  walk(DOCS);
  const blocks = new Map();
  for (const f of files) {
    for (const m of readFileSync(f, "utf8").matchAll(/^```mermaid\n([\s\S]*?)^```/gm)) blocks.set(mermaidKey(m[1]), m[1]);
  }
  return blocks;
}

async function renderMermaid(browser, only) {
  const dir = join(OUT, "mermaid");
  mkdirSync(dir, { recursive: true });
  const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 1200, height: 900 } });
  await page.setContent(`<html><body style="margin:0;background:#fff"><div id="box" style="display:inline-block;padding:12px"></div></body></html>`);
  await page.addScriptTag({ path: join(ROOT, "node_modules/mermaid/dist/mermaid.min.js") });
  await page.evaluate((font) => window.mermaid.initialize({ startOnLoad: false, theme: "default", fontFamily: font, flowchart: { htmlLabels: true } }), FONT);
  let n = 0, failed = 0;
  for (const [key, code] of mermaidBlocks()) {
    const file = join(dir, `${key}.png`);
    if (existsSync(file) && !only) continue;
    try {
      await page.evaluate(async ([k, c]) => {
        const { svg } = await window.mermaid.render("m" + k, c);
        document.getElementById("box").innerHTML = svg;
      }, [key, code]);
      await (await page.$("#box")).screenshot({ path: file });
      n++;
    } catch (e) {
      failed++;
      console.warn(`  Mermaid 渲染失败（${key}）：${String(e).split("\n")[0]}\n${code.split("\n").slice(0, 3).join("\n")}`);
    }
  }
  await page.close();
  console.log(`Mermaid 图：新渲染 ${n} 张${failed ? `，失败 ${failed} 张` : ""}`);
}

async function renderSims(browser) {
  const dir = join(OUT, "sims");
  mkdirSync(dir, { recursive: true });
  const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 960, height: 640 } });
  let n = 0;
  for (const e of readdirSync(join(DOCS, "sims"), { withFileTypes: true })) {
    const html = join(DOCS, "sims", e.name, "main.html");
    if (!e.isDirectory() || e.name === "common" || !existsSync(html)) continue;
    await page.goto(pathToFileURL(html).href);
    await page.waitForTimeout(600);   // 等动画的初始绘制完成
    await page.screenshot({ path: join(dir, `${e.name}.png`), fullPage: true });
    n++;
  }
  await page.close();
  console.log(`交互动画截图：${n} 张`);
}

const browser = await chromium.launch();
await renderMermaid(browser, process.argv.includes("--force"));
await renderSims(browser);
await browser.close();
console.log(`已输出到 ${OUT}`);
