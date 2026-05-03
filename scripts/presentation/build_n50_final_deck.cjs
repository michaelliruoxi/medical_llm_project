const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const PptxGenJS = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT_PPTX = path.join(ROOT, "MedQuAD_Robustness_Final_Presentation_n50_updated.pptx");
const PREVIEW_DIR = path.join(ROOT, "artifacts", "n50_final_previews");
const ARTIFACT_TOOL = "C:\\Users\\Owner\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules\\@oai\\artifact-tool\\dist\\artifact_tool.mjs";

const W = 13.333;
const H = 7.5;
const M = 0.52;
const FONT = "Aptos";
const MONO = "Consolas";
const C = {
  ink: "17202A",
  muted: "52616B",
  paper: "F7FAFC",
  white: "FFFFFF",
  line: "D9E2E8",
  teal: "127C7A",
  teal2: "0E5F60",
  blue: "3867D6",
  amber: "E0A13A",
  coral: "C94C4C",
  green: "2D9D78",
  plum: "7A4B8F",
  dark: "25323D",
  paleTeal: "E7F4F2",
  paleBlue: "EAF0FF",
  paleAmber: "FFF4DA",
  paleCoral: "FCEBEB",
  paleGreen: "EAF7F1",
};

function csvParse(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  const header = rows.shift() || [];
  return rows.filter((r) => r.length === header.length).map((r) => {
    const out = {};
    header.forEach((h, idx) => { out[h] = r[idx]; });
    return out;
  });
}

function loadCsv(rel) {
  return csvParse(fs.readFileSync(path.join(ROOT, rel), "utf8"));
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function avg(values) {
  const xs = values.map(num).filter((v) => Number.isFinite(v));
  return xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
}

function round(v, digits = 2) {
  return Number(v).toFixed(digits);
}

function cleanModelName(name) {
  return String(name || "")
    .replace(/_/g, " ")
    .replace("Mixtral-8x7B-Instruct-v0.1 (4bit)", "Mixtral 8x7B")
    .replace("Mixtral-8x7B-Instruct-v0.1 (4bit)", "Mixtral 8x7B")
    .replace("Mistral-7B-Instruct-v0.3", "Mistral 7B")
    .replace("Llama-3.1-8B-Instruct", "Llama 3.1 8B")
    .replace("Phi-3-medium-4k-instruct (4bit)", "Phi-3 med")
    .replace("Qwen2.5-14B-Instruct (4bit)", "Qwen2.5 14B")
    .replace("Qwen2.5-32B-Instruct (4bit)", "Qwen2.5 32B")
    .replace("Qwen3-32B (4bit)", "Qwen3 32B")
    .replace("gemma-4-31B-it (4bit)", "Gemma 4 31B")
    .replace("gemma-2-9b-it", "Gemma 2 9B")
    .replace("BioMistral-7B", "BioMistral")
    .replace("gpt-5.4", "GPT-5.4");
}

function loadModeSamples(mode) {
  const dir = path.join(ROOT, "data", "outputs", "benchmarks", mode);
  return fs.readdirSync(dir)
    .filter((name) => /^samples_.*\.csv$/i.test(name))
    .flatMap((name) => loadCsv(path.join("data", "outputs", "benchmarks", mode, name)));
}

function groupBy(rows, key) {
  const out = new Map();
  for (const row of rows) {
    const k = row[key] || "";
    if (!out.has(k)) out.set(k, []);
    out.get(k).push(row);
  }
  return out;
}

const fixedModels = loadCsv(path.join("data", "outputs", "benchmarks", "fixed_repair", "model_comparison.csv"));
const selfModels = loadCsv(path.join("data", "outputs", "benchmarks", "self_repair", "model_comparison.csv"));
const fixedSamples = loadModeSamples("fixed_repair");
const selfSamples = loadModeSamples("self_repair");
const allSamples = { fixed_repair: fixedSamples, self_repair: selfSamples };

function modeAverages(rows) {
  return {
    clean: avg(rows.map((r) => r.geval_clean)),
    noisy: avg(rows.map((r) => r.geval_noisy)),
    repaired: avg(rows.map((r) => r.geval_repaired)),
  };
}

function noiseAverages(rows) {
  const result = [];
  for (const [noise, group] of groupBy(rows, "noise_type").entries()) {
    result.push({
      noise,
      clean: avg(group.map((r) => r.geval_clean)),
      noisy: avg(group.map((r) => r.geval_noisy)),
      repaired: avg(group.map((r) => r.geval_repaired)),
      delta: avg(group.map((r) => num(r.geval_repaired) - num(r.geval_noisy))),
    });
  }
  return result.sort((a, b) => a.noise.localeCompare(b.noise));
}

function metricWinCounts(rows, metric) {
  return rows.reduce((acc, row) => {
    const delta = num(row[`${metric}_repaired`]) - num(row[`${metric}_noisy`]);
    if (delta > 1e-9) acc.improved += 1;
    else if (delta < -1e-9) acc.worse += 1;
    else acc.flat += 1;
    return acc;
  }, { improved: 0, worse: 0, flat: 0 });
}

function sampleRow(mode, modelFile, questionId) {
  const rows = loadCsv(path.join("data", "outputs", "benchmarks", mode, modelFile));
  return rows.find((r) => String(r.question_id) === String(questionId));
}

const modeAvg = {
  fixed_repair: modeAverages(fixedSamples),
  self_repair: modeAverages(selfSamples),
};
const noiseFixed = noiseAverages(fixedSamples);
const noiseSelf = noiseAverages(selfSamples);

const evidence = {
  air: sampleRow("fixed_repair", "samples_gpt-5.4.csv", 18),
  steroids: sampleRow("fixed_repair", "samples_gpt-5.4.csv", 28),
  abortion: sampleRow("fixed_repair", "samples_gpt-5.4.csv", 2),
  anal: sampleRow("self_repair", "samples_Mixtral-8x7B-Instruct-v0.1_(4bit).csv", 30),
};

function addBg(slide, color = C.paper) {
  slide.background = { color };
}

function addFooter(slide, n, total) {
  slide.addShape(pptx.ShapeType.line, { x: M, y: 7.05, w: W - 2 * M, h: 0, line: { color: C.line, width: 0.6 } });
  slide.addText("MedQuAD Robustness | Group 2 | n=50 final presentation", {
    x: M, y: 7.12, w: 7.9, h: 0.18, margin: 0,
    fontFace: FONT, fontSize: 6.8, color: C.muted,
  });
  slide.addText(`${n} / ${total}`, {
    x: W - 1.15, y: 7.12, w: 0.6, h: 0.18, margin: 0,
    fontFace: FONT, fontSize: 6.8, color: C.muted, align: "right",
  });
}

function addHeader(slide, kicker, title, subtitle, n, total) {
  addText(slide, kicker, M, 0.35, 4.8, 0.23, { size: 7.8, color: C.teal2, bold: true, caps: true });
  addText(slide, title, M, 0.68, 8.9, 0.62, { size: 24, bold: true, color: C.ink, fit: "shrink" });
  if (subtitle) addText(slide, subtitle, M, 1.27, 10.8, 0.34, { size: 10.5, color: C.muted, fit: "shrink" });
  addFooter(slide, n, total);
}

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: opts.margin ?? 0.05,
    fontFace: opts.fontFace || FONT,
    fontSize: opts.size || 10,
    color: opts.color || C.ink,
    bold: Boolean(opts.bold),
    italic: Boolean(opts.italic),
    align: opts.align || "left",
    valign: opts.valign || "top",
    breakLine: false,
    fit: opts.fit,
    paraSpaceAfterPt: opts.paraSpaceAfterPt ?? 0,
    bullet: opts.bullet,
    rotate: opts.rotate,
  });
}

function addRect(slide, x, y, w, h, fill, line = fill, radius = false, transparency = 0) {
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: radius ? 0.06 : 0,
    fill: { color: fill, transparency },
    line: { color: line, width: line === "none" ? 0 : 0.7, transparency: line === "none" ? 100 : 0 },
  });
}

function addPill(slide, text, x, y, w, fill, color = C.ink) {
  addRect(slide, x, y, w, 0.28, fill, fill, true);
  addText(slide, text, x + 0.08, y + 0.06, w - 0.16, 0.16, {
    size: 7.5, color, bold: true, align: "center", margin: 0,
  });
}

function addMetric(slide, label, value, x, y, w, color) {
  addText(slide, value, x, y, w, 0.42, { size: 20, color, bold: true, align: "center", margin: 0 });
  addText(slide, label, x, y + 0.46, w, 0.26, { size: 7.5, color: C.muted, align: "center", margin: 0, fit: "shrink" });
}

function addSmallTable(slide, headers, rows, x, y, widths, rowH, opts = {}) {
  const totalW = widths.reduce((a, b) => a + b, 0);
  addRect(slide, x, y, totalW, rowH, opts.headerFill || C.dark, opts.headerFill || C.dark);
  let cx = x;
  headers.forEach((h, i) => {
    addText(slide, h, cx + 0.05, y + 0.06, widths[i] - 0.1, rowH - 0.08, {
      size: opts.headerSize || 7.7, color: C.white, bold: true, margin: 0, fit: "shrink",
    });
    cx += widths[i];
  });
  rows.forEach((r, ri) => {
    const ry = y + rowH * (ri + 1);
    addRect(slide, x, ry, totalW, rowH, ri % 2 ? C.white : "F3F7F9", C.line);
    let c = x;
    r.forEach((cell, ci) => {
      addText(slide, String(cell), c + 0.05, ry + 0.055, widths[ci] - 0.1, rowH - 0.08, {
        size: opts.bodySize || 7.3,
        color: ci === 0 ? C.ink : C.muted,
        bold: ci === 0 && opts.boldFirst,
        margin: 0,
        fit: "shrink",
      });
      c += widths[ci];
    });
  });
}

function displayText(text) {
  return String(text || "")
    .replace(/\*\*/g, "")
    .replace(/#{1,6}\s*/g, "")
    .replace(/^\s*[-*]\s+/gm, "")
    .replace(/\b\d+\.\s+/g, "")
    .replace(/[“”]/g, '"')
    .replace(/[’]/g, "'")
    .replace(/â€“|–|—/g, "-")
    .replace(/\s+/g, " ")
    .trim();
}

function excerpt(text, max = 230) {
  const s = displayText(text);
  if (s.length <= max) return s;
  return `${s.slice(0, max - 3).trim()}...`;
}

function scoreBand(slide, row, x, y, w) {
  const clean = num(row.geval_clean);
  const noisy = num(row.geval_noisy);
  const repaired = num(row.geval_repaired);
  addText(slide, `G-Eval: ${clean} clean | ${noisy} noisy | ${repaired} repaired`, x, y, w, 0.18, {
    size: 8.2, color: C.ink, bold: true, margin: 0,
  });
  addText(slide, `med_coverage: ${round(num(row.med_coverage_clean), 1)} -> ${round(num(row.med_coverage_noisy), 1)} -> ${round(num(row.med_coverage_repaired), 1)}`, x, y + 0.24, w, 0.18, {
    size: 7.8, color: C.muted, margin: 0,
  });
}

function answerPanel(slide, label, text, x, y, w, h, fill, accent) {
  addRect(slide, x, y, w, h, fill, C.line, true);
  addPill(slide, label, x + 0.14, y + 0.13, 1.1, accent, C.white);
  addText(slide, excerpt(text, 360), x + 0.18, y + 0.52, w - 0.36, h - 0.66, {
    size: 8.2, color: C.ink, fit: "shrink", margin: 0.02,
  });
}

function addProgressFlow(slide, x, y, w) {
  const steps = [
    ["Clean medical question", C.paleBlue, C.blue],
    ["Noisy user question", C.paleAmber, C.amber],
    ["Repaired question", C.paleTeal, C.teal],
    ["Model answer", C.paleGreen, C.green],
    ["Score vs reference", C.paleCoral, C.coral],
  ];
  const gap = 0.14;
  const sw = (w - gap * 4) / 5;
  steps.forEach(([label, fill, accent], i) => {
    const sx = x + i * (sw + gap);
    addRect(slide, sx, y, sw, 0.72, fill, accent, true);
    addText(slide, label, sx + 0.08, y + 0.19, sw - 0.16, 0.24, {
      size: 8.1, color: C.ink, bold: true, align: "center", fit: "shrink", margin: 0,
    });
    if (i < steps.length - 1) {
      addText(slide, ">", sx + sw + 0.035, y + 0.2, 0.08, 0.22, { size: 12, color: C.muted, bold: true, margin: 0 });
    }
  });
}

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Group 2";
pptx.subject = "MedQuAD Robustness n=50 Experiments";
pptx.title = "Medical LLM Robustness to Noisy User Inputs";
pptx.company = "Columbia University";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: FONT,
  bodyFontFace: FONT,
  lang: "en-US",
};

const totalSlides = 19;
let sn = 0;
function newSlide(bg = C.paper) {
  const s = pptx.addSlide();
  addBg(s, bg);
  sn += 1;
  return s;
}

// 1. Cover
{
  const slide = newSlide(C.dark);
  addText(slide, "PRACTICUM IN LLM EVALUATION | FINAL PRESENTATION", M, 0.45, 5.3, 0.24, { size: 8, color: "B9CED7", bold: true, margin: 0 });
  addText(slide, "Medical LLM\nRobustness", M, 1.08, 5.9, 1.45, { size: 38, color: C.white, bold: true, margin: 0, fit: "shrink" });
  addText(slide, "to noisy user inputs", M, 2.55, 4.6, 0.36, { size: 17, color: "DCE9ED", margin: 0 });
  addText(slide, "We test whether prompt repair actually helps medical QA models answer messy real-user questions.", M, 3.35, 5.4, 0.74, { size: 14, color: "F2F6F7", margin: 0, fit: "shrink" });
  addRect(slide, 7.15, 0.85, 4.9, 4.9, "31424E", "31424E", false);
  addRect(slide, 7.55, 1.25, 4.1, 0.55, C.paleBlue, C.blue, true);
  addText(slide, "Clean medical question", 7.78, 1.42, 2.6, 0.17, { size: 9, color: C.dark, bold: true, margin: 0 });
  addRect(slide, 7.55, 2.12, 4.1, 0.55, C.paleAmber, C.amber, true);
  addText(slide, "Messy user question", 7.78, 2.29, 2.6, 0.17, { size: 9, color: C.dark, bold: true, margin: 0 });
  addRect(slide, 7.55, 2.99, 4.1, 0.55, C.paleTeal, C.teal, true);
  addText(slide, "Repair rewrite", 7.78, 3.16, 2.6, 0.17, { size: 9, color: C.dark, bold: true, margin: 0 });
  addRect(slide, 7.55, 3.86, 4.1, 0.55, C.paleGreen, C.green, true);
  addText(slide, "Answer + evaluate", 7.78, 4.03, 2.6, 0.17, { size: 9, color: C.dark, bold: true, margin: 0 });
  addText(slide, "n=50 pilot | 11 models | fixed_repair + self_repair", M, 5.95, 5.3, 0.32, { size: 12, color: "DCE9ED", bold: true, margin: 0 });
  addText(slide, "Group 2 | Columbia University", M, 6.35, 4.0, 0.26, { size: 9, color: "B9CED7", margin: 0 });
}

// 2. Motivation
{
  const slide = newSlide();
  addHeader(slide, "01 | Problem", "Real users do not ask clean benchmark questions.", "Medical QA systems see typos, vague wording, missing context, and patient-language phrasing.", sn, totalSlides);
  addText(slide, "A clean-input score tells us how a model behaves in the lab. It does not tell us what happens when a patient asks a messy question at midnight.", M, 1.9, 5.9, 0.86, { size: 17, color: C.ink, bold: true, fit: "shrink" });
  addRect(slide, 7.1, 1.72, 5.1, 4.25, C.white, C.line, true);
  addText(slide, "Why this matters", 7.45, 2.05, 2.6, 0.3, { size: 14, bold: true, color: C.teal2, margin: 0 });
  addText(slide, "In healthcare, a vague or misread question is not just a UX problem. It can change the medical concepts the model retrieves, the risks it mentions, and the confidence a user takes away.", 7.45, 2.55, 4.2, 1.0, { size: 12, color: C.ink, fit: "shrink" });
  addText(slide, "Research question", 7.45, 4.05, 2.8, 0.28, { size: 10, bold: true, color: C.muted, caps: true, margin: 0 });
  addText(slide, "Can a prompt-repair step recover answer quality lost under noisy user input?", 7.45, 4.4, 4.3, 0.7, { size: 15, color: C.ink, bold: true, fit: "shrink" });
  addMetric(slide, "models", "11", M, 4.9, 1.4, C.teal);
  addMetric(slide, "questions/model", "50", M + 1.7, 4.9, 1.7, C.blue);
  addMetric(slide, "noise types", "5", M + 3.7, 4.9, 1.4, C.amber);
}

// 3. Simple explainer
{
  const slide = newSlide(C.white);
  addHeader(slide, "02 | Audience bridge", "The project in one sentence", "We take real medical questions, make them messy like real user input, try to repair them, then test whether LLM answers get better or worse.", sn, totalSlides);
  addProgressFlow(slide, M, 2.0, 12.25);
  const defs = [
    ["Noise", "A controlled way of making the question harder: typo, vague phrasing, missing detail, layperson wording, or overgeneralization.", C.paleAmber, C.amber],
    ["Repair", "A rewrite step that tries to make the messy question clearer without answering it.", C.paleTeal, C.teal],
    ["Robustness", "Whether the model still gives a good answer when the user question is messy.", C.paleBlue, C.blue],
    ["Recovery", "Whether repair moves the answer back toward the clean-question answer and reference.", C.paleGreen, C.green],
  ];
  defs.forEach(([term, def, fill, accent], i) => {
    const x = M + (i % 2) * 6.15;
    const y = 3.25 + Math.floor(i / 2) * 1.35;
    addRect(slide, x, y, 5.65, 0.9, fill, accent, true);
    addText(slide, term, x + 0.18, y + 0.17, 1.3, 0.22, { size: 11, bold: true, color: accent, margin: 0 });
    addText(slide, def, x + 1.42, y + 0.14, 3.95, 0.44, { size: 8.8, color: C.ink, fit: "shrink", margin: 0 });
  });
}

// 4. Experiment design
{
  const slide = newSlide();
  addHeader(slide, "03 | Design", "Two n=50 experiments separate fairness from realism.", "The project now compares a controlled repair setup with a practical self-repair workflow.", sn, totalSlides);
  addText(slide, "Pipeline A", M, 1.95, 1.2, 0.22, { size: 9, bold: true, color: C.muted, margin: 0 });
  addText(slide, "Clean question -> answer model -> clean answer", 1.9, 1.88, 6.1, 0.36, { size: 15, bold: true, color: C.blue, margin: 0 });
  addText(slide, "Pipeline B", M, 2.75, 1.2, 0.22, { size: 9, bold: true, color: C.muted, margin: 0 });
  addText(slide, "Noisy question -> answer model -> noisy answer", 1.9, 2.68, 6.1, 0.36, { size: 15, bold: true, color: C.amber, margin: 0 });
  addText(slide, "Pipeline C", M, 3.55, 1.2, 0.22, { size: 9, bold: true, color: C.muted, margin: 0 });
  addText(slide, "Noisy question -> repair -> answer model -> repaired answer", 1.9, 3.48, 7.6, 0.36, { size: 15, bold: true, color: C.teal, margin: 0 });
  addRect(slide, M, 4.8, 5.75, 1.05, C.white, C.line, true);
  addText(slide, "fixed_repair", M + 0.22, 5.05, 1.55, 0.24, { size: 12, bold: true, color: C.teal2, margin: 0 });
  addText(slide, "All 11 models see the same clean, noisy, and repaired questions. Best for apples-to-apples comparison.", M + 1.8, 4.98, 3.55, 0.4, { size: 9.2, color: C.ink, fit: "shrink", margin: 0 });
  addRect(slide, 6.95, 4.8, 5.75, 1.05, C.white, C.line, true);
  addText(slide, "self_repair", 7.17, 5.05, 1.55, 0.24, { size: 12, bold: true, color: C.plum, margin: 0 });
  addText(slide, "All models see the same noisy questions, then each model repairs its own input. Best for practical workflow behavior.", 8.55, 4.98, 3.65, 0.4, { size: 9.2, color: C.ink, fit: "shrink", margin: 0 });
}

// 5. Setup
{
  const slide = newSlide();
  addHeader(slide, "04 | Setup", "What was tested", "The completed n=50 pilot uses 11 answer models, five evenly assigned noise types, and 11 metrics per pipeline.", sn, totalSlides);
  const metrics = [
    ["11", "answer models", "GPT-5.4 plus open-weight local/API models"],
    ["50", "questions/model", "same pilot scale across both modes"],
    ["5", "noise types", "10 examples each per model"],
    ["550", "rows/mode", "11 models x 50 questions"],
  ];
  metrics.forEach(([v, l, d], i) => {
    const x = M + i * 3.05;
    addText(slide, v, x, 1.85, 1.0, 0.45, { size: 24, bold: true, color: [C.teal, C.blue, C.amber, C.coral][i], margin: 0 });
    addText(slide, l, x, 2.35, 2.55, 0.22, { size: 10, bold: true, color: C.ink, margin: 0 });
    addText(slide, d, x, 2.68, 2.55, 0.34, { size: 8, color: C.muted, fit: "shrink", margin: 0 });
  });
  addSmallTable(slide,
    ["Noise type", "What it simulates"],
    [
      ["typos_grammar", "spelling, punctuation, casing, grammar slips"],
      ["ambiguity", "blurred or underspecified medical detail"],
      ["layperson", "patient-style wording replacing technical terms"],
      ["incomplete", "missing condition, body part, timeframe, or qualifier"],
      ["overgeneralization", "specific query broadened into a generic health question"],
    ],
    M, 3.65, [2.1, 5.0], 0.44, { boldFirst: true });
  addSmallTable(slide,
    ["Model families", "Examples"],
    [
      ["Closed/API", "GPT-5.4"],
      ["Open instruction", "Llama 3.1, Mistral, Mixtral, Qwen, Gemma, Phi"],
      ["Medical-domain", "BioMistral-7B"],
      ["Quantized local", "4-bit and Q6_K variants"],
    ],
    8.05, 3.65, [1.55, 3.15], 0.44, { boldFirst: true });
}

// 6. Metrics list
{
  const slide = newSlide(C.white);
  addHeader(slide, "05 | Evaluation", "Metrics used in the project", "The deck reports several metric families because no single metric captures medical QA robustness.", sn, totalSlides);
  addSmallTable(slide,
    ["Family", "Metric", "What it measures"],
    [
      ["Lexical overlap", "BLEU", "word n-gram overlap with the reference answer"],
      ["Lexical overlap", "chrF", "character n-gram F-score, more tolerant of wording/spelling shifts"],
      ["Lexical overlap", "ROUGE-L", "longest common subsequence overlap"],
      ["Lexical overlap", "Token F1", "normalized token precision/recall overlap"],
      ["Lexical overlap", "Exact Match", "strict normalized full-answer match; mostly zero for free-form QA"],
      ["Semantic answer", "BERTScore", "embedding-based similarity between answer and reference"],
      ["Question intent", "Intent Preservation", "embedding similarity between clean question and noisy/repaired question"],
      ["Judge score", "G-Eval", "GPT-5.4-mini score from 1-5 for answer quality"],
      ["Medical content", "med_coverage", "medical-term recall vs the reference answer"],
      ["Medical content", "med_precision", "medical-term precision in the model answer"],
      ["Medical content", "med_f1", "balance of medical-term precision and coverage"],
    ],
    M, 1.78, [2.0, 2.0, 7.25], 0.39, { bodySize: 6.8, headerSize: 7.5, boldFirst: true });
}

// 7. How to read metrics
{
  const slide = newSlide();
  addHeader(slide, "06 | Evaluation", "How to read the metrics", "A metric can improve because the question rewrite is cleaner, even if the final answer loses medical substance.", sn, totalSlides);
  const lenses = [
    ["Lexical metrics", "similar words", "BLEU, chrF, ROUGE-L, Token F1, Exact Match", C.blue, C.paleBlue],
    ["Semantic metrics", "similar meaning", "BERTScore and judge-style semantic similarity", C.teal, C.paleTeal],
    ["Intent preservation", "same question intent", "Does the noisy or repaired question still point to the clean one?", C.amber, C.paleAmber],
    ["Domain metrics", "medical content retained", "med_coverage, med_precision, med_f1 from a MedQuAD-derived lexicon", C.coral, C.paleCoral],
    ["G-Eval", "judged answer quality", "GPT-5.4-mini sees question, reference answer, and model answer, then scores 1-5", C.green, C.paleGreen],
  ];
  lenses.forEach(([name, signal, detail, accent, fill], i) => {
    const y = 1.82 + i * 0.78;
    addRect(slide, M, y, 12.1, 0.56, fill, accent, true);
    addText(slide, name, M + 0.18, y + 0.13, 2.15, 0.18, { size: 9.5, bold: true, color: accent, margin: 0 });
    addText(slide, signal, M + 2.55, y + 0.13, 2.25, 0.18, { size: 9.5, bold: true, color: C.ink, margin: 0 });
    addText(slide, detail, M + 4.95, y + 0.13, 6.85, 0.18, { size: 8.2, color: C.muted, fit: "shrink", margin: 0 });
  });
  addText(slide, "Key reading rule: repair can clean up the question while still hurting the answer. That is why the deck pairs intent metrics with answer-quality and medical-content metrics.", M, 6.05, 11.6, 0.44, { size: 13, color: C.ink, bold: true, fit: "shrink" });
}

// 8. Headline result
{
  const slide = newSlide();
  addHeader(slide, "07 | Headline result", "The n=50 pilot does not show systematic repair improvement.", "Average G-Eval falls after noise; repair does not consistently recover it.", sn, totalSlides);
  const chartData = [
    { name: "Clean", labels: ["fixed_repair", "self_repair"], values: [modeAvg.fixed_repair.clean, modeAvg.self_repair.clean] },
    { name: "Noisy", labels: ["fixed_repair", "self_repair"], values: [modeAvg.fixed_repair.noisy, modeAvg.self_repair.noisy] },
    { name: "Repaired", labels: ["fixed_repair", "self_repair"], values: [modeAvg.fixed_repair.repaired, modeAvg.self_repair.repaired] },
  ];
  slide.addChart(pptx.ChartType.bar, chartData, {
    x: M, y: 1.78, w: 7.6, h: 4.3,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 10,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 9,
    valAxisMinVal: 0,
    valAxisMaxVal: 5,
    valGridLine: { color: C.line, transparency: 30 },
    showLegend: true,
    legendPos: "b",
    showValue: false,
    showCatName: true,
    showTitle: false,
    chartColors: [C.blue, C.amber, C.teal],
  });
  addRect(slide, 8.65, 1.92, 3.8, 3.75, C.white, C.line, true);
  addText(slide, "Mode averages", 8.95, 2.2, 2.4, 0.25, { size: 14, bold: true, color: C.ink, margin: 0 });
  addText(slide, `fixed_repair\n${round(modeAvg.fixed_repair.clean)} clean -> ${round(modeAvg.fixed_repair.noisy)} noisy -> ${round(modeAvg.fixed_repair.repaired)} repaired`, 8.95, 2.78, 3.25, 0.72, { size: 13, color: C.teal2, bold: true, fit: "shrink" });
  addText(slide, `self_repair\n${round(modeAvg.self_repair.clean)} clean -> ${round(modeAvg.self_repair.noisy)} noisy -> ${round(modeAvg.self_repair.repaired)} repaired`, 8.95, 3.88, 3.25, 0.72, { size: 13, color: C.plum, bold: true, fit: "shrink" });
  addText(slide, "Interpretation: repair is not a blanket safety layer. It sometimes helps, but the average n=50 answer-quality signal stays mixed.", 8.95, 5.0, 3.1, 0.42, { size: 9.2, color: C.muted, fit: "shrink" });
}

// 9. Model comparison
{
  const slide = newSlide(C.white);
  addHeader(slide, "08 | Model comparison", "Which models recover G-Eval after repair?", "Repaired-minus-noisy G-Eval varies sharply by model and by repair mode.", sn, totalSlides);
  const fixedByModel = new Map(fixedModels.map((r) => [cleanModelName(r.Model), num(r.geval_repaired) - num(r.geval_noisy)]));
  const selfByModel = new Map(selfModels.map((r) => [cleanModelName(r.Model), num(r.geval_repaired) - num(r.geval_noisy)]));
  const labels = [...new Set([...fixedByModel.keys(), ...selfByModel.keys()])];
  labels.sort((a, b) => (selfByModel.get(b) || 0) - (selfByModel.get(a) || 0));
  slide.addChart(pptx.ChartType.bar, [
    { name: "fixed_repair", labels, values: labels.map((l) => fixedByModel.get(l) || 0) },
    { name: "self_repair", labels, values: labels.map((l) => selfByModel.get(l) || 0) },
  ], {
    x: M, y: 1.75, w: 8.25, h: 4.85,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 7.2,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 8,
    valAxisMinVal: -0.7,
    valAxisMaxVal: 1.4,
    valGridLine: { color: C.line, transparency: 30 },
    showLegend: true,
    legendPos: "b",
    showValue: false,
    chartColors: [C.teal, C.plum],
  });
  addRect(slide, 9.05, 2.0, 3.35, 3.15, C.paper, C.line, true);
  addText(slide, "What stands out", 9.28, 2.25, 2.1, 0.24, { size: 13, bold: true, color: C.ink, margin: 0 });
  addText(slide, "Self-repair has two large positive outliers: Qwen3 (+1.32) and Mixtral (+0.66).\n\nMost other models are flat or worse after repair.\n\nFixed repair is more controlled but has smaller gains: only Qwen3 and BioMistral improve G-Eval.", 9.28, 2.72, 2.8, 1.72, { size: 8.8, color: C.muted, fit: "shrink" });
}

// 10. Noise comparison
{
  const slide = newSlide();
  addHeader(slide, "09 | Noise breakdown", "Repair behaves differently by noise type.", "Bars show repaired-minus-noisy G-Eval. Values above zero mean repair helped answer quality.", sn, totalSlides);
  const labels = noiseFixed.map((r) => r.noise.replace("_", " "));
  const selfLookup = new Map(noiseSelf.map((r) => [r.noise, r.delta]));
  slide.addChart(pptx.ChartType.bar, [
    { name: "fixed_repair", labels, values: noiseFixed.map((r) => r.delta) },
    { name: "self_repair", labels, values: noiseFixed.map((r) => selfLookup.get(r.noise) || 0) },
  ], {
    x: M, y: 1.78, w: 8.4, h: 4.8,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 8,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 8,
    valAxisMinVal: -0.35,
    valAxisMaxVal: 0.15,
    valGridLine: { color: C.line, transparency: 20 },
    showLegend: true,
    legendPos: "b",
    chartColors: [C.teal, C.plum],
  });
  addText(slide, "Reading", 9.15, 2.05, 1.2, 0.24, { size: 13, bold: true, color: C.ink, margin: 0 });
  addText(slide, "Self-repair helps typos/grammar and incomplete questions slightly on average.\n\nLayperson phrasing and overgeneralization remain difficult: repair often preserves or amplifies the wrong framing.\n\nAverages hide examples where repair helps strongly and examples where it fails completely.", 9.15, 2.48, 3.0, 2.0, { size: 9.2, color: C.muted, fit: "shrink" });
}

// 11. Metric tension
{
  const slide = newSlide(C.white);
  addHeader(slide, "10 | Metric tension", "Repair cleans up intent more reliably than it improves answers.", "Across 11 models, intent preservation improves everywhere, but answer and domain metrics often move the other way.", sn, totalSlides);
  const metricRows = [
    ["G-Eval", metricWinCounts(fixedModels, "geval"), metricWinCounts(selfModels, "geval")],
    ["BERTScore", metricWinCounts(fixedModels, "bertscore"), metricWinCounts(selfModels, "bertscore")],
    ["med_coverage", metricWinCounts(fixedModels, "med_coverage"), metricWinCounts(selfModels, "med_coverage")],
    ["med_f1", metricWinCounts(fixedModels, "med_f1"), metricWinCounts(selfModels, "med_f1")],
    ["Intent preservation", metricWinCounts(fixedModels, "intent_preservation"), metricWinCounts(selfModels, "intent_preservation")],
  ];
  addSmallTable(slide,
    ["Metric", "fixed: improved/worse/flat", "self: improved/worse/flat"],
    metricRows.map(([name, f, s]) => [name, `${f.improved}/${f.worse}/${f.flat}`, `${s.improved}/${s.worse}/${s.flat}`]),
    M, 1.85, [3.0, 3.55, 3.55], 0.55, { bodySize: 9, headerSize: 8, boldFirst: true });
  addRect(slide, 1.0, 5.35, 11.2, 0.72, C.paleAmber, C.amber, true);
  addText(slide, "Core pattern", 1.25, 5.56, 1.5, 0.18, { size: 10, bold: true, color: C.amber, margin: 0 });
  addText(slide, "The repair step usually makes the question closer to the original intent. That does not guarantee the answer is better.", 2.85, 5.52, 8.8, 0.24, { size: 11, bold: true, color: C.ink, margin: 0, fit: "shrink" });
}

// Evidence slides
function addEvidenceSlide({ kicker, title, row, cleanLabel, noisyLabel, repairedLabel, takeaway, n }) {
  const slide = newSlide();
  addHeader(slide, kicker, title, `${row.model} | ${row.noise_type} noise | question_id ${row.question_id}`, sn, totalSlides);
  addText(slide, "Question path", M, 1.78, 1.4, 0.18, { size: 8.2, color: C.muted, bold: true, margin: 0 });
  addText(slide, excerpt(row.question_clean, 115), M, 2.05, 3.35, 0.36, { size: 8.3, color: C.blue, bold: true, fit: "shrink" });
  addText(slide, "->", 4.0, 2.11, 0.25, 0.2, { size: 11, color: C.muted, bold: true, margin: 0 });
  addText(slide, excerpt(row.question_noisy, 130), 4.35, 2.05, 3.35, 0.36, { size: 8.3, color: C.amber, bold: true, fit: "shrink" });
  addText(slide, "->", 7.82, 2.11, 0.25, 0.2, { size: 11, color: C.muted, bold: true, margin: 0 });
  addText(slide, excerpt(row.question_repaired, 150), 8.15, 2.05, 4.1, 0.36, { size: 8.3, color: C.teal2, bold: true, fit: "shrink" });
  answerPanel(slide, cleanLabel, row.answer_clean, M, 2.85, 3.8, 2.55, C.white, C.blue);
  answerPanel(slide, noisyLabel, row.answer_noisy || "[blank answer]", 4.78, 2.85, 3.8, 2.55, C.white, C.amber);
  answerPanel(slide, repairedLabel, row.answer_repaired || "[blank answer]", 9.02, 2.85, 3.18, 2.55, C.white, C.teal);
  scoreBand(slide, row, M, 5.75, 4.4);
  addText(slide, takeaway, 5.1, 5.62, 6.85, 0.54, { size: 11.2, color: C.ink, bold: true, fit: "shrink" });
}

addEvidenceSlide({
  kicker: "11 | Evidence example",
  title: "Incomplete wording can make the answer collapse.",
  row: evidence.air,
  cleanLabel: "CLEAN ANSWER",
  noisyLabel: "NOISY ANSWER",
  repairedLabel: "REPAIRED",
  takeaway: "The noisy question asks for 'bad air stuff' and GPT-5.4 answers with a fragment. Repair restores much of the answer quality.",
});

addEvidenceSlide({
  kicker: "12 | Evidence example",
  title: "Noisy wording can shift the topic.",
  row: evidence.steroids,
  cleanLabel: "CLEAN ANSWER",
  noisyLabel: "NOISY ANSWER",
  repairedLabel: "REPAIRED",
  takeaway: "The clean topic is anabolic steroids; the noisy answer drifts into generic supplements like protein powder and creatine.",
});

addEvidenceSlide({
  kicker: "13 | Evidence example",
  title: "Layperson phrasing changes the frame.",
  row: evidence.abortion,
  cleanLabel: "CLEAN ANSWER",
  noisyLabel: "NOISY ANSWER",
  repairedLabel: "REPAIRED",
  takeaway: "Here repair helps: it restores a concise medical distinction between spontaneous and induced abortion.",
});

addEvidenceSlide({
  kicker: "14 | Evidence example",
  title: "A better judge score can still lose medical content.",
  row: evidence.anal,
  cleanLabel: "CLEAN ANSWER",
  noisyLabel: "NOISY ANSWER",
  repairedLabel: "REPAIRED",
  takeaway: "Mixtral self-repair improves G-Eval from 3 to 5, but med_coverage falls. This is why the metric stack matters.",
});

// 16. Interpretation
{
  const slide = newSlide();
  addHeader(slide, "15 | Interpretation", "What the n=50 evidence means", "The project is not asking whether repair can produce nicer questions. It asks whether nicer questions produce better medical answers.", sn, totalSlides);
  const points = [
    ["Repair is useful, but bounded.", "It can fix surface noise and some missing-context cases, yet it cannot reliably recover medical meaning once the user prompt has drifted."],
    ["Question repair adds another failure mode.", "The repair model can commit to the wrong interpretation or broaden the target before the answer model ever sees it."],
    ["Metric choice changes the story.", "Intent scores often improve while G-Eval and medical-term coverage do not. The safest read uses multiple lenses."],
  ];
  points.forEach(([head, body], i) => {
    const y = 1.9 + i * 1.35;
    addText(slide, `0${i + 1}`, M, y, 0.45, 0.3, { size: 14, bold: true, color: [C.teal, C.coral, C.blue][i], margin: 0 });
    addText(slide, head, 1.08, y, 4.1, 0.28, { size: 16, bold: true, color: C.ink, margin: 0 });
    addText(slide, body, 1.08, y + 0.42, 9.9, 0.4, { size: 11.2, color: C.muted, fit: "shrink", margin: 0 });
  });
}

// 17. Limitations
{
  const slide = newSlide(C.white);
  addHeader(slide, "16 | Limitations", "What this pilot cannot yet claim", "The deck is deliberately conservative: these are completed n=50 pilot experiments, not the final 1,000-question study.", sn, totalSlides);
  const rows = [
    ["Pilot scale", "50 questions per model. Good for signal-finding, not final statistical claim-making."],
    ["Synthetic noise", "LLM-generated noise is controlled, but real patient language can be messier and multi-modal."],
    ["Single dataset", "MedQuAD is curated NIH/NLM content, not patient portal messages or EHR notes."],
    ["Judge model overlap", "G-Eval uses GPT-5.4-mini; the repair/noise stack also uses GPT-family components."],
    ["Domain metric limits", "Medical-term coverage catches lost terms, but it does not judge clinical harm or relevance."],
  ];
  addSmallTable(slide, ["Limit", "Why it matters"], rows, M, 1.86, [2.2, 8.6], 0.65, { bodySize: 9.2, headerSize: 8.5, boldFirst: true });
}

// 18. Next steps
{
  const slide = newSlide();
  addHeader(slide, "17 | Next steps", "What would turn the pilot into a stronger study", "The next phase should test scale, repair validation, and clinical safety rather than only adding more aggregate metrics.", sn, totalSlides);
  const steps = [
    ["Scale to n=1,000", "Run the same fixed_repair and self_repair structure on a larger MedQuAD sample with confidence intervals."],
    ["Validate repair outputs", "Reject rewrites that answer the question, ask for more context, or inject facts not present in the noisy input."],
    ["Human/clinical review", "Pair automatic scores with a harm-tier rubric for misleading, unsafe, or clinically dangerous answers."],
    ["Noise-specific policy", "Deploy repair selectively: typo/incomplete questions may benefit; overgeneralized or ambiguous prompts may need clarification instead."],
  ];
  steps.forEach(([head, body], i) => {
    const x = M + (i % 2) * 6.1;
    const y = 1.95 + Math.floor(i / 2) * 1.55;
    addRect(slide, x, y, 5.55, 1.06, C.white, C.line, true);
    addText(slide, head, x + 0.2, y + 0.18, 2.55, 0.24, { size: 13, bold: true, color: [C.teal, C.blue, C.amber, C.coral][i], margin: 0 });
    addText(slide, body, x + 0.2, y + 0.54, 4.9, 0.36, { size: 8.8, color: C.muted, fit: "shrink", margin: 0 });
  });
  addText(slide, "Practical takeaway: prompt repair should be evaluated as a conditional tool, not assumed to be a universal safety layer.", M, 5.65, 10.8, 0.38, { size: 15, bold: true, color: C.ink, fit: "shrink" });
}

// 19. Thank you
{
  const slide = newSlide(C.dark);
  addText(slide, "Thank you.", M, 1.55, 4.8, 0.78, { size: 42, color: C.white, bold: true, margin: 0 });
  addText(slide, "Questions, pushback, and especially counter-evidence welcome.", M, 2.62, 6.0, 0.32, { size: 14, color: "DCE9ED", margin: 0 });
  addText(slide, "Group 2 | Practicum in LLM Evaluation", M, 5.6, 4.6, 0.22, { size: 9, color: "B9CED7", margin: 0 });
  addText(slide, "MedQuAD Robustness | n=50 final presentation", M, 5.95, 4.6, 0.22, { size: 9, color: "B9CED7", margin: 0 });
  addRect(slide, 7.3, 1.35, 4.4, 3.4, "31424E", "31424E", true);
  addText(slide, "Main answer", 7.75, 1.85, 2.0, 0.26, { size: 13, color: C.white, bold: true, margin: 0 });
  addText(slide, "Repair improved question intent in every model, but it did not systematically improve medical QA answer quality in the completed n=50 pilot.", 7.75, 2.35, 3.35, 1.05, { size: 17, color: "F2F6F7", bold: true, fit: "shrink", margin: 0 });
}

async function renderPreviews() {
  fs.rmSync(PREVIEW_DIR, { recursive: true, force: true });
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });
  const artifact = await import(pathToFileURL(ARTIFACT_TOOL).href);
  const pres = await artifact.PresentationFile.importPptx(fs.readFileSync(OUT_PPTX));
  const paths = [];
  for (let i = 0; i < pres.slides.count; i += 1) {
    const blob = await pres.slides.items[i].export("image/png");
    const ab = await blob.arrayBuffer();
    const out = path.join(PREVIEW_DIR, `slide_${String(i + 1).padStart(2, "0")}.png`);
    fs.writeFileSync(out, Buffer.from(ab));
    paths.push(out);
  }
  return paths;
}

async function main() {
  fs.mkdirSync(path.dirname(OUT_PPTX), { recursive: true });
  fs.rmSync(OUT_PPTX, { force: true });
  await pptx.writeFile({ fileName: OUT_PPTX });
  const previews = await renderPreviews();
  const summary = {
    pptx: OUT_PPTX,
    previews: PREVIEW_DIR,
    slideCount: totalSlides,
    fixedRows: fixedSamples.length,
    selfRows: selfSamples.length,
    fixedGEval: modeAvg.fixed_repair,
    selfGEval: modeAvg.self_repair,
  };
  fs.writeFileSync(path.join(PREVIEW_DIR, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  console.log(`Rendered ${previews.length} PNG previews.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
