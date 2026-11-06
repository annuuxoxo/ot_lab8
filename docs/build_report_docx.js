// Build docs/Assignment8_Report.docx from the repo's Markdown files.
// Usage:  npm install docx   (once, anywhere on NODE_PATH)   then   node docs/build_report_docx.js
// Structure follows the brief: Sections 8 (AI use), 9 (submission requirements) and 11 (final questions).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, ImageRun, PageBreak, Footer, PageNumber, TableOfContents, BorderStyle,
} = require("docx");

const ROOT = path.dirname(__dirname);
const read = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");
const PAGE_W = 9026;                         // A4 width minus 2 × 1" margins, in DXA
const FONT = "Calibri";

// ------------------------------------------------------------------ markdown helpers
// Split a markdown file into { heading: body } on lines starting with `prefix` (e.g. "## ").
function sections(md, prefix) {
  const out = { _intro: [] };
  let cur = "_intro";
  for (const line of md.split("\n")) {
    if (line.startsWith(prefix)) { cur = line.slice(prefix.length).trim(); out[cur] = []; }
    else out[cur].push(line);
  }
  for (const k of Object.keys(out)) out[k] = out[k].join("\n").trim();
  return out;
}
const pick = (secs, start) => {
  const k = Object.keys(secs).find((h) => h.startsWith(start));
  if (!k) throw new Error("missing section: " + start);
  return secs[k];
};

// Inline markdown -> TextRuns (bold, italic, code, links shown as text).
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|(?<![\w*])\*[^*\s][^*]*\*(?![\w*])|(?<![\w])_[^_]+_(?![\w]))/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Consolas", size: (base.size || 21) - 2, color: "333333" }));
    else if (t.startsWith("[")) {
      const lm = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(t);
      out.push(...runs(lm[1], base));
    } else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

let listInstance = 0;
function table(rows, fontSize) {
  const header = rows[0];
  const body = rows.slice(2);
  const n = header.length;
  // column widths proportional to (clipped) content length
  const len = Array.from({ length: n }, (_, j) =>
    Math.min(55, Math.max(6, ...rows.filter((_, i) => i !== 1).map((r) => (r[j] || "").replace(/[*`]/g, "").length))));
  const total = len.reduce((a, b) => a + b, 0);
  const widths = len.map((l) => Math.floor((l / total) * PAGE_W));
  widths[n - 1] += PAGE_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (txt, j, head) => new TableCell({
    width: { size: widths[j], type: WidthType.DXA }, borders,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF4" } : undefined,
    children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: runs(txt || "", { size: fontSize, bold: head || undefined }) })],
  });
  return new Table({
    width: { size: PAGE_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((c, j) => cell(c, j, true)) }),
      ...body.map((r) => new TableRow({ cantSplit: true, children: Array.from({ length: n }, (_, j) => cell(r[j], j, false)) }))],
  });
}

// Markdown block -> docx children. headingShift maps "#" level to output heading level.
function blocks(md, { headingShift = 0, tableFont = 17 } = {}) {
  const out = [];
  const lines = md.split("\n");
  let para = [];
  let listOpen = false;
  const flush = () => { if (para.length) { out.push(new Paragraph({ children: runs(para.join(" ")) })); para = []; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const s = line.trim();
    const item = /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(line);
    if (!s) { flush(); if (!(lines[i + 1] || "").match(/^\s+\S|^\s*([-*]|\d+\.)\s/)) listOpen = false; continue; }
    if (s.startsWith("```")) {
      flush();
      let j = i + 1;
      while (j < lines.length && !lines[j].trim().startsWith("```")) {
        out.push(new Paragraph({ spacing: { before: 0, after: 0 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "F4F4F4" },
          children: [new TextRun({ text: lines[j] || " ", font: "Consolas", size: 17 })] }));
        j++;
      }
      i = j; continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(s);
    if (h) {
      flush(); listOpen = false;
      const lvl = Math.min(3, h[1].length + headingShift);
      out.push(new Paragraph({ heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][lvl - 1], children: runs(h[2].replace(/\*\*/g, "")) }));
      continue;
    }
    if (s === "---") { flush(); continue; }
    if (s.startsWith("|")) {
      flush();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        i++;
      }
      i--;
      out.push(table(rows, tableFont));
      out.push(new Paragraph({ spacing: { before: 0, after: 60 }, children: [] }));
      continue;
    }
    if (item) {
      flush();
      const level = Math.min(2, Math.floor(item[1].length / 2));
      const numbered = /\d/.test(item[2][0]);
      if (!listOpen || numbered && level === 0 && item[2] === "1.") listInstance++;
      listOpen = true;
      out.push(new Paragraph({ numbering: { reference: numbered ? "numbers" : "bullets", level, instance: listInstance },
        spacing: { before: 0, after: 40 }, children: runs(item[3]) }));
      continue;
    }
    if (listOpen && /^\s+\S/.test(line)) {           // continuation of the previous list item
      const prev = out.pop();
      out.push(prev);
      out.push(new Paragraph({ indent: { left: 720 }, spacing: { before: 0, after: 40 }, children: runs(s) }));
      continue;
    }
    listOpen = false;
    para.push(s);
  }
  flush();
  return out;
}

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const P = (t) => new Paragraph({ children: runs(t) });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(rel, caption) {
  const file = path.join(ROOT, "results", rel);
  const { w, h } = pngSize(file);
  const width = Math.min(600, w), height = Math.round((h / w) * width);
  const maxH = 330;
  const scale = height > maxH ? maxH / height : 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
      transformation: { width: Math.round(width * scale), height: Math.round(height * scale) }, altText: { title: caption, description: caption, name: rel } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(`${caption} (\`results/${rel}\`)`, { size: 18, italics: true }) }),
  ];
}

// ------------------------------------------------------------------ content
const report = sections(read("docs/report.md"), "## ");
const trackE = sections(read("trackE_case_study/README.md"), "## ");
const readme = sections(read("README.md"), "## ");
const failure = read("docs/failure_log.md").split("\n").filter((l) => l.trim().startsWith("|")).join("\n");
const checklist = read("docs/chapter6_checklist.md").split("\n").filter((l) => l.trim().startsWith("|")).join("\n");
const toolCmp = read("docs/tool_comparison.md").replace(/^# .*\n/, "");
const trackKeys = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J."];
const trackFolders = {
  "A.": "trackA_continuous", "B.": "trackB_discrete", "C.": "trackC_quantum_opt", "D.": "trackD_submodular_parallel_distributed",
  "E.": "trackE_case_study", "F.": "trackF_compiler", "G.": "trackG_xv6", "H.": "trackH_dbms", "I.": "trackI_quantum_algos_qml", "J.": "trackJ_software_eng",
};

const children = [];
// title page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 120 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "OPTIMIZATION TECHNIQUES | LAB ASSIGNMENT", size: 22, color: "555555" })] }),
  new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: [new TextRun("Assignment 8: Optimize Anything")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "14-Day Vibe-Coding Challenge: Technical Report", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Main case study (Track E): university exam timetabling", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 900 }, children: [new TextRun({ text: "All ten compulsory tracks A–J", size: 22, color: "555555" })] }),
  ...["Name: ______________________________", "Roll number: ______________________", "Repository: https://github.com/<your-username>/ot_lab8",
    "Date: November 2026"].map((t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: t, size: 22 })] })),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: "1F3864" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  P("_Right-click the table and choose “Update Field” if the page numbers are missing._"),
);

// 1. Introduction
children.push(H1("1. Introduction"), ...blocks(report._intro.replace(/^# .*\n/, "")),
  H2("Hardware and software"), ...blocks(readme["Hardware used for all measurements"]),
  P("Software: Python 3.12 (Homebrew, arm64), NumPy, SciPy, PyTorch, PuLP 2.9 (CBC), Qiskit 2.6 + Aer, PennyLane, dwave-neal, minorminer, DuckDB, SQLite, QEMU + riscv64-elf-gcc. Exact versions are in `requirements.txt`."),
  H2("How the brief's problem-solving process (Section 4) was applied"),
  ...blocks(`| step | how it was done |
|---|---|
| 1. Discover the process | Each track starts from a measured inefficiency (Section 4, question 1) |
| 2. Establish the baseline | Every track has a named baseline measured on the same hardware and data (first-fit, round-robin, no-index SQLite, O0 IR, full test suite, plain GD, greedy, …) |
| 3. Formulate the problem | Decision variables, objective and constraints are stated per track (Section 2 for the case study, Section 3 for the others) |
| 4. Select the method | Strategy-selection tables in Section 2.5 (case study) and Track D (Chapter 6 methods) |
| 5. Use vibe coding | Claude Code generated the code; every prompt, revision and fix is logged (Section 6) |
| 6. Experiment and compare | Fair comparisons: same data, budget, tuning grid and hardware; repeated runs for stochastic methods |
| 7. Analyze trade-offs | Each track reports its trade-offs and negative results; summary in Section 7 |`));

// 2. Main case study
children.push(H1("2. Main case study (Track E): university exam timetabling"),
  P("Code: `trackE_case_study/timetabling.py`, `e_experiments.py` · Results: `results/trackE_case_study/` · Tests: `tests/test_trackE.py`"),
  H2("2.1 Problem statement and evidence"), ...blocks(pick(trackE, "1.")),
  H2("2.2 Mathematical formulation and assumptions"), ...blocks(pick(trackE, "2.")),
  H2("2.3 Baseline and methods"), ...blocks(pick(trackE, "3.")),
  H2("2.4 Results"), ...blocks(pick(trackE, "4.")),
  ...figure("trackE_case_study/e_objective.png", "Figure 1: Objective per method, mean ± std over 3 seeds"),
  H2("2.5 Strategy selection"), ...blocks(pick(trackE, "5.")),
  H2("2.6 Correctness"), ...blocks(pick(trackE, "6.")),
  H2("2.7 Limitations"), ...blocks(pick(trackE, "Limitations")));

// Track-specific excerpts the brief asks for: the xv6 patch (G) and DBMS query plans (H).
function EXTRA_G() {
  const patch = read("trackG_xv6/mlfq_scheduler.patch").split("\n");
  const files = patch.filter((l) => l.startsWith("diff --git")).map((l) => l.split(" ")[2].slice(2));
  const start = patch.findIndex((l) => l.startsWith("+// Called from the timer interrupt"));
  const end = patch.findIndex((l, i) => i > start && l === "+}");
  const code = patch.slice(start, end + 1).map((l) => l.slice(1)).join("\n");
  return [new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Patch excerpt (trackG_xv6/mlfq_scheduler.patch)")] }),
    P(`The patch changes ${files.length} files against xv6-riscv commit \`e93dabe\`: ` + files.map((f) => "`" + f + "`").join(", ") + ". Below is the core of the policy: the timer-interrupt hook that demotes a process that used its whole slice, and preempts it early when a higher-priority process is runnable."),
    ...blocks("```\n" + code + "\n```")];
}
function EXTRA_H() {
  const plans = read("results/trackH_dbms/h_query_plans.txt").split("\n");
  const q10 = (header) => {
    const h = plans.findIndex((l) => l.startsWith("===== " + header));
    const s = plans.findIndex((l, i) => i > h && l.startsWith("Q10_"));
    const e = plans.findIndex((l, i) => i > s && /^\S/.test(l));
    return plans.slice(s + 1, e).map((l) => l.trim()).join("\n");
  };
  return [new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Query plans for Q10 (results/trackH_dbms/h_query_plans.txt)")] }),
    P("**No indexes (baseline):** SQLite builds temporary automatic indexes and Bloom filters."), ...blocks("```\n" + q10("baseline") + "\n```"),
    P("**25% budget (greedy = DP = ILP set):** the new index on `lineitem(l_orderkey)` replaces the automatic index."), ...blocks("```\n" + q10("greedy (benefit/size), budget 25%") + "\n```"),
    P("**All 13 indexes:** the plan starts from `orders` by date range. `o_orderdate` on its own produced a plan that exceeded the 5 s cap, but with `l_orderkey` present it is the fastest plan."), ...blocks("```\n" + q10("all candidates") + "\n```")];
}
const EXTRAS = { "G.": EXTRA_G, "H.": EXTRA_H };

// 3. Tracks A–J
children.push(H1("3. Experiment tracks A–J"),
  P("Each subsection summarizes one compulsory track: its objective, baseline, results and correctness check. It links to the code folder and the result files. The full tables are in each track's README.md."));
for (const key of trackKeys) {
  const title = Object.keys(report).find((h) => h.startsWith(key));
  let body = report[title];
  if (key === "E.") body = "This is the main case study; the full write-up is Section 2. Summary:\n\n" + body;
  children.push(H2(title.replace(/\s*\(\[README\]\([^)]*\)[^)]*\)/, "")),
    P(`Code: \`${trackFolders[key]}/\` · Results: \`results/${trackFolders[key]}/\` · Tests: \`tests/test_track${key[0]}.py\``),
    ...blocks(body), ...(EXTRAS[key] ? EXTRAS[key]() : []));
}

// 4. Final questions
children.push(H1("4. Answers to the final questions (brief Section 11)"),
  ...blocks(pick(report, "Answers to the 10 final questions")));

// 5. Chapter 6 checklist
children.push(H1("5. Chapter 6 coverage checklist"),
  P("Every taught method mapped to its experiment, code location and result; also in `docs/chapter6_checklist.md`. “Simulated” = run on a simulator or as a single-machine simulation; “measured” = timed or counted on real hardware (Apple M4, or QEMU for xv6)."),
  ...blocks(checklist, { tableFont: 15 }));

// 6. AI use
children.push(H1("6. Responsible AI use (brief Section 8)"),
  H2("6.1 AI tools register"), ...blocks(readme["AI tools register"]),
  H2("6.2 Prompt logs and commit traceability"),
  P("`prompts/` has one sub-folder per track (and `setup/`). Each log holds the original prompt, an excerpt of the AI response, the revised prompt and corrections, the commit hash and a verification note. Commits are tagged `[AI:claude-code]` for AI-generated work and `[manual]` for the student's own edits, so `git log` shows what was generated and what was corrected. Secrets scan (2026-10-30): no API keys, tokens or passwords in the files or the git history."),
  H2("6.3 External code and libraries"), ...blocks(readme["External code and libraries"]),
  H2("6.4 Failure log: AI-generated code that was wrong, unsafe, inefficient or misleading"),
  P("20 cases (at least 3 are required); also in `docs/failure_log.md`."),
  ...blocks(failure, { tableFont: 14 }),
  H2("6.5 Tool comparison"), ...blocks(toolCmp.replace(/^## /gm, "### ")));

// 7. Limitations, trade-offs, next steps
children.push(H1("7. Limitations, trade-offs and next steps"),
  H2("7.1 Self-review"), ...blocks(pick(report, "Review notes")),
  H2("7.2 Main trade-offs found"),
  ...blocks(`- **Exact vs heuristic:** ILP and B&B prove optimality on small instances (B, E, F, H, J), but local search and greedy get within 0–6% in milliseconds and are the only option at scale (E at 160 courses).
- **The model matters more than the solver:** non-additive index benefits (H), order-dependent test costs (J) and pass interactions (F) changed results more than the choice between greedy, DP and ILP.
- **Quality vs cost of verification:** a 2.5× faster test suite misses 9% of the planted faults (J); MLFQ helps interactive jobs but hurts short batch jobs (G).
- **Quantum vs classical:** the query-count advantages are real in theory, but every quantum result was slower or no better in practice, because of simulation cost, noise and barren plateaus (C, I). Parallel speedups are real but bounded (D).`),
  H2("7.3 Next steps"),
  ...blocks(`- **E:** use real (anonymized) enrolment data; add invigilators and student-preference constraints; parallel restarts of local search; compare with a commercial MIP solver's multi-threaded branch-and-bound.
- **J:** branch coverage or mutation-based selection, to recover the 3 missed faults; incremental re-profiling of changed files in CI; a larger project.
- **H:** PostgreSQL and join-order selection (H2), plus interaction-aware index selection (e.g. a quadratic knapsack).
- **G:** a second kernel feature (copy-on-write fork) and multi-CPU scheduling benchmarks.
- **C and I:** run the shallow circuits on IBM Quantum hardware to replace the noise model with real device data.`));

// 8. Reproducibility
children.push(H1("8. Reproducibility"),
  H2("Setup"), ...blocks(readme["Setup"]),
  H2("Running each track"), ...blocks(readme["How to run each track"]),
  P("Fixed seeds are used everywhere. Every number in this report comes from a CSV in `results/` produced by the track's `run.py`. Timing results depend on the machine; Section 1 gives the hardware."));

// Appendix: plots
const FIGS = [
  ["trackA_continuous/a1_saddle.png", "Figure 2: Track A, optimizer paths on the saddle function"],
  ["trackB_discrete/b1_knapsack.png", "Figure 3: Track B, knapsack runtime and greedy gap vs n"],
  ["trackC_quantum_opt/c1_qaoa.png", "Figure 4: Track C, QAOA approximation ratio vs depth (simulated)"],
  ["trackC_quantum_opt/c2_barren_plateau.png", "Figure 5: Track C, barren plateau (simulated)"],
  ["trackD_submodular_parallel_distributed/d1_scheduling.png", "Figure 6: Track D, parallel speedup per strategy (measured)"],
  ["trackD_submodular_parallel_distributed/d2_fedavg.png", "Figure 7: Track D, FedAvg vs centralized (simulated)"],
  ["trackI_quantum_algos_qml/i1_grover_vs_linear.png", "Figure 8: Track I, Grover vs linear search"],
  ["trackJ_software_eng/j_minimization.png", "Figure 9: Track J, test-suite minimization"],
];
children.push(H1("Appendix: key plots"));
for (const [rel, cap] of FIGS) children.push(...figure(rel, cap));

// ------------------------------------------------------------------ document
const heading = (size, before) => ({ run: { font: FONT, size, bold: true, color: "1F3864" },
  paragraph: { spacing: { before, after: 100 }, keepNext: true } });
const doc = new Document({
  creator: "Assignment 8", title: "Assignment 8: Optimize Anything, Technical Report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 21 }, paragraph: { spacing: { after: 100, line: 264 } } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 52, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(32, 0), paragraph: { spacing: { before: 0, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(26, 240), paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(22, 180), paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [0, 1, 2].map((l) => ({ level: l, format: LevelFormat.BULLET, text: ["•", "◦", "▪"][l], alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360 * (l + 1), hanging: 260 } } } })) },
      { reference: "numbers", levels: [0, 1, 2].map((l) => ({ level: l, format: LevelFormat.DECIMAL, text: `%${l + 1}.`, alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360 * (l + 1), hanging: 300 } } } })) },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Assignment 8: Optimize Anything · page ", size: 16, color: "777777" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "777777" })] })] }) },
    children,
  }],
});

const outPath = path.join(ROOT, "docs", "Assignment8_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath, Math.round(buf.length / 1024), "KB"); });
