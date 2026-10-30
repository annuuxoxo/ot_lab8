"""Build docs/report.pdf from docs/report.md: python docs/build_report.py

A small Markdown-to-HTML converter for the subset the report uses (headings, paragraphs, nested bullet and
numbered lists, tables, bold, inline code, links, rules), plus an appendix with one key plot per track.
The HTML is printed to PDF with headless Google Chrome.
"""
import html
import os
import re
import subprocess
import sys

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

FIGURES = [
    ("A", "trackA_continuous/a1_saddle.png", "Optimizer paths on the saddle function: Newton converges to the saddle point."),
    ("B", "trackB_discrete/b1_knapsack.png", "Knapsack runtime and greedy gap vs n."),
    ("C", "trackC_quantum_opt/c1_qaoa.png", "QAOA approximation ratio vs depth p (simulated)."),
    ("C", "trackC_quantum_opt/c2_barren_plateau.png", "Barren plateau: gradient variance vs number of qubits (simulated)."),
    ("D", "trackD_submodular_parallel_distributed/d1_scheduling.png", "Parallel speedup per scheduling strategy (measured, M4)."),
    ("D", "trackD_submodular_parallel_distributed/d2_fedavg.png", "FedAvg vs centralized training (simulated)."),
    ("E", "trackE_case_study/e_objective.png", "Exam timetabling objective per method (mean ± std, 3 seeds)."),
    ("I", "trackI_quantum_algos_qml/i1_grover_vs_linear.png", "Grover vs linear search: oracle queries and simulation time."),
    ("J", "trackJ_software_eng/j_minimization.png", "Test-suite minimization: test time, wall time, mutants killed."),
]


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])_([^_]+)_(?![\w])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def convert(md):
    out, para, lists = [], [], []          # lists: stack of (indent, tag)
    lines = md.split("\n")
    i = 0

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    def close_lists(to_indent=-1):
        while lists and lists[-1][0] > to_indent:
            out.append(f"</li></{lists.pop()[1]}>")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        item = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if not stripped:
            flush_para()
            # a blank line ends a list unless the next non-blank line continues it
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or not (re.match(r"^\s*([-*]|\d+\.)\s", lines[j]) or lines[j].startswith("  ")):
                close_lists()
        elif stripped.startswith("```"):
            flush_para()
            close_lists()
            j = i + 1
            block = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                block.append(lines[j])
                j += 1
            out.append("<pre>" + html.escape("\n".join(block)) + "</pre>")
            i = j
        elif re.match(r"^#{1,4} ", stripped):
            flush_para()
            close_lists()
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{inline(stripped[level:].strip())}</h{level}>")
        elif stripped == "---":
            flush_para()
            close_lists()
            out.append("<hr>")
        elif stripped.startswith("|"):
            flush_para()
            if not lists:
                close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            i -= 1
            t = ["<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>"]
            for r in rows[2:]:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("".join(t) + "</tbody></table>")
        elif item:
            flush_para()
            indent, marker, text = len(item.group(1)), item.group(2), item.group(3)
            tag = "ol" if marker[0].isdigit() else "ul"
            if lists and indent < lists[-1][0]:
                close_lists(indent)
            if not lists or indent > lists[-1][0]:
                out.append(f"<{tag}><li>")
                lists.append((indent, tag))
            else:
                out.append("</li><li>")
            out.append(inline(text))
        elif lists and line.startswith(" "):
            out.append(" " + inline(stripped))       # continuation of a list item
        else:
            close_lists()
            para.append(stripped)
        i += 1
    flush_para()
    close_lists()
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 9.5pt; line-height: 1.4; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
table { border-collapse: collapse; margin: 5pt 0 8pt; font-size: 8.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 5pt; vertical-align: top; } th { background: #f0f0f0; }
code { font-family: Menlo, monospace; font-size: 8.5pt; background: #f4f4f4; padding: 0 2pt; }
pre { background: #f4f4f4; padding: 6pt; font-size: 8pt; white-space: pre-wrap; }
ul, ol { margin: 3pt 0 6pt; padding-left: 16pt; } li { margin: 1pt 0; }
a { color: #1a4fa0; text-decoration: none; } hr { border: 0; border-top: 1px solid #ccc; margin: 12pt 0; }
figure { margin: 8pt 0 12pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 240pt; } figcaption { font-size: 8.5pt; color: #444; }
"""


def main():
    md = open(os.path.join(DOCS, "report.md")).read()
    body = convert(md)
    figs = ["<h2>Appendix: key plots</h2>"]
    for track, rel, caption in FIGURES:
        path = os.path.join(ROOT, "results", rel)
        if os.path.exists(path):
            figs.append(f'<figure><img src="file://{path}"><figcaption>Track {track}: {html.escape(caption)} '
                        f"(<code>results/{rel}</code>)</figcaption></figure>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Assignment 8 report</title>"
            f"<style>{CSS}</style></head><body>{body}{''.join(figs)}</body></html>")
    html_path = os.path.join(DOCS, "report.html")
    open(html_path, "w").write(page)
    pdf = os.path.join(DOCS, "report.pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf}", f"file://{html_path}"],
                   check=True, capture_output=True, timeout=120)
    print("wrote", pdf, os.path.getsize(pdf) // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
