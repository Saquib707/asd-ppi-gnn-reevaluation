"""
Step 8: pre-submission compliance check (ICACECT 2027 / IEEE conference rules).

Fails on anything that would get the paper desk-rejected or embarrass it:
more than 6 pages, a page size other than US Letter, more than 5 MB,
non-embedded fonts, leftover draft markers ([[...]], TODO) or unresolved
references ([?], ??) in the PDF text, undefined references or citations or
overfull boxes in the LaTeX log, pending macros in generated/numbers.tex,
IEEE template guidance text, and page numbers.

Usage: python 08_check.py [path/to/main.pdf]
Exit status 1 if any check fails.
"""
import os
import re
import subprocess
import sys

from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.join(HERE, "..", "latex")
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LATEX, "main.pdf")
LOG = os.path.splitext(PDF)[0] + ".log"
TEMPLATE_TEXT = ["Identify applicable funding agency", "dept. name of organization",
                 "Conference Paper Title", "Sub-titles are not captured",
                 "Please ensure that all template text is removed"]

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def main():
    reader = PdfReader(PDF)
    n = len(reader.pages)
    check("at most 6 pages", n <= 6, f"{n} pages")
    box = reader.pages[0].mediabox
    w, h = float(box.width), float(box.height)
    check("US Letter (612 x 792 pt)", abs(w - 612) < 1 and abs(h - 792) < 1,
          f"{w:.0f} x {h:.0f} pt")
    size = os.path.getsize(PDF)
    check("file size at most 5 MB", size <= 5 * 1024 * 1024, f"{size / 1e6:.2f} MB")

    out = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
    rows = [l for l in out.splitlines()[2:] if l.strip()]
    missing = [l.split()[0] for l in rows if l.split()[-5] != "yes"]
    check("all fonts embedded", rows and not missing,
          f"{len(rows)} fonts" + (f"; not embedded: {missing}" if missing else ""))

    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n".join(pages)
    for label, pat in (("draft markers [[...]]", r"\[\["), ("TODO", r"TODO"),
                       ("unresolved citation [?]", r"\[\?\]"),
                       ("unresolved reference ??", r"\?\?")):
        hits = re.findall(pat, text)
        check(f"no {label} in PDF", not hits, f"{len(hits)} found" if hits else "")
    tmpl = [t for t in TEMPLATE_TEXT if t.lower() in text.lower()]
    check("no IEEE template guidance text", not tmpl, "; ".join(tmpl))
    if "anonymous" in os.path.basename(PDF).lower():
        leaks = [w for w in ("Mazid", "Warsi", "Sharma", "Fathima", "mazidgaba", "ORCID",
                             "gmail", "manuu", "jnu.ac.in", "Warangal", "github.com")
                 if w.lower() in text.lower()]
        check("anonymous copy reveals no author identity", not leaks, "; ".join(leaks))
    else:
        check("named copy has real authors", "Anonymous Submission" not in text)
    numbered = [i + 1 for i, t in enumerate(pages)
                if t.strip().splitlines() and t.strip().splitlines()[-1].strip() == str(i + 1)]
    check("no page numbers", not numbered, f"on pages {numbered}" if numbered else "")

    log = open(LOG, encoding="latin-1").read() if os.path.exists(LOG) else ""
    check("LaTeX log present", bool(log), LOG)
    for label, pat in (("undefined references", r"LaTeX Warning: Reference .* undefined"),
                       ("undefined citations", r"LaTeX Warning: Citation .* undefined"),
                       ("LaTeX errors", r"^! "),
                       ("overfull boxes", r"Overfull \\[hv]box")):
        hits = re.findall(pat, log, flags=re.M)
        check(f"no {label}", not hits, f"{len(hits)} found" if hits else "")

    nums = os.path.join(LATEX, "generated", "numbers.tex")
    pending = re.findall(r"\\newcommand\{\\(\w+)\}\{\\TODO", open(nums).read()) \
        if os.path.exists(nums) else ["numbers.tex missing"]
    check("no pending result macros", not pending,
          f"{len(pending)} pending: {', '.join(pending[:6])}" if pending else "")

    width = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
