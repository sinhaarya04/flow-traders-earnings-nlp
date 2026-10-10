"""Extract the CEO commentary (management narrative) block from each quarterly release."""
import re, pathlib

TXT = pathlib.Path("data/txt")
OUT = pathlib.Path("data/commentary")
OUT.mkdir(exist_ok=True)

QUARTERS = ["2023Q4","2024Q1","2024Q2","2024Q4","2025Q1","2025Q2","2025Q3","2025Q4"]

# headings that mark the END of the CEO letter
END_MARKERS = [
    "Outlook",
    "Preliminary Financial Calendar",
    "Financial Calendar",
    "Financial calendar",
    "Analyst Conference Call and Webcast",
    "Contact Details",
    "About Flow Traders",
]

def extract(text: str) -> str:
    lines = text.splitlines()
    # find start: first line containing an opening curly quote that begins a sentence of prose
    start = None
    for i, ln in enumerate(lines):
        if "\u201c" in ln and len(ln.strip()) > 40:  # opening curly quote + substantial text
            start = i
            break
    if start is None:
        return ""
    # find end: first END_MARKER heading after start
    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if any(s.startswith(m) for m in END_MARKERS):
            end = j
            break
    block = "\n".join(lines[start:end])
    # the CEO letter always ends with a closing curly quote; trim anything after the last one
    last_q = block.rfind("\u201d")
    if last_q != -1:
        block = block[: last_q + 1]
    # tidy whitespace, drop page-footer noise lines
    block = re.sub(r"For immediate release.*?CE[ST]T", " ", block)
    block = re.sub(r"\n\s*\d+\s*\n", "\n", block)          # lone page numbers
    block = re.sub(r"[ \t]+", " ", block)
    block = re.sub(r"\n{2,}", "\n", block).strip()
    return block

for q in QUARTERS:
    raw = (TXT / f"{q}.txt").read_text(encoding="utf-8", errors="ignore")
    block = extract(raw)
    (OUT / f"{q}.txt").write_text(block, encoding="utf-8")
    words = len(block.split())
    print(f"{q}: {words:>4} words | start: {block[:70]!r}")
    print(f"        end: ...{block[-70:]!r}\n")
