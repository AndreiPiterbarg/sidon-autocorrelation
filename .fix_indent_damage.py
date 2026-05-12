"""Restore .py/.lean files damaged by the over-eager emoji-strip script.

The bug was `re.sub(r' {2,}', ' ', text)` which collapsed every multi-space run
into a single space, destroying Python and Lean indentation.

This pulls the pre-strip content from HEAD~1 (the commit before the cleanup
restructure) and writes it back with emojis stripped — without touching
whitespace.
"""
import os
import re
import subprocess
import sys

EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F000-\U0001F2FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE0F"
    "\U0001F1E6-\U0001F1FF"
    "]"
)


def get_blob(rev: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout


def main(paths: list[str]) -> int:
    fixed = 0
    skipped = []
    for p in paths:
        # Try HEAD~1 first (pre-strip), then HEAD as fallback
        raw = get_blob("HEAD~1", p) or get_blob("HEAD", p)
        if raw is None:
            skipped.append((p, "no HEAD~1 or HEAD copy"))
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
        new_text = EMOJI.sub("", text)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        fixed += 1
        print(f"fixed {p}")
    print(f"--- {fixed} fixed, {len(skipped)} skipped")
    for p, why in skipped:
        print(f"SKIP {p}: {why}")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
