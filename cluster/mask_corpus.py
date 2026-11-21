"""
Build a content-word-masked copy of the corpus.

Every word NOT on the function-word list is replaced with a placeholder
("X"); function words and punctuation are kept. Run the normal experiments
on the masked copy to test whether the signal survives with topic removed.
--invert makes the opposite control (keep content words, mask function words):
if THAT still classifies well, topic/vocabulary is carrying signal.

Usage (from the cluster/ folder, where prepare_corpus.py created corpus/federalist/):
    python mask_corpus.py --out corpus/federalist_masked
    python mask_corpus.py --out corpus/federalist_content_only --invert
Then point any experiment at the copy with --corpus-root, e.g.
    sbatch submit_job.sh balanced --corpus-root corpus/federalist_masked ...
    python traditional_stylometry.py --corpus-root corpus/federalist_masked --tag _masked
"""
import argparse
import glob
import os
import re

FUNCTION_WORDS = set("""
a about above across after afterwards again against all almost alone along already also
although always am among amongst an and another any anyone anything anywhere are around as at
back be became because become becomes becoming been before beforehand behind being below beside
besides between beyond both but by can cannot could did do does doing done down during each
either else elsewhere enough even ever every everyone everything everywhere except few first for
former formerly from further had has have having he hence her here hereafter hereby herein hereupon
hers herself him himself his how however i if in indeed instead into is it its itself last latter
least less many may me meanwhile might mine more moreover most mostly much must my myself namely
neither never nevertheless next no nobody none nor not nothing now nowhere of off often on once one
only onto or other others otherwise our ours ourselves out over own per perhaps rather same seem
seemed seeming seems several shall she should since so some somehow someone something sometimes
somewhere still such than that the their theirs them themselves then thence there thereafter thereby
therefore therein thereof thereon thereupon these they this those though through throughout thus to
together too toward towards under unless until up upon us very was we well were what whatever when
whence whenever where whereas whereby wherein whereupon wherever whether which while whilst whither
who whoever whole whom whose why will with within without would yet you your yours yourself
yourselves
""".split())

PLACEHOLDER = "X"


def clean_text(raw):  # same as stylometry_lib.clean_text
    text = re.sub(r"\[.*?\]", "", raw)
    text = re.sub(r"PUBLIUS\.?\s*$", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def mask_text(text, invert=False):
    kept = masked = 0

    def repl(m):
        nonlocal kept, masked
        is_fw = m.group(0).lower() in FUNCTION_WORDS
        if is_fw != invert:
            kept += 1
            return m.group(0)
        masked += 1
        return PLACEHOLDER

    return re.sub(r"[A-Za-z']+", repl, text), kept, masked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="corpus/federalist")
    p.add_argument("--out", required=True)
    p.add_argument("--invert", action="store_true",
                   help="Keep content words, mask function words (control).")
    a = p.parse_args()

    tot_kept = tot_masked = 0
    for sub in sorted(os.listdir(a.src)):
        d = os.path.join(a.src, sub)
        if not os.path.isdir(d):
            continue
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        for path in sorted(glob.glob(os.path.join(d, "*.txt"))):
            with open(path, encoding="utf-8", errors="ignore") as f:
                out, k, m = mask_text(clean_text(f.read()), a.invert)
            with open(os.path.join(a.out, sub, os.path.basename(path)), "w", encoding="utf-8") as f:
                f.write(out)
            tot_kept += k
            tot_masked += m
    total = tot_kept + tot_masked
    print(f"Wrote {a.out}: kept {tot_kept/total:.1%} of word tokens, masked {tot_masked/total:.1%}.")


if __name__ == "__main__":
    main()
