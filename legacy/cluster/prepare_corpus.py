"""
prepare_corpus.py

Downloads the Federalist Papers from Project Gutenberg and splits them into
85 individual files, sorted into author folders under corpus/federalist/.

IMPORTANT for cluster use: run this ONCE on a LOGIN NODE, not inside a
submitted GPU job. Many HPC clusters (including most SLURM-managed systems)
restrict or block outbound internet access from compute nodes for security
reasons — login nodes almost always have internet access, compute nodes
often don't. Running this script here, before submitting any training job,
avoids that problem entirely: once corpus/federalist/ exists on disk, the
training scripts never need network access at all.

Usage (on a login node):
    python prepare_corpus.py
"""

import os
import re
import glob
import requests

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1404/pg1404.txt"
CORPUS_ROOT = "corpus/federalist"

HAMILTON_SOLO = set([1] + list(range(6, 10)) + list(range(11, 14)) + list(range(15, 18)) +
                     list(range(21, 37)) + list(range(59, 62)) + list(range(65, 86)))
MADISON_SOLO = set([10, 14] + list(range(37, 49)))
JAY = set([2, 3, 4, 5, 64])
JOINT = set([18, 19, 20])
DISPUTED = set(list(range(49, 59)) + [62, 63])


def folder_for(number):
    if number in HAMILTON_SOLO:
        return "hamilton"
    if number in MADISON_SOLO:
        return "madison"
    if number in JAY:
        return "jay"
    if number in JOINT:
        return "joint"
    if number in DISPUTED:
        return "disputed"
    return None


def main():
    print("Downloading Federalist Papers from Project Gutenberg...")
    response = requests.get(GUTENBERG_URL, timeout=30)
    response.raise_for_status()
    raw_text = response.text
    print(f"Downloaded {len(raw_text):,} characters.")

    with open("federalist_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Strip Gutenberg boilerplate header/footer
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start_idx = raw_text.find(start_marker)
    end_idx = raw_text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        body_start = raw_text.find("\n", start_idx) + 1
        body_text = raw_text[body_start:end_idx]
    else:
        print("WARNING: could not find Gutenberg markers, using full text "
              "(check for boilerplate contamination).")
        body_text = raw_text
    print(f"Body text length: {len(body_text):,} characters")

    # Split into individual papers by header
    header_pattern = re.compile(r"FEDERALIST\.?\s+No\.?\s+(\d+)", re.IGNORECASE)
    matches = list(header_pattern.finditer(body_text))
    print(f"Found {len(matches)} header matches (expect at least 85).")

    papers = {}
    for i, match in enumerate(matches):
        number = int(match.group(1))
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(body_text)
        text = body_text[section_start:section_end].strip()
        if number not in papers or len(text) > len(papers[number]):
            papers[number] = text

    print(f"Distinct paper numbers recovered: {len(papers)}")
    missing = sorted(set(range(1, 86)) - set(papers.keys()))
    if missing:
        print(f"WARNING: missing papers {missing} — inspect federalist_raw.txt "
              f"and adjust the header_pattern regex above.")

    # Sanity-check the attribution lists
    assert len(HAMILTON_SOLO) == 51, f"Hamilton count wrong: {len(HAMILTON_SOLO)}"
    assert len(MADISON_SOLO) == 14, f"Madison count wrong: {len(MADISON_SOLO)}"
    assert len(JAY) == 5
    assert len(JOINT) == 3
    assert len(DISPUTED) == 12

    for folder in ["hamilton", "madison", "jay", "joint", "disputed"]:
        os.makedirs(os.path.join(CORPUS_ROOT, folder), exist_ok=True)

    written_count = 0
    for number, text in papers.items():
        folder = folder_for(number)
        if folder is None:
            print(f"WARNING: paper {number} not in any known category, skipping.")
            continue
        out_path = os.path.join(CORPUS_ROOT, folder, f"federalist_{number:02d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        written_count += 1

    print(f"\nWrote {written_count} paper files into {CORPUS_ROOT}/")
    for folder in ["hamilton", "madison", "jay", "joint", "disputed"]:
        n = len(glob.glob(os.path.join(CORPUS_ROOT, folder, "*.txt")))
        print(f"  {folder}: {n} files")

    print("\nCorpus ready. You can now submit training jobs without needing "
          "network access on compute nodes.")


if __name__ == "__main__":
    main()
