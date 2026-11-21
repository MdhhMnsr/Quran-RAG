import time
from pathlib import Path

from search import search

HERE = Path(__file__).parent
PAUSE = {True:45, False:25}

def parse_expected(text): 
    expected = []
    for part in text.split(","): 
        part = part.strip()
        if ":" not in part: 
            expected.append((int(part), 1, 999))
            continue
        surah, ayahs = part.split(":")
        if "-" in ayahs: 
            start,  end = ayahs.split("-")
        else:
            start = end = ayahs
        expected.append((int(surah), int(start), int(end)))
    return expected 

def is_hit(ref,expected): 
    surah, ayah = map(int, ref.split(":"))
    return any(s == surah and start <= ayah <= end for s, start, end in expected)

def load_questions():
    rows = []
    for line in (HERE / "questions.txt").read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        question, refs = line.rsplit("|", 1)
        refs = refs.strip()
        if refs == "none":
            continue
        rows.append((question.strip(), parse_expected(refs)))
    return rows

def run(rows,rerank): 
    hits = 0
    for question, expected in rows: 
        found, _ = search(question, rerank=rerank)
        refs = [v["ref"] for v in found]
        hit = any(is_hit(r, expected) for r in refs)
        hits += hit
        print(f"{'HIT' if hit else 'MISS'} {question[:45]:<45} {','.join(refs)}")
        time.sleep(PAUSE[rerank])
    print(f"Hit rate: {hits}/{len(rows)}\n")
    return hits

if __name__ == "__main__":
    rows = load_questions()
    print("=== Vector only ===")
    run(rows, rerank=False)
    print("=== Vector + reranker ===")
    run(rows, rerank=True) 
