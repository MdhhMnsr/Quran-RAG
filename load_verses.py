import os
import re
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = Path(os.environ.get("CTM_DATA", HERE / "data"))
PRICE_PER_MILLION = 0.06 
HARAKAT = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")

def arabic_plain(text):
    return HARAKAT.sub("", text).replace("\u0671", "\u0627")

def load_verses(): 
    verses = []
    for number in range(1,115): 
        surah = json.loads((DATA / f"{number}.json").read_text(encoding="utf-8"))
        for ayah in surah["ayahs"]:
            verses.append({
                "ref": f"{number}:{ayah['number']}",
                "surah":surah["name"],
                "english": ayah["english"],
                "malay": ayah["malay"],
                "arabic": ayah["arabic"],
            })
    return verses
def text_to_embed(verse): 
    return verse["english"] + "\n" + verse["malay"]

if __name__ == "__main__":
    verses = load_verses()
    characters = sum(len(text_to_embed(v)) for v in verses)
    tokens = characters / 3 
    print(f"Verses: {len(verses)}")
    print(f"Characters: {characters:,}")
    print(f"Tokens, rough estimate: {tokens:,.0f}")
    print(f"Value at list price: ${tokens / 1_000_000 * PRICE_PER_MILLION:.4f}")
    print(verses[0]) 
          