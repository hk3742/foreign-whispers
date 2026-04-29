from foreign_whispers.reranking import get_shorter_translations
c = get_shorter_translations(
    source_text="The Strait of Hormuz is critical.",
    baseline_es="El estrecho de Ormuz es un punto critico, sin embargo es fundamental para el suministro.",
    target_duration_s=4.0,
)
print(f"Candidates: {len(c)}")
for x in c:
    print(f"  [{x.char_count}] {x.text}")
    print(f"       {x.brevity_rationale}")
