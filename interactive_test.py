import sys
sys.path.insert(0, 'src')
from defense.zedd import build_powergrid_baseline
from attack_generator import AttackGenerator
import re

z = build_powergrid_baseline()
gen = AttackGenerator()
dataset = gen.load('data/attacks/attack_dataset.json')

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def detect_sentence_level(zedd_detector, text):
    sentences = split_sentences(text)
    if not sentences:
        return zedd_detector.detect(text)
    scores = []
    for sentence in sentences:
        result = zedd_detector.detect(sentence)
        scores.append((result.similarity_score, sentence))
    min_score, min_sentence = min(scores, key=lambda x: x[0])
    flagged = min_score < zedd_detector.drift_threshold
    return flagged, min_score, min_sentence

expected_zedd = [s for s in dataset if 'zedd' in s.expected_blocked_by]
benign = [s for s in dataset if s.category == 'benign']

print("=== SENTENCE-LEVEL ZEDD — ATTACK SCORES ===")
caught = 0
for sample in expected_zedd:
    flagged, score, worst_sentence = detect_sentence_level(z, sample.wrapped_attack)
    status = "CAUGHT" if flagged else "MISSED"
    if flagged: caught += 1
    print(f"{status} [{score:.3f}] {sample.attack_id} | worst: {worst_sentence[:60]}")
print(f"\nRecall: {caught}/{len(expected_zedd)} = {caught/len(expected_zedd):.3f}")

print("\n=== SENTENCE-LEVEL ZEDD — BENIGN SCORES ===")
fp = 0
for sample in benign:
    flagged, score, worst_sentence = detect_sentence_level(z, sample.wrapped_attack)
    flag = "← FP" if flagged else ""
    if flagged: fp += 1
    print(f"  {sample.attack_id}: {score:.3f} {flag} | worst: {worst_sentence[:60]}")
print(f"\nFalse positives: {fp}/{len(benign)}")


def split_sentences_filtered(text, min_chars=50):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= min_chars]

def detect_sentence_level_v2(zedd_detector, text, min_chars=50):
    sentences = split_sentences_filtered(text, min_chars)
    if not sentences:
        # All sentences too short — fall back to document level
        result = zedd_detector.detect(text)
        return result.flagged, result.similarity_score, text[:60]
    scores = []
    for sentence in sentences:
        result = zedd_detector.detect(sentence)
        scores.append((result.similarity_score, sentence))
    min_score, min_sentence = min(scores, key=lambda x: x[0])
    flagged = min_score < zedd_detector.drift_threshold
    return flagged, min_score, min_sentence

print("=== SENTENCE-LEVEL v2 (min 50 chars) — ATTACKS ===")
caught = 0
for sample in expected_zedd:
    flagged, score, worst = detect_sentence_level_v2(z, sample.wrapped_attack)
    status = "CAUGHT" if flagged else "MISSED"
    if flagged: caught += 1
    print(f"{status} [{score:.3f}] {sample.attack_id} | worst: {worst[:60]}")
print(f"\nRecall: {caught}/{len(expected_zedd)} = {caught/len(expected_zedd):.3f}")

print("\n=== SENTENCE-LEVEL v2 — BENIGN ===")
fp = 0
for sample in benign:
    flagged, score, worst = detect_sentence_level_v2(z, sample.wrapped_attack)
    flag = "← FP" if flagged else ""
    if flagged: fp += 1
    print(f"  {sample.attack_id}: {score:.3f} {flag} | worst: {worst[:60]}")
print(f"\nFalse positives: {fp}/{len(benign)}")

for min_chars in [30, 35, 40, 45]:
    caught = 0
    fp = 0
    for sample in expected_zedd:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sample.wrapped_attack.strip()) if len(s.strip()) >= min_chars]
        if not sentences:
            result = z.detect(sample.wrapped_attack)
            flagged = result.flagged
        else:
            scores = [(z.detect(s).similarity_score, s) for s in sentences]
            min_score = min(scores, key=lambda x: x[0])[0]
            flagged = min_score < z.drift_threshold
        if flagged: caught += 1

    for sample in benign:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sample.wrapped_attack.strip()) if len(s.strip()) >= min_chars]
        if not sentences:
            result = z.detect(sample.wrapped_attack)
            flagged = result.flagged
        else:
            scores = [(z.detect(s).similarity_score, s) for s in sentences]
            min_score = min(scores, key=lambda x: x[0])[0]
            flagged = min_score < z.drift_threshold
        if flagged: fp += 1

    print(f"min_chars={min_chars}: recall={caught}/{len(expected_zedd)}={caught/len(expected_zedd):.3f}, FP={fp}/{len(benign)}")