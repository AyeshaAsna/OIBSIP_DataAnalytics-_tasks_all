import os
import re
import string
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"
DATASET_PATH = os.path.join(BASE_DIR, "creditcard.csv (1).zip")


# ---------- Text corpus (for autocomplete/autocorrect) ----------

SAMPLE_TEXT = """
Autocomplete and autocorrect systems are widely used in messaging applications,
search engines, and text editors. These systems improve user experience by predicting
the next word and correcting spelling mistakes in real time. Natural language processing
techniques such as tokenization, n-gram language models, and edit distance are commonly
used to build these systems. Evaluation of autocomplete and autocorrect involves measuring
accuracy, precision, recall, and the impact on typing speed and user satisfaction.
Modern approaches may also include neural language models such as recurrent networks
and transformers, but classical methods remain useful and interpretable.
"""


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # keep basic punctuation for sentence boundaries if needed
    for ch in string.punctuation:
        if ch not in [".", "!", "?"]:
            text = text.replace(ch, "")
    return text.strip()


def tokenize(text: str):
    return text.split()


def build_language_model(tokens, n: int = 2):
    """
    Build n-gram counts (here we primarily use bigrams for autocomplete).
    """
    ngram_counts = Counter()
    context_counts = Counter()

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        context = ngram[:-1]
        ngram_counts[ngram] += 1
        context_counts[context] += 1

    return ngram_counts, context_counts


def autocomplete_next_word(prefix, ngram_counts, context_counts, top_k=5):
    """
    Given the last word in prefix, suggest next-word candidates using bigram probabilities.
    """
    tokens = tokenize(normalize_text(prefix))
    if not tokens:
        return []
    last = tokens[-1]
    context = (last,)

    candidates = []
    for (w1, w2), count in ngram_counts.items():
        if (w1,) == context:
            prob = count / context_counts[context]
            candidates.append((w2, prob))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


# ---------- Autocorrect based on edit distance ----------

def edits1(word):
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def known(words, vocabulary):
    return {w for w in words if w in vocabulary}


def edits2(word, vocabulary):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in vocabulary}


def autocorrect(word, vocab_counts, max_candidates=5):
    word = word.lower()
    vocabulary = set(vocab_counts.keys())

    candidates = []
    if word in vocabulary:
        candidates.append((word, vocab_counts[word]))
    else:
        # one edit distance
        c1 = known(edits1(word), vocabulary)
        c2 = edits2(word, vocabulary) if not c1 else set()
        possible = c1 or c2
        candidates = [(w, vocab_counts[w]) for w in possible]

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_candidates]


# ---------- Visualization helpers ----------

def plot_word_frequency(vocab_counts, top_n=15):
    most_common = vocab_counts.most_common(top_n)
    words, counts = zip(*most_common)

    plt.figure(figsize=(8, 4))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} words in corpus")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_autocomplete_suggestions(prefix, suggestions):
    if not suggestions:
        print(f"No suggestions for prefix: '{prefix}'")
        return

    words = [w for w, _ in suggestions]
    probs = [p for _, p in suggestions]

    plt.figure(figsize=(6, 4))
    plt.bar(words, probs)
    plt.ylabel("Probability")
    plt.title(f"Autocomplete suggestions for: '{prefix}'")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_autocorrect_candidates(word, candidates):
    if not candidates:
        print(f"No autocorrect candidates for word: '{word}'")
        return

    words = [w for w, _ in candidates]
    scores = [c for _, c in candidates]

    plt.figure(figsize=(6, 4))
    plt.bar(words, scores)
    plt.ylabel("Frequency in corpus")
    plt.title(f"Autocorrect candidates for: '{word}'")
    plt.tight_layout()
    plt.show()
    plt.close()


# ---------- Main pipeline ----------

def main():
    print("Base directory:", BASE_DIR)
    if os.path.exists(DATASET_PATH):
        print(f"Found dataset file (not used for NLP text corpus): {DATASET_PATH}")
    else:
        print("Warning: specified dataset zip not found, continuing with internal text corpus.")

    # Preprocessing
    print("\nNormalizing and tokenizing sample text corpus...")
    norm_text = normalize_text(SAMPLE_TEXT)
    tokens = tokenize(norm_text)
    print("Total tokens:", len(tokens))

    # Vocabulary and counts
    vocab_counts = Counter(tokens)
    print("\nVocabulary size:", len(vocab_counts))

    # Build bigram model for autocomplete
    print("\nBuilding bigram language model for autocomplete...")
    bigram_counts, context_counts = build_language_model(tokens, n=2)
    print("Number of unique bigrams:", len(bigram_counts))

    # Example autocomplete queries
    prefixes = [
        "autocomplete and",
        "these systems",
        "natural language",
        "evaluation of",
        "neural",
    ]

    for prefix in prefixes:
        suggestions = autocomplete_next_word(prefix, bigram_counts, context_counts, top_k=5)
        print(f"\nAutocomplete suggestions for '{prefix}':")
        for w, p in suggestions:
            print(f"  {w} (prob={p:.3f})")
        plot_autocomplete_suggestions(prefix, suggestions)

    # Example autocorrect queries
    typo_words = ["autocorrext", "langauge", "moden", "transofrmers", "evluation"]
    for w in typo_words:
        candidates = autocorrect(w, vocab_counts, max_candidates=5)
        print(f"\nAutocorrect candidates for '{w}':")
        for cand, freq in candidates:
            print(f"  {cand} (freq={freq})")
        plot_autocorrect_candidates(w, candidates)

    # Word frequency visualization
    plot_word_frequency(vocab_counts, top_n=15)

    print("\nAll autocomplete/autocorrect analytics tasks (task5-L2) completed.")


if __name__ == "__main__":
    main()

