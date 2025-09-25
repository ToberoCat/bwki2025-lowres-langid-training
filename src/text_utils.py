import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from typing import Iterable, Tuple, Dict, List, Set

import numpy as np
from simhash import simhash

_URL_RE = re.compile(r"""(?xi)\b(?:https?://|www\.)[^\s<>]+""")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_MENTION_RE = re.compile(r"(^|[\s])@[A-Za-z0-9_]{2,}")
_HASHTAG_RE = re.compile(r"(^|[\s])#[^\s#]+")
_DIGIT_RE = re.compile(r"\d")
_WHITESPACE_RE = re.compile(r"\s+")

def clean_text(
    s: str,
    *,
    lowercase: bool = True,
    normalize_digits: bool = True,
    strip_urls: bool = True,
    strip_emails: bool = True,
    strip_mentions: bool = True,
    strip_hashtags: bool = False,
) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    if strip_urls:
        s = _URL_RE.sub(" ", s)
    if strip_emails:
        s = _EMAIL_RE.sub(" ", s)
    if strip_mentions:
        s = _MENTION_RE.sub(" ", s)
    if strip_hashtags:
        s = _HASHTAG_RE.sub(" ", s)
    if normalize_digits:
        s = _DIGIT_RE.sub("0", s)
    if lowercase:
        s = s.lower()
    s = s.replace("\r", " ").replace("\n", " ")
    return _WHITESPACE_RE.sub(" ", s).strip()


def _shingles_ngrams(text: str, n: int = 3) -> List[str]:
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def compute_simhash(text: str, ngram: int = 3) -> int:
    feats = _shingles_ngrams(text, ngram)
    return simhash(feats)


def hamming_distance(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def hamming_distance_vec(sigs: np.ndarray, sig: int) -> np.ndarray:
    xor = np.bitwise_xor(sigs, np.uint64(sig)).astype(np.uint64)
    return np.unpackbits(xor.view(np.uint8), axis=1).sum(axis=1)


def lsh_band_keys(sig: int, bands: int = 8) -> List[Tuple[int, int]]:
    width = 64 // bands
    rem = 64 - width * bands
    keys = []
    pos = 0
    for b in range(bands):
        w = width + (rem if b == bands - 1 else 0)
        mask = ((1 << w) - 1) << pos
        keys.append((b, (sig & mask) >> pos))
        pos += w
    return keys


_SCRIPT_RANGES = {
    "Latin": [(0x0000, 0x007F), (0x0080, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Greek": [(0x0370, 0x03FF)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    "Hebrew": [(0x0590, 0x05FF)],
    "Devanagari": [(0x0900, 0x097F)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    "Hiragana": [(0x3040, 0x309F)],
    "Katakana": [(0x30A0, 0x30FF)],
    "Han": [(0x4E00, 0x9FFF)],
}


def _script_of_char(ch: str) -> str | None:
    cp = ord(ch)
    for name, ranges in _SCRIPT_RANGES.items():
        for a, b in ranges:
            if a <= cp <= b:
                return name
    return None


def dominant_script(text: str) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for ch in text:
        if ch.isdigit() or ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S", "C"):
            continue
        s = _script_of_char(ch)
        if s:
            counts[s] += 1
    if not counts:
        return "Other"
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    arr = np.fromiter((ch.isdigit() for ch in text), dtype=np.int32)
    return arr.sum() / max(1, len(text))


def has_diacritics(text: str) -> bool:
    nfd = unicodedata.normalize("NFD", text)
    return any(unicodedata.combining(ch) for ch in nfd)


def len_bucket(
    text: str,
    *,
    word_thr_short: int = 8,
    word_thr_long: int = 20,
    char_thr_short: int = 40,
    char_thr_long: int = 120,
) -> str:
    words = len(text.split())
    chars = len(text)
    if words <= word_thr_short or chars <= char_thr_short:
        return "SHORT"
    if words >= word_thr_long and chars >= char_thr_long:
        return "LONG"
    return "MEDIUM"


def digit_bucket(
    r: float,
    *,
    none_eps: float = 0.0,
    low_thr: float = 0.05,
    med_thr: float = 0.20,
) -> str:
    if r <= none_eps:
        return "NONE"
    if r < low_thr:
        return "LOW"
    if r < med_thr:
        return "MED"
    return "HIGH"


def build_meta_tokens(
    text: str,
    *,
    include_script: bool = True,
    include_length: bool = True,
    include_digitratio: bool = True,
    word_thr_short: int = 8,
    word_thr_long: int = 20,
    char_thr_short: int = 40,
    char_thr_long: int = 120,
    digit_low: float = 0.05,
    digit_med: float = 0.20,
) -> str:
    feats = []
    if include_script:
        feats.append(f"__feat_SCRIPT={dominant_script(text)}")
    if include_length:
        feats.append(
            f"__feat_LEN={len_bucket(text, word_thr_short, word_thr_long, char_thr_short, char_thr_long)}"
        )
    if include_digitratio:
        feats.append(f"__feat_DIGITRATIO={digit_bucket(digit_ratio(text), low_thr=digit_low, med_thr=digit_med)}")
    return " " + " ".join(feats) if feats else ""


_dedup_cache: Dict[int, Tuple[List[int], Dict[str, int]]] = {}


def _compute_simhash_batch(texts_batch: List[str], ngram: int) -> List[int]:
    return [compute_simhash(text, ngram=ngram) for text in texts_batch]


def compute_batch_parallel(sub_batches: List[List[str]], ngram: int, workers: int) -> List[int]:
    results: List[int] = []
    if workers > 1 and len(sub_batches) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            res = executor.map(_compute_simhash_batch, sub_batches, [ngram] * len(sub_batches))
            results = list(chain.from_iterable(res))
    else:
        results = list(chain.from_iterable([_compute_simhash_batch(b, ngram) for b in sub_batches]))
    return results


def near_deduplicate(
    texts: Iterable[str],
    langs: Iterable[str] = None,
    *,
    bands: int = 8,
    ham_thresh: int = 3,
    ngram: int = 3,
    workers: int = 4,
    batch_size: int | None = None,
    memory_limit_gb: float = 8.0,
) -> Tuple[List[int], Dict[str, int]]:

    texts_list = list(texts)
    langs_list = list(langs) if langs is not None else [""] * len(texts_list)
    n_texts = len(texts_list)

    if n_texts == 0:
        return [], {"near_duplicates_dropped": 0}
    if len(langs_list) != n_texts:
        raise ValueError("`texts` and `langs` must have the same length")

    lang_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, lang in enumerate(langs_list):
        lang_to_indices[str(lang) if lang else ""].append(i)

    kept_global: List[int] = []
    total_dropped = 0

    for lang, indices in lang_to_indices.items():
        sub_texts = [texts_list[i] for i in indices]
        n_sub = len(sub_texts)
        if n_sub == 0:
            continue

        local_batch_size = batch_size or max(5000, min(25000, int(memory_limit_gb * 1e6 / 200)))

        sample_size = min(100, n_sub)
        content_sample = tuple(sub_texts[::max(1, n_sub // sample_size)][:sample_size])
        content_hash = hash((content_sample, bands, ham_thresh, ngram))
        if content_hash in _dedup_cache:
            kept_local, stats = _dedup_cache[content_hash]
            kept_global.extend(indices[i] for i in kept_local)
            total_dropped += stats["near_duplicates_dropped"]
            continue

        buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        kept: List[int] = []
        kept_sigs: List[int] = []

        for start in range(0, n_sub, local_batch_size):
            end = min(start + local_batch_size, n_sub)
            slice_texts = sub_texts[start:end]

            sub_batch_size = max(1, len(slice_texts) // workers)
            sub_batches = [slice_texts[i:i + sub_batch_size] for i in range(0, len(slice_texts), sub_batch_size)]

            sigs = compute_batch_parallel(sub_batches, ngram, workers)
            sigs_np = np.array(sigs, dtype=np.uint64)

            for i_local, sig_int in enumerate(sigs_np):
                i_global = start + i_local
                if any(hamming_distance(sig_int, kept_sig) <= ham_thresh for kept_sig in kept_sigs):
                    continue
                candidate_idxs: Set[int] = set()
                for key in lsh_band_keys(int(sig_int), bands=bands):
                    candidate_idxs.update(buckets.get(key, []))
                is_duplicate = False
                for cand_local in candidate_idxs:
                    if hamming_distance(sig_int, kept_sigs[cand_local]) <= ham_thresh:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    kept_idx = len(kept)
                    kept.append(i_global)
                    kept_sigs.append(int(sig_int))
                    for key in lsh_band_keys(int(sig_int), bands=bands):
                        buckets[key].append(kept_idx)

        dropped = n_sub - len(kept)
        total_dropped += dropped
        stats = {"near_duplicates_dropped": dropped}
        if len(_dedup_cache) < 32:
            _dedup_cache[content_hash] = (kept, stats)
        else:
            _dedup_cache.clear()
            _dedup_cache[content_hash] = (kept, stats)

        kept_global.extend(indices[i] for i in kept)

    return kept_global, {"near_duplicates_dropped": total_dropped}
def is_short(txt: str, short_word_threshold: int, short_char_threshold: int) -> bool:
    return (len(txt.split()) <= short_word_threshold) or (len(txt) <= short_char_threshold)
