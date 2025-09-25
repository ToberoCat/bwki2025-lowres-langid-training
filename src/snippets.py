import random

def word_snippets(text: str, n: int, wmin: int, wmax: int) -> list[str]:
    """Sample n word-span snippets (inclusive length range)."""
    words = text.split()
    if not words or n <= 0:
        return []
    out = []
    for _ in range(n):
        L = max(wmin, min(wmax, len(words)))
        L = random.randint(wmin, L)
        start = 0 if len(words) <= L else random.randint(0, len(words) - L)
        out.append(" ".join(words[start:start + L]))
    return out

def char_snippets(text: str, n: int, cmin: int, cmax: int) -> list[str]:
    """Sample n character-span snippets (inclusive length range)."""
    if not text or n <= 0:
        return []
    out = []
    Lmax = min(len(text), cmax)
    for _ in range(n):
        L = random.randint(cmin, max(cmin, Lmax))
        start = 0 if len(text) <= L else random.randint(0, len(text) - L)
        out.append(text[start:start + L])
    return out


def region_snippets(text: str, n: int, rmin: int, rmax: int) -> list[str]:
    if not text or n <= 0:
        return []
    out = []
    Lmax = min(len(text), rmax)
    for _ in range(n):
        L = random.randint(rmin, max(rmin, Lmax))
        start = 0 if len(text) <= L else random.randint(0, len(text) - L)
        out.append(text[start:start + L])
    return out