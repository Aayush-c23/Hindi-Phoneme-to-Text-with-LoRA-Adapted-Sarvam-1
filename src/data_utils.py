"""Prompt contract + viseme-noise helpers for Hindi phoneme-to-text."""

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

VISEME_GROUPS = [
    ["P", "B", "M"],
    ["PH", "BH"],
    ["TD", "THD", "DD", "DHD", "N"],
    ["TR", "THR", "DR", "DHR", "NR"],
    ["K", "KH", "G", "GH", "NG"],
    ["CH", "CHH", "J", "JH", "NY"],
]
VISEME_SWAP = {}
for _group in VISEME_GROUPS:
    for _token in _group:
        VISEME_SWAP[_token] = _group


def make_prompt(phoneme_string: str) -> str:
    return (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phoneme_string.strip()}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )


def inject_noise(phonemes, rng, swap_p=0.12, drop_p=0.03, dup_p=0.02):
    """Simulate visual-speech errors: viseme swaps, drops, duplications."""
    out = []
    for token in phonemes:
        if token == "|":
            out.append(token)
            continue
        roll = rng.random()
        if roll < drop_p:
            continue
        if roll < drop_p + dup_p:
            out.extend([token, token])
            continue
        if roll < drop_p + dup_p + swap_p and token in VISEME_SWAP:
            alternatives = [t for t in VISEME_SWAP[token] if t != token]
            if alternatives:
                out.append(rng.choice(alternatives))
                continue
        out.append(token)
    return out
