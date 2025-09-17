from typing import Final

_NUM: Final = "num__"
_CAT: Final = "cat__"

def canonical_name(raw: str) -> str:
    """
    Map any scikit-learn feature name to the single, canonical form
        • numeric →  booked_min
        • categorical → procedure_id=G014065
    Handles names that already lost their sklearn prefix and those
    that carry a grouping tag like 'main='.
    """
    # 1 ▸ strip standard prefixes
    if raw.startswith(_NUM):
        return raw[len(_NUM):]            # numeric → done
    if raw.startswith(_CAT):
        raw = raw[len(_CAT):]             # leave cat for next steps

    # 2 ▸ drop optional encoder group tags (e.g. 'main=')
    if raw.startswith("main="):
        raw = raw[len("main="):]

    # 3 ▸ categorical: replace last '_' with '='
    if "_" in raw:
        left, _, right = raw.rpartition("_")
        return f"{left}={right}"

    return raw
