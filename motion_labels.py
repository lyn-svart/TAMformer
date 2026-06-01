"""Fixed 21-class motion vocabulary (aligned with TrackJSONAdapter in data_generator.py)."""

MOTION_TO_CLASS = {
    'opposite': 0,
    'crossing-tocenter': 1,
    'crossing-outward': 2,
    'stopped': 3,
    'approaching': 4,
    'ra-merge': 5,
    'ra-exit': 6,
    'ra': 7,
    'leaving': 8,
    'og-exit': 9,
    'passed': 10,
    'passing': 11,
    'og-r2l': 12,
    'og-l2r': 13,
    'tc-l2r': 14,
    'tc-r2l': 15,
    'tc-merge': 16,
    'parked': 17,
    'following': 18,
    'ra-yield': 19,
    'intent to cross': 20,
}

NUM_MOTION_CLASSES = len(MOTION_TO_CLASS)

# id -> canonical label string (for reports)
CLASS_ID_TO_NAME = {v: k for k, v in MOTION_TO_CLASS.items()}


def motion_to_class(motion):
    """Map JSON motion string to class id; unknown/missing -> stopped (3)."""
    if motion is None:
        return MOTION_TO_CLASS['stopped']
    key = str(motion).strip().lower()
    return MOTION_TO_CLASS.get(key, MOTION_TO_CLASS['stopped'])
