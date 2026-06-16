import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------
# Config
# ----------------------------
A_DIR = "samples_A2G_Pured/aerial"  # given aerial UAV images
B_DIR = "samples_A2G_Pured/ground"  # candidate ground images
OUT_JSON = "annotations_A2G_Collaboration_Recognition.json"
N_DISTRACTORS = 3
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TASK_NAME = "A2G_Collaboration_Recognition"
QUESTION_TEXT = (
    "The given image is an aerial UAV image. Images 2-5 are ground-view "
    "candidate images corresponding to options A-D, respectively. Which ground "
    "image was captured at the same location as the aerial image and can "
    "collaborate with it? Answer with only one option letter: A, B, C, or D. "
    "Provide no explanation."
)
REVERSE_QUESTION_TEXT = (
    "The given image is a ground-view image. Images 2-5 are aerial UAV "
    "candidate images corresponding to options A-D, respectively. Which aerial "
    "UAV image was captured at the same location as the ground image and can "
    "collaborate with it? Answer with only one option letter: A, B, C, or D. "
    "Provide no explanation."
)


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def stem_no_ext(p: Path) -> str:
    return p.stem


def list_image_stems(image_dir: Path) -> List[str]:
    return sorted(stem_no_ext(p) for p in image_dir.iterdir() if is_image(p))


def ground_candidates_for_aerial_stem(aerial_stem: str) -> List[str]:
    """
    Build possible matching ground stems from an aerial stem.
    Mirrors the A2G_VT.py pairing rules and also covers names like
    scene_3_East_of_Campus_drone_000100 -> scene_3_East_of_Campus_ground_000100.
    """
    candidates = [
        aerial_stem.replace("droneView", "groundView"),
        aerial_stem.replace("_drone_", "_ground_"),
        aerial_stem.replace("drone", "ground"),
        aerial_stem,
    ]

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def build_ground_index(ground_dir: Path) -> Dict[str, str]:
    """Lowercase ground stem -> original ground stem."""
    idx = {}
    for stem in list_image_stems(ground_dir):
        idx[stem.lower()] = stem
    return idx


def find_matching_ground_stem(aerial_stem: str, ground_idx: Dict[str, str]) -> Optional[str]:
    for candidate in ground_candidates_for_aerial_stem(aerial_stem):
        matched = ground_idx.get(candidate.lower())
        if matched:
            return matched
    return None


def make_item_aerial_query(
    aerial_stem: str,
    correct_ground_stem: str,
    ground_pool: List[str],
    rng: random.Random,
    n_distractors: int = 3,
) -> Dict[str, str]:
    distractor_pool = [s for s in ground_pool if s != correct_ground_stem]
    if len(distractor_pool) < n_distractors:
        raise ValueError(
            f"Not enough ground distractors (need {n_distractors}, got {len(distractor_pool)})."
        )

    distractors = rng.sample(distractor_pool, n_distractors)
    options = [correct_ground_stem] + distractors
    rng.shuffle(options)

    labels = ["A", "B", "C", "D"]
    option_map = {f"Option {labels[i]}": options[i] for i in range(4)}
    correct_label = labels[options.index(correct_ground_stem)]

    return {
        "Image": aerial_stem,
        **option_map,
        "Task": TASK_NAME,
        "User": QUESTION_TEXT,
        "Answer": correct_label,
    }


def make_item_ground_query(
    ground_stem: str,
    correct_aerial_stem: str,
    aerial_pool: List[str],
    rng: random.Random,
    n_distractors: int = 3,
) -> Dict[str, str]:
    distractor_pool = [s for s in aerial_pool if s != correct_aerial_stem]
    if len(distractor_pool) < n_distractors:
        raise ValueError(
            f"Not enough aerial distractors (need {n_distractors}, got {len(distractor_pool)})."
        )

    distractors = rng.sample(distractor_pool, n_distractors)
    options = [correct_aerial_stem] + distractors
    rng.shuffle(options)

    labels = ["A", "B", "C", "D"]
    option_map = {f"Option {labels[i]}": options[i] for i in range(4)}
    correct_label = labels[options.index(correct_aerial_stem)]

    return {
        "Image": ground_stem,
        **option_map,
        "Task": TASK_NAME,
        "User": REVERSE_QUESTION_TEXT,
        "Answer": correct_label,
    }


def main():
    rng = random.Random(SEED)
    a_dir = Path(A_DIR)
    b_dir = Path(B_DIR)

    if not a_dir.exists():
        raise FileNotFoundError(f"A_DIR not found: {a_dir}")
    if not b_dir.exists():
        raise FileNotFoundError(f"B_DIR not found: {b_dir}")

    aerial_stems = list_image_stems(a_dir)
    ground_pool = list_image_stems(b_dir)
    ground_idx = build_ground_index(b_dir)

    if len(ground_pool) < (1 + N_DISTRACTORS):
        raise ValueError(
            f"Ground image pool too small: {len(ground_pool)} "
            f"(need at least {1 + N_DISTRACTORS})."
        )

    reverse_items = []
    skipped_no_match = 0

    for aerial_stem in aerial_stems:
        correct_ground_stem = find_matching_ground_stem(aerial_stem, ground_idx)
        if not correct_ground_stem:
            skipped_no_match += 1
            continue

        item = make_item_ground_query(
            ground_stem=correct_ground_stem,
            correct_aerial_stem=aerial_stem,
            aerial_pool=aerial_stems,
            rng=rng,
            n_distractors=N_DISTRACTORS,
        )
        reverse_items.append(item)

    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"Existing output is not a JSON array: {out_path}")
    else:
        items = []

    existing_reverse_keys = {
        (item.get("Image"), item.get("User"))
        for item in items
        if item.get("Task") == TASK_NAME and item.get("User") == REVERSE_QUESTION_TEXT
    }

    appended = []
    for item in reverse_items:
        key = (item["Image"], item["User"])
        if key in existing_reverse_keys:
            continue
        items.append(item)
        appended.append(item)
        existing_reverse_keys.add(key)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Appended {len(appended)} reverse items to: {out_path}")
    print(f"Total items in output: {len(items)}")
    print(f"Skipped (no matching ground image): {skipped_no_match}")


if __name__ == "__main__":
    main()
