import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

REGION_TASKS = {
    "Region_Recognition",
    "Distance_Comparison",
    "Anomaly_Detection",
    "A2A_Occlusion_Removal",
    "A2G_View_Translation",
}
OPTION_TASKS = {
    "A2A_Collaboration_Recognition",
    "A2G_Collaboration_Recognition",
}
PAIR_TASKS = {
    "A2A_Shared_Association",
    "A2G_Shared_Association",
}
PARTIAL_REGION_TASKS = {
    "Region_Recognition",
    "Anomaly_Detection",
}
DEFAULT_OPENAI_MOTION_JUDGE_BASE_URL = "https://www.autodl.art/api/v1"
DEFAULT_GEMINI_MOTION_JUDGE_BASE_URL = "https://www.autodl.art/api/v1/gemini"
DEFAULT_MOTION_JUDGE_MODEL = "gpt-5.4-mini"
API_REFERENCE_PATH = Path("API.py")


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
    return rows


def task_name_from_id(question_id: str) -> str:
    return question_id.rsplit("_", 1)[0]


def extract_ints(text: str):
    return [int(x) for x in re.findall(r"-?\d+", text or "")]


def extract_floats(text: str):
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [float(x) for x in re.findall(pattern, text or "")]


def parse_region_labels(text: str):
    return extract_ints(text)


def parse_oclock(text: str):
    nums = extract_ints(text)
    return nums[0] if nums else None


def parse_direction_label(text: str):
    nums = extract_ints(text)
    return nums[0] if nums else None


def parse_option(text: str):
    match = re.search(r"\b([A-D])\b", (text or "").upper())
    return match.group(1) if match else None


def parse_pairs(text: str):
    pairs = re.findall(r"(\d+)\s*-\s*(\d+)", text or "")
    return [(int(a), int(b)) for a, b in pairs]


def import_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai\n"
            "Install it with:\n"
            "  pip install -U openai\n"
        ) from exc
    return OpenAI


def import_gemini():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: google-genai\n"
            "Install them with:\n"
            "  pip install -U google-genai\n"
        ) from exc
    return genai, types


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def text_tokens(text: str):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def token_f1_metrics(pred_text: str, gt_text: str):
    pred_counter = Counter(text_tokens(pred_text))
    gt_counter = Counter(text_tokens(gt_text))
    matched = sum((pred_counter & gt_counter).values())
    pred_total = sum(pred_counter.values())
    gt_total = sum(gt_counter.values())
    precision = 0.0 if pred_total == 0 else matched / pred_total
    recall = 0.0 if gt_total == 0 else matched / gt_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "matched_tokens": matched,
        "pred_tokens": pred_total,
        "gt_tokens": gt_total,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "exact_match": normalize_text(pred_text) == normalize_text(gt_text),
    }


def pair_set_metrics(pred_pairs, gt_pairs):
    pred_counter = Counter(pred_pairs)
    gt_counter = Counter(gt_pairs)
    matched = sum((pred_counter & gt_counter).values())
    pred_total = sum(pred_counter.values())
    gt_total = sum(gt_counter.values())
    precision = 0.0 if pred_total == 0 else matched / pred_total
    recall = 0.0 if gt_total == 0 else matched / gt_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "matched_pairs": matched,
        "pred_pairs": pred_total,
        "gt_pairs": gt_total,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "exact_match": pred_counter == gt_counter,
    }


def region_set_metrics(pred_labels, gt_labels):
    pred_set = set(pred_labels)
    gt_set = set(gt_labels)
    matched = len(pred_set & gt_set)
    pred_total = len(pred_set)
    gt_total = len(gt_set)
    precision = 0.0 if pred_total == 0 else matched / pred_total
    recall = 0.0 if gt_total == 0 else matched / gt_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    oversized_prediction = pred_total > gt_total
    score = 0.0 if oversized_prediction else recall
    return {
        "matched_regions": matched,
        "pred_regions": pred_total,
        "gt_regions": gt_total,
        "region_precision": precision,
        "region_recall": recall,
        "region_f1": f1,
        "region_score": score,
        "oversized_prediction": oversized_prediction,
        "exact_match": pred_set == gt_set,
    }


def parse_bbox(text: str):
    nums = extract_ints(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (x1, y1, x2, y2)


def bbox_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_geometry_metrics(box1, box2):
    if box1 is None or box2 is None:
        return {
            "iou": 0.0,
            "center_score": 0.0,
            "size_score": 0.0,
            "combined_score": 0.0,
        }

    iou = bbox_iou(box1, box2)
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    aw = max(ax2 - ax1, 1e-6)
    ah = max(ay2 - ay1, 1e-6)
    bw = max(bx2 - bx1, 1e-6)
    bh = max(by2 - by1, 1e-6)

    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    # Normalize center offset by GT box diagonal so position and size are both considered.
    gt_diag = math.hypot(bw, bh)
    center_dist = math.hypot(acx - bcx, acy - bcy)
    center_score = max(0.0, 1.0 - center_dist / max(gt_diag, 1e-6))

    width_ratio = min(aw, bw) / max(aw, bw)
    height_ratio = min(ah, bh) / max(ah, bh)
    size_score = (width_ratio + height_ratio) / 2.0

    # Composite score: overlap is primary, but center/size consistency also matters.
    combined_score = 0.5 * iou + 0.25 * center_score + 0.25 * size_score
    return {
        "iou": iou,
        "center_score": center_score,
        "size_score": size_score,
        "combined_score": combined_score,
    }


def circular_angle_error(pred_angle: float, gt_angle: float):
    diff = abs(pred_angle - gt_angle) % 360.0
    return min(diff, 360.0 - diff)


def normalize_angle(angle: float):
    return angle % 360.0


def zero_safe_relative_error(abs_error: float, gt_abs_value: float):
    if math.isclose(gt_abs_value, 0.0, abs_tol=1e-12):
        return 0.0 if math.isclose(abs_error, 0.0, abs_tol=1e-12) else None
    return abs_error / gt_abs_value


def camera_transformation_metrics(pred: str, gt: str):
    pred_vals = extract_floats(pred)
    gt_vals = extract_floats(gt)
    metrics = {
        "pred_numeric_count": len(pred_vals),
        "gt_numeric_count": len(gt_vals),
        "answer_format": "invalid",
        "parse_valid": False,
        "angle_error": math.inf,
        "distance_abs_error": math.inf,
        "angle_rel_error": None,
        "distance_rel_error": None,
        "codabench_final_score": None,
        "codabench_valid": False,
    }
    if len(pred_vals) == 1 and len(gt_vals) == 1:
        answer_format = "angle_only"
    elif len(pred_vals) == 2 and len(gt_vals) == 2:
        answer_format = "angle_distance"
    else:
        return metrics

    pred_angle = pred_vals[0]
    gt_angle = gt_vals[0]
    pred_angle_norm = normalize_angle(pred_angle)
    gt_angle_norm = normalize_angle(gt_angle)
    angle_error = circular_angle_error(pred_angle_norm, gt_angle_norm)
    metrics.update(
        {
            "answer_format": answer_format,
            "parse_valid": True,
            "pred_angle": pred_angle,
            "gt_angle": gt_angle,
            "pred_angle_norm": pred_angle_norm,
            "gt_angle_norm": gt_angle_norm,
            "angle_error": angle_error,
        }
    )
    if answer_format == "angle_only":
        return metrics

    pred_distance = pred_vals[1]
    gt_distance = gt_vals[1]
    distance_abs_error = abs(pred_distance - gt_distance)
    angle_rel_error = zero_safe_relative_error(angle_error, abs(gt_angle_norm))
    distance_rel_error = zero_safe_relative_error(distance_abs_error, abs(gt_distance))
    codabench_valid = angle_rel_error is not None and distance_rel_error is not None

    metrics.update(
        {
            "pred_distance": pred_distance,
            "gt_distance": gt_distance,
            "distance_abs_error": distance_abs_error,
            "angle_rel_error": angle_rel_error,
            "distance_rel_error": distance_rel_error,
            "codabench_final_score": (
                (angle_rel_error + distance_rel_error) / 2.0 if codabench_valid else None
            ),
            "codabench_valid": codabench_valid,
        }
    )
    return metrics


def attach_ground_truth(rows, annotations_path: Path):
    if all(row.get("gt") is not None for row in rows):
        return rows
    annotations = load_jsonl(annotations_path)
    gt_by_id = {row["id"]: row.get("GT", "") for row in annotations}
    patched_rows = []
    missing_ids = []
    for row in rows:
        if row.get("gt") is not None:
            patched_rows.append(row)
            continue
        qid = row.get("question_id")
        if qid not in gt_by_id:
            missing_ids.append(qid)
            patched_rows.append(row)
            continue
        patched_rows.append({**row, "gt": gt_by_id[qid]})
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        suffix = "" if len(missing_ids) <= 5 else ", ..."
        raise ValueError(f"Missing GT for {len(missing_ids)} prediction rows: {preview}{suffix}")
    return patched_rows


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def avg_or_none(values):
    return None if not values else sum(values) / len(values)


def extract_json_object(text: str):
    if not text:
        raise ValueError("Empty judge response")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]}")
    return json.loads(match.group(0))


def load_api_reference_defaults(path: Path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    defaults = {}
    key_match = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', text)
    if key_match:
        defaults["api_key"] = key_match.group(1)
    base_url_match = re.search(r'["\']base_url["\']\s*:\s*["\']([^"\']+)["\']', text)
    if base_url_match:
        defaults["base_url"] = base_url_match.group(1)
    return defaults


def infer_motion_judge_provider(model: str) -> str:
    model_name = (model or "").strip().lower()
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        return "openai"
    if "gemini" in model_name:
        return "gemini"
    return "openai"


def resolve_motion_judge_base_url(provider: str, base_url: str, api_defaults: dict) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        if provider == "gemini":
            return api_defaults.get("base_url", DEFAULT_GEMINI_MOTION_JUDGE_BASE_URL)
        return DEFAULT_OPENAI_MOTION_JUDGE_BASE_URL
    if provider == "openai" and candidate.endswith("/gemini"):
        return DEFAULT_OPENAI_MOTION_JUDGE_BASE_URL
    return candidate


def extract_judge_score(parsed):
    score_keys = ("score", "semantic_score", "similarity_score", "similarity", "rating")
    for key in score_keys:
        if key in parsed:
            return max(0.0, min(1.0, float(parsed[key])))
    for value in parsed.values():
        if isinstance(value, (int, float)):
            numeric = float(value)
            if 0.0 <= numeric <= 1.0:
                return numeric
    raise KeyError(f"score not found in judge response keys: {sorted(parsed.keys())}")


class MotionJudge:
    def __init__(
        self,
        mode: str,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        cache_path: Path,
        max_tokens: int,
        temperature: float,
        retries: int,
        retry_delay: float,
        fallback_on_error: bool,
    ):
        self.mode = mode
        self.provider = provider
        self.model = model
        self.cache_path = cache_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.retry_delay = retry_delay
        self.fallback_on_error = fallback_on_error
        self.cache = load_json(cache_path, {}) if mode == "gpt_54_mini" else {}
        self.client = None
        self.genai_types = None
        if mode == "gpt_54_mini":
            if provider == "openai":
                OpenAI = import_openai()
                self.client = OpenAI(base_url=base_url, api_key=api_key)
            elif provider == "gemini":
                genai, types = import_gemini()
                self.genai_types = types
                self.client = genai.Client(
                    api_key=api_key,
                    http_options={"base_url": base_url},
                )
            else:
                raise SystemExit(f"Unsupported motion judge provider: {provider}")

    def score(self, question_id: str, gt: str, pred: str):
        if self.mode == "token_f1":
            metrics = token_f1_metrics(pred, gt)
            return {
                "score": metrics["token_f1"],
                "correct": metrics["exact_match"],
                "meta": {**metrics, "judge_mode": self.mode},
            }

        cache_key = json.dumps(
            {"question_id": question_id, "gt": gt, "prediction": pred, "model": self.model},
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            score = extract_judge_score(cached)
            return {
                "score": score,
                "correct": normalize_text(pred) == normalize_text(gt),
                "meta": {
                    "judge_mode": self.mode,
                    "semantic_score": score,
                    "judge_reason": cached.get("reason", ""),
                    "cache_hit": True,
                },
            }

        system_prompt = (
            "You are an evaluator for drone video motion descriptions. "
            "Compare a prediction against the ground truth semantically. "
            "Score from 0.0 to 1.0 based on whether the prediction captures the same drone motion, "
            "camera angle change, object/frame shift, and final viewing focus. "
            "Be strict about contradictions, but allow paraphrases. "
            "Return only compact JSON: {\"score\": <number between 0 and 1>, \"reason\": \"<short reason>\"}."
        )
        user_prompt = (
            f"Question ID: {question_id}\n"
            f"Ground truth: {gt}\n"
            f"Prediction: {pred}\n"
            "Evaluate semantic similarity and return the JSON only."
        )

        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                parsed = self._judge_once(system_prompt, user_prompt)
                score = extract_judge_score(parsed)
                reason = str(parsed.get("reason", "")).strip()
                self.cache[cache_key] = {"score": score, "reason": reason}
                dump_json(self.cache_path, self.cache)
                return {
                    "score": score,
                    "correct": normalize_text(pred) == normalize_text(gt),
                    "meta": {
                        "judge_mode": self.mode,
                        "semantic_score": score,
                        "judge_reason": reason,
                        "cache_hit": False,
                        "judge_attempts": attempt,
                    },
                }
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_delay * attempt)

        if self.fallback_on_error:
            metrics = token_f1_metrics(pred, gt)
            return {
                "score": metrics["token_f1"],
                "correct": metrics["exact_match"],
                "meta": {
                    **metrics,
                    "judge_mode": "token_f1_fallback",
                    "judge_error": str(last_error),
                },
            }

        raise RuntimeError(
            f"Motion judge failed after {self.retries} attempts for {question_id}: {last_error}"
        ) from last_error

    def _judge_once(self, system_prompt: str, user_prompt: str):
        if self.mode == "gpt_54_mini":
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or "{}"
                return extract_json_object(content)
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self.genai_types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )
            content = getattr(response, "text", "") or "{}"
            return extract_json_object(content)
        raise ValueError(f"Unsupported motion judge mode: {self.mode}")


def evaluate_row(
    row,
    camera_threshold: float,
    camera_distance_threshold: float,
    bbox_score_threshold: float,
    motion_judge: MotionJudge,
):
    task = task_name_from_id(row["question_id"])
    gt = row.get("gt", "")
    pred = row.get("prediction", "")

    if task in PARTIAL_REGION_TASKS:
        metrics = region_set_metrics(parse_region_labels(pred), parse_region_labels(gt))
        return {
            "task": task,
            "correct": metrics["exact_match"],
            "score": metrics["region_score"],
            "meta": metrics,
        }

    if task in REGION_TASKS:
        correct = parse_region_labels(pred) == parse_region_labels(gt)
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": {},
        }

    if task == "Direction_Recognition":
        correct = parse_oclock(pred) == parse_oclock(gt)
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": {},
        }

    if task == "Path_Planning":
        correct = parse_direction_label(pred) == parse_direction_label(gt)
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": {},
        }

    if task in OPTION_TASKS:
        correct = parse_option(pred) == parse_option(gt)
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": {},
        }

    if task in PAIR_TASKS:
        metrics = pair_set_metrics(parse_pairs(pred), parse_pairs(gt))
        return {
            "task": task,
            "correct": metrics["exact_match"],
            "score": metrics["pair_f1"],
            "meta": metrics,
        }

    if task == "A2A_Object_Matching":
        gt_bbox = parse_bbox(gt)
        pred_bbox = parse_bbox(pred)
        if gt_bbox is not None:
            metrics = bbox_geometry_metrics(pred_bbox, gt_bbox)
            correct = metrics["combined_score"] >= bbox_score_threshold
            return {
                "task": task,
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "meta": {**metrics, "subtype": "bbox"},
            }
        correct = parse_region_labels(pred) == parse_region_labels(gt)
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": {"subtype": "region"},
        }

    if task == "A2A_Camera_Transformation":
        metrics = camera_transformation_metrics(pred, gt)
        if metrics["answer_format"] == "angle_only":
            correct = metrics["angle_error"] <= camera_threshold
        elif metrics["answer_format"] == "angle_distance":
            correct = (
                metrics["angle_error"] <= camera_threshold
                and metrics["distance_abs_error"] <= camera_distance_threshold
            )
        else:
            correct = False
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": metrics,
        }

    if task == "A2G_Path_Planning":
        correct = parse_direction_label(pred) == parse_direction_label(gt)
        return {
            "task": task,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "meta": {},
        }

    if task == "Motion_Understanding":
        metrics = motion_judge.score(row["question_id"], gt, pred)
        return {
            "task": task,
            "correct": metrics["correct"],
            "score": metrics["score"],
            "meta": metrics,
        }

    metrics = token_f1_metrics(pred, gt)
    return {
        "task": task,
        "correct": metrics["exact_match"],
        "score": metrics["token_f1"],
        "meta": {**metrics, "fallback_metric": "token_f1"},
    }


def safe_pct(a: int, b: int):
    return 0.0 if b == 0 else 100.0 * a / b


def build_summary(args, total, total_correct, total_score, per_task):
    summary = {
        "predictions": args.predictions,
        "annotations": args.annotations,
        "total_samples": total,
        "overall": {
            "score_pct": safe_pct(total_score, total),
            "score_sum": total_score,
            "exact_match_accuracy_pct": safe_pct(total_correct, total),
            "exact_match_correct": total_correct,
        },
        "tasks": {},
    }

    for task in sorted(per_task):
        stats = per_task[task]
        task_summary = {
            "total": stats["total"],
            "score_pct": safe_pct(stats["score_sum"], stats["total"]),
            "score_sum": stats["score_sum"],
            "exact_match_accuracy_pct": safe_pct(stats["correct"], stats["total"]),
            "exact_match_correct": stats["correct"],
        }

        if stats["bbox_scores"]:
            task_summary["bbox_metrics"] = {
                "threshold": args.bbox_score_threshold,
                "mean_iou": avg_or_none(stats["ious"]),
                "mean_center_score": avg_or_none(stats["center_scores"]),
                "mean_size_score": avg_or_none(stats["size_scores"]),
                "mean_composite_score": avg_or_none(stats["bbox_scores"]),
            }

        if stats["angle_errors"]:
            valid_errors = [x for x in stats["angle_errors"] if math.isfinite(x)]
            if valid_errors:
                task_summary["angle_metrics"] = {
                    "threshold": args.camera_threshold,
                    "mean_absolute_error": avg_or_none(valid_errors),
                    "within_5_deg_pct": safe_pct(sum(x <= 5 for x in valid_errors), len(valid_errors)),
                    "within_10_deg_pct": safe_pct(sum(x <= 10 for x in valid_errors), len(valid_errors)),
                    "within_20_deg_pct": safe_pct(sum(x <= 20 for x in valid_errors), len(valid_errors)),
                }

        camera_format_total = (
            stats["camera_angle_distance"]
            + stats["camera_angle_only"]
            + stats["camera_invalid"]
        )
        if camera_format_total:
            task_summary["camera_format_counts"] = {
                "angle_distance": stats["camera_angle_distance"],
                "angle_only": stats["camera_angle_only"],
                "invalid": stats["camera_invalid"],
            }

        if stats["codabench_final_scores"]:
            task_summary["codabench_camera_metrics"] = {
                "final_score": avg_or_none(stats["codabench_final_scores"]),
                "distance_rel_error": avg_or_none(stats["distance_rel_errors"]),
                "angle_rel_error": avg_or_none(stats["angle_rel_errors"]),
                "valid_samples": len(stats["codabench_final_scores"]),
                "invalid_samples": stats["codabench_invalid"],
                "mean_distance_abs_error": avg_or_none(stats["distance_abs_errors"]),
                "distance_threshold": args.camera_distance_threshold,
            }

        if stats["pair_f1s"]:
            macro_precision = sum(stats["pair_precisions"]) / len(stats["pair_precisions"])
            macro_recall = sum(stats["pair_recalls"]) / len(stats["pair_recalls"])
            macro_f1 = sum(stats["pair_f1s"]) / len(stats["pair_f1s"])
            micro_precision = 0.0 if stats["pred_pairs"] == 0 else stats["matched_pairs"] / stats["pred_pairs"]
            micro_recall = 0.0 if stats["gt_pairs"] == 0 else stats["matched_pairs"] / stats["gt_pairs"]
            micro_f1 = 0.0 if micro_precision + micro_recall == 0 else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            task_summary["pair_metrics"] = {
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }

        if stats["region_f1s"]:
            task_summary["region_metrics"] = {
                "macro_precision": avg_or_none(stats["region_precisions"]),
                "macro_recall": avg_or_none(stats["region_recalls"]),
                "macro_f1": avg_or_none(stats["region_f1s"]),
            }

        if stats["semantic_scores"]:
            task_summary["semantic_metrics"] = {
                "mean_semantic_score": avg_or_none(stats["semantic_scores"]),
            }

        if stats["token_f1s"]:
            task_summary["token_metrics"] = {
                "macro_precision": avg_or_none(stats["token_precisions"]),
                "macro_recall": avg_or_none(stats["token_recalls"]),
                "macro_f1": avg_or_none(stats["token_f1s"]),
            }

        summary["tasks"][task] = task_summary

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpatialUAV predictions.")
    parser.add_argument(
        "--predictions",
        default="./predictions/predictions.jsonl",
        help="Prediction JSONL file path.",
    )
    parser.add_argument(
        "--annotations",
        default="./SpatialUAV/annotations.jsonl",
        help="Annotation JSONL file path used to fill missing GT values.",
    )
    parser.add_argument(
        "--camera-threshold",
        type=float,
        default=10.0,
        help="Camera transformation accuracy threshold in degrees.",
    )
    parser.add_argument(
        "--camera-distance-threshold",
        type=float,
        default=10.0,
        help="Camera transformation accuracy threshold in meters.",
    )
    parser.add_argument(
        "--bbox-score-threshold",
        type=float,
        default=0.5,
        help="Composite threshold for Object Matching bbox answers.",
    )
    parser.add_argument(
        "--motion-judge-mode",
        choices=["gpt_54_mini", "token_f1"],
        default="gpt_54_mini",
        help="How to score Motion_Understanding. Use gpt_54_mini for external API semantic judging, or token_f1 as fallback.",
    )
    parser.add_argument(
        "--motion-judge-provider",
        choices=["auto", "openai", "gemini"],
        default="auto",
        help="API provider used by the external motion judge. auto maps gpt-* models to openai and gemini-* models to gemini.",
    )
    parser.add_argument(
        "--motion-judge-base-url",
        default="",
        help="API base URL for the motion judge. If empty, choose the default route from the provider.",
    )
    parser.add_argument(
        "--motion-judge-api-key",
        default="",
        help="API key for the motion judge endpoint. If empty, use AUTODL_API_KEY or API.py.",
    )
    parser.add_argument(
        "--motion-judge-model",
        default=DEFAULT_MOTION_JUDGE_MODEL,
        help="Served model name used for semantic judging.",
    )
    parser.add_argument(
        "--motion-judge-cache",
        default="./eval_cache/motion_understanding_gpt54mini_scores.json",
        help="Cache JSON file for motion-judge scores.",
    )
    parser.add_argument(
        "--motion-judge-max-tokens",
        type=int,
        default=128,
        help="Max output tokens for the motion judge response.",
    )
    parser.add_argument(
        "--motion-judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the motion judge.",
    )
    parser.add_argument(
        "--motion-judge-retries",
        type=int,
        default=3,
        help="Retry count for motion judge API failures.",
    )
    parser.add_argument(
        "--motion-judge-retry-delay",
        type=float,
        default=2.0,
        help="Base sleep seconds between motion judge retries.",
    )
    parser.add_argument(
        "--motion-judge-fallback-on-error",
        action="store_true",
        help="Fallback to token_f1 if the external motion judge still fails after retries.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save evaluation summary as JSON.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable per-sample progress logs for Motion_Understanding judge calls.",
    )
    args = parser.parse_args()

    if args.motion_judge_model == "token_f1" and args.motion_judge_mode == "gpt_54_mini":
        print(
            "Warning: --motion-judge-model token_f1 was interpreted as "
            "--motion-judge-mode token_f1. Use --motion-judge-mode token_f1 explicitly.",
            flush=True,
        )
        args.motion_judge_mode = "token_f1"

    api_defaults = load_api_reference_defaults(API_REFERENCE_PATH)
    if args.motion_judge_provider == "auto":
        args.motion_judge_provider = infer_motion_judge_provider(args.motion_judge_model)
    args.motion_judge_base_url = resolve_motion_judge_base_url(
        provider=args.motion_judge_provider,
        base_url=args.motion_judge_base_url,
        api_defaults=api_defaults,
    )
    if not args.motion_judge_api_key:
        args.motion_judge_api_key = os.getenv("AUTODL_API_KEY", "") or api_defaults.get("api_key", "")
    if args.motion_judge_mode == "gpt_54_mini" and not args.motion_judge_api_key:
        raise SystemExit(
            "Missing motion judge API key. Pass --motion-judge-api-key, set AUTODL_API_KEY, or configure API.py."
        )

    rows = load_jsonl(Path(args.predictions))
    rows = attach_ground_truth(rows, Path(args.annotations))
    motion_total = sum(1 for row in rows if task_name_from_id(row["question_id"]) == "Motion_Understanding")
    motion_seen = 0
    motion_judge = MotionJudge(
        mode=args.motion_judge_mode,
        provider=args.motion_judge_provider,
        base_url=args.motion_judge_base_url,
        api_key=args.motion_judge_api_key,
        model=args.motion_judge_model,
        cache_path=Path(args.motion_judge_cache),
        max_tokens=args.motion_judge_max_tokens,
        temperature=args.motion_judge_temperature,
        retries=args.motion_judge_retries,
        retry_delay=args.motion_judge_retry_delay,
        fallback_on_error=args.motion_judge_fallback_on_error,
    )
    per_task = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "score_sum": 0.0,
            "ious": [],
            "center_scores": [],
            "size_scores": [],
            "bbox_scores": [],
            "angle_errors": [],
            "distance_abs_errors": [],
            "angle_rel_errors": [],
            "distance_rel_errors": [],
            "codabench_final_scores": [],
            "codabench_invalid": 0,
            "camera_angle_distance": 0,
            "camera_angle_only": 0,
            "camera_invalid": 0,
            "pair_precisions": [],
            "pair_recalls": [],
            "pair_f1s": [],
            "matched_pairs": 0,
            "pred_pairs": 0,
            "gt_pairs": 0,
            "token_precisions": [],
            "token_recalls": [],
            "token_f1s": [],
            "region_precisions": [],
            "region_recalls": [],
            "region_f1s": [],
            "semantic_scores": [],
        }
    )

    total = 0
    total_correct = 0
    total_score = 0.0

    for row in rows:
        task = task_name_from_id(row["question_id"])
        started_at = None
        if task == "Motion_Understanding":
            motion_seen += 1
            started_at = time.time()
            if not args.quiet_progress:
                judge_label = (
                    args.motion_judge_mode
                    if args.motion_judge_mode == "token_f1"
                    else args.motion_judge_model
                )
                print(
                    f"[Motion_Understanding {motion_seen}/{motion_total}] "
                    f"judging {row['question_id']} with {judge_label}...",
                    flush=True,
                )

        result = evaluate_row(
            row,
            args.camera_threshold,
            args.camera_distance_threshold,
            args.bbox_score_threshold,
            motion_judge,
        )
        if task == "Motion_Understanding" and not args.quiet_progress:
            elapsed = time.time() - started_at
            meta = result["meta"].get("meta", {})
            if meta.get("judge_mode") == "token_f1":
                cache_status = "local token_f1"
            elif meta.get("cache_hit"):
                cache_status = "cache hit"
            else:
                cache_status = "api call"
            print(
                f"[Motion_Understanding {motion_seen}/{motion_total}] "
                f"done {row['question_id']}: score={result['score']:.3f}, "
                f"{cache_status}, {elapsed:.1f}s",
                flush=True,
            )

        per_task[task]["total"] += 1
        per_task[task]["score_sum"] += result["score"]
        total += 1
        total_score += result["score"]
        if result["correct"]:
            per_task[task]["correct"] += 1
            total_correct += 1

        meta = result["meta"]
        if "iou" in meta:
            per_task[task]["ious"].append(meta["iou"])
        if "center_score" in meta:
            per_task[task]["center_scores"].append(meta["center_score"])
        if "size_score" in meta:
            per_task[task]["size_scores"].append(meta["size_score"])
        if "combined_score" in meta:
            per_task[task]["bbox_scores"].append(meta["combined_score"])
        if "angle_error" in meta:
            per_task[task]["angle_errors"].append(meta["angle_error"])
        if "distance_abs_error" in meta and math.isfinite(meta["distance_abs_error"]):
            per_task[task]["distance_abs_errors"].append(meta["distance_abs_error"])
        if meta.get("angle_rel_error") is not None:
            per_task[task]["angle_rel_errors"].append(meta["angle_rel_error"])
        if meta.get("distance_rel_error") is not None:
            per_task[task]["distance_rel_errors"].append(meta["distance_rel_error"])
        if meta.get("codabench_final_score") is not None:
            per_task[task]["codabench_final_scores"].append(meta["codabench_final_score"])
        if "codabench_valid" in meta and not meta["codabench_valid"]:
            per_task[task]["codabench_invalid"] += 1
        if meta.get("answer_format") == "angle_distance":
            per_task[task]["camera_angle_distance"] += 1
        elif meta.get("answer_format") == "angle_only":
            per_task[task]["camera_angle_only"] += 1
        elif "answer_format" in meta:
            per_task[task]["camera_invalid"] += 1
        if "pair_precision" in meta:
            per_task[task]["pair_precisions"].append(meta["pair_precision"])
        if "pair_recall" in meta:
            per_task[task]["pair_recalls"].append(meta["pair_recall"])
        if "pair_f1" in meta:
            per_task[task]["pair_f1s"].append(meta["pair_f1"])
        if "matched_pairs" in meta:
            per_task[task]["matched_pairs"] += meta["matched_pairs"]
        if "pred_pairs" in meta:
            per_task[task]["pred_pairs"] += meta["pred_pairs"]
        if "gt_pairs" in meta:
            per_task[task]["gt_pairs"] += meta["gt_pairs"]
        if "token_precision" in meta:
            per_task[task]["token_precisions"].append(meta["token_precision"])
        if "token_recall" in meta:
            per_task[task]["token_recalls"].append(meta["token_recall"])
        if "token_f1" in meta:
            per_task[task]["token_f1s"].append(meta["token_f1"])
        if "region_precision" in meta:
            per_task[task]["region_precisions"].append(meta["region_precision"])
        if "region_recall" in meta:
            per_task[task]["region_recalls"].append(meta["region_recall"])
        if "region_f1" in meta:
            per_task[task]["region_f1s"].append(meta["region_f1"])
        if "semantic_score" in meta:
            per_task[task]["semantic_scores"].append(meta["semantic_score"])

    print(f"Predictions: {args.predictions}")
    print(f"Total samples: {total}")
    print(f"Overall score: {safe_pct(total_score, total):.2f}% ({total_score:.2f}/{total})")
    print(f"Overall exact match accuracy: {safe_pct(total_correct, total):.2f}% ({total_correct}/{total})")
    print("")

    for task in sorted(per_task):
        stats = per_task[task]
        acc = safe_pct(stats["score_sum"], stats["total"])
        print(f"[{task}]")
        print(f"  score: {acc:.2f}% ({stats['score_sum']:.2f}/{stats['total']})")
        print(f"  exact match accuracy: {safe_pct(stats['correct'], stats['total']):.2f}% ({stats['correct']}/{stats['total']})")

        if stats["bbox_scores"]:
            mean_iou = sum(stats["ious"]) / len(stats["ious"])
            mean_center = sum(stats["center_scores"]) / len(stats["center_scores"])
            mean_size = sum(stats["size_scores"]) / len(stats["size_scores"])
            mean_bbox_score = sum(stats["bbox_scores"]) / len(stats["bbox_scores"])
            print(
                f"  bbox correct if composite score >= {args.bbox_score_threshold:.2f} "
                "(0.5*IoU + 0.25*center + 0.25*size)"
            )
            print(f"  mean IoU: {mean_iou:.4f}")
            print(f"  mean center score: {mean_center:.4f}")
            print(f"  mean size score: {mean_size:.4f}")
            print(f"  mean composite score: {mean_bbox_score:.4f}")

        if stats["angle_errors"]:
            valid_errors = [x for x in stats["angle_errors"] if math.isfinite(x)]
            if valid_errors:
                mae = sum(valid_errors) / len(valid_errors)
                acc_5 = safe_pct(sum(x <= 5 for x in valid_errors), len(valid_errors))
                acc_10 = safe_pct(sum(x <= 10 for x in valid_errors), len(valid_errors))
                acc_20 = safe_pct(sum(x <= 20 for x in valid_errors), len(valid_errors))
                print(f"  mean absolute angle error: {mae:.2f} degrees")
                print(f"  within 5 degrees: {acc_5:.2f}%")
                print(f"  within 10 degrees: {acc_10:.2f}%")
                print(f"  within 20 degrees: {acc_20:.2f}%")

        camera_format_total = (
            stats["camera_angle_distance"]
            + stats["camera_angle_only"]
            + stats["camera_invalid"]
        )
        if camera_format_total:
            print(
                "  camera answer formats: "
                f"{stats['camera_angle_distance']} angle+distance, "
                f"{stats['camera_angle_only']} angle-only, "
                f"{stats['camera_invalid']} invalid"
            )
            print(
                f"  angle-only correct if angle <= {args.camera_threshold:.2f} degrees; "
                f"angle+distance correct if angle <= {args.camera_threshold:.2f} degrees "
                f"and distance <= {args.camera_distance_threshold:.2f} meters"
            )

        if stats["codabench_final_scores"]:
            final_score = sum(stats["codabench_final_scores"]) / len(stats["codabench_final_scores"])
            angle_rel_error = sum(stats["angle_rel_errors"]) / len(stats["angle_rel_errors"])
            distance_rel_error = sum(stats["distance_rel_errors"]) / len(stats["distance_rel_errors"])
            mean_distance_abs_error = (
                sum(stats["distance_abs_errors"]) / len(stats["distance_abs_errors"])
                if stats["distance_abs_errors"]
                else math.nan
            )
            print("  Codabench camera metrics are lower-is-better")
            print(f"  final_score: {final_score:.6f}")
            print(f"  distance_rel_error: {distance_rel_error:.6f}")
            print(f"  angle_rel_error: {angle_rel_error:.6f}")
            print(f"  valid Codabench samples: {len(stats['codabench_final_scores'])}")
            print(f"  invalid Codabench samples: {stats['codabench_invalid']}")
            print(
                f"  mean absolute distance error: {mean_distance_abs_error:.2f} meters"
            )

        if stats["pair_f1s"]:
            macro_precision = sum(stats["pair_precisions"]) / len(stats["pair_precisions"])
            macro_recall = sum(stats["pair_recalls"]) / len(stats["pair_recalls"])
            macro_f1 = sum(stats["pair_f1s"]) / len(stats["pair_f1s"])
            micro_precision = (
                0.0 if stats["pred_pairs"] == 0 else stats["matched_pairs"] / stats["pred_pairs"]
            )
            micro_recall = 0.0 if stats["gt_pairs"] == 0 else stats["matched_pairs"] / stats["gt_pairs"]
            micro_f1 = (
                0.0
                if micro_precision + micro_recall == 0
                else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            )
            print("  shared association is evaluated as an unordered set of directed pairs")
            print("  task score uses pair F1")
            print(f"  macro pair precision: {macro_precision:.4f}")
            print(f"  macro pair recall: {macro_recall:.4f}")
            print(f"  macro pair F1: {macro_f1:.4f}")
            print(f"  micro pair precision: {micro_precision:.4f}")
            print(f"  micro pair recall: {micro_recall:.4f}")
            print(f"  micro pair F1: {micro_f1:.4f}")

        if stats["region_f1s"]:
            macro_precision = sum(stats["region_precisions"]) / len(stats["region_precisions"])
            macro_recall = sum(stats["region_recalls"]) / len(stats["region_recalls"])
            macro_f1 = sum(stats["region_f1s"]) / len(stats["region_f1s"])
            print("  region answers are evaluated as sets")
            print("  task score uses set recall only when |pred| <= |gt|; otherwise score = 0")
            print(f"  macro region precision: {macro_precision:.4f}")
            print(f"  macro region recall: {macro_recall:.4f}")
            print(f"  macro region F1: {macro_f1:.4f}")

        if stats["semantic_scores"]:
            mean_semantic = sum(stats["semantic_scores"]) / len(stats["semantic_scores"])
            print(f"  semantic judge mean score: {mean_semantic:.4f}")

        if stats["token_f1s"]:
            macro_precision = sum(stats["token_precisions"]) / len(stats["token_precisions"])
            macro_recall = sum(stats["token_recalls"]) / len(stats["token_recalls"])
            macro_f1 = sum(stats["token_f1s"]) / len(stats["token_f1s"])
            print("  text tasks are evaluated with token-level overlap")
            print(f"  macro token precision: {macro_precision:.4f}")
            print(f"  macro token recall: {macro_recall:.4f}")
            print(f"  macro token F1: {macro_f1:.4f}")

        print("")

    if args.output_json:
        summary = build_summary(args, total, total_correct, total_score, per_task)
        dump_json(Path(args.output_json), summary)
        print(f"Saved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()
