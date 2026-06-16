import argparse
import base64
import json
import logging
import mimetypes
import os
import random
import re
import sys
import time
import types
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_ANNOTATIONS = Path("SpatialUAV/annotations.jsonl")
DEFAULT_OUTPUTS = {
    "autodl": Path("predictions/predictions_autodl_api.jsonl"),
    "cambrian": Path("predictions/predictions_cambrian_s_7b.jsonl"),
    "internvl35": Path("predictions/predictions_internvl35_8b.jsonl"),
    "qwen": Path("predictions/predictions_qwen_oai.jsonl"),
    "spatialvlm": Path("predictions/predictions_spatialvlm.jsonl"),
    "vst": Path("predictions/predictions_vst_7b_sft.jsonl"),
}
DEFAULT_MODELS = {
    "cambrian": "/path/to/Cambrian-S-7B",
    "internvl35": "/path/to/InternVL3_5-8B",
    "qwen": "/path/to/Qwen3.6-35B-A3B",
    "spatialvlm": "/path/to/Qwen2.5-VL-3B-Instruct",
    "vst": "/path/to/VST-7B-SFT",
}
DEFAULT_BASE_URLS = {
    "openai": "https://www.autodl.art/api/v1",
    "anthropic": "https://www.autodl.art/api/v1/anthropic",
    "gemini": "https://www.autodl.art/api/v1/gemini",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VST_THINK_SYSTEM_PROMPT = (
    "You are a helpful assistant. You should first think about the reasoning process in "
    "the mind and then provide the user with the answer. The reasoning process is "
    "enclosed within <think> </think> tags, i.e. <think> reasoning process here "
    "</think> answer here."
)
INTERNVL_R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:
1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.
Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_completed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid:
                completed.add(qid)
    return completed


def sample_evenly(items: Sequence[str], limit: Optional[int]) -> List[str]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]

    last = len(items) - 1
    indices = []
    for i in range(limit):
        idx = round(i * last / (limit - 1))
        if not indices or idx != indices[-1]:
            indices.append(idx)

    selected = [items[i] for i in indices]
    while len(selected) < limit:
        selected.append(items[-1])
    return selected[:limit]


def resolve_image_paths(
    image_paths: Iterable[str], base_dir: Path, image_limit: Optional[int]
) -> List[Path]:
    resolved = []
    for rel_path in image_paths:
        path = Path(rel_path)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append((base_dir / path).resolve())
            candidates.append((base_dir.parent / path).resolve())
            candidates.append(path.resolve())

        for candidate in candidates:
            if candidate.exists():
                resolved.append(candidate)
                break
        else:
            tried = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"Image not found: {rel_path}. Tried: {tried}")

    return [Path(x) for x in sample_evenly([str(x) for x in resolved], image_limit)]


def should_keep(row: dict, task_prefixes: Optional[Sequence[str]]) -> bool:
    if not task_prefixes:
        return True
    qid = row["id"]
    return any(qid.startswith(prefix) for prefix in task_prefixes)


def strip_thinking(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def local_or_remote_model(model: str):
    model_path = Path(model).expanduser()
    return model_path if model_path.exists() else model


def optional_pixels(value: Optional[int]) -> Optional[int]:
    return None if value is None or value < 0 else value


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def read_image_as_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def import_pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: Pillow\n"
            "Install it with:\n"
            "  pip install -U Pillow"
        ) from exc
    return Image, ImageOps


def read_resized_jpeg_payload(path: Path, max_side: int, jpeg_quality: int) -> bytes:
    Image, ImageOps = import_pillow()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.convert("RGBA").split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(resized_size, Image.Resampling.LANCZOS)

        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )
        return output.getvalue()


def get_torch_dtype(torch, dtype_name: str):
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    if dtype_name in ("fp32", "float32"):
        return torch.float32
    return "auto"


class AutoDLRunner:
    name = "autodl"

    def __init__(self, args):
        self.args = args

    def setup(self):
        if not self.args.api_key:
            raise SystemExit("Missing API key. Pass --api-key or set AUTODL_API_KEY.")
        if self.args.anthropic_image_max_side is not None and self.args.anthropic_image_max_side < 1:
            raise SystemExit("--anthropic-image-max-side must be a positive integer.")
        if not 1 <= self.args.anthropic_image_jpeg_quality <= 95:
            raise SystemExit("--anthropic-image-jpeg-quality must be between 1 and 95.")
        if self.args.provider == "auto":
            self.args.provider = self.infer_provider_from_model(self.args.model)
        if self.args.base_url is None:
            self.args.base_url = DEFAULT_BASE_URLS[self.args.provider]

    def print_summary(self, rows: Sequence[dict]) -> None:
        print(f"Backend: {self.name}")
        print(f"Provider: {self.args.provider}")
        print(f"Base URL: {self.args.base_url}")
        print(f"Model: {self.args.model}")
        print(f"Samples: {len(rows)}")
        print(f"Images enabled: {not self.args.text_only}")
        if self.args.image_limit:
            print(f"Per-sample image cap: {self.args.image_limit}")
        print(f"Writing predictions to: {self.args.output}")

    @staticmethod
    def infer_provider_from_model(model: str) -> str:
        model_name = (model or "").strip().lower()
        if model_name.startswith("claude-"):
            return "anthropic"
        if "gemini" in model_name:
            return "gemini"
        if model_name.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        raise ValueError(
            f"Could not infer provider from model: {model}. "
            "Pass --provider explicitly with one of: openai, anthropic, gemini."
        )

    @staticmethod
    def import_openai():
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: openai\n"
                "Install it with:\n"
                "  pip install -U openai"
            ) from exc
        return OpenAI

    @staticmethod
    def import_anthropic():
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: anthropic\n"
                "Install it with:\n"
                "  pip install -U anthropic"
            ) from exc
        return Anthropic

    @staticmethod
    def import_gemini():
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: google-genai\n"
                "Install it with:\n"
                "  pip install -U google-genai"
            ) from exc
        return genai, types

    def read_anthropic_image_source(self, path: Path) -> dict:
        if self.args.anthropic_image_max_side is None:
            return {
                "type": "base64",
                "media_type": guess_mime_type(path),
                "data": read_image_as_base64(path),
            }
        image_bytes = read_resized_jpeg_payload(
            path,
            self.args.anthropic_image_max_side,
            self.args.anthropic_image_jpeg_quality,
        )
        return {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }

    @staticmethod
    def coerce_openai_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(item.text)
            return "".join(parts).strip()
        return "" if content is None else str(content)

    def build_openai_messages(self, question: str, image_paths: Sequence[Path]) -> List[dict]:
        messages = []
        if self.args.system_prompt.strip():
            messages.append({"role": "system", "content": self.args.system_prompt})
        if not self.args.text_only and image_paths:
            content = [{"type": "text", "text": question}]
            for path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{guess_mime_type(path)};base64,{read_image_as_base64(path)}"
                        },
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": question})
        return messages

    def build_anthropic_messages(self, question: str, image_paths: Sequence[Path]) -> List[dict]:
        if not self.args.text_only and image_paths:
            content = [{"type": "text", "text": question}]
            for path in image_paths:
                content.append({"type": "image", "source": self.read_anthropic_image_source(path)})
            return [{"role": "user", "content": content}]
        return [{"role": "user", "content": question}]

    def build_gemini_contents(self, question: str, image_paths: Sequence[Path], types_module):
        parts = []
        if not self.args.text_only and image_paths:
            for path in image_paths:
                parts.append(
                    types_module.Part.from_bytes(
                        data=path.read_bytes(),
                        mime_type=guess_mime_type(path),
                    )
                )
        parts.append(types_module.Part.from_text(text=question))
        return [types_module.Content(role="user", parts=parts)]

    def call_openai_api(self, question: str, image_paths: Sequence[Path]) -> str:
        OpenAI = self.import_openai()
        client = OpenAI(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            timeout=self.args.timeout,
            max_retries=self.args.max_retries,
        )
        request_kwargs = {
            "model": self.args.model,
            "messages": self.build_openai_messages(question, image_paths),
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        if self.args.presence_penalty != 0.0:
            request_kwargs["presence_penalty"] = self.args.presence_penalty
        if self.args.top_k is not None:
            request_kwargs["extra_body"] = {"top_k": self.args.top_k}
        response = client.chat.completions.create(**request_kwargs)
        text = response.choices[0].message.content if response.choices else ""
        return self.coerce_openai_text(text)

    def call_anthropic_api(self, question: str, image_paths: Sequence[Path]) -> str:
        Anthropic = self.import_anthropic()
        client = Anthropic(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            timeout=self.args.timeout,
            max_retries=self.args.max_retries,
        )
        request_kwargs = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "messages": self.build_anthropic_messages(question, image_paths),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        if self.args.system_prompt.strip():
            request_kwargs["system"] = self.args.system_prompt
        response = client.messages.create(**request_kwargs)
        text_parts = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return "".join(text_parts).strip()

    def call_gemini_api(self, question: str, image_paths: Sequence[Path]) -> str:
        genai, types_module = self.import_gemini()
        client = genai.Client(
            api_key=self.args.api_key,
            http_options={"base_url": self.args.base_url},
        )
        config_kwargs = {
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_output_tokens": self.args.max_tokens,
        }
        if self.args.top_k is not None:
            config_kwargs["top_k"] = self.args.top_k
        if self.args.system_prompt.strip():
            config_kwargs["system_instruction"] = self.args.system_prompt
        response = client.models.generate_content(
            model=self.args.model,
            contents=self.build_gemini_contents(question, image_paths, types_module),
            config=types_module.GenerateContentConfig(**config_kwargs),
        )
        return getattr(response, "text", "") or ""

    def predict(self, row: dict, question: str, image_paths: Sequence[Path]) -> Tuple[str, dict]:
        if self.args.provider == "openai":
            return self.call_openai_api(question, image_paths), {"provider": self.args.provider}
        if self.args.provider == "anthropic":
            return self.call_anthropic_api(question, image_paths), {"provider": self.args.provider}
        if self.args.provider == "gemini":
            return self.call_gemini_api(question, image_paths), {"provider": self.args.provider}
        raise ValueError(f"Unsupported provider: {self.args.provider}")


class CambrianRunner:
    name = "cambrian"

    def __init__(self, args):
        self.args = args

    @staticmethod
    def add_cambrian_repo(repo_path: Optional[Path]) -> None:
        candidates = []
        if repo_path is not None:
            candidates.append(repo_path)
        env_repo = os.environ.get("CAMBRIAN_REPO")
        if env_repo:
            candidates.append(Path(env_repo))
        candidates.extend([Path.cwd(), Path.cwd() / "cambrian-s", Path.cwd().parent / "cambrian-s"])

        for candidate in candidates:
            repo_root = candidate.expanduser().resolve()
            if (repo_root / "cambrian").is_dir():
                sys.path.insert(0, str(repo_root))
                return

        if repo_path is not None or env_repo:
            searched = ", ".join(str(path.expanduser()) for path in candidates)
            raise FileNotFoundError(
                "Could not find Cambrian-S repo root containing a `cambrian` package. "
                f"Searched: {searched}"
            )

    @staticmethod
    def install_ezcolorlog_fallback() -> None:
        if "ezcolorlog" in sys.modules:
            return
        try:
            __import__("ezcolorlog")
            return
        except ImportError:
            pass

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        module = types.ModuleType("ezcolorlog")
        module.root_logger = logging.getLogger()
        module.root_loggger = module.root_logger
        module.log_stdout = logging.StreamHandler()

        def setup_logging(*args, **kwargs):
            return module.root_logger

        module.setup_logging = setup_logging
        sys.modules["ezcolorlog"] = module

    def import_dependencies(self):
        self.add_cambrian_repo(self.args.cambrian_repo)
        self.install_ezcolorlog_fallback()
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies for local Cambrian-S inference.\n"
                "Install the Cambrian-S environment first, including torch and pillow.\n"
                f"Original import error: {exc}"
            ) from exc

        try:
            from cambrian.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from cambrian.conversation import conv_templates
            from cambrian.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
            from cambrian.model.builder import load_pretrained_model
        except ImportError as exc:
            raise SystemExit(
                "Could not import Cambrian-S modules.\n"
                "Clone https://github.com/cambrian-mllm/cambrian-s and pass its root with:\n"
                "  --cambrian-repo /path/to/cambrian-s\n"
                "or set CAMBRIAN_REPO=/path/to/cambrian-s.\n"
                f"Original import error: {exc}"
            ) from exc

        return (
            torch,
            Image,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
            conv_templates,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
            load_pretrained_model,
        )

    def setup(self):
        (
            self.torch,
            self.Image,
            self.DEFAULT_IMAGE_TOKEN,
            self.IMAGE_TOKEN_INDEX,
            self.conv_templates,
            self.get_model_name_from_path,
            self.process_images,
            self.tokenizer_image_token,
            self.load_pretrained_model,
        ) = self.import_dependencies()
        self.model_source = local_or_remote_model(self.args.model)
        model_name = self.get_model_name_from_path(str(self.model_source))
        self.tokenizer, self.model, self.image_processor, _ = self.load_pretrained_model(
            str(self.model_source),
            None,
            model_name,
            device_map=self.args.device,
        )
        self.model.config.video_max_frames = self.args.video_max_frames
        self.model.config.video_fps = self.args.video_fps
        self.model.config.video_force_sample = self.args.video_force_sample
        self.model.config.miv_token_len = self.args.miv_token_len
        self.model.config.si_token_len = self.args.si_token_len
        self.model.config.image_aspect_ratio = self.args.image_aspect_ratio
        self.model.config.anyres_max_subimages = self.args.anyres_max_subimages
        self.model.eval()

    def print_summary(self, rows: Sequence[dict]) -> None:
        print(f"Backend: {self.name}")
        print(f"Model path: {self.model_source}")
        print(f"Model id: {self.args.model}")
        print(f"Model class: {self.model.__class__.__name__}")
        print(f"Samples: {len(rows)}")
        print(f"Device: {self.args.device}")
        print(f"Conversation template: {self.args.conv_template}")
        print(f"Writing predictions to: {self.args.output}")

    def prepare_visuals(self, image_paths: Sequence[Path]):
        images = [self.Image.open(path).convert("RGB") for path in image_paths]
        use_pad = len(images) > 1
        return self.process_images(images, self.image_processor, self.model.config, use_pad=use_pad)

    def build_prompt(self, question: str, image_count: int):
        assert "<image>" not in question
        image_prefix = self.DEFAULT_IMAGE_TOKEN * max(image_count, 1)
        prompt_question = f"{image_prefix}\n{question}"
        conv = self.conv_templates[self.args.conv_template].copy()
        conv.append_message(conv.roles[0], prompt_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
        return input_ids

    def move_visuals(self, visual_tensors):
        moved = []
        for tensor in visual_tensors:
            if self.torch.cuda.is_available() and self.args.device.startswith("cuda"):
                moved.append(tensor.half().to(self.args.device))
            else:
                moved.append(tensor.float().to(self.args.device))
        return moved

    def predict(self, row: dict, question: str, image_paths: Sequence[Path]) -> Tuple[str, dict]:
        visual_tensors, visual_sizes = self.prepare_visuals(image_paths)
        input_ids = self.build_prompt(question, len(image_paths))
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids.to(self.args.device),
                images=self.move_visuals(visual_tensors),
                image_sizes=visual_sizes,
                use_cache=True,
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_tokens,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip(), {}


class InternVL35Runner:
    name = "internvl35"

    def __init__(self, args):
        self.args = args

    @staticmethod
    def import_dependencies():
        try:
            import torch
            import torchvision.transforms as T
            from PIL import Image
            from torchvision.transforms.functional import InterpolationMode
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies for local InternVL3.5 inference.\n"
                "Install them with:\n"
                "  pip install -U torch torchvision transformers accelerate pillow\n"
                "InternVL3.5 recommends transformers>=4.52.1.\n"
                f"Original import error: {exc}"
            ) from exc
        return torch, T, Image, InterpolationMode, AutoModel, AutoTokenizer

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: Sequence[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> Tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def build_transform(self, input_size: int):
        return self.T.Compose(
            [
                self.T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                self.T.Resize((input_size, input_size), interpolation=self.InterpolationMode.BICUBIC),
                self.T.ToTensor(),
                self.T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def dynamic_preprocess(
        self,
        image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))

        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image_file: Path):
        image = self.Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=self.args.input_size)
        images = self.dynamic_preprocess(
            image,
            image_size=self.args.input_size,
            use_thumbnail=True,
            max_num=self.args.max_tiles,
        )
        pixel_values = [transform(image) for image in images]
        return self.torch.stack(pixel_values)

    def setup(self):
        self.torch, self.T, self.Image, self.InterpolationMode, AutoModel, AutoTokenizer = (
            self.import_dependencies()
        )
        if self.args.device_map == "":
            self.args.device_map = None
        self.model_source = local_or_remote_model(self.args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            trust_remote_code=True,
            use_fast=False,
        )
        model_kwargs = {
            "torch_dtype": get_torch_dtype(self.torch, self.args.dtype),
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.args.device_map:
            model_kwargs["device_map"] = self.args.device_map
        if self.args.use_flash_attn:
            model_kwargs["use_flash_attn"] = True
        if self.args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        try:
            self.model = AutoModel.from_pretrained(self.model_source, **model_kwargs).eval()
        except Exception:
            if not self.args.use_flash_attn:
                raise
            model_kwargs.pop("use_flash_attn", None)
            self.model = AutoModel.from_pretrained(self.model_source, **model_kwargs).eval()
        if self.args.enable_thinking:
            self.model.system_message = INTERNVL_R1_SYSTEM_PROMPT
        self.input_device = self.infer_input_device()
        self.input_dtype = get_torch_dtype(self.torch, self.args.dtype)
        if self.input_dtype == "auto":
            self.input_dtype = self.torch.bfloat16

    def infer_input_device(self):
        if self.torch.cuda.is_available():
            return self.torch.device("cuda", self.torch.cuda.current_device())
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cpu")

    def print_summary(self, rows: Sequence[dict]) -> None:
        print(f"Backend: {self.name}")
        print(f"Model path: {self.model_source}")
        print(f"Model id: {self.args.model}")
        print(f"Model class: {self.model.__class__.__name__}")
        print(f"Samples: {len(rows)}")
        print(f"Image tile size: {self.args.input_size}")
        print(f"Max tiles per image: {self.args.max_tiles}")
        print(f"Input device: {self.input_device}")
        print(f"Thinking mode: {'on' if self.args.enable_thinking else 'off'}")
        print(f"Writing predictions to: {self.args.output}")

    def build_inputs(self, image_paths: Sequence[Path], question: str, row: dict):
        if not image_paths:
            return None, question, None
        pixel_values_list = []
        num_patches_list = []
        for image_path in image_paths:
            pixel_values = self.load_image(image_path)
            num_patches_list.append(pixel_values.size(0))
            pixel_values_list.append(pixel_values)

        pixel_values = self.torch.cat(pixel_values_list, dim=0).to(
            dtype=self.input_dtype,
            device=self.input_device,
        )
        if len(image_paths) == 1:
            prompt = f"<image>\n{question}"
        elif row["id"].startswith("Motion_Understanding"):
            prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(image_paths)))
            prompt = prefix + question
        else:
            prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(image_paths)))
            prompt = prefix + question
        return pixel_values, prompt, num_patches_list

    def predict(self, row: dict, question: str, image_paths: Sequence[Path]) -> Tuple[str, dict]:
        pixel_values, prompt, num_patches_list = self.build_inputs(image_paths, question, row)
        generation_config = {
            "max_new_tokens": self.args.max_tokens,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            generation_config["temperature"] = self.args.temperature
            generation_config["top_p"] = self.args.top_p
            if self.args.top_k is not None:
                generation_config["top_k"] = self.args.top_k

        chat_kwargs = {
            "tokenizer": self.tokenizer,
            "pixel_values": pixel_values,
            "question": prompt,
            "generation_config": generation_config,
        }
        if num_patches_list is not None:
            chat_kwargs["num_patches_list"] = num_patches_list
        with self.torch.no_grad():
            return self.model.chat(**chat_kwargs), {}


class QwenRunner:
    name = "qwen"

    def __init__(self, args):
        self.args = args

    @staticmethod
    def import_dependencies():
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies for local Qwen inference.\n"
                "Install them with:\n"
                "  pip install -U torch transformers accelerate\n"
                f"Original import error: {exc}"
            ) from exc
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = None
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None
        return (
            torch,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
            AutoProcessor,
            AutoTokenizer,
            LogitsProcessor,
            LogitsProcessorList,
            process_vision_info,
        )

    @staticmethod
    def apply_chat_template(template_owner, messages: List[dict], enable_thinking: bool) -> str:
        try:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @staticmethod
    def build_logits_processor(LogitsProcessor, LogitsProcessorList, penalty: float):
        if penalty == 0.0:
            return None

        class PresencePenaltyLogitsProcessor(LogitsProcessor):
            def __init__(self, presence_penalty: float):
                self.presence_penalty = presence_penalty

            def __call__(self, input_ids, scores):
                for batch_idx in range(input_ids.shape[0]):
                    seen_token_ids = input_ids[batch_idx].unique()
                    scores[batch_idx, seen_token_ids] -= self.presence_penalty
                return scores

        return LogitsProcessorList([PresencePenaltyLogitsProcessor(penalty)])

    @staticmethod
    def build_messages(
        image_paths: Sequence[Path],
        question: str,
        multimodal: bool,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> List[dict]:
        if multimodal:
            content = []
            for path in image_paths:
                image_content = {"type": "image", "image": str(path)}
                if min_pixels is not None:
                    image_content["min_pixels"] = min_pixels
                if max_pixels is not None:
                    image_content["max_pixels"] = max_pixels
                content.append(image_content)
            content.append({"type": "text", "text": question})
            return [{"role": "user", "content": content}]
        return [{"role": "user", "content": question}]

    def setup(self):
        (
            self.torch,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
            AutoProcessor,
            AutoTokenizer,
            LogitsProcessor,
            LogitsProcessorList,
            self.process_vision_info,
        ) = self.import_dependencies()
        self.model_source = local_or_remote_model(self.args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source, trust_remote_code=True)
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_source, trust_remote_code=True)
        except Exception:
            self.processor = None

        model_classes = []
        if self.processor is not None:
            model_classes.extend(
                cls
                for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq)
                if cls is not None
            )
        model_classes.append(AutoModelForCausalLM)

        last_error = None
        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(
                    self.model_source,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise last_error
        self.model.eval()
        self.multimodal = self.processor is not None and self.process_vision_info is not None
        self.logits_processor = self.build_logits_processor(
            LogitsProcessor,
            LogitsProcessorList,
            self.args.presence_penalty,
        )

    def print_summary(self, rows: Sequence[dict]) -> None:
        print(f"Backend: {self.name}")
        print(f"Model path: {self.model_source}")
        print(f"Model id: {self.args.model}")
        print(f"Model class: {self.model.__class__.__name__}")
        print(f"Samples: {len(rows)}")
        print(f"Thinking mode: {'on' if self.args.enable_thinking else 'off'}")
        print(f"Multimodal mode: {'on' if self.multimodal else 'off'}")
        print(f"Writing predictions to: {self.args.output}")

    def resolve_input_device(self) -> str:
        if self.torch.cuda.is_available():
            return "cuda"
        return str(self.model.device)

    def prepare_multimodal_inputs(self, messages: List[dict]):
        if self.process_vision_info is None:
            raise RuntimeError(
                "qwen_vl_utils is required for multimodal local inference.\n"
                "Install it with:\n"
                "  pip install qwen-vl-utils[decord]==0.0.8"
            )
        text = self.apply_chat_template(self.processor, messages, self.args.enable_thinking)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.resolve_input_device())

    def prepare_text_inputs(self, messages: List[dict]):
        prompt = self.apply_chat_template(self.tokenizer, messages, self.args.enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def decode_generated(self, output_ids, prompt_length: int) -> str:
        generated_ids = output_ids[0][prompt_length:]
        if self.processor is not None:
            return self.processor.batch_decode(
                [generated_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def predict(self, row: dict, question: str, image_paths: Sequence[Path]) -> Tuple[str, dict]:
        if image_paths and not self.multimodal:
            raise RuntimeError(
                "The loaded model directory does not expose multimodal preprocessing. "
                "Local inference for SpatialUAV requires a processor plus qwen_vl_utils."
            )
        messages = self.build_messages(
            image_paths,
            question,
            multimodal=self.multimodal,
            min_pixels=self.args.min_pixels,
            max_pixels=self.args.max_pixels,
        )
        if self.multimodal:
            inputs = self.prepare_multimodal_inputs(messages)
        else:
            inputs = self.prepare_text_inputs(messages)
        prompt_length = inputs["input_ids"].shape[1]
        generate_kwargs = {
            "max_new_tokens": self.args.max_tokens,
            "do_sample": self.args.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.logits_processor is not None:
            generate_kwargs["logits_processor"] = self.logits_processor
        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
            if self.args.top_k is not None:
                generate_kwargs["top_k"] = self.args.top_k
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        return self.decode_generated(output_ids, prompt_length), {}


class SpatialVLMRunner:
    name = "spatialvlm"

    def __init__(self, args):
        self.args = args

    @staticmethod
    def import_dependencies():
        try:
            import torch
            import transformers
            import qwen_vl_utils
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies for SpatialVLM inference. Install them first, for example:\n"
                "  pip install torch torchvision torchaudio\n"
                "  pip install transformers accelerate qwen-vl-utils[decord]==0.0.8\n"
                f"Original import error: {exc}"
            ) from exc
        return (
            torch,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            process_vision_info,
            transformers.__version__,
            getattr(qwen_vl_utils, "__version__", "unknown"),
        )

    @staticmethod
    def build_messages(image_paths: Sequence[Path], question: str) -> List[dict]:
        content = [{"type": "image", "image": str(path)} for path in image_paths]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def setup(self):
        (
            self.torch,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            self.process_vision_info,
            self.transformers_version,
            self.qwen_vl_utils_version,
        ) = self.import_dependencies()
        random.seed(self.args.seed)
        if self.torch.cuda.is_available():
            self.torch.manual_seed(self.args.seed)
            self.torch.cuda.manual_seed_all(self.args.seed)
        self.model_source = local_or_remote_model(self.args.model)
        print(f"Loading model: {self.model_source}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_source,
            trust_remote_code=True,
            revision=self.args.revision,
            force_download=self.args.force_download,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_source,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            revision=self.args.revision,
            force_download=self.args.force_download,
        )
        self.model.eval()

    def get_input_device(self):
        if self.torch.cuda.is_available():
            return "cuda"
        return str(self.model.device)

    def prepare_inputs(self, messages: List[dict]):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.get_input_device())

    @staticmethod
    def summarize_inputs(inputs) -> str:
        parts = []
        for key, value in inputs.items():
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            parts.append(f"{key}={shape}")
        return ", ".join(parts)

    def print_summary(self, rows: Sequence[dict]) -> None:
        print(f"Backend: {self.name}")
        print(f"Model path: {self.model_source}")
        print(f"Model id: {self.args.model}")
        print(f"Samples: {len(rows)}")
        if self.args.debug:
            print(f"transformers: {self.transformers_version}")
            print(f"qwen_vl_utils: {self.qwen_vl_utils_version}")
            print(f"model.config.transformers_version: {getattr(self.model.config, 'transformers_version', 'unknown')}")
            print(f"processor class: {self.processor.__class__.__name__}")
            print(f"model class: {self.model.__class__.__name__}")
        print(f"Writing predictions to: {self.args.output}")

    def predict(self, row: dict, question: str, image_paths: Sequence[Path]) -> Tuple[str, dict]:
        messages = self.build_messages(image_paths, question)
        inputs = self.prepare_inputs(messages)
        if self.args.debug:
            print(f"Debug input keys: {self.summarize_inputs(inputs)}")
        generation_kwargs = {
            "max_new_tokens": self.args.max_tokens,
            "do_sample": self.args.temperature > 0,
        }
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if self.args.temperature > 0:
            generation_kwargs["temperature"] = self.args.temperature
            generation_kwargs["top_p"] = self.args.top_p
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        prediction = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return prediction, {}


class VSTRunner:
    name = "vst"

    def __init__(self, args):
        self.args = args

    @staticmethod
    def import_dependencies():
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies for local VST-7B-SFT inference.\n"
                "Install them with:\n"
                "  pip install -U torch torchvision transformers accelerate pillow qwen-vl-utils[decord]\n"
                "For flash_attention_2, install flash-attn in the target CUDA environment.\n"
                f"Original import error: {exc}"
            ) from exc
        return torch, Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info

    @staticmethod
    def is_motion_row(row: dict) -> bool:
        return row["id"].startswith("Motion_Understanding")

    def build_messages(self, image_paths: Sequence[Path], question: str, row: dict) -> List[dict]:
        messages = []
        if self.args.enable_thinking:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": VST_THINK_SYSTEM_PROMPT}],
                }
            )

        if self.args.motion_as_video and self.is_motion_row(row) and len(image_paths) > 1:
            video_item = {
                "type": "video",
                "video": [str(path) for path in image_paths],
                "nframes": len(image_paths),
                "fps": self.args.sample_fps,
            }
            if optional_pixels(self.args.video_min_pixels) is not None:
                video_item["min_pixels"] = optional_pixels(self.args.video_min_pixels)
            if optional_pixels(self.args.video_max_pixels) is not None:
                video_item["max_pixels"] = optional_pixels(self.args.video_max_pixels)
            if optional_pixels(self.args.video_total_pixels) is not None:
                video_item["total_pixels"] = optional_pixels(self.args.video_total_pixels)
            content = [video_item]
        else:
            content = []
            for path in image_paths:
                image_item = {"type": "image", "image": str(path)}
                if optional_pixels(self.args.min_pixels) is not None:
                    image_item["min_pixels"] = optional_pixels(self.args.min_pixels)
                if optional_pixels(self.args.max_pixels) is not None:
                    image_item["max_pixels"] = optional_pixels(self.args.max_pixels)
                content.append(image_item)

        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        return messages

    def setup(self):
        if self.args.device_map == "":
            self.args.device_map = None
        self.torch, Qwen2_5_VLForConditionalGeneration, AutoProcessor, self.process_vision_info = (
            self.import_dependencies()
        )
        self.model_source = local_or_remote_model(self.args.model)
        model_kwargs = {
            "torch_dtype": "auto" if self.args.dtype == "auto" else getattr(self.torch, self.args.dtype),
            "device_map": self.args.device_map,
        }
        if self.args.attn_implementation:
            model_kwargs["attn_implementation"] = self.args.attn_implementation
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_source,
                **model_kwargs,
            )
        except Exception:
            if self.args.attn_implementation != "flash_attention_2":
                raise
            model_kwargs.pop("attn_implementation", None)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_source,
                **model_kwargs,
            )
        self.model.eval()
        processor_kwargs = {}
        if optional_pixels(self.args.min_pixels) is not None:
            processor_kwargs["min_pixels"] = optional_pixels(self.args.min_pixels)
        if optional_pixels(self.args.max_pixels) is not None:
            processor_kwargs["max_pixels"] = optional_pixels(self.args.max_pixels)
        self.processor = AutoProcessor.from_pretrained(self.model_source, **processor_kwargs)

    def resolve_input_device(self) -> str:
        if self.torch.cuda.is_available():
            return "cuda"
        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:
            return "cpu"

    def prepare_inputs(self, messages: List[dict]):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.resolve_input_device())

    def print_summary(self, rows: Sequence[dict]) -> None:
        print(f"Backend: {self.name}")
        print(f"Model path: {self.model_source}")
        print(f"Model id: {self.args.model}")
        print(f"Model class: {self.model.__class__.__name__}")
        print(f"Samples: {len(rows)}")
        print(f"Thinking mode: {'on' if self.args.enable_thinking else 'off'}")
        print(f"Motion as video: {'on' if self.args.motion_as_video else 'off'}")
        print(f"Writing predictions to: {self.args.output}")

    def predict(self, row: dict, question: str, image_paths: Sequence[Path]) -> Tuple[str, dict]:
        messages = self.build_messages(image_paths, question, row)
        inputs = self.prepare_inputs(messages)
        prompt_length = inputs["input_ids"].shape[1]
        generate_kwargs = {
            "max_new_tokens": self.args.max_tokens,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
            if self.args.top_k is not None:
                generate_kwargs["top_k"] = self.args.top_k
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = output_ids[0][prompt_length:]
        prediction = self.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return prediction, {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpatialUAV inference with a selected backend/model."
    )
    parser.add_argument(
        "--backend",
        choices=("autodl", "cambrian", "internvl35", "qwen", "spatialvlm", "vst"),
        required=True,
        help="Inference backend/model family to use.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Path to the SpatialUAV JSONL annotations file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file compatible with eval_spatialuav.py.",
    )
    parser.add_argument(
        "--model",
        "--model-id",
        dest="model",
        default=None,
        help="Model path, Hugging Face model id, or API model name.",
    )
    parser.add_argument(
        "--task-prefix",
        nargs="*",
        default=None,
        help="Optional task id prefixes to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on sample count.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N filtered samples.")
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Uniformly subsample each sample's images to at most this many.",
    )
    parser.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--provider", choices=("auto", "openai", "anthropic", "gemini"), default="auto")
    parser.add_argument("--api-key", default=os.getenv("AUTODL_API_KEY", ""))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--anthropic-image-max-side", type=int, default=None)
    parser.add_argument("--anthropic-image-jpeg-quality", type=int, default=85)
    parser.add_argument("--presence-penalty", type=float, default=0.0)

    parser.add_argument("--cambrian-repo", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conv-template", default="qwen_2")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--video-force-sample", action="store_true")
    parser.add_argument("--miv-token-len", type=int, default=196)
    parser.add_argument("--si-token-len", type=int, default=729)
    parser.add_argument("--image-aspect-ratio", default="pad")
    parser.add_argument("--anyres-max-subimages", type=int, default=9)

    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32", "bfloat16", "float16", "float32"),
        default="bf16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--no-flash-attn", dest="use_flash_attn", action="store_false")
    parser.set_defaults(use_flash_attn=True)
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--motion-as-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--video-min-pixels", type=int, default=-1)
    parser.add_argument("--video-max-pixels", type=int, default=-1)
    parser.add_argument("--video-total-pixels", type=int, default=-1)

    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUTS[args.backend]
    if args.model is None:
        if args.backend == "autodl":
            parser.error("--model is required for --backend autodl")
        args.model = DEFAULT_MODELS[args.backend]
    if args.max_tokens is None:
        args.max_tokens = 1280 if args.backend == "vst" else 256 if args.backend == "spatialvlm" else 512
    if args.backend == "vst" and args.dtype == "bf16":
        args.dtype = "bfloat16"
    if args.backend in ("qwen", "vst"):
        for option in ("min_pixels", "max_pixels"):
            value = getattr(args, option)
            if value is not None and value <= 0 and args.backend == "qwen":
                parser.error(f"--{option.replace('_', '-')} must be positive")
        if args.min_pixels is not None and args.max_pixels is not None and args.min_pixels > args.max_pixels:
            parser.error("--min-pixels must not exceed --max-pixels")
    if args.backend == "vst":
        if args.min_pixels is None:
            args.min_pixels = 256 * 28 * 28
        if args.max_pixels is None:
            args.max_pixels = 1280 * 28 * 28
    return args


def build_runner(args):
    runners = {
        "autodl": AutoDLRunner,
        "cambrian": CambrianRunner,
        "internvl35": InternVL35Runner,
        "qwen": QwenRunner,
        "spatialvlm": SpatialVLMRunner,
        "vst": VSTRunner,
    }
    return runners[args.backend](args)


def select_rows(args) -> List[dict]:
    rows = load_jsonl(args.annotations)
    rows = [row for row in rows if should_keep(row, args.task_prefix)]
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    completed_ids = read_completed_ids(args.output) if args.resume else set()
    if completed_ids:
        rows = [row for row in rows if row["id"] not in completed_ids]
    return rows


def main():
    args = parse_args()
    runner = build_runner(args)
    runner.setup()
    rows = select_rows(args)
    if not rows:
        print("No samples to run after filtering/resume.")
        return
    runner.print_summary(rows)
    if args.image_limit:
        print(f"Per-sample image cap: {args.image_limit}")

    for index, row in enumerate(rows, start=1):
        question = row["conversations"][0]["value"]
        image_paths = []
        try:
            image_paths = resolve_image_paths(
                row.get("image", []),
                args.annotations.parent,
                args.image_limit,
            )
            raw_prediction, extra = runner.predict(row, question, image_paths)
            prediction = strip_thinking(raw_prediction)
            output_row = {
                "question_id": row["id"],
                "question": question,
                "gt": row.get("GT", ""),
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "source": row.get("source", ""),
                "image": [str(path) for path in image_paths],
                "model_id": args.model,
            }
            output_row.update(extra)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            output_row = {
                "question_id": row["id"],
                "question": question,
                "gt": row.get("GT", ""),
                "prediction": "",
                "source": row.get("source", ""),
                "image": [str(path) for path in image_paths] if image_paths else row.get("image", []),
                "model_id": args.model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.backend == "autodl":
                output_row["provider"] = args.provider

        append_jsonl(args.output, output_row)
        print(f"[{index}/{len(rows)}] {row['id']} -> {output_row['prediction']}")

        if args.backend == "autodl" and args.sleep_seconds > 0 and index < len(rows):
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
