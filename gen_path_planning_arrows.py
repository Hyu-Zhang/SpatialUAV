import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INPUT_DIR = Path("samples_A2G_Path_Planning_sampled/ground")
OUTPUT_DIR = Path("samples_A2G_Path_Planning/ground")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Six directions with a strict 30-degree interval, ordered from right to left.
ANGLES_DEG = [15, 45, 75, 105, 135, 165]
COLORS = [
    (225, 70, 70),
    (238, 139, 67),
    (232, 185, 64),
    (76, 171, 131),
    (65, 145, 197),
    (132, 106, 215),
]


def list_images(input_dir: Path):
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name.lower(),
    )


def load_font(size: int):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def arrow_endpoint(origin, angle_deg, image_size, margin):
    width, height = image_size
    ox, oy = origin
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = -math.sin(rad)
    ground_top = height * 0.56

    limits = []
    if dx > 0:
        limits.append((width - margin - ox) / dx)
    elif dx < 0:
        limits.append((margin - ox) / dx)
    if dy < 0:
        limits.append((ground_top - oy) / dy)

    max_len = min(v for v in limits if v > 0)
    length = min(width * 0.26, height * 0.42, max_len * 0.92)
    return ox + dx * length, oy + dy * length


def draw_arrow(overlay: Image.Image, origin, end, color, width):
    draw = ImageDraw.Draw(overlay)
    ox, oy = origin
    ex, ey = end
    outline_width = width + max(3, width // 2)
    outline_color = (20, 24, 28, 185)
    arrow_color = (*color, 230)
    draw.line((ox, oy, ex, ey), fill=outline_color, width=outline_width)
    draw.line((ox, oy, ex, ey), fill=arrow_color, width=width)

    angle = math.atan2(ey - oy, ex - ox)
    head_len = width * 4.0
    head_width = width * 2.6
    back_x = ex - head_len * math.cos(angle)
    back_y = ey - head_len * math.sin(angle)
    perp_x = math.cos(angle + math.pi / 2)
    perp_y = math.sin(angle + math.pi / 2)
    head = [
        (ex, ey),
        (back_x + head_width * perp_x, back_y + head_width * perp_y),
        (back_x - head_width * perp_x, back_y - head_width * perp_y),
    ]
    outline_head = [
        (ex, ey),
        (back_x + (head_width + 3) * perp_x, back_y + (head_width + 3) * perp_y),
        (back_x - (head_width + 3) * perp_x, back_y - (head_width + 3) * perp_y),
    ]
    draw.polygon(outline_head, fill=outline_color)
    draw.polygon(head, fill=arrow_color)


def draw_label(overlay: Image.Image, origin, end, label, color, font):
    draw = ImageDraw.Draw(overlay)
    ox, oy = origin
    ex, ey = end
    lx = ox + (ex - ox) * 0.66
    ly = oy + (ey - oy) * 0.66
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    radius = max(tw, th) * 0.66 + 8
    draw.ellipse(
        (lx - radius, ly - radius, lx + radius, ly + radius),
        fill=(255, 255, 255, 232),
        outline=(24, 28, 32, 210),
        width=2,
    )
    draw.text((lx - tw / 2, ly - th / 2 - bbox[1] / 2), label, fill=(*color, 255), font=font)


def annotate_image(input_path: Path, output_dir: Path) -> Path:
    image = Image.open(input_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    margin = max(28, min(width, height) // 22)
    origin = (width / 2, height - margin * 0.72)
    line_width = max(4, min(width, height) // 95)
    font = load_font(max(20, min(width, height) // 22))

    band_height = max(80, int(height * 0.24))
    draw.rectangle((0, height - band_height, width, height), fill=(0, 0, 0, 34))

    origin_radius = max(7, line_width)
    draw.ellipse(
        (
            origin[0] - origin_radius,
            origin[1] - origin_radius,
            origin[0] + origin_radius,
            origin[1] + origin_radius,
        ),
        fill=(255, 255, 255, 235),
        outline=(24, 28, 32, 220),
        width=2,
    )

    for idx, (angle, color) in enumerate(zip(ANGLES_DEG, COLORS), start=1):
        end = arrow_endpoint(origin, angle, image.size, margin)
        draw_arrow(overlay, origin, end, color, line_width)
        draw_label(overlay, origin, end, str(idx), color, font)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_path_planning.png"
    annotated = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    annotated.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--all", action="store_true", help="Process all images instead of only the first one.")
    parser.add_argument("--limit", type=int, default=1, help="Number of sorted images to process when --all is not set.")
    args = parser.parse_args()

    images = list_images(args.input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    selected = images if args.all else images[: args.limit]
    outputs = [annotate_image(path, args.output_dir) for path in selected]

    print(f"processed={len(outputs)}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
