"""
generators/t1_typographic.py
─────────────────────────────
Type 1 — Typographic Injection
Renders the malicious instruction as typed text on a white background.
The vision encoder reads it as visual content; text-channel safety
filters never see it. Black-box, zero model access required.
"""

import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class TypographicGenerator:
    def __init__(self, cfg: dict, images_dir: Path):
        t1 = cfg["attack_types"]["t1_typographic"]
        self.font_path = t1.get("font_path")
        self.font_size = t1.get("font_size", 20)
        self.image_size = tuple(t1.get("image_size", [512, 512]))
        self.text_color = tuple(t1.get("text_color", [0, 0, 0]))
        self.bg_color = tuple(t1.get("bg_color", [255, 255, 255]))
        self.out_dir = images_dir / "t1_typographic"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        if self.font_path:
            return ImageFont.truetype(self.font_path, size)
        # Try common system fonts; fall back to PIL default
        fallbacks = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        for path in fallbacks:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def generate(self, instruction: str, sample_id: str) -> Path:
        """
        Render `instruction` as black text on white background.
        Returns path to saved image.
        """
        img = Image.new("RGB", self.image_size, color=self.bg_color)
        draw = ImageDraw.Draw(img)
        font = self._load_font(self.font_size)

        # Wrap text to fit image width (with padding)
        max_chars_per_line = max(10, self.image_size[0] // (self.font_size // 2))
        lines = []
        for paragraph in instruction.split("\n"):
            wrapped = textwrap.wrap(paragraph, width=max_chars_per_line)
            lines.extend(wrapped if wrapped else [""])

        # Calculate total text block height
        line_height = self.font_size + 6
        total_height = len(lines) * line_height
        margin_x = 20
        margin_y = max(20, (self.image_size[1] - total_height) // 2)

        for i, line in enumerate(lines):
            y = margin_y + i * line_height
            draw.text((margin_x, y), line, fill=self.text_color, font=font)

        out_path = self.out_dir / f"{sample_id}.png"
        img.save(out_path, "PNG")
        return out_path
