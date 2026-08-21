"""
generators/t4_metadata.py
──────────────────────────
Type 4 — Indirect Metadata Injection
Embeds the malicious instruction in EXIF metadata fields of a
completely benign-looking image. Only effective when the LVLM
pipeline reads image metadata — agent deployments, RAG systems,
automated image-processing pipelines.
Zero stealthiness cost since the image itself is unmodified.
"""

import io
import json
import random
from pathlib import Path

import piexif
from PIL import Image


EXIF_FIELDS_MAP = {
    # field_name: (IFD_key, tag_id, encoding)
    "UserComment": ("Exif", piexif.ExifIFD.UserComment, "utf-8"),
    "ImageDescription": ("0th", piexif.ImageIFD.ImageDescription, "ascii"),
    "XPComment": ("0th", piexif.ImageIFD.XPComment, "utf-16-le"),
}

BENIGN_DESCRIPTIONS = [
    "Family vacation photo, summer 2024",
    "Product catalog image for internal use",
    "Conference presentation slide screenshot",
    "Dashboard screenshot - Q3 analytics review",
    "Team offsite group photo - do not distribute",
    "System architecture diagram for documentation",
    "Screenshot for bug report #4821",
    "Marketing asset — approved for social media",
]


class MetadataGenerator:
    def __init__(self, cfg: dict, images_dir: Path):
        t4 = cfg["attack_types"]["t4_metadata"]
        self.exif_fields = t4.get("exif_fields", ["UserComment", "ImageDescription"])
        self.base_images_dir = Path(t4.get("base_images_dir", "./assets/benign_images"))
        self.out_dir = images_dir / "t4_metadata"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_random_benign(self) -> Image.Image:
        benign_files = (
            list(self.base_images_dir.glob("*.jpg")) +
            list(self.base_images_dir.glob("*.jpeg")) +
            list(self.base_images_dir.glob("*.png"))
        )
        if not benign_files:
            # Fallback: generate a plain coloured image
            color = tuple(random.randint(80, 200) for _ in range(3))
            img = Image.new("RGB", (512, 512), color=color)
            return img
        return Image.open(random.choice(benign_files)).convert("RGB")

    def _encode_field(self, field_name: str, text: str) -> bytes:
        """Encode text correctly for the specific EXIF field type."""
        encoding = EXIF_FIELDS_MAP[field_name][2]
        if field_name == "UserComment":
            # UserComment requires a character code prefix (ASCII: b'ASCII\x00\x00\x00')
            return b"ASCII\x00\x00\x00" + text.encode("ascii", errors="replace")
        return text.encode(encoding, errors="replace")

    def generate(self, instruction: str, sample_id: str) -> tuple[Path, dict]:
        """
        Embed `instruction` into EXIF metadata of a benign image.
        Returns (image_path, metadata_fields_dict).
        """
        base_img = self._load_random_benign()

        # Build EXIF dict
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        # Always add a benign ImageDescription to look normal
        benign_desc = random.choice(BENIGN_DESCRIPTIONS)
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = benign_desc.encode("ascii")

        # Pick which fields to inject into (random subset of configured fields)
        fields_to_inject = random.sample(
            self.exif_fields, k=random.randint(1, len(self.exif_fields))
        )
        metadata_record = {}

        for field_name in fields_to_inject:
            if field_name not in EXIF_FIELDS_MAP:
                continue
            ifd_key, tag_id, _ = EXIF_FIELDS_MAP[field_name]
            encoded = self._encode_field(field_name, instruction)
            exif_dict[ifd_key][tag_id] = encoded
            metadata_record[field_name] = instruction

        # Save image with EXIF
        out_path = self.out_dir / f"{sample_id}.jpg"
        exif_bytes = piexif.dump(exif_dict)

        img_bytes = io.BytesIO()
        base_img.save(img_bytes, format="JPEG", exif=exif_bytes, quality=95)
        out_path.write_bytes(img_bytes.getvalue())

        return out_path, metadata_record
