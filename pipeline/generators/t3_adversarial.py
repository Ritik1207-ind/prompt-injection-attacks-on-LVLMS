import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class AdversarialGenerator:
    def __init__(self, cfg, images_dir):
        t3 = cfg["attack_types"]["t3_adversarial"]
        self.clip_model_name = t3.get("clip_model_name", "openai/clip-vit-large-patch14")
        self.epsilon = t3.get("epsilon", 0.05)
        self.alpha = t3.get("alpha", 0.005)
        self.num_steps = t3.get("num_steps", 200)
        self.base_images_dir = Path(t3.get("base_images_dir", "./assets/benign_images"))
        self.out_dir = images_dir / "t3_adversarial"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load_clip(self):
        if self._model is None:
            print(f"[T3] Loading CLIP model: {self.clip_model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._model = CLIPModel.from_pretrained(self.clip_model_name, torch_dtype=torch.float16).to(self.device)
            self._model.eval()
            print(f"[T3] CLIP loaded on {self.device}")

    def _extract_tensor(self, out):
        """Safely extract a float tensor from either a raw tensor or model output object."""
        if isinstance(out, torch.Tensor):
            return out.float()
        # BaseModelOutputWithPooling or similar
        if hasattr(out, "image_embeds"):
            return out.image_embeds.float()
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output.float()
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :].float()
        raise ValueError(f"Cannot extract tensor from {type(out)}: {dir(out)}")

    def _get_target_embedding(self, instruction):
        inputs = self._processor(text=[instruction], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            feats = self._extract_tensor(out)
            return F.normalize(feats, dim=-1)

    def _load_random_benign(self):
        benign_files = list(self.base_images_dir.glob("*.jpg")) + list(self.base_images_dir.glob("*.png"))
        if not benign_files:
            arr = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            return Image.fromarray(arr)
        return Image.open(random.choice(benign_files)).convert("RGB")

    def pgd_attack(self, base_img, target_embedding):
        inputs = self._processor(images=base_img, return_tensors="pt").to(self.device)
        x_orig = inputs["pixel_values"].clone().detach().float()
        delta = torch.zeros_like(x_orig, requires_grad=False)

        for step in range(self.num_steps):
            delta.requires_grad_(True)
            x_adv = (x_orig + delta).half()
            raw = self._model.get_image_features(pixel_values=x_adv)
            img_features = F.normalize(self._extract_tensor(raw), dim=-1)
            target = F.normalize(target_embedding.float(), dim=-1)
            loss = -F.cosine_similarity(img_features, target).mean()
            loss.backward()
            with torch.no_grad():
                delta = delta + self.alpha * delta.grad.sign()
                delta = delta.clamp(-self.epsilon, self.epsilon)
                delta = ((x_orig + delta).clamp(0, 1) - x_orig).detach()
            if step % 50 == 0:
                print(f"  [PGD] step {step}/{self.num_steps} | loss={loss.item():.4f}")

        return (x_orig + delta).detach()

    def generate(self, instruction, sample_id):
        self._load_clip()
        base_img = self._load_random_benign()
        target_emb = self._get_target_embedding(instruction)
        with torch.enable_grad():
            adv_tensor = self.pgd_attack(base_img, target_emb)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
        img_t = (adv_tensor.cpu() * std + mean).clamp(0, 1)
        adv_img = Image.fromarray((img_t[0].permute(1,2,0).numpy()*255).astype(np.uint8))
        out_path = self.out_dir / f"{sample_id}.png"
        adv_img.save(out_path, "PNG")
        return out_path
