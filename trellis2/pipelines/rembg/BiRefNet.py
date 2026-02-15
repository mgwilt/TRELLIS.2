from typing import *
import os
import warnings
from transformers import AutoConfig, AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download
import torch
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
from ...utils.hf_local import resolve_hf_path


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        model_source = resolve_hf_path(model_name)
        self.model = self._load_model_without_meta_init(model_source)
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _resolve_weight_path(self, model_source: str) -> str:
        if os.path.isdir(model_source):
            safetensors_path = os.path.join(model_source, "model.safetensors")
            pytorch_path = os.path.join(model_source, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                return safetensors_path
            if os.path.exists(pytorch_path):
                return pytorch_path
            raise FileNotFoundError(
                f"No weight file found in {model_source}. Expected model.safetensors or pytorch_model.bin."
            )

        try:
            return hf_hub_download(model_source, "model.safetensors")
        except Exception:
            return hf_hub_download(model_source, "pytorch_model.bin")

    def _load_state_dict(self, weight_path: str) -> dict:
        if weight_path.endswith(".safetensors"):
            return load_file(weight_path)
        try:
            return torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(weight_path, map_location="cpu")

    def _load_model_without_meta_init(self, model_source: str):
        """
        Work around transformers>=5 meta-device init for custom remote-code models
        that call Tensor.item() during __init__.
        """
        try:
            config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
            model = AutoModelForImageSegmentation.from_config(config, trust_remote_code=True)
            weight_path = self._resolve_weight_path(model_source)
            state_dict = self._load_state_dict(weight_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"BiRefNet loaded with missing={len(missing)} unexpected={len(unexpected)} keys.",
                    RuntimeWarning,
                )
            return model
        except Exception:
            # Fallback for older/newer transformers behavior.
            return AutoModelForImageSegmentation.from_pretrained(model_source, trust_remote_code=True)
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
