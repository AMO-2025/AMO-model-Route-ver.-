import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from io import BytesIO
import timm
import face_recognition
import numpy as np
import os
import requests
from urllib.parse import urlparse

class EmotionPredictor:
    def __init__(self, model_path, use_adapter=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.use_adapter = use_adapter
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        model = timm.create_model('convnext_tiny_in22ft1k', pretrained=False, num_classes=len(self.emotion_labels))
        if self.use_adapter:
            self._inject_adapter(model)

        if model_path.startswith("http://") or model_path.startswith("https://"):
            response = requests.get(model_path)
            response.raise_for_status()  
            state_dict = torch.load(BytesIO(response.content), map_location=self.device)
        else:
            state_dict = torch.load(model_path, map_location=self.device)

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _crop_face(self, image: Image.Image, margin=20):
        img_np = np.array(image)
        face_locations = face_recognition.face_locations(img_np)

        if not face_locations:
            return None

        top, right, bottom, left = face_locations[0]
        h, w = img_np.shape[:2]
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(h, bottom + margin)
        right = min(w, right + margin)

        cropped = img_np[top:bottom, left:right]
        
        return Image.fromarray(cropped)
    
    def _predict(self, image: Image.Image):
        face = self._crop_face(image)
        if face is None:
            face = image
        input_tensor = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()
        pred_label = self.emotion_labels[pred_idx]
        pred_confidence = probs[pred_idx]
        prob_dict = {label: float(prob) for label, prob in zip(self.emotion_labels, probs)}
        return pred_label, pred_confidence, prob_dict

    def predict_from_path(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        return self._predict(image)

    def predict_from_bytes(self, image_bytes: bytes):
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return self._predict(image)

    class _Adapter(nn.Module):
        def __init__(self, dim, reduction=4):
            super().__init__()
            self.down = nn.Conv2d(dim, dim // reduction, kernel_size=1)
            self.act = nn.ReLU()
            self.up = nn.Conv2d(dim // reduction, dim, kernel_size=1)

        def forward(self, x):
            return self.up(self.act(self.down(x)))

    def _inject_adapter(self, model, reduction=4):
        for name, module in model.named_modules():
            if module.__class__.__name__ == "ConvNeXtBlock":
                dim = module.conv_dw.out_channels
                adapter = self._Adapter(dim, reduction=reduction).to(self.device)
                for param in adapter.parameters():
                    param.requires_grad = True
                module.adapter = adapter
                orig_forward = module.forward

                def new_forward(x, orig_forward=orig_forward, adapter=adapter):
                    return orig_forward(x) + adapter(x)

                module.forward = new_forward
