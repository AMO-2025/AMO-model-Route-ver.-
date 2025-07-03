import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from io import BytesIO
import timm

class EmotionPredictor:
    def __init__(self, model_path='/Users/chaewon/AMO-model-Route-ver.-/DeiT-Tiny_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=len(self.emotion_labels))
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

    def _predict(self, image: Image.Image):
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
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