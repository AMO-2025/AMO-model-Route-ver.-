import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import torch.nn.functional as F
from io import BytesIO

class EmotionPredictor:
    def __init__(self, model_path='./weight/efficient_dartmouth_final.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.val_transforms = self._get_transforms()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    def _load_model(self, model_path):
        model = efficientnet_b0(weights='DEFAULT')
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier[1].in_features, 7)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_from_path(self, image_path):
        """
        Predict emotion from an image file path
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = self.emotion_labels[pred_idx]
            confidence = probs[0][pred_idx].item()
        
        return pred_label, confidence

    def predict_from_bytes(self, image_data):
        """
        Predict emotion from image bytes
        """
        image = Image.open(BytesIO(image_data)).convert('RGB')
        input_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = self.emotion_labels[pred_idx]
            confidence = probs[0][pred_idx].item()
        
        return pred_label, confidence
