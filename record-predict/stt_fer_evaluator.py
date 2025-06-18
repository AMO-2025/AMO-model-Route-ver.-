import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile
import whisper
from groq import Groq
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True

class STTFEREvaluator:
    def __init__(self, 
                 emotion_model_path='./weight/efficient_fer2013_pretrained.pth',
                 groq_api_key=None,
                 device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_model = whisper.load_model("base")
        self._init_emotion_model(emotion_model_path)
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    def _init_emotion_model(self, model_path):
        self.emotion_model = models.efficientnet_b0(pretrained=True)
        self.emotion_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.emotion_model.classifier[1].in_features, 7)
        )
        
        if os.path.exists(model_path):
            self.emotion_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        
        self.emotion_model = self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def transcribe_audio(self, audio_path):
        """
        audio2text
        """ 
        result = self.whisper_model.transcribe(audio_path, language="ko")
        return result["text"]

    def fer_emotion(self, image_path):
        """
        fer
        
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.emotion_model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = self.emotion_labels[pred_idx]
            confidence = probs[0][pred_idx].item()
            
        return pred_label, confidence

    def fer_from_bytes(self, image_data):
        image = Image.open(BytesIO(image_data)).convert('RGB')
        input_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.emotion_model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = self.emotion_labels[pred_idx]
            confidence = probs[0][pred_idx].item()
            
        return pred_label, confidence

    def evaluate_emotional_alignment(self, transcribed_text, image_emotion, scenario):
        """
        stt-fer-eval alignment 
        """
  
        prompt = f"""
You are an evaluator of emotional appropriateness in context.

Scenario:
"{scenario}"

User said (via speech): "{transcribed_text}"
Their facial expression was: "{image_emotion}"

Step 1: Identify the emotion implied in the speech. Use Ekman's 6 basic emotions: happy, sad, angry, fear, surprise, disgust.

Step 2: Evaluate the emotional alignment between the speech and facial expression using:
1. Emotion Label Match (same emotion)
2. Valence Match (positive/negative)
3. Arousal Match (low/medium/high activation)
4. Contextual Fit (appropriate emotion for the scenario)

Score each from 0 to 25. Total is 100.

Only return:
LabelScore, ValenceScore, ArousalScore, ContextScore, TotalScore, Reason (one sentence).

Important instructions:
- Only output raw scores and a short reason.
- DO NOT restate the inputs.
- Format:
LabelScore: [0-25]
ValenceScore: [0-25]
ArousalScore: [0-25]
ContextScore: [0-25]
TotalScore: [0-100]
Reason: [short sentence]
"""

        response = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def evaluate_complete(self, audio_path, image_path, scenario): 
        """
        stt-fer-eval process  
        """
        
        transcribed_text = self.transcribe_audio(audio_path)
        image_emotion, confidence = self.predict_emotion(image_path)
        alignment_evaluation = self.evaluate_emotional_alignment(transcribed_text, image_emotion, scenario)        
        return {
            "transcribed_text": transcribed_text,
            "image_emotion": image_emotion,
            "emotion_confidence": confidence,
            "alignment_evaluation": alignment_evaluation
        }

    def set_groq_api_key(self, api_key):
        """Groq API key"""
        self.groq_client = Groq(api_key=api_key) 