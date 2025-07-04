import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from io import BytesIO
import timm

class EmotionPredictor:
    def __init__(self, model_path='/Users/chaewon/AMO-model-Route-ver.-/weight/ConvNeXt-Tiny_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # get gpu or cpu 
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised'] # emotion class based on Ekman study 
        self.model = self._load_model(model_path) # load model 
        self.transform = self._get_transform() # get transform (image resize, ..)

    def _load_model(self, model_path): #loading model 
        model = timm.create_model('convnext_tiny_in22ft1k', pretrained=False, num_classes=len(self.emotion_labels)) # convnext-tiny model, no pretrain (we have our own parameters), classes = 7 (based on Ekman study)
        state_dict = torch.load(model_path, map_location=self.device) # load model weight in model_path
        model.load_state_dict(state_dict)
        model = model.to(self.device) # move model to gpu / cpu 
        model.eval() # make model to eval mode 
        return model

    def _get_transform(self): # transform image to tensor 
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _predict(self, image: Image.Image): # predict emotion from image 
        input_tensor = self.transform(image).unsqueeze(0).to(self.device) # transform image to tensor and add batch dimension 
        with torch.no_grad(): # no need to calculate gradient 
            output = self.model(input_tensor) 
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
        
        pred_idx = probs.argmax() # get index of max probability 
        pred_label = self.emotion_labels[pred_idx] # get emotion label from index 
        pred_confidence = probs[pred_idx] # get confidence from probability 
        prob_dict = {label: float(prob) for label, prob in zip(self.emotion_labels, probs)} # get probability dictionary 

        return pred_label, pred_confidence, prob_dict # return emotion label, confidence, probability dictionary 

    def predict_from_path(self, image_path: str): # predict emotion from image path 
        image = Image.open(image_path).convert('RGB') # open image 
        return self._predict(image)

    def predict_from_bytes(self, image_bytes: bytes): # predict emotion from image bytes 
        image = Image.open(BytesIO(image_bytes)).convert('RGB') # open image from bytes 
        return self._predict(image) # return emotion label, confidence, probability dictionary 