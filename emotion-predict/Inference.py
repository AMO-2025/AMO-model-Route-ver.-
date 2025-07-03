from emotion import EmotionPredictor

predictor = EmotionPredictor("/Users/chaewon/AMO-model-Route-ver.-/DeiT-Tiny_half_best.pth")
label, intensity, distribution = predictor.predict_from_path("/Users/chaewon/AMO-model-Route-ver.-/disgust.jpg")

print (label, intensity, distribution)