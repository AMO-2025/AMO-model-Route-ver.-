from emotion import EmotionPredictor

predictor = EmotionPredictor("/Users/chaewon/AMO-model-Route-ver.-/weight/ConvNeXt-Tiny_best.pth") # used model: convnext-tiny 
label, intensity, distribution = predictor.predict_from_path("/Users/chaewon/AMO-model-Route-ver.-/image-file/smile.png") # put image path ! 

print (label, intensity, distribution) 