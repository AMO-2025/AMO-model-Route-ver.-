from emotion import EmotionPredictor
import requests

url = "https://amo-model.s3.ap-northeast-2.amazonaws.com/weight.pth"
picture = "/Users/chaewon/AMO-model-Route-ver.-/image-file/surprised.png" # from front-end

predictor = EmotionPredictor(
    model_path=url,
    use_adapter=True
)

label, intensity, distribution = predictor.predict_from_path(picture)
print(label, intensity, distribution)