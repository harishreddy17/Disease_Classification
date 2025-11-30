from src.predictor import Predictor

predictor = Predictor(model_path="resnet_onion.pth")

# Test with a sample image
image_path = "tests/sample_images/healthy1.jpg"  # Replace with your image path
image_tensor = predictor.preprocess(image_path)
class_name = predictor.predict(image_tensor)
print(f"Predicted class: {class_name}")
