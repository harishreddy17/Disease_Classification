# tests/test_predictor.py
from src.predictor import Predictor
from config import load_config, BaseConfig

def test_predictor(image_path: str):
    # Load config dynamically
    config = load_config(model_name="resnet18")  # You can change model here

    # Initialize predictor with config
    predictor = Predictor(model_path=f"{config.model_dir}/resnet_onion.pth", config=config)

    # Preprocess the image
    image_tensor = predictor.preprocess(image_path)

    # Make prediction
    class_name = predictor.predict(image_tensor)
    print(f"Predicted class: {class_name}")


if __name__ == "__main__":
    sample_image = "tests/sample_images/healthy1.jpg"  # replace with actual image path
    test_predictor(sample_image)

