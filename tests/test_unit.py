from src.predictor import Predictor

def test_predictor_load():
    predictor = Predictor(model_path="resnet_onion.pth")
    assert predictor.model_wrapper.model is not None
