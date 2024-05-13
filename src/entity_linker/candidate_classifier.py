# KLUE/ROBERRTA-small
# KLUE/ROBERRTA-base
# KLUE/ROBERRTA-large

from transformers import 

class CandidateClassifier:
    def __init__(
        self,
        backbone: str,
        model_path: str = None,
        pretrained: bool = True,
    ):
        self.backbone = backbone
        self.model_path = model_path
        self.pretrained = pretrained

        self.model = None

        if not backbone in [
            "KLUE/ROBERTA-small",
            "KLUE/ROBERTA-base",
            "KLUE/ROBERTA-large",
        ]:
            raise ValueError("Invalid backbone model")
        
        if model_path:


    def load_model_from_local(self, path=None):
        pass

    def load_model_from_huggingface(self, model_name):
        pass

    def predict(self, entity, candidate):
        return self.model.predict(candidate)

    def train(self, dataloader, args):
        pass

    def eval(self, dataloader):
        pass

    def save_model(self, path):
        pass
