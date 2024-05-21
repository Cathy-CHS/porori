# KLUE/ROBERRTA-small
# KLUE/ROBERRTA-base
# KLUE/ROBERRTA-large

import lightning as L
import transformers

roberta_base_config = transformers.RobertaConfig.from_pretrained("KLUE/ROBERTA-base")


class CandidateClassifier(L.LightningModule):
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
            self.load_model_from_local(model_path)
        elif pretrained:
            self.load_model_from_huggingface(backbone)
        else:
            self.load_model_from_huggingface(backbone)
            # TODO: reset model weights

    def load_model_from_local(self, path=None):
        pass

    def load_model_from_huggingface(self, model_name) -> transformers.PreTrainedModel:
        pass

    def predict(self, entity, candidate):
        return self.model.predict(candidate)

    def train(self, dataloader, args):
        pass

    def eval(self, dataloader):
        pass

    def save_model(self, path):
        pass
