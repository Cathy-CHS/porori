from src.relationship.korre import KingKorre
import wandb

if __name__ == "__main__":
    
    run = wandb.init()
    artifact = run.use_artifact('miiin/KingKorre/model-c94p3yja:v20', type='model')
    artifact_dir = artifact.download()
