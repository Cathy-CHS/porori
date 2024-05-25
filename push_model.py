from src.relationship.korre import KingKorre
import wandb

if __name__ == "__main__":
    import wandb

    run = wandb.init()
    artifact = run.use_artifact("miiin/KingKorre/model-vbtvnvxf:v22", type="model")
    artifact_dir = artifact.download()
