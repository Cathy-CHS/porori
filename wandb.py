import wandb

run = wandb.init()
artifact = run.use_artifact('miiin/KingKorre/model-49dcf2e4:v2', type='model')
artifact_dir = artifact.download()