import wandb

def load_to_WB():
    """ 
    Загрузить датасет на W&B.
     
    Загружаем shift-cv-winter-2023.zip на сайт, используем эту функцию только один раз.
    """
    DIR = './data/'

    run = wandb.init(project="pipeline_competition", job_type="dataset")
    dataset = wandb.Artifact('my-dataset', type='dataset')
    dataset.add_file(DIR + 'shift-cv-winter-2023.zip')
    # Log the artifact to save it as an output of this run 
    run.log_artifact(dataset)
    wandb.finish()


if __name__ == '__main__':
    load_to_WB()