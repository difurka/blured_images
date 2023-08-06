# EN

A pipeline for finding blurry images is presented. The Inception_V3 model is being trained.


Content:
- configs - configs for training
- notebooks - are a notebook .ipynb to run
- src - main scripts
- wandb - logs
- makefile - for quick launch:
     - make - download dataset from W&B and start training,
     - make train - train the model,
     - make validate - display roc_auc for the validation dataset,
     - make predict - create and save submission.csv,
     - make sweep - selection of the best hyperparameters using W&B
     - clean - delete all folders created during training: data, outs, wandb

Directories will appear:
- artefacts - files from the competition from Kaggle will be uploaded here. Download from https://wandb.ai
- outs - files resulting from training (best weights), results of the submission.csv model

View logs: https://wandb.ai/balakinakate2022/pipeline_competition/overview?workspace=user-balakinakate2022

Experiments were carried out with different values of learning rate and batch_size using sweep from WandB. Results can be seen at https://wandb.ai/balakinakate2022/pipeline_competition/runs/mf7wze6n?workspace=user-balakinakate2022

# RU

Представлен пайплаин обучения модели для нахождения размытых изображений. Дообучается модель Inception_V3.


Содержание:
- configs - конфиги для обучения
- notebooks - находятся ноутбук .ipynb для запуска
- src - основные скрипты
- wandb - логи
- makefile - для быстрого запуска: 
    - make - загрузить датасет с W&B и запустить обучение, 
    - make train - обучить модель,
    - make validate - вывести roc_auc для валидационного датасета, 
    - make predict - создать и сохранить submission.csv, 
    - make sweep - подбор лучших гиперпараметров с помощью W&B
    - clean - удаление всех созданных во время обучения папок: data, outs, wandb

Появятся директории:
- artefacts - сюда будут загружены файлы с соревнования из Kaggle. Загрузка из https://wandb.ai
- outs - файлы, получающиеся в результате обучения (лучшие веса), результаты работы модели submission.csv

Посмотреть логи: https://wandb.ai/balakinakate2022/pipeline_competition/overview?workspace=user-balakinakate2022

Проведены эксперименты с различными значениями learning rate и batch_size, используя sweep от WandB. Результаты можно видеть на https://wandb.ai/balakinakate2022/pipeline_competition/runs/mf7wze6n?workspace=user-balakinakate2022
