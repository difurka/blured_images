.PHONY: all train validate predict clean load_data get_data

# загрузка датасета и запуск обучения
all: get_data train

# запуск обучения
train:
	python3 train.py

# вывести roc_auc для валидационного датасета
validate:
	python3 validate.py

# создать и сохранить submission.csv
predict:
	python3 predict.py

# подбор лучших гиперпараметров с помощью W&B
sweep:
	python3 sweep.py

# удаление всех созданных во время обучения папок
clean:
	rm -rf data
	rm -rf outs
	rm -rf wandb
	rm -rf artifacts
	rm -rf src/__pycache__ configs/__pycache__ 

# загрузка датасета на сайт W&B
load_data:
	python3 src/load_data_to_wandb.py

# загрузка датасета с сайта W&B
get_data:
	python3 src/load_data_from_wandb.py
