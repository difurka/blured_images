"""Load datasets with transforms."""


from pathlib import Path, PosixPath
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from src.load_data_from_wandb import load_from_WB
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# режимы датасета 
DATA_MODES = ['train', 'val', 'test']
# все изображения масштабируем к размеру 299*299 px
RESCALE_SIZE = 299

class CastomDataset(Dataset):
    """
    Датасет картинок, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files: np.array,  mode: str, data_labels: pd.core.frame.DataFrame = None, 
                 transform: transforms.Compose=None):
        """
        Конструктор датасета.

        Args:
            files (np.array): список путей до изображений
            mode (str): тип датасета из ['train', 'val', 'test']
            data_labels (pd.core.frame.DataFrame, optional): _description_. Defaults to None.
            transform (transforms.Compose, optional): преобразования датасета

        Raises:
            NameError: возникает в случае неправильного типа датамета
        """
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode
        self.transform = transform
        self.data_labels = data_labels

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError
        self.len_ = len(self.files)

        # загружем метки файлов
        if self.mode != 'test':
            self.labels = torch.tensor([np.array(self.data_labels[self.data_labels.iloc[:, 0] == path.name].iloc[:,1])[0] \
                        for path in self.files],dtype=torch.long)  
                        
    def __len__(self) -> int:
        """
        Количество элементов в датасете.

        Returns:
            int: количество элементов 
        """
        return self.len_
    
    def load_sample(self, file: PosixPath) -> Image.Image:
        """
        Загружает изображение, находящееся по пути file.

        Args:
            file (PosixPath): путь до изображения

        Returns:
            Image.Image: изображение
        """
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index: int) -> Tuple[np.array, int]:
        """
        Возвращает элемент датасета.

        Args:
            index (int): индекс элемента датасета

        Returns:
            Tuple[np.array, int]: изображение и размыто/неразмыто
        """
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        if self.transform:
            x = self.transform(x)
        if self.mode == 'test':
            return x
        else:
            y = self.labels[index]
            return x, y
        
    def _prepare_sample(self, image: Image.Image) -> np.array:
        """
        Уменьшение размера изображения.

        Args:
            image (Image.Image): входящее изображение

        Returns:
            np.array: уменьшенное изображение
        """
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)
    

def get_datasets() -> Tuple[CastomDataset, CastomDataset, CastomDataset]:
    """
    Create datasets for train, validate, predict.

    Returns:
        Tuple[CastomDataset, CastomDataset, CastomDataset]: train_dataset, val_dataset, test_dataset
    """
    DIR = './data/'
    TRAIN_DIR = Path(DIR + 'train/train')
    TEST_DIR = Path(DIR + 'test/test')

    train_val_files = list(TRAIN_DIR.rglob('*.jpg'))
    test_files = list(TEST_DIR.rglob('*.jpg'))

    data_labels = pd.read_csv(DIR + 'train.csv')
    data_labels[['blur']] = data_labels[['blur']].astype('long')

    transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    train_val_files = list(TRAIN_DIR.rglob('*.jpg'))
    test_files = list(TEST_DIR.rglob('*.jpg'))

    train_val_labels = [np.array(data_labels[data_labels.iloc[:, 0] == path.name].iloc[:,1])[0] for path in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

    train_dataset = CastomDataset(train_files, data_labels=data_labels, mode='train', transform=transform_train)
    val_dataset = CastomDataset(val_files, data_labels=data_labels, mode='val', transform=transform_test)
    test_dataset = CastomDataset(test_files, mode='test', transform=transform_test)

    return train_dataset, val_dataset, test_dataset
