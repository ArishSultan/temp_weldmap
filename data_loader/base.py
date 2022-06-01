import os
import cv2
import uuid
from path import Path
from typing import List


class BaseDataLoader:
    def get_name(self):
        return None

    def get_processed_dir(self):
        return None

    def load(self):
        for item in self.get_raw_directories():
            self.transform_directory(item)
            print()

    def get_raw_directories(self):
        return []

    def transform_directory(self, path: Path):
        pass

    @staticmethod
    def get_processed_paths(path: Path):
        out_dir = Path('./data') / 'processed'

        image_dir = out_dir / 'img'
        label_dir = out_dir / 'gt'
        label_file = label_dir / 'words.txt'

        if not out_dir.exists():
            out_dir.mkdir()
        if not image_dir.exists():
            image_dir.mkdir()
        if not label_dir.exists():
            label_dir.mkdir()

        return out_dir.basename().upper(), image_dir, label_file


def load_dataset(loaders: List[BaseDataLoader], force=False):
    out_dir = Path('./data') / 'processed'
    if out_dir.exists():
        if len(out_dir.listdir()) > 0:
            if not force:
                return

            out_dir.rmtree()
            out_dir.mkdir()
    else:
        out_dir.mkdir()

    for loader in loaders:
        name = loader.get_name()

        if name is None:
            raise 'asd'

        print(f'Loading "{name}" Dataset')
        loader.load()
        print()
