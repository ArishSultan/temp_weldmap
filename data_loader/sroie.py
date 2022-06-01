import numpy as np

from .base import *
from PIL import Image


class SROIEDataLoader(BaseDataLoader):
    def get_name(self):
        return "SROIED"

    def get_processed_dir(self):
        return Path('./data') / 'processed'

    def get_raw_directories(self):
        return (Path('./data') / 'raw' / 'sroie').listdir()

    def transform_directory(self, path: Path):
        images_dir = path / 'img'
        labels_dir = path / 'box'

        labels = labels_dir.listdir()
        dir_name, out_image, out_label = BaseDataLoader.get_processed_paths(path)
        label_file = open(out_label, 'a', encoding='utf-8')

        for i in range(len(labels)):
            label = labels[i]
            boxes = SROIEDataLoader.__read_txt(label)

            processed_labels = []
            for j in range(len(boxes)):
                print(f'\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r', end='')
                print(f'[{dir_name}] '
                      f'IMAGE: {i + 1}/{len(labels)}, '
                      f'BOX: {j + 1}/{len(boxes)}', end='')

                text = SROIEDataLoader.__handle_entry(
                    boxes[j],
                    out_image,
                    images_dir / label.basename().replace('.txt', '.jpg')
                )

                processed_labels.append(text + '\n')
            label_file.writelines(processed_labels)

    @staticmethod
    def __handle_entry(entry, out: Path, image_path: Path):
        x, y, w, h, text = entry

        # Fix some erroneous headers
        image_pil = Image.open(image_path).convert('RGB')
        image = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2GRAY)[y:h, x:w]

        if len(image) > 0:
            filename = str(uuid.uuid4()) + '.png'
            cv2.imwrite(out / filename, image)
            print(out)
            print(out.exists())
            exit(0)

        return filename + '|' + text

    @staticmethod
    def __read_txt(path: Path):
        boxes = []
        with open(path, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(',')
                if not len(parts) > 1:
                    continue

                boxes.append((int(parts[0]), int(parts[1]), int(parts[4]),
                              int(parts[5]), ' '.join(parts[8:])))
        return boxes
