import json
from .base import *


class ScannedDataLoader(BaseDataLoader):
    def get_name(self):
        return "Scanned"

    def get_raw_directories(self):
        return (Path('./data') / 'raw' / 'scanned').listdir()

    def transform_directory(self, path: Path):
        images_dir = path / 'images'
        labels_dir = path / 'annotations'

        labels = labels_dir.listdir()
        dir_name, out_image, out_label = BaseDataLoader.get_processed_paths(path)

        for i in range(len(labels)):
            label = labels[i]
            boxes = ScannedDataLoader.__read_json(label)['form']

            processed_labels = []
            for j in range(len(boxes)):
                print(f'\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r', end='')
                print(f'[{dir_name}] '
                      f'IMAGE: {i + 1}/{len(labels)}, '
                      f'BOX: {j + 1}/{len(boxes)}', end='')

                text = ScannedDataLoader.__handle_entry(
                    boxes[j],
                    out_image,
                    images_dir / label.basename().replace('.json', '.png')
                )

                processed_labels.append(text)

            text = '\n'.join(processed_labels)
            with open(out_label, 'a', encoding='utf-8') as file:
                file.write(text)

    @staticmethod
    def __handle_entry(entry, out: Path, image_path: Path):
        text = entry["text"]
        x, y, w, h = entry["box"]

        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)[y:h, x:w]
        filename = str(uuid.uuid4()) + '.png'
        cv2.imwrite(out / filename, image)
        print(out / filename)

        return filename + '|' + text

    @staticmethod
    def __read_json(path: Path):
        with open(path, 'r', encoding="utf-8") as file:
            return json.load(file)
