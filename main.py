import uuid
import editdistance
from htr.src.main import *
from htr.src.model import Model

# 4863 is of the original

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'gen_report', 'infer'], default='infer')
    parser.add_argument('--line_mode', help='Line Mode.', type=bool, default=True)
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing dataset.', type=Path, required=False)
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size)

        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, DecoderType.BestPath)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), DecoderType.BestPath, must_restore=True)
        infer(model, args.img_file)
    elif args.mode == 'gen_report':
        model = Model(char_list_from_file(), DecoderType.BestPath, must_restore=True)

        # Load all the images and their
        im_labels = []
        with open('./data/weldmap/gt/labels.txt') as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    parts = line.split('|')
                    im_labels.append((parts[0], parts[1]))

        distance = 0
        results = []
        print('Generating report')
        for sample in im_labels:
            item = dict()
            text, conf = infer(model, Path(sample[0]))

            dist = editdistance.eval(text, sample[1])
            item['conf'] = float(conf)
            item['pred'] = text
            item['orig'] = sample[1]
            item['diff'] = dist
            distance += dist

            results.append(item)

        with open(f'./report_{uuid.uuid4()}.json', 'w') as file:
            json.dump({
                'distance': distance,
                'results': results
            }, file)

    else:
        raise 'Unknown `mode` specified'


main()
