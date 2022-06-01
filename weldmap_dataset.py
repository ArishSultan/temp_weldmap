import os
import time

import cv2
import fitz
import mimetypes
import numpy as np

from PIL import Image
from path import Path

PAGE_ZOOM = 3
VERTICAL_LINE_MORPH_KERNEL = (1, 10)
HORIZONTAL_LINE_MORPH_KERNEL = (10, 2)

RAW_DATA = Path('./data/raw/')
PROCESSED_DATA = Path('./data/processed/')
INTERMEDIATE_DATA = Path('./data/intermediate/')

INPUT_FILES = [RAW_DATA / 'Monsanto.pdf']


def extract_page_image(path: Path, page_number=None, zoom_factor=PAGE_ZOOM):
    if not path.exists():
        raise "Provided file %s does not exist" % path

    if not mimetypes.guess_type(path)[0] == "application/pdf":
        raise "Provided file %s is not a valid PDF" % path

    document = fitz.Document(str(path))

    if page_number is not None:
        pages = document.pages(page_number - 1, page_number)
    else:
        pages = document.pages()

    for page in pages:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
        np_arr = np.frombuffer(pixmap.samples, dtype=np.uint8) \
            .reshape(pixmap.h, pixmap.w, pixmap.n)

        yield np.ascontiguousarray(np_arr[..., [2, 1, 0]])


def find_lines(morphed_img):
    result = cv2.findContours(morphed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sometimes the [cv.findContours] returns a tuple of either len 3 or len 2
    # the following line fixes this by extracting only the tuple of `contours`.
    return result[0] if len(result) == 2 else result[1]


def morph_image(img, kernel_shape):
    # A kernel of shape [cv.MORPH_RECT] is used since it is the most suitable
    # option for detecting lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)

    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)


def filter_contours(contours):
    vertical = []
    horizontal = []

    x_list = []
    y_list = []
    w_list = []
    h_list = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x_list.append(x)
        y_list.append(y)
        w_list.append(w)
        h_list.append(h)

    for i in range(6):
        op = (h_list, vertical) if i % 2 == 0 else (w_list, horizontal)
        index = np.argmax(op[0])

        op[1].append((x_list[index], y_list[index],
                      x_list[index] + w_list[index],
                      y_list[index] + h_list[index]))

        del x_list[index]
        del y_list[index]
        del w_list[index]
        del h_list[index]

    return vertical, horizontal


def rgb_to_binary(image, return_grey=False):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        grey,
        0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    return (thresh, grey) if return_grey else thresh


def prepare_intermediate_states(sample_name):
    root = INTERMEDIATE_DATA / sample_name

    table = root / 'table'
    table_lines = root / 'lines'

    morphed = root / 'morphed'

    contoured = root / 'contoured'
    contoured_raw = contoured / 'raw'
    contoured_filtered = contoured / 'filtered'

    root.mkdir()
    table.mkdir()
    morphed.mkdir()
    contoured.mkdir()
    table_lines.mkdir()
    contoured_raw.mkdir()
    contoured_filtered.mkdir()

    return root, morphed, contoured_raw, contoured_filtered, table, table_lines


def remove_colored_content(image):
    # optimize performance of content below
    w, h, _ = image.shape

    tb = image[:, :, 0] + 10
    tg = image[:, :, 1] + 10
    tr = image[:, :, 2] + 10
    for i in range(w):
        for j in range(h):
            if image[i, j, 0] > tr[i, j] and image[i, j, 0] > tg[i, j] or \
                    image[i, j, 1] > tb[i, j] and image[i, j, 1] > tr[i, j] or \
                    image[i, j, 2] > tb[i, j] and image[i, j, 2] > tg[i, j]:
                image[i, j, :] = 255


def process_image(image, save_states=False):
    sample_name = f'smp_{time.time()}'

    if save_states:
        m_d, mo_d, cr_d, cf_d, t_d, tl_d = prepare_intermediate_states(sample_name)
        cv2.imwrite(m_d / 'original.png', image)

    # Converts image to binary using OTSU's adaptive threshold
    b_img = rgb_to_binary(image)

    if save_states:
        cv2.imwrite(m_d / 'binary.png', b_img)

    # Detect vertical & horizontal lines using morphology
    v_morph = morph_image(b_img, VERTICAL_LINE_MORPH_KERNEL)
    h_morph = morph_image(b_img, HORIZONTAL_LINE_MORPH_KERNEL)

    if save_states:
        cv2.imwrite(mo_d / 'v_morph.png', v_morph)
        cv2.imwrite(mo_d / 'h_morph.png', h_morph)

    # Detect vertical & horizontal contours using morphed images
    v_contours = find_lines(v_morph)
    h_contours = find_lines(h_morph)

    if save_states:
        temp = image.copy()
        cv2.drawContours(temp, v_contours, -1, (0, 0, 255), 5)

        cv2.imwrite(cr_d / 'v_lines.png', temp)

        temp = image.copy()
        cv2.drawContours(temp, h_contours, -1, (0, 0, 255), 5)
        cv2.imwrite(cr_d / 'h_lines.png', temp)

        cv2.drawContours(temp, v_contours, -1, (0, 0, 255), 5)
        cv2.imwrite(cr_d / 'lines.png', temp)

    # Filter out the most valuable contours
    vertical, horizontal = filter_contours(v_contours + h_contours)

    if save_states:
        temp1 = image.copy()
        temp2 = image.copy()
        temp3 = image.copy()

        for rect in vertical:
            cv2.rectangle(temp1, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 5)

        for rect in horizontal:
            cv2.rectangle(temp2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 5)

        for rect in vertical + horizontal:
            cv2.rectangle(temp3, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 5)

        cv2.imwrite(cf_d / 'lines.png', temp3)
        cv2.imwrite(cf_d / 'v_lines.png', temp1)
        cv2.imwrite(cf_d / 'h_lines.png', temp2)

    # Sort horizontals based on height
    vertical = sorted(vertical, key=lambda item: item[0])
    horizontal = sorted(horizontal, key=lambda item: item[1])

    # Crop the required table portion from image
    start_x = min(vertical[1][0], vertical[1][2])
    end_x = min(vertical[2][0], vertical[2][2])

    start_y = max(horizontal[0][1], horizontal[0][3])
    end_y = min(horizontal[1][1], horizontal[1][3])

    cv2.drawContours(image, v_contours + h_contours, -1, (255, 255, 255), 3)

    image = image[start_y: end_y, start_x: end_x]

    if save_states:
        cv2.imwrite(t_d / 'table.png', image)

    # Keep only the grey content and remove all the high intensity R, G, B content
    remove_colored_content(image)

    if save_states:
        cv2.imwrite(t_d / 'table_clean.png', image)

    result_path = PROCESSED_DATA / sample_name
    os.mkdir(result_path)

    index = 1
    for line in _detect_lines(image):
        pil_image = _pad_line(_crop_line(image, line))
        img, g_img = rgb_to_binary(np.array(pil_image.convert('RGB')), True)

        morphed = morph_image(img, (1, 3))
        count = np.sum(morphed > 0)

        if save_states:
            pil_image.save(tl_d / f'{index}.png')
            cv2.imwrite(tl_d / 'f{index}_morphed_f{count}.png', morphed)

        if count > 0:
            # last_zero_x = None
            # last_zero_y = None
            # first_zero_x = None
            # first_zero_y = None
            #
            # for i in range(g_img.shape[1]):
            #     grey_pixel = min(g_img[:, i])
            #
            #     if grey_pixel < 127:
            #         if first_zero_y is None:
            #             first_zero_y = i
            #         last_zero_y = i
            #
            # for i in range(g_img.shape[0]):
            #     grey_pixel = min(g_img[i, :])
            #
            #     if grey_pixel < 127:
            #         if first_zero_x is None:
            #             first_zero_x = i
            #         last_zero_x = i
            #
            # g_img = g_img[first_zero_x:last_zero_x + 1, first_zero_y:last_zero_y + 1]

            cv2.imwrite(result_path / f'{index}.png', img)
            index += 1


def _detect_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[0], 1))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    boxes = []
    for c in cnts:
        boxes.append(cv2.boundingRect(c))

    return sorted(boxes, key=lambda x: x[1])


def _crop_line(img, contours: (int, int, int, int)):
    x, y, w, h = contours
    return img[y:y + h, x:x + w]


def _pad_line(input_image):
    size = input_image.shape
    new_size = size[1], size[0] + 20

    old_image = Image.fromarray(input_image)
    new_image = Image.new("L", new_size, 255)
    new_image.paste(old_image, (
        0,
        new_size[1] // 2 - size[0] // 2
    ))
    return new_image


def main():
    for item in RAW_DATA.listdir():
        if '.pdf' not in item:
            continue

        PROCESSED_DATA.rmtree()
        PROCESSED_DATA.mkdir()

        # TODO: Add specific Page range here.
        # Start range from 1.

        for i in range(1, 11):
            image = next(extract_page_image(item, i))

            # TODO: set save_states=True to get intermediate states.
            process_image(image, save_states=False)

        with open(PROCESSED_DATA / 'labels.txt', 'w') as labels:
            for folder in PROCESSED_DATA.listdir():
                if folder.isfile():
                    continue
                for image in sorted(folder.listdir(), key=lambda x: int(x.basename().name.split('.')[0])):
                    labels.write(f'{image}|\n')


if __name__ == "__main__":
    main()
