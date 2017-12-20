import os
import cv2
import time
import pytesseract
import multiprocessing as mp
from PIL import Image
import unicodedata
import numpy as np


def extract_text_process(Image_obj,box_texts,boxes):
    print "{} boxes to process".format(len(boxes))

    bordersize = 20
    for box in boxes[:5]:
        if box != [0, 0, 0, 0]:
            x, y, w, h = box
            cropped_image = Image_obj[y: y + h, x: x + w]
            cropped_image = cv2.copyMakeBorder(cropped_image, top=bordersize, bottom=bordersize, left=bordersize,
                                               right=bordersize,
                                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # print type(cropped_image)
            recognised_text = pytesseract.image_to_string(Image.fromarray(cropped_image), lang="eng", config="-psm 6")
            recognised_text = unicodedata.normalize('NFKD', recognised_text.replace("\n", "\n")).encode('ascii',
                                                                                                        'ignore')
            box_data = {"box":box,"text":recognised_text}
            box_texts.put(box_data)

    # for box in boxes:
    #     if box!=[0,0,0,0]:
    #         x,y,w,h = box
    #         cropped_image = Image_obj[y: y + h, x: x + w]
    #         # print type(cropped_image)
    #         recognised_text = pytesseract.image_to_string(Image.fromarray(cropped_image),lang="eng")
    #         recognised_text = unicodedata.normalize('NFKD', recognised_text.replace("\n","")).encode('ascii','ignore')
    #         box_data = {"box":box,"text":recognised_text}
    #         box_texts.put(box_data)



def get_text_from_bounding_boxes_mp(image_path,boxes,console):
    box_texts = mp.Queue()
    image_obj = cv2.imread(image_path)
    smooth_image = remove_noise_and_smooth(image_path)
    image_show(smooth_image,"Smoothed image")
    draw_boxes(image_obj,boxes)
    image_obj_bw = cv2.adaptiveThreshold(smooth_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
    num_cores = mp.cpu_count()
    print "Using {} cores".format(num_cores)
    new_boxes=[[] for core in range(num_cores)]
    for i,box in enumerate(boxes):
        new_boxes[i%num_cores].append(box)

    processes = [mp.Process(target=extract_text_process, args=(image_obj_bw,box_texts,boxes_chunk)) for boxes_chunk in new_boxes]
    for p in processes:
        p.start()

    for p in processes:
        p.join()



    recognised_boxes = []
    while not box_texts.empty():
        recognised_boxes.append(box_texts.get())
    print "Recognised boxes:",len(recognised_boxes)

    result = {"image_path":image_path,"boxes":recognised_boxes,"image_size":(image_obj.shape[0],image_obj.shape[1])}

    return result

def image_show(image_object,image_name="Image"):
    cv2.imshow(image_name, image_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_boxes(image,boxes,color=(255,0,0)):
    for box in boxes:
        cv2.rectangle(image,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),color,2)
    image_show(image)


def image_smoothening(img):
    # ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def get_text_from_bounding_boxes(image_path, boxes, console=False):
    image_obj = cv2.imread(image_path)
    smooth_image = remove_noise_and_smooth(image_path)
    image_show(smooth_image,"Smoothed image")
    draw_boxes(image_obj,boxes)
    image_obj_bw = cv2.adaptiveThreshold(smooth_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
    result = {"image_path":image_path,"boxes":[]}
    bordersize=20
    for box in boxes[:5]:
        if box!=[0,0,0,0]:
            x,y,w,h = box
            cropped_image = image_obj_bw[y: y + h, x: x + w]
            cropped_image = cv2.copyMakeBorder(cropped_image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # print type(cropped_image)
            recognised_text = pytesseract.image_to_string(Image.fromarray(cropped_image),lang="eng",config="-psm 6")
            recognised_text = unicodedata.normalize('NFKD', recognised_text.replace("\n","\n")).encode('ascii','ignore')
            print recognised_text
            image_show(cropped_image)

            # box_data = {"box":box,"text":recognised_text}
            # result["boxes"].append(box_data)
            if console:
                image_show(cropped_image)
    result["image_size"]=(image_obj.shape[0],image_obj.shape[1])
    return result

def get_boxes_faster(image_path,console=False):
    # print image_path
    image_object = cv2.imread(image_path)
    # image_show(image_object=image_object)
    image_object_gray = cv2.cvtColor(image_object, cv2.COLOR_BGR2GRAY)

    image_object_bw = cv2.adaptiveThreshold(image_object_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            115, 1)
    start_time = time.time()
    mser = cv2.MSER_create(_min_area=10, _max_area=1000)

    regions, np_boxes = mser.detectRegions(image_object_bw)
    boxes = []
    for np_box in np_boxes:
        boxes.append(list(np_box))

    boxes.sort(key=lambda x: x[1])

    if len(boxes)==0:
        return []

    boxes_bins = []
    boxes_bin = []
    dy = 50
    y_val = boxes[0][1]
    y2_val = boxes[0][1]+boxes[0][3]

    for i, box in enumerate(boxes):
        if box[1] <= y_val + dy or box[1]+box[3] <= y2_val + dy:
            boxes_bin.append(box)
        else:
            boxes_bins.append(boxes_bin)
            boxes_bin = [box]
            y_val = box[1]


    new_boxes = []
    for boxes_bin in boxes_bins:
        min_x = 100000000
        min_y = 100000000
        max_x = 0
        max_y = 0
        for box in boxes_bin:
            [x, y, w, h] = box
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x + w > max_x:
                max_x = x + w
            if y + h > max_y:
                max_y = y + h
        new_boxes.append([min_x, min_y, (max_x - min_x), (max_y - min_y)])

    print "Number of boxes:",len(new_boxes)

    print "Time taken to get bounding boxes with get_boxes_faster:", time.time() - start_time
    if console:
        draw_boxes(image_object.copy(), boxes)
        draw_boxes(image_object.copy(), new_boxes)

    return new_boxes


if __name__ == '__main__':
    images_folder = "./tests/"
    images_list = os.listdir(images_folder)

    # for image_path in ["qpl.jpg"]:
    #     for f in range(1, 20, 5):
    #         for o in range(10):
    #             textclean(os.path.join(images_folder,image_path), f, o)
    #             start_time = time.time()
    #             boxes=get_boxes_faster("out.jpg",console=False)
    #             if len(boxes)>0:
    #                 result = get_text_from_bounding_boxes("out.jpg", boxes, console=False)
    #                 print "Completed in: ",time.time()-start_time
    #                 print result
    # for image_path in images_list[:1]:
    for image_path in ["blog_screenshot.jpg","scanned_image.jpg","qpl2.jpg"]:
        start_time = time.time()
        boxes=get_boxes_faster(os.path.join(images_folder,image_path),console=False)
        result = get_text_from_bounding_boxes_mp(os.path.join(images_folder,image_path), boxes, console=False)
        print "Completed in: ",time.time()-start_time
        print result
