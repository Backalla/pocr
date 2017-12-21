import os
import cv2
import time
import pytesseract
import multiprocessing as mp
from PIL import Image
import unicodedata
import numpy as np
from matplotlib import pyplot as plt


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
        cv2.rectangle(image,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),color,1)
    image_show(image)
    cv2.imwrite("boxes.jpg",image)

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

def box_analysis(boxes):
    all_x = sorted([x[0] for x in boxes])
    all_y = sorted([x[1] for x in boxes])
    all_h = sorted([x[3] for x in boxes])
    # plt.plot(all_x)
    # plt.ylabel("X-coordinates")
    # plt.show()
    # plt.plot(all_y)
    # plt.ylabel("Y-coordinates")
    # plt.show()
    #
    # delta_y = []
    # for i in range(len(all_y)-1):
    #     delta_y.append(all_y[i+1]-all_y[i])
    # plt.plot(delta_y)
    # plt.ylabel("delta Y   "+str(sum(delta_y)/len(delta_y)))
    # plt.show()
    # plt.plot(sorted(delta_y))
    # plt.ylabel("delta Y   " + str(sum(delta_y) / len(delta_y)))
    # plt.show()

def fill_boxes(image_obj,boxes,color=(0,0)):
    polygons=[]
    percent_of_size = 0.2
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        dx = int(percent_of_size * w)
        dy = int(percent_of_size * h)
        cv2.rectangle(image_obj,(x-dx,y),(x+w+dx,y+h),(0,0,0),cv2.FILLED)
        # polygon = [[x-dx,y-dy],[x+w+dx,y-dy],[x+w+dx,y+h+dy],[x-dx,y+h+dy]]
        # polygons.append(polygon)

    # for polygon in polygons:
    #     cv2.rectangle(image_obj,tuple(polygon[0]),tuple(polygon[2]),(0,0,0),cv2.FILLED)
    # polygons = np.array(polygons)
    # return cv2.fillPoly(image_obj,polygons,color)
    return image_obj

def get_better_boxes(image_path,console=False):
    min_contour_size = 50
    image_obj = cv2.imread(image_path)
    image_show(image_obj,"Original image")
    smoothed_image = remove_noise_and_smooth(image_path)
    print smoothed_image.shape
    image_show(smoothed_image,"Smoothed image")
    start_time = time.time()
    mser = cv2.MSER_create(_min_area=10, _max_area=1000)

    regions, np_boxes = mser.detectRegions(smoothed_image)
    boxes = []
    for np_box in np_boxes:
        boxes.append(list(np_box))

    filled_smooth_image = cv2.cvtColor(fill_boxes(image_obj.copy(),boxes),cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(filled_smooth_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_obj.copy(),contours,-1,(0,255,0),2)
    print "Found {} countours".format(len(contours))
    contour_areas = [c for c in contours if cv2.contourArea(c) > min_contour_size]
    print "Filtered to {} contours".format(len(contour_areas))
    contour_boxes=[]
    for contour in contour_areas:
        contour_boxes.append(cv2.boundingRect(contour))


    image_show(image_obj)

    draw_boxes(image_obj.copy(),contour_boxes)
    box_analysis(boxes)
    return contour_boxes

if __name__ == '__main__':
    images_folder = "./tests/"
    images_list = os.listdir(images_folder)

    for image_path in ["agreement.jpg","qpl5.jpg","qpl.jpg","blog_screenshot.jpg","scanned_image.jpg","qpl2.jpg","qpl3.jpg","qpl4.jpg"]:
        start_time = time.time()
        boxes=get_better_boxes(os.path.join(images_folder,image_path),console=True)
        # result = get_text_from_bounding_boxes_mp(os.path.join(images_folder,image_path), boxes, console=False)
        # print "Completed in: ",time.time()-start_time
        # print result
