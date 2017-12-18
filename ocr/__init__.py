import cv2
import time
import pytesseract
from PIL import Image
import unicodedata


def image_show(image_object,image_name="Image"):
    cv2.imshow(image_name, image_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_boxes(image,boxes,color=(255,0,0)):
    for box in boxes:
        cv2.rectangle(image,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),color,2)

    image_show(image)


def area(rect):
    x1,y1,w,h = rect
    return float(w*h)

def intersection(rect1,rect2):
    x1,y1,w1,h1=rect1
    x2,y2,w2,h2=rect2
    xi = max(x1,x2)
    yi = max(y1,y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2+h2) - yi
    if wi < 0 or hi < 0:
        return (0,0,0,0)
    return (xi,yi,wi,hi)


def get_character_bounding_boxes(image_path,console=False):

    # Your image path i-e receipt path
    img = cv2.imread(image_path)
    if console:
        image_show(img)
    # print img.shape
    # img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # image_show(img_bw)


    # detect regions in gray scale image
    start_time=time.time()
    mser = cv2.MSER_create(_min_area=10,_max_area=1000)

    regions, np_boxes = mser.detectRegions(img_bw)
    boxes = []
    for np_box in np_boxes:
        boxes.append(list(np_box))
    if console:
        draw_boxes(img.copy(), boxes)
    start_time=time.time()
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi==[0,0,0,0]:
            continue
        # img_box = img.copy()
        # draw_boxes(img_box,[boxi],(255,0,0))
        for j in range(i+1,len(boxes)):
            boxj=boxes[j]
            if boxj==[0,0,0,0]:
                continue
            if boxi == [0, 0, 0, 0]:
                break
            # draw_boxes(img_box, [boxj], (0, 255, 0))

            # boxiou = iou(tuple(boxi),tuple(boxj))
            box_intersection = intersection(boxi,boxj)
            # print area(box_intersection),area(boxi),boxj, area(boxj)
            if area(box_intersection) == min(area(boxi),area(boxj)):
                if area(boxj)>area(boxi):
                    small=i
                else:
                    small=j
                # print "inside"
                boxes[small]=[0, 0,0,0]

    # print boxes
    # draw_boxes(img.copy(),boxes)
    dx = 150
    dy = 10
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi == [0, 0, 0, 0]:
            continue
        # img_box = img.copy()
        # draw_boxes(img_box,[boxi],(0,255,0))
        for j in range(i+1,len(boxes)):
            boxj = boxes[j]
            # draw_boxes(img_box, [boxj], (0, 0, 255))
            if boxj == [0, 0, 0, 0]:
                continue
            if boxi == [0, 0, 0, 0]:
                break

            if abs(boxj[0]-boxi[0])<dx and abs(boxj[1]-boxi[1])<dy:
                xb,yb = min(boxj[0],boxi[0]),min(boxj[1],boxi[1])
                wb = max(boxj[0]+boxj[2],boxi[0]+boxi[2]) - xb
                hb = max(boxj[1]+boxj[3],boxi[1]+boxi[3]) - yb
                boxes[i] = [xb,yb,wb,hb]
                boxes[j] = [xb,yb,wb,hb]

    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi==[0,0,0,0]:
            continue
        img_box = img.copy()
        # draw_boxes(img_box,[boxi],(255,0,0))
        for j in range(i+1,len(boxes)):
            boxj=boxes[j]
            if boxj==[0,0,0,0]:
                continue
            if boxi == [0, 0, 0, 0]:
                break
            # draw_boxes(img_box, [boxj], (0, 255, 0))

            # boxiou = iou(tuple(boxi),tuple(boxj))
            box_intersection = intersection(boxi,boxj)
            # print area(box_intersection),area(boxi),boxj, area(boxj)
            if area(box_intersection) == min(area(boxi),area(boxj)):
                if area(boxj)>area(boxi):
                    small=i
                else:
                    small=j
                # print "inside"
                boxes[small]=[0, 0,0,0]

    # print boxes
    print "Time taken to get bounding boxes with get_character_bounding_boxes:", time.time() - start_time

    if console:
        draw_boxes(img.copy(), boxes)
    return boxes


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


def get_text_from_bounding_boxes(image_path, boxes, console=False):
    image_obj = cv2.imread(image_path)
    image_obj_gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)

    image_obj__bw = cv2.adaptiveThreshold(image_obj_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    result = {"image_path":image_path,"boxes":[]}
    for box in boxes:
        if box!=[0,0,0,0]:
            x,y,w,h = box
            cropped_image = image_obj__bw[y: y + h, x: x + w]
            # print type(cropped_image)
            recognised_text = pytesseract.image_to_string(Image.fromarray(cropped_image),lang="eng")
            recognised_text = unicodedata.normalize('NFKD', recognised_text.replace("\n","")).encode('ascii','ignore')
            box_data = {"box":box,"text":recognised_text}
            result["boxes"].append(box_data)
            if console:
                image_show(cropped_image)
    result["image_size"]=(image_obj.shape[0],image_obj.shape[1])
    return result


def do_ocr(image_path,console):
    start_time=time.time()
    # boxes = get_character_bounding_boxes(image_path,console)
    boxes = get_boxes_faster(image_path,console)
    result = get_text_from_bounding_boxes(image_path,boxes,console)

    result["time_taken"] = str(time.time()-start_time)
    return result




if __name__ == '__main__':
    filename = "../tests/blog_screenshot.jpg"
    filename = "../tests/scanned_image.jpg"
    # filename = "../tests/scanned_image2.png"
    do_ocr(filename,True)
