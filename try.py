import os
import cv2
import time

def image_show(image_object,image_name="Image"):
    cv2.imshow(image_name, image_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_boxes(image,boxes,color=(255,0,0)):
    for box in boxes:
        cv2.rectangle(image,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),color,2)
    image_show(image)


def get_boxes(image_path):
    # print image_path
    image_object = cv2.imread(image_path)
    # image_show(image_object=image_object)
    image_object_gray = cv2.cvtColor(image_object, cv2.COLOR_BGR2GRAY)

    image_object_bw = cv2.adaptiveThreshold(image_object_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    start_time = time.time()
    mser = cv2.MSER_create(_min_area=10,_max_area=1000)

    regions, np_boxes = mser.detectRegions(image_object_bw)
    boxes = []
    for np_box in np_boxes:
        boxes.append(list(np_box))

    boxes.sort(key=lambda x: x[1])


    boxes_bins = []
    boxes_bin = []
    dy=10
    y_val = boxes[0][1]
    for i,box in enumerate(boxes):
        if box[1]<= y_val+dy:
            boxes_bin.append(box)
        else:
            boxes_bins.append(boxes_bin)
            boxes_bin=[box]
            y_val = box[1]



    new_boxes = []
    for boxes_bin in boxes_bins:
        min_x = 100000000
        min_y = 100000000
        max_x = 0
        max_y = 0
        for box in boxes_bin:
            [x,y,w,h] = box
            if x<min_x:
                min_x=x
            if y<min_y:
                min_y=y
            if x+w>max_x:
                max_x=x+w
            if y+h>max_y:
                max_y=y+h
        new_boxes.append([min_x,min_y,(max_x-min_x),(max_y-min_y)])


    print "Time taken to get bounding boxes:",time.time()-start_time
    draw_boxes(image_object.copy(), boxes)
    draw_boxes(image_object.copy(), new_boxes)

    return new_boxes


if __name__ == '__main__':
    images_folder = "./tests/question_papers/"
    images_list = os.listdir(images_folder)
    for image_path in images_list[:1]:
        get_boxes(os.path.join(images_folder,image_path))