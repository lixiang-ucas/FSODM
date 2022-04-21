# from msilib.schema import Error
import cv2
import numpy as np
import os

def drawBoundingBox(img, point_left_top, point_right_bottom, class_name = "ship", 
                    line_color=(0, 255, 0), line_thickness=2, line_type=4):
    """_summary_
    Args:
        img (cv2.imread return): cv2.imread return
        point_left_top (tuple): the left top point coordinate
        point_right_bottom (tuple): the right bottom point coordinate
        class_name (String): the class name of the object
        line_color (tuple, optional): Define the color of the linecolor. Defaults to (0, 255, 0).
        line_thickness (int, optional): Define the thickness of the line. Defaults to 2.
        line_type (int, optional): Define the type of the line. Defaults to 4.

    Returns:
        cv2.imread: the image with bounding box drew
    """    
    try:
        cv2.rectangle(img, point_left_top, point_right_bottom, line_color, line_thickness, line_type)
    except TypeError:
        print(point_left_top, point_right_bottom, 'Not integer!')
        quit()
    # get text box size
    text_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    # get text box right bottom coordinate
    textbottom = point_left_top + np.array(list(text_size))
    # draw text rectangle
    cv2.rectangle(img, point_left_top, tuple(textbottom), line_color, -1)
    # get text offset
    text_left_top = (point_left_top[0], point_left_top[1] + (text_size[1]/2 + 4))
    cv2.putText(img, class_name, text_left_top, cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    return img

if __name__ == '__main__':
    annotation_path = '/home/xinyang/project/FSODM/results/metayolov3_nwpu_novel0_neg1/ene001900/comp4_det_test_ship.txt'
    # img_path = '/home/xinyang/project/FSODM/NWPU_Root/evaluation/images/686__1__0___0.jpg'
    # img_output_path = '/home/xinyang/project/FSODM/visualize/output.jpg'
    img_output_dir = '/home/xinyang/project/FSODM/visualize/'
    img_input_dir = '/home/xinyang/project/FSODM/NWPU_Root/evaluation/images'
    line_color = (0, 255, 0) # line color
    line_thickness = 2
    line_type = 4
    class_name = "ship"
    # img = cv2.imread(img_path)
    # point_left_top = (449, 303)
    # point_right_bottom = (553, 406)
    # img = drawBoundingBox(img, point_left_top, point_right_bottom, class_name)
    # cv2.imwrite(img_output_path, img)
    with open(annotation_path, 'r') as anno:
        processing_image = ''
        # img = cv2.imread()
        point_list = []
        line_count = len(anno.readlines())
    with open(annotation_path, 'r') as anno:   
        # print(line_count)
        for i, line in enumerate(anno):
            print(i)
            split_list = line.split()
            img_name = split_list[0]
            # first convert frome String to float, then to integer
            point_left_top = (int(float(split_list[2])), int(float(split_list[3])))
            point_right_bottom = (int(float(split_list[4])), int(float(split_list[5])))
            if i == 0:
                processing_image = img_name
                point_list.append((point_left_top, point_right_bottom))
            elif i != 0:
                if img_name != processing_image or i == line_count - 1:
                    img_path = os.path.join(img_input_dir, processing_image+'.jpg')
                    try :
                        img = cv2.imread(img_path)
                        print(img_path)
                    except FileNotFoundError:
                        print(img_path, 'input image path not exists!')
                        # raise FileNotFoundError('input image path not exists!')
                        # quit()
                    else:
                        for point in point_list:
                            img = drawBoundingBox(img, point[0], point[1])
                        img_output_path = os.path.join(img_output_dir, processing_image+'.jpg')
                        try :
                            cv2.imwrite(img_output_path, img)
                        except FileNotFoundError:
                            print(img_output_path, 'output image path not exists!')
                            # quit()
                            # raise FileNotFoundError('output image path not exists!')
                        else:
                            print(img_output_path)
                            processing_image = img_name
                            point_list = []
                            point_list.append((point_left_top, point_right_bottom))
                else:
                    point_list.append((point_left_top, point_right_bottom))
