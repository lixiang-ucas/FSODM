import cv2

img_name = 'C:/project/few_shot_detector/NWPU/positive image set/079.jpg'
anno_name = 'C:/project/few_shot_detector/NWPU/ground truth/079.txt'
training_classes = ['airplane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                    'ground-track-field', 'harbor', 'bridge', 'vehicle']

with open(anno_name, 'r') as anno:
    objs = [line.strip().split(',') for line in anno.readlines()]

img = cv2.imread(img_name)

for obj in objs:
    cls = int(obj[4]) - 1
    if cls == 0:
        color = (0, 230, 0)
        rec_length = 120
    elif cls == 3:
        color = (0, 0, 230)
        rec_length = 120
    elif cls == 4:
        color = (230, 0, 0)
        rec_length = 150
    else:
        color = (0, 230, 230)
        rec_length = 126
    x1 = int(obj[0].strip('('))
    x2 = int(obj[2].strip('('))
    y1 = int(obj[1].strip(')'))
    y2 = int(obj[3].strip(')'))
    # box_size = min(x2 - x1, y2 - y1) // 50
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # 标注文本
    cv2.rectangle(img, (x1, y1 - 15), (x1 + rec_length, y1), color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = training_classes[cls]
    cv2.putText(img, text, (x1 + 2, y1 - 4), font, 0.4, (0, 0, 0), 1)
    cv2.imwrite('C:/project/few_shot_detector/002.jpg', img)