import json
from collections import defaultdict
import numpy as np


"""hyper parameters"""
# images_dir_path = "coco/images/train2017"
json_file_path = "coco/labels/instances_train2017.json"
out_path = "coco/labels/train2017"

"""load json file"""
# name_box_id = defaultdict(list)
# id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)
annotations = data['annotations']
imgs = data["images"]
for img in imgs:
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    for annotation in annotations:
        id = annotation['image_id']
        if img_id == id:
            with open(out_path + '/%012d.txt'%id, 'a') as f:
                # name = "coco/images/train2017/%012d.jpg" %id
                cat = annotation['category_id']

                if cat >= 1 and cat <= 11:
                    cat = cat - 1
                elif cat >= 13 and cat <= 25:
                    cat = cat - 2
                elif cat >= 27 and cat <= 28:
                    cat = cat - 3
                elif cat >= 31 and cat <= 44:
                    cat = cat - 5
                elif cat >= 46 and cat <= 65:
                    cat = cat - 6
                elif cat == 67:
                    cat = cat - 7
                elif cat == 70:
                    cat = cat - 9
                elif cat >= 72 and cat <= 82:
                    cat = cat - 10
                elif cat >= 84 and cat <= 90:
                    cat = cat - 11
                category = repr(cat) + " "
                bboxes = np.array(annotation["bbox"])
                bboxes[2:] = bboxes[:2] + bboxes[2:]
                # bboxes[0] = (bboxes[0] + bboxes[2]/2)/img_width
                # bboxes[1] = (bboxes[1] + bboxes[3]/2)/img_height
                # bboxes[2] = bboxes[2] / img_width
                # bboxes[3] = bboxes[3] / img_height
                category += ' '.join(map(repr, bboxes))
                category += '\n'
                f.write(category)



