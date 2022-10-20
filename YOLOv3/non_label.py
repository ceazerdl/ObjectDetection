import json


json_file_path = "coco/labels/instances_train2017.json"
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)
annotations = data['annotations']
# print(len(data["annotations"]))
idx = 0
for annotation in annotations:
    # idx = 0
    if "bbox" in annotation:
        idx += 1
print(idx)
