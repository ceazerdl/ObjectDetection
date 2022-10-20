import sys
import os
import glob
import xml.etree.ElementTree as T

class_list = ['face', 'face_mask']
xml_path = r'face_mask\\test_labels_xml'
save_txt_path = r'face_mask\\test_labels_wh'


def get_txt_files(xml_path, save_txt_path, class_list):
    os.makedirs(save_txt_path, exist_ok=True)
    xml_file_list = glob.glob(xml_path + '/*.xml')
    assert len(xml_file_list), "Error: no .xml files found in ground-truth"
    for xml_file in xml_file_list:
        with open(save_txt_path + '\\' + os.path.basename(xml_file).replace(".xml", ".txt"), "w") as new_f:
            root = T.parse(xml_file).getroot()
            for size_wh in root.findall('size'):
                w = size_wh.find('width').text
                h = size_wh.find('height').text
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                if obj_name in class_list:
                    obj_name = class_list.index(obj_name)
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    new_f.write(w + ' ' + h + ' ' + repr(obj_name) + " " + left + " " + top + " " + right + " " + bottom + '\n')
    print("Conversion completed!")


if __name__ == '__main__':
    get_txt_files(xml_path, save_txt_path, class_list)