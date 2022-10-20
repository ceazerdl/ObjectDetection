import glob


def get_file(in_path, out_path):
    img_file_list = glob.glob(in_path + "/*.jpg")
    # glob.glob返回所有匹配的文件路径列表 e.g.：['.\\glob_.py', '.\\transfer_learning.py']
    img_file_list = [x.replace('\\', '/') + "\n" for x in img_file_list]
    with open(out_path + "\\" + "test.txt", "w") as f:
        f.writelines(img_file_list)


if __name__ == "__main__":
    in_path = r"face_mask\\test_images"
    out_path = r"face_mask"
    get_file(in_path, out_path)