import os
import glob

for i in range (0,61):
    if i<=9:
        path = input("Dataset\Training\Training\000{}".format(i))
    if i >9:
        path = input("Dataset\Training\Training\000{}".format(i))

    for filename in glob.glob(os.path.join(path, '*.ppm')):
        file_obj = open(filename, "r", encoding="utf-8")
        cv2.imshow("file_obj",file_obj)