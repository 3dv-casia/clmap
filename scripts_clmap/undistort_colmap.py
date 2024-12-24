import os
import time


def undistort_colmap(image_path, input_path, output_path):
    cmd = 'colmap image_undistorter --image_path {0} --input_path {1} --output_path {2} --output_type COLMAP'.format(
        image_path, input_path, output_path)
    print(cmd)
    os.system(cmd)
    time.sleep(1.0)
