import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rishikesh/Desktop/drone_hover/install/drone_img_processing'
