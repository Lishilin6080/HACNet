import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    model = YOLO('/weights/best.pt') # 第三步：选择P-R提取数据文件夹中训练好的权重路径
    model.val(data='/ultralytics/cfg/datasets/persimmon.yaml',#第四步：类别文件的路径
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=64,
              project='runs/val',
              name='exp',
              )
