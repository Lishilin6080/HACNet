# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'\ultralytics\cfg\models\FJ-HACNet.yaml')
    model.load('yolo12n.pt') 
    model.train(data=r'\ultralytics\cfg\datasets\TinyPerson.yaml',
                imgsz=640,
                epochs=1,
                batch=64,
                workers=4,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
