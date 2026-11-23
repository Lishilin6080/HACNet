# -*- coding: utf-8 -*-

from ultralytics import YOLO
from collections import Counter


if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'\runs\train\exp417\weights\best.pt')
    results = model.predict(source=r'\data\枣测试集\images',
                  save=True,
                  show=False,
                  )

    # ========== 新增：分类计数 ==========
    # 获取类别名称映射
    names = model.names  # {0: 'cls0', 1: 'cls1', ...}

    # 全数据集总计
    total_counter = Counter()
    # 每张图片的计数
    per_image_counter = {}

    for r in results:
        # r.boxes.cls 是每张图中检测到的类别 id 张量
        if r.boxes is None or len(r.boxes) == 0:
            per_image_counter[r.path] = {}
            continue

        cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
        c = Counter(cls_ids)
        total_counter.update(c)

        img_name = r.path.split('\\')[-1].split('/')[-1]
        per_image_counter[img_name] = {
            names[k]: v for k, v in c.items()
        }

    # ---------- 打印每一张图片的统计 ----------
    print("\n========== 每张图片检测结果 ==========")
    for img_name, counts in per_image_counter.items():
        print(f"\n[{img_name}]")
        if not counts:
            print("  未检测到目标")
        else:
            for cls_name, num in counts.items():
                print(f"  {cls_name}: {num} 颗")

    # ---------- 打印整个测试集的总统计 ----------
    print("\n========== 整个测试集分类总计 ==========")
    grand_total = 0
    for cls_id, num in sorted(total_counter.items()):
        print(f"  {names[cls_id]}: {num} 颗")
        grand_total += num
    print(f"  -----------------------------")
    print(f"  总计: {grand_total} 颗")
    # =====================================
