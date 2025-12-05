# import torch
# import ultralytics;
# print(torch.cuda.is_available())
# print(torch.__version__)
# x = torch.rand(5, 3)
# print(x)
#
# print(ultralytics.__version__)
#
# print(torch.version.cuda)
# print(torch.cuda.get_arch_list())

# from ultralytics import YOLO
#
# model = YOLO("yolov8n.pt")
# results = model("https://ultralytics.com/images/bus.jpg")
# results[0].show()
##-----------------------------------------
##all for test the workflow


from ultralytics import YOLO
import torch
import os

# if __name__ == '__main__':
#     print("GPU is available：", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print(f"use GPU：{torch.cuda.get_device_name(0)}")
#         torch.backends.cudnn.benchmark = True
#     else:
#         print("Warning: No GPU detected, training may be very slow.")
#
#     model = YOLO('yolov8n.pt')
#
#
#     PROJECT_DIR = r"D:\ic\SIOT\clothes_identify_new\model_final"
#     EXPERIMENT_NAME = "four_clothes_types_model"
#
#     os.makedirs(os.path.join(PROJECT_DIR, EXPERIMENT_NAME), exist_ok=True)
#
#     print("\n--- Start training ---")
#     results = model.train(
#         data=r"D:\ic\SIOT\dataset_new_new\data.yaml",
#         epochs=300,
#         imgsz=640,
#         batch=4,
#         patience=100,
#         lr0=0.01,
#         optimizer='AdamW',
#         seed=42,
#         workers=2,
#         device=0,
#
#         save=True,
#         exist_ok=False,
#         project=PROJECT_DIR,
#         name=EXPERIMENT_NAME,
#
#         val=True
#     )
#
#     print("\n--- Training completed, starting test set evaluation ---")
#     metrics = model.val(
#         data=r"D:\ic\SIOT\dataset_new_new\data.yaml",
#         split='test'
#     )
#
#     print(f"\nTest set evaluation results：")
#     print(f"mAP@0.5 (All categories): {metrics.box.map50:.4f}")
#     print(f"mAP@0.5:0.95 (All categories): {metrics.box.map:.4f}")
#
#     print("\nmAP@0.5 for each category:")
#     if metrics.names and metrics.box.class_map50:
#         for i, name in enumerate(metrics.names):
#             if i < len(metrics.box.class_map50):
#                 print(f"  - {name}: {metrics.box.class_map50[i]:.4f}")
#             else:
#                 print(f"  - catagory {i} ({name}): Insufficient evaluation data or indexing issues")
#     else:
#         print("Unable to obtain mAP@0.5 for each category, please check the dataset and evaluation process.")
from ultralytics import YOLO
import torch
import os

DATA_YAML_PATH = r"D:\ic\SIOT\dataset_new_new\data.yaml"

PROJECT_DIR = r"D:\ic\SIOT\clothes_identify_new\model_final"
EXPERIMENT_NAME = "four_clothes_types_final"
# ---------------------------------------------

if __name__ == '__main__':
    print("--- 1. Check env and GPU ---")
    print(f"GPU is available：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"use GPU：{gpu_name}")
        torch.backends.cudnn.benchmark = True
        DEVICE_ID = 0
    else:
        print("Warning: No GPU detected, training may be very slow.")
        DEVICE_ID = 'cpu'

    os.makedirs(os.path.join(PROJECT_DIR, EXPERIMENT_NAME), exist_ok=True)
    print(f"Save result to: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME)}")

    model = YOLO('yolov8n.pt')

    print("\n--- 3. Start training ---")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=300,
        imgsz=640,
        batch=4,
        patience=100,
        lr0=0.01,
        optimizer='AdamW',

        save=True,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        device=DEVICE_ID,
        workers=8,
        val=True
    )

    print("\n--- 4. Training completed, starting test set evaluation ---")

    best_weights_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')
    if os.path.exists(best_weights_path):
        final_model = YOLO(best_weights_path)
    else:
        final_model = model

    metrics = final_model.val(
        data=DATA_YAML_PATH,
        split='test'
    )

    print(f"\n--- Test set evaluation results：---")
    print(f"mAP@0.5 (All categories): {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 (All categories): {metrics.box.map:.4f}")

    print("\nmAP@0.5 for each category:")
    class_map50_list = metrics.box.all_ap[:, 0]
    names = metrics.names
    ap_classes = metrics.box.ap_class_index
    for i, class_index in enumerate(ap_classes):
        class_name = names[class_index]
        map50_value = class_map50_list[i]

        print(f"  - {class_name}: {map50_value:.4f}")