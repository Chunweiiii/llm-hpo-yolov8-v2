from ultralytics import YOLO

# ===== 設定區 =====
EPOCHS   = 100
IMGSZ    = 640
BATCH    = 16
DATA     = "C:/YOLO_agent_project/datasets/traffic_light/data.yaml"   # 改成絕對路徑
PROJECT  = "C:/YOLO_agent_project/runs"                               # 新增：絕對路徑
RUN_NAME = "baseline_s2"
SEED     = 2
# --- 以下是要讓 LLM 調整的超參數 ---
OPTIMIZER = "AdamW"      # 新增：關掉 auto，參數才會生效
LR0       = 0.001429
MOMENTUM  = 0.9
WEIGHT_DECAY = 0.0005
# ==================

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=8,
        seed=SEED,
        deterministic=True,
        optimizer=OPTIMIZER,          # 新增
        lr0=LR0,                      # 新增
        momentum=MOMENTUM,            # 新增
        weight_decay=WEIGHT_DECAY,    # 新增
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
    )

    print("=" * 40)
    print("mAP@0.5      :", results.box.map50)
    print("mAP@0.5:0.95 :", results.box.map)
    print("=" * 40)

if __name__ == "__main__":
    main()