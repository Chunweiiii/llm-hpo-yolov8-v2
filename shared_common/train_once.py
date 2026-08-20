from ultralytics import YOLO
import csv, time, os

from search_space import FROZEN

DATA    = "C:/YOLO_agent_project/datasets/traffic_light/data.yaml"
PROJECT = "C:/YOLO_agent_project/runs"

FIXED = dict(
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    deterministic=True,
    project=PROJECT,
    exist_ok=True,
    verbose=False,
    plots=False,
)


def read_log(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        return [{k: float(v) for k, v in row.items()} for row in reader]


def train_once(hyp, run_name, seed=0, epochs=100, data_path=None):
    t0 = time.time()

    data = data_path if data_path is not None else DATA

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data,
        **FIXED,
        **FROZEN,
        name=run_name,
        seed=seed,
        epochs=epochs,
        **hyp,
    )

    save_dir = os.path.join(PROJECT, run_name)
    log = read_log(os.path.join(save_dir, "results.csv"))

    names = results.names
    per_class = {names[i]: round(float(v), 4) for i, v in enumerate(results.box.maps)}

    return {
        "run_name":  run_name,
        "seed":      seed,
        "epochs":    epochs,
        "hyp":       hyp,
        "map50":     round(float(results.box.map50), 4),
        "map5095":   round(float(results.box.map), 4),
        "per_class": per_class,
        "minutes":   round((time.time() - t0) / 60, 2),
        "save_dir":  save_dir,
        "log":       log,
    }


if __name__ == "__main__":
    from search_space import default_hyp

    out = train_once(default_hyp(), run_name="tmp_test", epochs=3)

    print("=" * 50)
    print("mAP50   :", out["map50"])
    print("mAP50-95:", out["map5095"])
    print("per_class:", out["per_class"])
    print("耗時    :", out["minutes"], "min")
    print("log 列數:", len(out["log"]))
    print("=" * 50)