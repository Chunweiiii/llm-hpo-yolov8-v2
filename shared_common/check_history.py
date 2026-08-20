import json
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "llm_haiku_s0"

with open(f"history/{tag}.json", encoding="utf-8") as f:
    h = json.load(f)

print("=== 完整表格 ===")
for x in h:
    print(f"i{x['iteration']:2d} | {x['map5095']:.4f} | {x['verdict']:9s} | "
          f"base={str(x.get('base_iteration')):4s} | dup={str(x.get('duplicate_of')):4s}")

print("\n=== 搜尋空間覆蓋率 ===")
keys = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'box', 'cls', 'hsv_v', 'mosaic']
touched = set()
for x in h[1:]:
    for c in x.get('changes', []):
        touched.add(c.get('param'))
print("動過的參數:", touched)
print("覆蓋率:", len(touched), "/ 9")