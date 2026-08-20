# LLM-based Hyperparameter Optimization for YOLOv8 Traffic Light Detection

比較不同能力層級的語言模型,在超參數搜尋任務中是否能有效利用「訓練動態診斷」(而非僅最終分數)做出優於傳統方法的決策。以 YOLOv8n 紅綠燈偵測為測試任務,對照 Default / Random Search / Optuna(TPE)/ 四個 Claude 系列模型 + 一個本地開源模型,在相同的 10 次迭代預算下進行公平比較。

---

## 一、研究問題

> 在超參數最佳化中,讓 LLM 讀取「訓練動態診斷」而非僅「最終分數」,是否帶來實質效益?效益如何隨模型能力變化?

與前作(Zhang et al., "Using Large Language Models for Hyperparameter Optimization", NeurIPS 2023 Workshop)的核心差異:該研究的 LLM 回饋訊號是**最終表現分數**;本研究的回饋訊號是**訓練過程的結構化診斷**(平台期位置、loss-mAP 脫鉤現象、真假過擬合判斷等)。

---

## 二、核心結果

### 六方法對照(10 次迭代,固定預算)

| 方法 | 最佳 mAP50-95 | 出現輪次 | Duplicate rate | 搜尋空間覆蓋率 |
|---|---|---|---|---|
| Random Search | 0.7106 | iter 1 | - | - |
| Optuna (TPE) | 0.7111 | iter 8 | - | - |
| Qwen2.5-Coder-7B (本地) | 0.7049 | iter 7 | 40% (4/10) | 3/9 |
| Claude Haiku 4.5 | 0.7153 | iter 5 | 0% | 8/9 |
| **Claude Sonnet 5** | **0.7187** | iter 5 | 0% | **9/9** |
| Claude Opus 5 | 0.7121 | iter 4 | 0% | 9/9 |

**Default baseline (3-seed):** mAP50-95 = 0.7024 ± 0.0026
**判斷門檻:** 改善需超過 0.005(約 2σ)才視為統計上顯著

詳見 `convergence_comparison.png`(六方法的 best-so-far 收斂曲線)與 `summary_table.png`(彙總表圖片版)。

### 最佳配置的 3-seed 驗證

Sonnet 5 於 iteration 5 找到的配置,重跑 3 個 seed(與 baseline 相同方法論)驗證其穩健性:

```
最佳配置: mAP50-95 = 0.7129 ± 0.0078
baseline:  mAP50-95 = 0.7024 ± 0.0026
改善幅度:  0.0105(超過 2σ 門檻,統計上顯著)
```

**注意:** 改善顯著,但標準差同時擴大(0.0026 → 0.0078)。細部分析(`final_verification.json`)顯示,擴大主因是黃燈類別(樣本數最少,n=28)在不同 seed 間的自然波動,而非該配置引入了系統性不穩定——綠燈類別(該配置主要改善目標)在三個 seed 間高度一致(0.686–0.709)。

### 客觀行為指標的斷崖式差異

Duplicate rate(重複配置比例)與搜尋空間覆蓋率,在 Qwen 與三個 Claude 模型之間呈現遠比推理品質差異更懸殊的斷層(40% vs 0%;3/9 vs 8-9/9)。這暗示模型能力對「能否將診斷正確轉化為新穎決策」的影響,可能比對「文字描述準確度」更顯著。

### 質性觀察(個案分析,非系統性評分)

以下為對特定迭代深入檢視後記錄的具體現象,用於提供推理行為的具體感受,非模型排序依據:

- **Qwen(iteration 3)**:提出與歷史紀錄不符的參數演變描述,並重複建議先前已證明無效的配置
- **Haiku(iteration 3→4)**:配置未改善後,準確引用具體訓練曲線數值並調整策略幅度,而非重複相同建議
- **Sonnet 5 / Opus 5**:均曾將 YOLOv8 實際使用的線性學習率排程誤稱為「cosine schedule」,但此機制描述誤用並未影響其處方與訓練動態觀察之間的邏輯一致性
- **Opus 5(iteration 9→10)**:主動宣告前一輪假說已被推翻,並指出先前一輪實驗因同時改動兩個參數導致結果無法歸因的方法論問題

---

## 三、實驗設計

### 資料集

- 來源:Roboflow(`traffic-light-for-yolo/traffic-light-reconizetion`, v10)
- 切分:train 509 / valid 145 / test 73,共 727 張
- 類別:green light / red light / yellow light(3 類)
- Preprocessing 僅開 Auto-Orient;Augmentation 全部關閉(增生強度為本實驗的搜尋變數之一,不可預先烤入資料)

### 搜尋空間(9 個參數)

| 參數 | 範圍 | 預設值 |
|---|---|---|
| lr0 | [0.0002, 0.005] | 0.001429 |
| lrf | [0.005, 0.2] | 0.01 |
| momentum | [0.85, 0.98] | 0.9 |
| weight_decay | [0.0001, 0.002] | 0.0005 |
| warmup_epochs | [1.0, 5.0] | 3.0 |
| box | [5.0, 10.0] | 7.5 |
| cls | [0.3, 1.5] | 0.5 |
| hsv_v | [0.2, 0.6] | 0.4 |
| mosaic | [0.5, 1.0] | 1.0 |

**刻意排除** `hsv_h`(色相擾動,會破壞交通號誌類別的語意正確性)與 `flipud`(垂直翻轉,物理上不合理)。凍結參數見 `search_space.py` 的 `FROZEN`。

### 固定設定(四種方法共用,保證公平比較)

```
model=yolov8n.pt, epochs=100, imgsz=640, batch=16
optimizer=AdamW(明確指定，避免 ultralytics 'auto' 模式覆蓋自訂 lr0/momentum)
deterministic=True
```

### 使用的模型與 API 版本

```
本地:  qwen2.5-coder:7b (Ollama, temperature=0)
API:   claude-haiku-4-5-20251001 (temperature=0)
       claude-sonnet-5 (adaptive thinking, temperature 已棄用, max_tokens=8000)
       claude-opus-5 (adaptive thinking, 同上)
```

**注意:** Sonnet 5 / Opus 5 屬新一代 adaptive thinking 模型,`max_tokens` 需涵蓋內部思考過程,故對這兩個模型特別設為 8000(其餘模型為 2000);且兩者均已棄用 `temperature` 參數,呼叫時不可傳入,否則回傳 400 錯誤。

---

## 四、環境安裝

### 系統需求

- Windows 10/11,NVIDIA GPU(本專案於 RTX 5060 Laptop 驗證,Blackwell 架構 sm_120)
- Python 3.12

### 安裝步驟

```bash
python -m venv venv
venv\Scripts\activate

# ⚠️ 順序不可顛倒：先裝 torch 再裝 ultralytics
# ⚠️ Blackwell (RTX 50 系列) 必須使用 cu128 以上版本，cu118/cu121/cu124 會報錯
#    "no kernel image is available for execution on the device"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

python -m pip install -r requirements.txt
```

### 環境驗證(必做,不能跳過)

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); x=torch.rand(1000,1000).cuda(); print((x@x).sum().item())"
```

結尾須為 `+cu130`(非 `+cpu`),且矩陣乘法須成功印出數字(僅 `cuda.is_available()==True` 不足以確認 GPU 真正可用)。

### API 金鑰設定

在專案根目錄建立 `.env`(不可提交版本控制、不可截圖分享):

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 已知環境陷阱

- **`pip install` 若遭 Windows Device Guard/WDAC 政策封鎖**,改用 `python -m pip install <package>`
- **pandas 可能被 Smart App Control 封鎖**(`DLL load failed`),本專案已改用 Python 內建 `csv` 模組,不依賴 pandas

---

## 五、檔案結構與資料流

```
search_space.py       固定：9 個超參數範圍 + FROZEN 凍結參數（LLM 與 Optuna 共用同一份定義）
train_once.py         固定：訓練封裝，吃 hyp dict → 回傳 result dict（四種方法共用同一介面）
summarize_log.py      固定：訓練 log 轉為 LLM 可讀的診斷文字（英文輸出）
llm_optimizer.py      可抽換後端（qwen/haiku/sonnet/opus）+ prompt 組裝 + JSON 解析防呆
prompts/program.md    LLM 指令模板（含 {{SEARCH_SPACE}} {{CURRENT_STATE}} {{HISTORY}} {{LOG_SUMMARY}} 佔位符）

run_experiment.py      LLM 主迴圈（斷點續跑、三態判定 improved/regressed/tie）
run_random.py          Random Search 主迴圈
run_optuna.py          Optuna (TPE) 主迴圈，n_startup_trials=3

check_history.py       讀取任一 history/*.json 並印出表格與搜尋空間覆蓋率
inspect_history.py     另一版本的歷史檢視工具
plot_convergence.py    繪製六方法收斂曲線圖
plot_summary_table.py  繪製彙總表格圖

history/*.json         每輪即時存檔的唯一真相來源，斷點續跑時自動從此接續
traces/*.txt            每輪 LLM 呼叫的完整 prompt 與原始回應（論文附錄材料）
runs/                   每次訓練的完整輸出（results.csv, args.yaml, weights/ 等）
runs_for_handoff/        精簡版：僅含每個方法的 baseline 與最佳輪次（交接用）
```

### 資料流

```
search_space.py
    ↓
[Random 抽樣 / Optuna TPE / LLM 推理] → 決定下一組超參數
    ↓
train_once.py（統一訓練介面，100 epochs）
    ↓
summarize_log.py（將訓練動態轉為診斷文字，僅 LLM 路徑使用）
    ↓
llm_optimizer.py（組裝 prompt、呼叫模型、驗證輸出，僅 LLM 路徑使用）
    ↓
history/*.json（記錄本輪結果，回饋進下一輪決策）
```

---

## 六、如何重跑一次完整實驗

### 單一方法(以 Sonnet 5 為例)

```bash
# 1. 先跑冒煙測試，確認管線通暢（約 4 分鐘）
# 編輯 run_experiment.py 最後一行：
#   run(backend="sonnet", n_iter=2, seed=0, epochs=10)
python run_experiment.py

# 2. 驗收：往上找 "optimizer:" 那行，確認沒有 "ignoring"
#    確認 JSON 解析無錯誤（無 "! JSON 解析失敗" 訊息）

# 3. 清除冒煙測試資料
del history\llm_sonnet_s0.json
for /d %i in (runs\llm_sonnet_s0_i*) do rmdir /s /q "%i"

# 4. 切換正式設定並執行（約 70-75 分鐘）
#   run(backend="sonnet", n_iter=10, seed=0, epochs=100)
python run_experiment.py
```

`backend` 可選:`qwen` / `haiku` / `sonnet` / `opus`(對應 `llm_optimizer.py` 的 `BACKENDS` 字典)。

### Random Search / Optuna

```bash
python run_random.py    # 約 75 分鐘
python run_optuna.py    # 約 75 分鐘
```

### 查看任一方法的結果

```bash
python check_history.py llm_sonnet_s0   # 不帶參數預設查 llm_haiku_s0
python check_history.py random_s0
python check_history.py optuna_s0
```

### 重新繪圖

```bash
python plot_convergence.py     # 輸出 convergence_comparison.png
python plot_summary_table.py   # 輸出 summary_table.png
```

---

## 七、已知的兩個致命陷阱(修改程式前務必了解)

### 陷阱一:`optimizer='auto'` 會靜默覆蓋自訂超參數

ultralytics 預設 `optimizer='auto'`,此模式下會**自動決定並覆蓋** `lr0`、`momentum`,且**不報錯**。log 中會出現:

```
optimizer: 'optimizer=auto' found, ignoring 'lr0=...' and 'momentum=...' ...
```

**驗收標準:** 訓練 log 的 `optimizer:` 那行**絕不能出現 "ignoring"**。此陷阱在開發過程中出現過兩次(`search_space.py` 定義了 `FROZEN` 但 `train_once.py` 未接入),每次新增訓練相關腳本都須複查。

### 陷阱二:Sonnet 5 / Opus 5 的 adaptive thinking 機制

新一代模型的 `max_tokens` 同時涵蓋內部思考與最終輸出。若配額不足,思考過程會耗盡配額,導致最終 JSON 未寫完即被截斷(`stop_reason: max_tokens`),此時 `content` 中僅有空的 `text` 區塊,程式不會報錯但會靜默失敗(退回預設參數)。已透過對這兩個模型特別設定 `max_tokens=8000` 解決,詳見 `llm_optimizer.py` 的 `call_anthropic`。

---

## 八、範圍與限制

### 明確排除的方向(維持研究範圍可控)

- 換用 YOLO26(會作廢現有全部 baseline 與雜訊門檻校準)
- 多智能體架構(Advisor/Evaluator/Optimizer 多代理人)
- 神經架構搜尋 / 動態 Focal Loss(YOLOv8 預設不使用 Focal Loss)

### 已知限制

- 搜尋預算(10 輪)相對 9 維空間而言遠未窮盡(理論網格點數 3^9 ≈ 19,683),此為刻意的實驗設計,用以模擬實務中的低預算場景,而非資源不足
- Sonnet 5 與 Opus 5 屬不同世代命名(對照 Haiku 4.5),因發布時間點巧合所致,已在方法論中如實記錄
- 推理品質未經系統性量化評分(缺乏獨立評分者,樣本集中於少數代表性迭代),相關觀察僅作個案分析呈現,未包裝為排序依據

### 附加/未來工作(依優先度排序)

1. Score-only 版本 prompt(移除訓練動態診斷,僅提供最終分數),用以隔離「讀 log」本身的效益
2. Optuna 20 輪版本,驗證 10 輪打平 Random Search 是否純粹因暖機期(`n_startup_trials=3`)占比過高
3. 建立客觀評分規準,由獨立評分者對全部迭代進行盲評,量化推理品質差異
4. 邊緣裝置部署驗證(Jetson TX2 / TensorRT 匯出),量測實際推論延遲
5. Rolling Window 門檻掃描實驗(離線分析,無需重新訓練)
6. 第二資料集驗證(火煙偵測,mAP50 baseline 約 56%,天花板遠高於本資料集,可驗證方法在不同改善空間下的行為)

---

## 九、成本參考

以本專案的 prompt 結構估算(單次 10 輪實驗,平均 input ≈ 2000-2500 tokens、output ≈ 400-1800 tokens):

| 模型 | 定價(每百萬 token,input/output) | 單次實驗估計成本 |
|---|---|---|
| Claude Haiku 4.5 | $1 / $5 | 約 $0.04 |
| Claude Sonnet 5 | $2-3 / $10-15(介紹價至 2026/8/31) | 約 $0.15-0.3 |
| Claude Opus 5 | $5 / $25 | 約 $0.2-0.3 |

**API 費用非本研究的實際限制,GPU 訓練時間(每方法 70-75 分鐘)才是主要成本瓶頸。**
