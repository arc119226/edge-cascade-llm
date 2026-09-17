#!/usr/bin/env python3
"""
EdgeCascadeLLM — 分散式推理效能的參數化數值模型

用途：在寫任何一行推理程式碼之前，先用第一性原理估算這個架構跑得多快、
瓶頸在哪、哪些設計選擇是划算的。文件中每一個量化宣稱都應該能對回這裡的
某一組參數。

核心物理：
  1. batch=1 解碼是「記憶體頻寬受限」的：一次算 K 個序列位置，權重只讀一次，
     FLOPs 變 K 倍。所以小的 K 幾乎免費 —— 但有上限（見 roofline_k_star）。
  2. 流水線平行「層」是嚴格資料依賴，無法平行；每次迭代都要走完整條鏈。
  3. 平行解碼（Jacobi / Lookahead / 投機解碼）平行的是「序列位置」，
     它減少的是「每個 token 要走幾趟流水線」，不是消除 pipeline bubble。

用法：
  python3 model.py                         # 預設情境總覽
  python3 model.py --model 70b --nodes 8 --rtt 50 --mbps 20
  python3 model.py --sweep k               # 掃描平行視窗 K（重現「vanilla Jacobi 最佳 K=1」）
  python3 model.py --sweep nodes
  python3 model.py --roofline              # 各裝置的免費平行視窗 K*
  python3 model.py --churn                 # 節點流失與冗餘成本
  python3 model.py --validate              # 對照 Petals 實測值
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 參數定義
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    """一個 LLM 的關鍵尺寸。"""

    name: str
    params: float  # 總參數量
    d_model: int  # 隱藏維度（決定 pipeline hop 的傳輸量）
    layers: int
    weight_bits: int = 4  # 量化後的權重位元數

    @property
    def weight_bytes(self) -> float:
        return self.params * self.weight_bits / 8

    @property
    def weight_gb(self) -> float:
        return self.weight_bytes / 1e9


@dataclass(frozen=True)
class Device:
    """一個參與節點的硬體能力。"""

    name: str
    bandwidth_gbps: float  # 記憶體頻寬 GB/s（決定讀權重的時間）
    tflops: float  # 有效 FLOPS（已折算 WebGPU 實際可達，非峰值）

    def roofline_k_star(self, model: Model, stage_frac: float = 1.0) -> float:
        """免費平行視窗上限 K*。

        K* = 讀權重時間 / 每個位置的 FLOPs 時間
        K <= K* 時：時間由讀權重決定，多算位置不要錢。
        K >  K* 時：轉為計算受限，每多一個位置就線性變慢。
        注意 stage_frac 會同時縮放分子與分母，所以 K* 與切分方式無關。
        """
        t_mem = model.weight_bytes * stage_frac / (self.bandwidth_gbps * 1e9)
        t_flop_per_k = 2 * model.params * stage_frac / (self.tflops * 1e12)
        return t_mem / t_flop_per_k

    def stage_time_ms(self, model: Model, stage_frac: float, k: int) -> float:
        """此裝置負責 stage_frac 比例的層、一次處理 k 個位置所需的時間。"""
        t_mem = model.weight_bytes * stage_frac / (self.bandwidth_gbps * 1e9)
        t_flop = 2 * model.params * stage_frac * k / (self.tflops * 1e12)
        return max(t_mem, t_flop) * 1000


@dataclass(frozen=True)
class Network:
    """節點之間的網路條件。"""

    name: str
    rtt_ms: float
    mbps: float  # 上行頻寬（P2P 的瓶頸通常在上行）
    overhead_ms: float = 15.0  # 每個 hop 的固定軟體開銷（見下）

    def hop_ms(self, payload_bytes: float) -> float:
        """單一 pipeline hop 的時間 = 單程延遲 + 傳輸時間 + 固定開銷。

        固定開銷涵蓋所有與 payload 大小和距離無關、但每經過一個節點就要付一次的成本：
          - 張量序列化 / 反序列化
          - 推理框架的 dispatch 開銷
          - WebGPU 的 kernel launch 開銷（瀏覽器環境特別明顯，見 arXiv 2604.02344）
          - GPU buffer 的上傳 / 下載與同步

        這一項在高延遲下被 RTT 蓋過，但在 LAN / 低延遲情境會變成主導項。
        預設 15ms 是對瀏覽器 + WebGPU 堆疊的估計；Petals（Python + PyTorch + 原生 CUDA）
        的實測隱含值約 35–40ms/hop —— 見 --validate。這個參數目前是本模型
        最沒有把握的一項，M2 里程碑要實測校正。
        """
        transfer_ms = (payload_bytes * 8 / 1e6) / self.mbps * 1000
        return self.rtt_ms / 2 + transfer_ms + self.overhead_ms


# ---------------------------------------------------------------------------
# 解碼方法：每次迭代平均能接受幾個 token
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeMethod:
    """平行解碼方法的接受率曲線。

    acceptance(K) = 平均每次迭代（每趟流水線）產出的 token 數。
    K=1 時所有方法都退化成 1.0（普通自迴歸）。

    ⚠ 這些是保守的文獻估計值，不是實測。M2 里程碑必須用實測值替換。
      但 §結論對這些數字不敏感 —— 見 --sweep k 的敏感度分析。
    """

    name: str
    gain_per_slot: float  # 每增加一個視窗位置的邊際接受量
    ceiling: float  # 接受率上限
    needs_draft_model: bool = False

    def acceptance(self, k: int) -> float:
        return min(self.ceiling, 1.0 + self.gain_per_slot * (k - 1))


# ---------------------------------------------------------------------------
# 預設值
# ---------------------------------------------------------------------------

MODELS = {
    "1b": Model("TinyLlama-1.1B", 1.1e9, 2048, 22),
    "8b": Model("Llama-3-8B", 8e9, 4096, 32),
    "32b": Model("Qwen2.5-32B", 32e9, 5120, 64),
    "70b": Model("Llama-3-70B", 70e9, 8192, 80),
    "176b": Model("BLOOM-176B", 176e9, 14336, 70),
}

DEVICES = {
    "phone": Device("旗艦手機 (WebGPU)", 68, 1.5),
    "laptop": Device("MacBook M3", 100, 4.0),
    "laptop-pro": Device("MacBook M3 Max", 400, 14.0),
    "desktop": Device("RTX 4090", 1000, 150.0),
}

NETWORKS = {
    "lan": Network("LAN / 同城 P2P", 20, 50),
    "wan": Network("一般家寬 WAN", 50, 20),
    "far": Network("跨國 / 行動網路", 120, 10),
}

METHODS = {
    # 樸素 Jacobi：文獻上實測加速極弱（約 1.05–1.2x）
    "jacobi": DecodeMethod("vanilla Jacobi", 0.03, 1.15),
    # Lookahead：Jacobi + n-gram pool + 驗證分支
    "lookahead": DecodeMethod("Lookahead (n-gram)", 0.09, 1.90),
    # EAGLE-2 級：需要訓練 draft head，但接受長度高得多
    "eagle": DecodeMethod("EAGLE-2 級 draft", 0.35, 4.50, needs_draft_model=True),
    # 對照組：完全不做平行解碼的普通流水線
    "pipeline": DecodeMethod("naive pipeline", 0.0, 1.0),
}

ACT_BITS = {"fp16": 16, "int8": 8, "int4": 4}

# Petals 堆疊（Python + PyTorch + 原生 CUDA）每個 hop 的固定開銷。
# 這個值不是獨立量來的，是從 Petals 的低延遲測量點（176B, 100Mbps, <5ms RTT
# -> 1.71 steps/s）反解出來的：在該條件下網路傳輸與 RTT 都很小，
# 剩下的時間差只能是每 hop 的固定成本。用它去預測同組的高延遲測量點，
# 算是一次獨立驗證。
PETALS_OVERHEAD_MS = 36.0


# ---------------------------------------------------------------------------
# 核心模擬
# ---------------------------------------------------------------------------


@dataclass
class Result:
    k: int
    acceptance: float
    compute_ms: float
    network_ms: float
    iter_ms: float
    tok_per_s: float
    payload_kb: float

    @property
    def network_share(self) -> float:
        return self.network_ms / self.iter_ms if self.iter_ms else 0.0


def simulate(
    model: Model,
    devices: list[Device],
    net: Network,
    method: DecodeMethod,
    k: int,
    act_bits: int = 8,
) -> Result:
    """模擬一次完整的流水線迭代。

    devices 的長度就是 P（節點數）。模型的層平均分配給各節點。
    """
    p = len(devices)
    stage_frac = 1.0 / p

    # 計算：各節點依序執行（嚴格資料依賴，無法重疊）
    compute_ms = sum(d.stage_time_ms(model, stage_frac, k) for d in devices)

    # 網路：一趟完整 traversal 有 P 個 hop（含把結果送回頭節點）
    payload_bytes = k * model.d_model * act_bits / 8
    network_ms = p * net.hop_ms(payload_bytes)

    iter_ms = compute_ms + network_ms
    acceptance = method.acceptance(k)

    return Result(
        k=k,
        acceptance=acceptance,
        compute_ms=compute_ms,
        network_ms=network_ms,
        iter_ms=iter_ms,
        tok_per_s=acceptance / iter_ms * 1000 if iter_ms else 0.0,
        payload_kb=payload_bytes / 1024,
    )


def best_k(
    model: Model,
    devices: list[Device],
    net: Network,
    method: DecodeMethod,
    k_max: int = 64,
    act_bits: int = 8,
) -> Result:
    """找出讓 tok/s 最大的視窗大小 K。"""
    return max(
        (simulate(model, devices, net, method, k, act_bits) for k in range(1, k_max + 1)),
        key=lambda r: r.tok_per_s,
    )


def pipeline_utilisation(p: int, concurrent_requests: int) -> float:
    """流水線利用率 = M / (M + P - 1)。

    這才是 pipeline bubble 的真正解法：多請求交錯。
    注意它只提升「吞吐」，不降低「單請求延遲」—— 與平行解碼正交。
    """
    return concurrent_requests / (concurrent_requests + p - 1)


def session_survival(p: int, per_node_dropout_per_min: float) -> float:
    """一分鐘內整條流水線都沒有節點掉線的機率。"""
    return (1 - per_node_dropout_per_min) ** p


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------


def make_devices(spec: str, p: int) -> list[Device]:
    """'phone,laptop' -> 循環填滿 P 個節點。"""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise SystemExit("--devices 不能為空")
    for n in names:
        if n not in DEVICES:
            raise SystemExit(f"未知裝置 {n!r}，可用：{', '.join(DEVICES)}")
    return [DEVICES[names[i % len(names)]] for i in range(p)]


def cmd_single(args) -> None:
    model = MODELS[args.model]
    devices = make_devices(args.devices, args.nodes)
    net = Network("自訂", args.rtt, args.mbps, args.overhead)
    act_bits = ACT_BITS[args.act]

    print(f"\n模型 {model.name}  int{model.weight_bits} = {model.weight_gb:.1f} GB")
    print(f"節點 P={args.nodes}（{', '.join(d.name for d in devices)}）")
    print(f"每節點權重 {model.weight_gb / args.nodes:.2f} GB")
    print(f"網路 RTT={args.rtt}ms  上行={args.mbps}Mbps  激活值 {args.act}\n")

    hdr = f"{'解碼方法':<22}{'最佳K':>6}{'接受/迭代':>10}{'計算ms':>9}{'網路ms':>9}{'迭代ms':>9}{'tok/s':>9}"
    print(hdr)
    print("-" * 74)
    for key in ("pipeline", "jacobi", "lookahead", "eagle"):
        m = METHODS[key]
        r = best_k(model, devices, net, m, args.k_max, act_bits)
        print(
            f"{m.name:<22}{r.k:>6}{r.acceptance:>10.2f}"
            f"{r.compute_ms:>9.0f}{r.network_ms:>9.0f}{r.iter_ms:>9.0f}{r.tok_per_s:>9.2f}"
        )

    base = best_k(model, devices, net, METHODS["pipeline"], args.k_max, act_bits)
    eag = best_k(model, devices, net, METHODS["eagle"], args.k_max, act_bits)
    print(f"\n投機解碼相對普通流水線加速：{eag.tok_per_s / base.tok_per_s:.2f}x")
    print(f"網路佔迭代時間比例：{eag.network_share * 100:.0f}%  "
          f"（>50% 代表瓶頸在網路，優化單機計算沒用）")


def cmd_sweep_k(args) -> None:
    model = MODELS[args.model]
    devices = make_devices(args.devices, args.nodes)
    net = Network("自訂", args.rtt, args.mbps, args.overhead)
    act_bits = ACT_BITS[args.act]

    print(f"\n掃描平行視窗 K — {model.name}, P={args.nodes}, "
          f"RTT={args.rtt}ms, {args.mbps}Mbps\n")
    print(f"{'K':>4} | " + " | ".join(f"{METHODS[m].name:>20}" for m in
                                      ("jacobi", "lookahead", "eagle")))
    print("-" * 76)
    for k in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
        if k > args.k_max:
            break
        cells = []
        for key in ("jacobi", "lookahead", "eagle"):
            r = simulate(model, devices, net, METHODS[key], k, act_bits)
            cells.append(f"{r.tok_per_s:>12.2f} tok/s")
        print(f"{k:>4} | " + " | ".join(f"{c:>20}" for c in cells))

    print("\n最佳 K：")
    for key in ("jacobi", "lookahead", "eagle"):
        r = best_k(model, devices, net, METHODS[key], args.k_max, act_bits)
        print(f"  {METHODS[key].name:<22} K={r.k:<3} -> {r.tok_per_s:.2f} tok/s")

    # 敏感度：結論對「接受率」這個最沒把握的假設有多敏感？
    #
    # 注意這裡問的不是「最佳 K 是不是 > 1」—— 那個問題會被每 hop 固定開銷
    # 左右（開銷大時，把 K 撐大來攤提開銷總是划算，即使接受率很爛）。
    # 決策相關的問題是：相對於完全不做平行解碼，到底快了幾倍？
    base = best_k(model, devices, net, METHODS["pipeline"], args.k_max, act_bits)
    print(f"\n敏感度分析 — 相對普通流水線（{base.tok_per_s:.2f} tok/s）的加速倍數")
    print("  問的是「值不值得為了它增加整套投機/驗證邏輯的複雜度」\n")
    print(f"  {'接受率上限':>12}{'最佳K':>7}{'tok/s':>9}{'加速':>8}   判斷")
    for ceiling in (1.10, 1.15, 1.25, 1.50, 2.00, 3.00, 4.50):
        m = DecodeMethod("試算", 0.35, ceiling)
        r = best_k(model, devices, net, m, args.k_max, act_bits)
        sp = r.tok_per_s / base.tok_per_s
        if sp < 1.15:
            verdict = "✗ 不值得，複雜度換不到效能"
        elif sp < 1.5:
            verdict = "~ 邊際，看實作成本"
        else:
            verdict = "✓ 明顯值得"
        print(f"  {ceiling:>12.2f}{r.k:>7}{r.tok_per_s:>9.2f}{sp:>7.2f}x   {verdict}")
    print("\n  文獻估計值：vanilla Jacobi ~1.15、Lookahead ~1.9、EAGLE-2 ~4.5")
    print("  -> 結論不依賴精確數字：只要 vanilla Jacobi 的接受率停在 1.2 以下，")
    print("     它就換不到足以支撐這套架構的加速；而 draft model 路線即使")
    print("     只達到文獻值的一半（~2.5），仍然明顯值得。")


def cmd_sweep_nodes(args) -> None:
    model = MODELS[args.model]
    act_bits = ACT_BITS[args.act]
    print(f"\n掃描節點數 P — {model.name} int{model.weight_bits} "
          f"= {model.weight_gb:.1f} GB，投機解碼\n")
    print(f"{'P':>3}{'每節點GB':>10}  " +
          "".join(f"{n.name:>20}" for n in NETWORKS.values()))
    print("-" * 76)
    for p in (2, 3, 4, 6, 8, 12, 16):
        devices = make_devices(args.devices, p)
        cells = []
        for net in NETWORKS.values():
            r = best_k(model, devices, net, METHODS["eagle"], args.k_max, act_bits)
            cells.append(f"{r.tok_per_s:>13.2f} tok/s")
        print(f"{p:>3}{model.weight_gb / p:>10.2f}  " + "".join(f"{c:>20}" for c in cells))
    print("\n節點越多，每節點記憶體壓力越小，但每趟 traversal 的 hop 數線性增加。")
    print("最佳 P 通常是「剛好塞得下」的最小值。")


def cmd_roofline(args) -> None:
    print("\n免費平行視窗 K* — 超過這個值，多算一個位置就要真的付 FLOPs\n")
    print("  K* = (讀權重時間) / (每位置 FLOPs 時間)，與切分方式無關\n")
    for mkey in ("8b", "32b", "70b"):
        model = MODELS[mkey]
        print(f"{model.name} (int{model.weight_bits}, {model.weight_gb:.1f} GB)")
        print(f"  {'裝置':<22}{'讀權重ms':>11}{'每位置ms':>11}{'K*':>8}")
        for d in DEVICES.values():
            t_mem = model.weight_bytes / (d.bandwidth_gbps * 1e9) * 1000
            t_fl = 2 * model.params / (d.tflops * 1e12) * 1000
            print(f"  {d.name:<22}{t_mem:>11.1f}{t_fl:>11.2f}{d.roofline_k_star(model):>8.1f}")
        print()
    print("關鍵：手機級裝置 K* 只有個位數。任何調度器都必須把 K 綁在該節點的 K* 上，")
    print("否則「平行解碼是免費的」這個前提就不成立了。")


def cmd_churn(args) -> None:
    print("\n流水線利用率 = M/(M+P-1)，M = 同時在途的請求數")
    print("（這才是 pipeline bubble 的解法 —— 與平行解碼正交）\n")
    ms = [1, 2, 4, 8, 16, 32]
    print(f"{'P':>4} |" + "".join(f"{f'M={m}':>9}" for m in ms))
    print("-" * 60)
    for p in (2, 4, 8, 16):
        print(f"{p:>4} |" + "".join(f"{pipeline_utilisation(p, m) * 100:>8.0f}%" for m in ms))

    print("\n\n節點流失：一分鐘內整條鏈路存活的機率\n")
    print(f"{'每節點每分鐘掉線率':>20} |" + "".join(f"{f'P={p}':>10}" for p in (2, 4, 8, 16)))
    print("-" * 64)
    for rate in (0.01, 0.02, 0.05, 0.10):
        cells = "".join(f"{session_survival(p, rate) * 100:>9.1f}%" for p in (2, 4, 8, 16))
        print(f"{rate * 100:>19.0f}% |" + cells)

    print("\n冗餘成本：要容忍掉線就必須熱備（接手者得「已經載入」該層權重）")
    for r in (1, 2, 3):
        print(f"  冗餘係數 R={r}: P=8 需要 {8 * r} 個線上節點服務 1 個請求，"
              f"有效算力利用率 {100 / r:.0f}%")


def cmd_payload(args) -> None:
    print("\n每個 pipeline hop 的激活值傳輸量 [K, d_model]\n")
    for mkey in ("8b", "32b", "70b"):
        m = MODELS[mkey]
        print(f"{m.name} (d_model={m.d_model})")
        print(f"  {'K':>4}" + "".join(f"{b:>12}" for b in ACT_BITS))
        for k in (1, 8, 16, 32, 64):
            cells = "".join(f"{k * m.d_model * b / 8 / 1024:>9.1f} KB"
                            for b in ACT_BITS.values())
            print(f"  {k:>4}" + cells)
        print()
    print("⚠ WebRTC DataChannel 單則訊息上限 256 KiB —— 大 K + fp16 會超過，需應用層分片。")
    print("⚠ 激活值有離群通道（中位數的 20–100 倍），per-tensor INT8 會毀掉品質。")
    print("  必須逐通道/分組量化，離群通道保留 fp16（LLM.int8 式）。")


def cmd_cold_start(args) -> None:
    print("\n冷啟動：每個節點要下載多少權重才能開始服務\n")
    for mkey in ("8b", "32b", "70b"):
        m = MODELS[mkey]
        print(f"{m.name} int{m.weight_bits} = {m.weight_gb:.1f} GB")
        print(f"  {'P':>4}{'每節點GB':>10}" +
              "".join(f"{f'{s}Mbps':>12}" for s in (100, 30, 10)))
        for p in (4, 8, 16):
            per = m.weight_gb / p
            cells = "".join(f"{per * 8 * 1000 / s / 60:>9.1f} 分" for s in (100, 30, 10))
            print(f"  {p:>4}{per:>10.2f}" + cells)
        print()
    print("這是真正的採用門檻：沒人會為了聊天等 20 分鐘下載。")
    print("對策：節點長駐 + OPFS 持久快取；v1 先用小模型把架構跑通。")


def cmd_validate(args) -> None:
    """對照已知的真實世界測量值，檢查模型是否落在正確量級。

    重點：必須用「與該次測量相符的參數」去比，而不是拿最佳情況的頭條數字
    配上悲觀的裝置假設。每個 case 都註明測量條件出處。
    """
    print("\n模型校驗 —— 對照 Petals 公開實測值")
    print("（Petals 節點是原生 GPU，非 WebGPU；且未使用投機解碼，")
    print("  故校驗一律用 pipeline / lookahead 設定，不用 EAGLE）\n")

    cases = [
        dict(
            label="BLOOM-176B，100 Mbit/s + 100ms 延遲 -> 1.23 steps/s",
            note="Petals 論文的頻寬/延遲敏感度實驗，條件明確，是最適合校驗的一筆",
            mkey="176b", p=12, devices="desktop", rtt=100, mbps=100, overhead=PETALS_OVERHEAD_MS,
            method="pipeline", observed=1.23,
        ),
        dict(
            label="BLOOM-176B，100 Mbit/s + <5ms 延遲 -> 1.71 steps/s",
            note="同一組實驗的低延遲端；用來檢查模型對 RTT 的敏感度方向正確",
            mkey="176b", p=12, devices="desktop", rtt=5, mbps=100, overhead=PETALS_OVERHEAD_MS,
            method="pipeline", observed=1.71,
        ),
    ]

    ok = True
    for c in cases:
        model = MODELS[c["mkey"]]
        devices = make_devices(c["devices"], c["p"])
        net = Network("", c["rtt"], c["mbps"], c["overhead"])
        r = best_k(model, devices, net, METHODS[c["method"]], args.k_max, 8)
        ratio = r.tok_per_s / c["observed"]
        passed = 0.5 <= ratio <= 2.0
        ok = ok and passed
        print(f"  {c['label']}")
        print(f"    條件：P={c['p']} × {DEVICES[c['devices']].name}, "
              f"RTT={c['rtt']}ms, {c['mbps']}Mbps")
        print(f"    {c['note']}")
        print(f"    預測 {r.tok_per_s:.2f} vs 實測 {c['observed']:.2f} tok/s"
              f"  (比值 {ratio:.2f})  {'✓ 通過' if passed else '✗ 偏離超過 2 倍'}\n")

    # petals.dev 的「70B 最高 6 tok/s」沒有載明網路條件，拿它做 pass/fail
    # 等於用自己猜的參數驗證自己的模型。改成反解：什麼條件才達得到？
    print("  [參考，不計入校驗] petals.dev 宣稱 Llama-2-70B 單批最高 6 tok/s，")
    print("  但未載明網路條件。反解達成該數字所需的條件：\n")
    model = MODELS["70b"]
    for p_ in (2, 4, 6, 8):
        devices = make_devices("desktop", p_)
        hit = None
        for rtt in range(0, 101, 5):
            net = Network("", rtt, 200, PETALS_OVERHEAD_MS)
            r = best_k(model, devices, net, METHODS["pipeline"], args.k_max, 8)
            if r.tok_per_s >= 6.0:
                hit = rtt
        msg = f"RTT <= {hit}ms" if hit is not None else "任何 RTT 都達不到（固定開銷已超標）"
        print(f"    P={p_}: {msg}")
    print("\n  -> 6 tok/s 只在「短鏈 + 低延遲」下成立。這反過來說明本專案")
    print("     為什麼要盡量壓低 P，以及為什麼每 hop 固定開銷這麼關鍵。\n")

    print("校驗標準：預測值需落在實測值的 0.5x–2x 之間。")
    print("模型只用來做數量級決策（選 P、選 K、判斷瓶頸在網路還是計算），")
    print("不用來做精確預測；真實數字要等 M2 里程碑實測。")
    print(f"\n結果：{'通過' if ok else '未通過 —— 模型參數或校驗條件需重新檢視'}")
    raise SystemExit(0 if ok else 1)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EdgeCascadeLLM 分散式推理效能數值模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    ap.add_argument("--model", default="70b", choices=sorted(MODELS))
    ap.add_argument("--nodes", type=int, default=8, help="節點數 P")
    ap.add_argument("--devices", default="laptop",
                    help=f"逗號分隔，循環填滿 P 個節點。可用：{', '.join(DEVICES)}")
    ap.add_argument("--rtt", type=float, default=50, help="節點間 RTT (ms)")
    ap.add_argument("--mbps", type=float, default=20, help="上行頻寬 (Mbps)")
    ap.add_argument("--act", default="int8", choices=sorted(ACT_BITS),
                    help="激活值傳輸精度")
    ap.add_argument("--overhead", type=float, default=15.0,
                    help="每個 pipeline hop 的固定軟體開銷 (ms)：序列化 + 框架 dispatch "
                         "+ WebGPU kernel launch。低延遲情境下會成為主導項。")
    ap.add_argument("--k-max", type=int, default=64, help="搜尋的最大平行視窗")
    ap.add_argument("--sweep", choices=("k", "nodes"), help="掃描某個維度")
    ap.add_argument("--roofline", action="store_true", help="各裝置的免費平行視窗 K*")
    ap.add_argument("--churn", action="store_true", help="流水線利用率與節點流失")
    ap.add_argument("--payload", action="store_true", help="每 hop 的傳輸量")
    ap.add_argument("--cold-start", action="store_true", help="權重下載時間")
    ap.add_argument("--validate", action="store_true", help="對照 Petals 實測值校驗")
    args = ap.parse_args()

    if args.validate:
        cmd_validate(args)
    elif args.roofline:
        cmd_roofline(args)
    elif args.churn:
        cmd_churn(args)
    elif args.payload:
        cmd_payload(args)
    elif args.cold_start:
        cmd_cold_start(args)
    elif args.sweep == "k":
        cmd_sweep_k(args)
    elif args.sweep == "nodes":
        cmd_sweep_nodes(args)
    else:
        cmd_single(args)


if __name__ == "__main__":
    main()
