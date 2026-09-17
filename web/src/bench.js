/**
 * 效能量測 —— 回答三個目前只能靠猜的問題。
 *
 * 這些量測沒辦法在開發容器裡做（那裡 navigator.gpu 根本不存在），
 * 所以做成一個網頁讓實際裝置的擁有者跑，把 JSON 貼回來校正 bench/model.py。
 *
 * Q1  每個 hop 的固定開銷是多少？
 *     bench/model.py 目前猜 15 ms，那是從 Petals 的 36 ms/hop 推的，
 *     但 Petals 是 Python + CUDA，瀏覽器 + WebGPU 完全是另一回事。
 *     這個值決定 P（節點數）的上限，是整個架構最敏感的參數之一。
 *
 * Q2  一次算多個位置划不划算？
 *     原本問的是「roofline 轉折點 K* 在哪」（理論推算手機約 5.5）。
 *     但實測發現 135M 模型在真實裝置上根本沒有轉折點 —— 總耗時對 k
 *     是一條直線（r²=0.9994），78.7% 的時間花在固定 dispatch 開銷上。
 *     所以改問「多算一個位置的邊際成本是多少」，見 analyseKStar()。
 *
 * ⚠ 這兩題共用同一次掃描，而且刻意如此 —— 見 sweepSeqLen() 的說明。
 *   舊版用「把 4 段收成 2 段」的方式製造不同 hop 數來量 Q1，那是錯的：
 *   收起來的流水線會跳過中間的層，所以時間差裡同時混了「少幾個 hop」和
 *   「少算幾層」，兩者在等大小切分下完全共線，事後無法分離。
 *
 * Q11 數值等價性是否逐 execution provider 成立？
 *     已知原生 ORT 誤差 7.7e-05、ORT-Web WASM 1.97e-4。WebGPU 未知。
 *     若不同節點用不同 EP 會導致流水線分歧，就必須強制全網統一。
 */

import { Pipeline, detectProviders, compareToReference } from './runner.js';

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function randomIds(seq, vocab) {
  const ids = new BigInt64Array(seq);
  for (let i = 0; i < seq; i++) ids[i] = BigInt(Math.floor(Math.random() * vocab));
  return ids;
}

/** 環境資訊：adapter、限制、功能、儲存配額。 */
export async function probeEnvironment({ powerPreference } = {}) {
  const providers = await detectProviders(powerPreference ? { powerPreference } : {});
  const env = {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemoryGB: navigator.deviceMemory ?? null,
    crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    providers,
  };
  if (navigator.storage?.estimate) {
    const { quota, usage } = await navigator.storage.estimate();
    env.storage = { quotaBytes: quota, usageBytes: usage };
  }
  return env;
}

/** 最小平方直線擬合。回傳斜率、截距與 r²（擬合品質，1 = 完美）。 */
function linearFit(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (ys[i] - (intercept + slope * xs[i])) ** 2;
    ssTot += (ys[i] - my) ** 2;
  }
  return { slope, intercept, r2: ssTot === 0 ? 1 : 1 - ssRes / ssTot };
}

/** 預設的序列長度階梯。取到 64 是因為 32 曾經不夠 —— 實測在 32 還在降。 */
const DEFAULT_KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/**
 * 掃描序列長度，同時取得 Q1 與 Q2 需要的全部原始資料。
 *
 * 一條**完整的**流水線（層數固定不變），只改每次餵進去的位置數 k。
 * 每個 k 記兩件事：整條流水線的總時間，以及每個 shard 各自的 session.run 時間
 * （Pipeline.run() 本來就逐 shard 記了 computeMs）。
 *
 * 有了這份資料：
 *   - 對 totalMs vs k 擬合 -> Q2 的 K* 與「固定成本佔比」
 *   - 對每個 shard 的 computeMs vs k 擬合，**截距就是該 shard 的固定 dispatch 成本**
 *     -> Q1，而且完全沒有混淆：層數從頭到尾沒變過。
 */
export async function sweepSeqLen(pipeline, manifest, {
  ks = DEFAULT_KS, maxK = 64, repeats = 3, onProgress,
} = {}) {
  const list = ks.filter((v, i, a) => v <= maxK && a.indexOf(v) === i).sort((a, b) => a - b);
  const vocab = manifest.vocab_size;
  const points = [];

  for (const [i, k] of list.entries()) {
    const totals = [];
    const perShard = [];
    for (let r = 0; r < repeats; r++) {
      const ids = randomIds(k, vocab);
      const t0 = performance.now();
      const out = await pipeline.run(ids, k);
      totals.push(performance.now() - t0);
      out.hops.forEach((h, j) => {
        (perShard[j] ??= []).push(h.computeMs);
      });
    }
    const ms = median(totals);
    points.push({
      k,
      totalMs: ms,
      msPerPosition: ms / k,
      shardMs: perShard.map(median),
    });
    onProgress?.({ done: i + 1, total: list.length, k });
  }
  return points;
}

/** 投機解碼實際會用到的猜測長度。超過這個範圍 draft 模型也猜不準。 */
const VERIFY_WINDOWS = [1, 4, 8, 16];

/**
 * Q2 — 一次算多個位置到底划不划算。
 *
 * 原本的問法是「roofline 轉折點 K* 在哪」：batch=1 解碼是記憶體頻寬受限的，
 * 一次算 K 個位置時權重只讀一次、FLOPs 變 K 倍，所以 K 小的時候幾乎「免費」，
 * 超過 K* 之後轉為計算受限。K* = 每位置耗時最低的那個 K。
 *
 * **實測發現這個問法在小模型上得不到答案。** 第一台真實裝置的資料是
 * `總耗時 = 27.95 + 4.23 × k`，r² = 0.9994 —— 一條完美直線。
 * 那代表 `每位置耗時 = 27.95/k + 4.23` 對所有 k 單調遞減，永遠沒有最小值，
 * 所以「K* = 掃描上限」不管上限拉到 64 還是 1024 都會成立。
 * 不是掃描不夠遠，是 135M 太小、78.7% 的時間花在 dispatch 上，
 * 記憶體頻寬根本不是瓶頸，roofline 的前提不成立。
 *
 * 所以這裡改報**投機解碼真正在意的東西**：
 *
 *   `marginalMsPerPosition` 多算一個位置的邊際成本（= 擬合斜率）
 *   `firstPositionMs`       第一個位置的成本（含整條流水線的固定開銷）
 *   `verifyWindow`          猜 γ 個字一次驗證，要花單一 token 的幾倍時間
 *   `knee`                  範圍內到底有沒有轉折點，而不是含糊的 censored
 *
 * γ=8 花 60.6 ms、是單一 token 35.5 ms 的 1.71 倍卻產出 8 個位置 ——
 * 這個數字直接回答「投機解碼划不划算」，而一個假的 K* 不會。
 */
export function analyseKStar(points) {
  // 嚴格小於：打平的區間會保留較小的 K，所以 kStar 落在上限就真的是「還在降」。
  let best = points[0];
  for (const p of points) if (p.msPerPosition < best.msPerPosition) best = p;

  const k1 = points.find((p) => p.k === 1);
  const maxKTested = points[points.length - 1].k;
  const censored = best.k === maxKTested;
  const fit = linearFit(points.map((p) => p.k), points.map((p) => p.totalMs));
  const fixedFractionAtK1 = fit && k1 ? fit.intercept / k1.totalMs : null;

  // censored 只說「最低點落在邊界」，沒說為什麼。這裡把兩種原因分開：
  //   observed      範圍內有真正的轉折點（最低點在內部）
  //   none-observed 資料是一條直線 —— 沒有轉折點可找，拉高上限也沒用
  //   unknown       擬合太差，說不準
  let knee;
  if (!censored) knee = 'observed';
  else if (fit && fit.r2 >= 0.99) knee = 'none-observed';
  else knee = 'unknown';

  // 要讓每位置耗時逼近漸近線到某個百分比，需要多大的 K。
  // 由 fixed/k + slope <= (1 + pct) * slope 解出 k >= fixed / (pct * slope)。
  const amortisation = [];
  if (fit && fit.slope > 0 && fit.intercept > 0) {
    for (const withinPct of [25, 10]) {
      amortisation.push({
        withinPct,
        k: Math.ceil(fit.intercept / ((withinPct / 100) * fit.slope)),
      });
    }
  }

  const verifyWindow = VERIFY_WINDOWS
    .map((gamma) => {
      const pt = points.find((p) => p.k === gamma);
      if (!pt || !k1) return null;
      return {
        gamma,
        totalMs: pt.totalMs,
        msPerPosition: pt.msPerPosition,
        vsSingleToken: pt.totalMs / k1.totalMs,
      };
    })
    .filter(Boolean);

  return {
    points,
    // 以下兩個是新的主角
    firstPositionMs: k1?.totalMs ?? null,
    marginalMsPerPosition: fit?.slope ?? null,
    verifyWindow,
    amortisation,
    knee,
    // kStar 保留是為了和舊報告對得上，但 knee !== 'observed' 時它沒有意義
    kStar: best.k,
    maxKTested,
    censored,
    msPerPositionAtK1: k1?.msPerPosition ?? null,
    msPerPositionAtKStar: best.msPerPosition,
    speedupAtKStar: k1 ? k1.msPerPosition / best.msPerPosition : null,
    fit: fit
      ? { fixedMs: fit.intercept, msPerPositionSlope: fit.slope, r2: fit.r2 }
      : null,
    fixedFractionAtK1,
    regime: fixedFractionAtK1 == null
      ? null
      : (fixedFractionAtK1 > 0.5 ? 'dispatch-bound' : 'memory-bound'),
  };
}

/**
 * Q1 — 每個 hop 的固定開銷，從同一次掃描導出。
 *
 * 對每個 shard 擬合 `computeMs = 固定成本 + 斜率 × k`。截距就是那一段
 * 「不管算幾個位置都要付」的成本：張量在 JS 與 wasm/GPU 之間搬進搬出、
 * session.run 的 dispatch。
 *
 * 這是**本機**成本。真實部署還要再加上網路往返，那一段由 bench/model.py
 * 另外算（--overhead 旗標）。這裡量的是「就算網路零延遲也躲不掉」的部分。
 *
 * 截距偶爾會是小負值 —— 那是擬合噪音，不是真的負成本，所以照實回報 r²
 * 讓讀的人自己判斷可信度，而不是偷偷夾到 0。
 */
export function deriveHopOverhead(points, manifest) {
  const ks = points.map((p) => p.k);
  const nShards = points[0]?.shardMs?.length ?? 0;
  const perShard = [];

  for (let i = 0; i < nShards; i++) {
    const fit = linearFit(ks, points.map((p) => p.shardMs[i]));
    perShard.push({
      index: manifest.shards?.[i]?.index ?? i,
      layers: manifest.shards?.[i]?.layers ?? null,
      fixedMs: fit?.intercept ?? null,
      slopeMsPerPosition: fit?.slope ?? null,
      r2: fit?.r2 ?? null,
    });
  }

  const fixed = perShard.map((x) => x.fixedMs).filter((v) => v != null);
  const totalFit = linearFit(ks, points.map((p) => p.totalMs));
  const sumShardFixedMs = fixed.length ? fixed.reduce((a, b) => a + b, 0) : null;
  const pipelineFixedMs = totalFit?.intercept ?? null;

  return {
    method: 'per-shard-intercept',
    shards: nShards,
    perShard,
    // 「每段」而不是「每 hop」—— 4 段之間只有 3 個交接，但要付 4 次
    // session.run 的固定成本。用 hop 當單位會少算一次，P 大的時候差很多。
    //
    // 中位數而非平均：第一段含 embedding、最後一段含 lm_head，
    // 兩端本來就比中間貴（實測第 4 段是其他段的兩倍），平均會被它們拉走。
    fixedMsPerShard: fixed.length ? median(fixed) : null,
    sumShardFixedMs,
    pipelineFixedMs,
    // 整條流水線的固定成本減掉各 shard session.run 的固定成本，
    // 剩下的是 JS 這一層的開銷：建張量、量化編解碼、await 排程。
    glueMs: pipelineFixedMs != null && sumShardFixedMs != null
      ? pipelineFixedMs - sumShardFixedMs
      : null,
  };
}

/**
 * Q11 — 同一組 shard 在不同 EP 下的數值差異。
 *
 * 如果 WebGPU 與 WASM 算出的 logits 差太多，混用不同 EP 的節點
 * 就會讓流水線產生分歧 —— 那會是很難查的錯：輸出看起來合理、只是慢慢偏掉。
 */
export async function compareProviders(ort, manifest, baseUrl, reference, eps, { repeats = 3 } = {}) {
  const results = [];
  let firstLogits = null;

  for (const ep of eps) {
    const pipe = new Pipeline(ort, manifest, baseUrl, { ep, scheme: 'none' });
    const entry = { ep };
    try {
      await pipe.load();
      const ids = BigInt64Array.from(reference.input_ids.flat().map(BigInt));
      const seq = reference.input_ids[0].length;

      // 第一次執行和之後的完全不是同一件事：WebGPU 要編譯 shader、
      // 兩邊都要做 kernel 特化與記憶體配置。實測 WebGPU 首次 269 ms，
      // 暖機後同樣長度只要 93 ms —— 把兩者混在一起報，會讓人得出
      // 「WebGPU 比 WASM 慢」這個錯誤結論。所以分開量、分開報。
      //
      // firstRunMs 本身也有用：那是一個節點加入網路後的暖機成本（M3 會在意）。
      const t0 = performance.now();
      const out = await pipe.run(ids, seq);
      entry.firstRunMs = performance.now() - t0;

      // 正確性比對用第一次的輸出，趁它還沒被後面的執行蓋掉
      Object.assign(entry, compareToReference(out.logits, out.dims, reference));

      const times = [];
      for (let r = 0; r < repeats; r++) {
        const t = performance.now();
        await pipe.run(ids, seq);
        times.push(performance.now() - t);
      }
      entry.ms = median(times);
      entry.warmupMs = entry.firstRunMs - entry.ms;

      // 也和第一個成功的 EP 互比 —— 那才是「不同節點會不會分歧」的直接答案
      if (firstLogits === null) {
        firstLogits = out.logits;
      } else {
        let d = 0;
        for (let i = 0; i < firstLogits.length; i++) {
          d = Math.max(d, Math.abs(out.logits[i] - firstLogits[i]));
        }
        entry.maxDiffVsFirstEp = d;
      }
      entry.ok = true;
    } catch (e) {
      entry.ok = false;
      entry.error = String(e).slice(0, 300);
    } finally {
      await pipe.dispose();
    }
    results.push(entry);
  }
  return results;
}

/** 把量測結果收成一份可貼回的報告。 */
export function buildReport(parts) {
  return {
    // v2：hopOverhead 改用逐 shard 截距法（v1 的子集合流水線量測是混淆的），
    //     kStar 多了 censored / regime / fit 欄位。
    // v3：kStar 改以邊際成本與驗證視窗為主（censored 本身回答不了問題），
    //     providerComparison 分開冷啟動與穩態。
    schemaVersion: 3,
    generatedAt: new Date().toISOString(),
    ...parts,
  };
}
