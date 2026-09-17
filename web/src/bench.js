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
 * Q2  平行視窗 K* 有多大？
 *     理論值：K* = 讀權重時間 / 每位置 FLOPs 時間。手機推算約 5.5。
 *     若實測 K* < 3，投機解碼的效益空間就太小，整條路線要重新評估。
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
export async function probeEnvironment() {
  const providers = await detectProviders();
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

/**
 * Q2 — 掃描平行視窗 K，找出 roofline 轉折點。
 *
 * 原理：batch=1 解碼是記憶體頻寬受限的，所以一次算 K 個位置時，
 * 權重只讀一次、FLOPs 變 K 倍。K 小的時候時間幾乎不變（「免費」），
 * 超過 K* 之後轉為計算受限，時間開始隨 K 線性成長。
 *
 * 作法：量每個 K 的「每位置耗時」。免費區間內這個值會隨 K 下降，
 * 到了 K* 之後會打平。取每位置耗時最低的 K 當作 K*。
 */
export async function measureKStar(pipeline, manifest, { maxK = 32, repeats = 3 } = {}) {
  const points = [];
  const vocab = manifest.vocab_size;

  for (const k of [1, 2, 3, 4, 6, 8, 12, 16, 24, maxK].filter((v, i, a) =>
    v <= maxK && a.indexOf(v) === i,
  )) {
    const times = [];
    for (let r = 0; r < repeats; r++) {
      const ids = randomIds(k, vocab);
      const t0 = performance.now();
      await pipeline.run(ids, k);
      times.push(performance.now() - t0);
    }
    const ms = median(times);
    points.push({ k, totalMs: ms, msPerPosition: ms / k });
  }

  // K* = 每位置耗時最低的那個 K。再往上加 K 就開始付真實計算成本。
  let best = points[0];
  for (const p of points) if (p.msPerPosition < best.msPerPosition) best = p;

  // 免費程度：K=1 與 K* 的每位置耗時比值。比值越大代表平行視窗越划算。
  const k1 = points.find((p) => p.k === 1);
  return {
    points,
    kStar: best.k,
    msPerPositionAtK1: k1?.msPerPosition ?? null,
    msPerPositionAtKStar: best.msPerPosition,
    speedupAtKStar: k1 ? k1.msPerPosition / best.msPerPosition : null,
  };
}

/**
 * Q1 — 每個 hop 的固定開銷。
 *
 * 把同一個模型切成不同段數跑，總計算量幾乎不變，多出來的時間就是
 * 「每多經過一個 shard 邊界要付的固定成本」：張量在 JS 與 wasm/GPU
 * 之間搬進搬出、session.run 的 dispatch、量化編碼解碼。
 *
 * 真實部署還要再加上網路往返，但那一段 bench/model.py 已經另外算了。
 * 這裡量的是「就算網路零延遲也躲不掉」的那部分。
 */
export async function measureHopOverhead(pipelines, { seq = 8, repeats = 5 } = {}) {
  const rows = [];
  for (const { shards, pipeline, manifest } of pipelines) {
    const times = [];
    for (let r = 0; r < repeats; r++) {
      const ids = randomIds(seq, manifest.vocab_size);
      const t0 = performance.now();
      await pipeline.run(ids, seq);
      times.push(performance.now() - t0);
    }
    rows.push({ shards, hops: shards - 1, totalMs: median(times) });
  }

  // 對 (hops, totalMs) 做最小平方線性迴歸，斜率就是每 hop 的固定開銷
  let overheadMsPerHop = null;
  if (rows.length >= 2) {
    const n = rows.length;
    const sx = rows.reduce((a, r) => a + r.hops, 0);
    const sy = rows.reduce((a, r) => a + r.totalMs, 0);
    const sxy = rows.reduce((a, r) => a + r.hops * r.totalMs, 0);
    const sxx = rows.reduce((a, r) => a + r.hops * r.hops, 0);
    const denom = n * sxx - sx * sx;
    if (denom !== 0) overheadMsPerHop = (n * sxy - sx * sy) / denom;
  }
  return { rows, overheadMsPerHop };
}

/**
 * Q11 — 同一組 shard 在不同 EP 下的數值差異。
 *
 * 如果 WebGPU 與 WASM 算出的 logits 差太多，混用不同 EP 的節點
 * 就會讓流水線產生分歧 —— 那會是很難查的錯：輸出看起來合理、只是慢慢偏掉。
 */
export async function compareProviders(ort, manifest, baseUrl, reference, eps) {
  const results = [];
  let firstLogits = null;

  for (const ep of eps) {
    const pipe = new Pipeline(ort, manifest, baseUrl, { ep, scheme: 'none' });
    const entry = { ep };
    try {
      await pipe.load();
      const ids = BigInt64Array.from(reference.input_ids.flat().map(BigInt));
      const seq = reference.input_ids[0].length;

      const t0 = performance.now();
      const out = await pipe.run(ids, seq);
      entry.ms = performance.now() - t0;
      Object.assign(entry, compareToReference(out.logits, out.dims, reference));

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
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    ...parts,
  };
}
