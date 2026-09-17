/**
 * `web/src/wire.js` 的單元測試 —— §4.5 制定的 frame 格式。
 *
 * 這支測試刻意完全不碰瀏覽器、不碰 WebRTC、不需要 `npm run build`：
 *   node --test test/wire.test.mjs
 * 理由是 §4.5.1 的分層 —— 分片在 frame 底下，所以 frame 層可以純粹用位元組驗。
 * M1/M2 的教訓是「先確定算得對，再加網路」；把格式測試綁在瀏覽器上，
 * 之後查 WebRTC 的問題時就分不清是傳錯還是編錯了。
 *
 * 七件事在這裡被釘死：
 *   1. fp16 編解碼的位元樣式（線路上兩端必須逐位元一致）
 *   2. round-trip 的位元組穩定性
 *   3. 與 M4 品質模擬器 `quant.js` 的數值綁定（§4.5.5，**不是無條件相等**）
 *   4. 位元組帳與 `bench/wire_size.py` 一致 —— 這一項才讓 §4.5 的表可信
 *   5. §4.5.2 的**絕對位元組位移**（第 8 節）—— 跨實作解析唯一的依據
 *   6. §4.5.5b 的飽和規則：有限的輸入不准變成 Inf/NaN，非有限的輸入不准被吞掉
 *   7. 壞掉的 header 必須丟出說得出下一步的錯誤，而不是回傳垃圾
 *
 * 第 5 項是對抗性審查加上的，而且它的理由值得記著：在它之前，把 writer 與 reader
 * 裡的欄位位移**同時**改掉，16 個測試全綠 —— 因為沒有任何一個測試站在「別家的
 * 解析器」那一邊。encode 與 peekHeader 互相同意，不等於 frame 符合 §4.5.2。
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

import {
  encode,
  decode,
  peekHeader,
  frameHeaderSize,
  SCHEME_IDS,
  SCHEME_NAMES,
  FRAME_MAGIC,
  FRAME_VERSION,
  floatToFp16,
  fp16ToFloat,
} from '../src/wire.js';

import { SCHEMES, perChannel, groupWise, fp16Roundtrip } from '../src/quant.js';

const repoRoot = path.resolve(import.meta.dirname, '..', '..');
const benchDir = path.join(repoRoot, 'bench');

/** §4.5.2 的名稱 -> quant.js / wire_size.py 的 key。三個地方三種叫法是既成事實。 */
const QUANT_KEY = {
  none: 'none',
  'per-channel': 'per-channel',
  'group-64': 'group-64',
  'per-ch+outlier': 'wire',
};

const OUTLIER_FRAC = 0.03; // quant.js:149 / wire_size.py:41
const QMAX = 127;

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

/**
 * 自己寫的 LCG（數值取自 Numerical Recipes）。
 *
 * 不用 `Math.random()`：不可重現的測試在這裡等於沒有測試 —— 量化的邊界效應
 * （值剛好落在 .5 捨入邊界）本來就是低機率事件，跑一次綠不代表下一次綠。
 * 用 Math.imul 是因為 32 位乘法在 double 裡會溢出精度。
 */
function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * 造一組有「離群 channel」結構的假激活值。
 *
 * 真實的 hidden state 不是均勻雜訊 —— §4.2 的整個結論（離群通道保 fp16 投報率極高）
 * 就是建立在「巨值集中在少數 channel」上。用純均勻雜訊測 per-ch+outlier
 * 等於在測一個不存在的情境，挑出來的離群集合會隨機漂移。
 */
function makeTensor(dModel, qLen, seed) {
  const rnd = lcg(seed);
  const x = new Float32Array(dModel * qLen);
  const heavy = new Float32Array(dModel);
  for (let c = 0; c < dModel; c++) heavy[c] = c % 31 === 0 ? 40 : 1;
  for (let t = 0; t < qLen; t++) {
    for (let c = 0; c < dModel; c++) {
      x[t * dModel + c] = (rnd() * 2 - 1) * 4 * heavy[c];
    }
  }
  return x;
}

/** 逐元素嚴格相等。用 `!==` 而不是 Object.is —— 見下面那段註解。 */
function assertExact(actual, expected, msg) {
  assert.equal(actual.length, expected.length, `${msg}：長度不同`);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(!Number.isNaN(actual[i]), `${msg}：第 ${i} 個值是 NaN`);
    if (actual[i] !== expected[i]) {
      assert.fail(`${msg}：第 ${i} 個值 ${actual[i]} !== ${expected[i]}`);
    }
  }
}

/**
 * ⚠ 上面刻意用 `!==` 而不是 `Object.is`，因為 -0 是真的會出現的。
 *
 * `quant.js` 的 `qdq` 在 `Math.round(-0.3)` 時拿到 -0，回傳 `-0 * scale` = -0，
 * 存進 Float32Array 仍然是 -0。線路這邊 -0 經過 Int8Array 會變成 +0
 * （TypedArray 存整數 -0 就是 0），解回來是 +0。
 * 數值上 -0 === +0，下游算 argmax / softmax 完全不受影響，
 * 所以這不是 bug，用 Object.is 去卡它只會製造假警報。
 */

/** §4.5.4 的區塊佈局，**獨立**照著 bench/wire_size.py 的 layout() 重寫一次。 */
function mirrorLayout(scheme, d, k, prec) {
  const s = prec === 'fp16' ? 2 : 4;
  const numel = d * k;
  if (scheme === 'none') return { body: 4 * numel, scales: 0, index: 0 };
  if (scheme === 'per-channel') return { body: numel, scales: d * s, index: 0 };
  if (scheme === 'group-64') {
    return { body: numel, scales: k * Math.ceil(d / 64) * s, index: 0 };
  }
  const nOut = Math.max(1, Math.floor(d * OUTLIER_FRAC));
  return {
    body: (d - nOut) * k + nOut * k * 2,
    scales: (d - nOut) * s,
    // §4.5.4 只定義了 u16 索引表。這裡原本跟著 wire_size.py 寫
    // `min(nOut*2, ceil(d/8))`（允許 bitmap），但那是 wire_size.py 錯 ——
    // 見下面位元組帳測試裡 d ∈ {1,2,4,8} 的那段說明。
    index: nOut * 2,
  };
}

function mirrorTotal(scheme, d, k, prec) {
  const L = mirrorLayout(scheme, d, k, prec);
  return L.body + L.scales + L.index;
}

/** 把 frame 裡的 scale 區塊讀回來（測試自己算位移，不呼叫 wire.js 的私有函式）。 */
function readWireScales(buf) {
  const h = peekHeader(buf);
  const dv = new DataView(buf);
  const s = h.scalePrecision === 'fp16' ? 2 : 4;
  const off = frameHeaderSize + (h.schemeId === 3 ? h.outlierCount * 2 : 0);
  const out = new Float64Array(h.scaleCount);
  for (let i = 0; i < h.scaleCount; i++) {
    out[i] = h.scalePrecision === 'fp16'
      ? fp16ToFloat(dv.getUint16(off + i * s, true))
      : dv.getFloat32(off + i * s, true);
  }
  return out;
}

/** 讀離群索引區塊。 */
function readOutlierIndex(buf) {
  const h = peekHeader(buf);
  const dv = new DataView(buf);
  const out = [];
  for (let j = 0; j < h.outlierCount; j++) out.push(dv.getUint16(frameHeaderSize + j * 2, true));
  return out;
}

const f32 = new Float32Array(1);
const fl32 = (v) => { f32[0] = v; return f32[0]; };

/** quant.js:105-131 挑離群的方式，原樣重寫一次當獨立對照。 */
function pickOutliers(x, dModel) {
  const nTok = x.length / dModel;
  const mag = new Float32Array(dModel);
  for (let t = 0; t < nTok; t++) {
    for (let c = 0; c < dModel; c++) {
      const a = Math.abs(x[t * dModel + c]);
      if (a > mag[c]) mag[c] = a;
    }
  }
  const nOut = Math.max(1, Math.floor(dModel * OUTLIER_FRAC));
  const idx = Array.from({ length: dModel }, (_, i) => i)
    .sort((a, b) => mag[b] - mag[a])
    .slice(0, nOut);
  return { mag, nOut, set: new Set(idx), sorted: idx.slice().sort((a, b) => a - b) };
}

/**
 * 與 `quant.js` 的輸出比對：非離群 channel 逐位元相等，離群 channel 容忍一個 fp16 ULP。
 *
 * 離群那條容差是 **有紀錄的分歧，不是隨便放寬**：`quant.js` 的 `fp16Roundtrip`
 * （quant.js:25-43）用 half-up 捨入而且把非正規數沖成 0，不是 IEEE 的
 * round-to-nearest-even。線路端不能跟著錯 —— 兩個節點各跑一次就會逐位元對不上。
 * 兩者永遠落在相鄰的兩個 fp16 值上，所以差距上限正好是一個 fp16 ULP = 2^-10 相對。
 * 讀到這裡如果覺得「容差比別處鬆是不是偷懶」：不是，是 quant.js 在這裡是錯的那一方。
 *
 * 式子裡的 `+ 2^-14` 是給 quant.js 沖掉的非正規數留的：|x| < 2^-14 時
 * quant.js 給 0、我們給正確的非正規數，絕對差不會超過 2^-14。
 */
function assertAgainstQuantJs(got, x, dModel, outlierSet, tag) {
  const want = SCHEMES.wire.fn(x, dModel);
  let maxRelOutlier = 0;
  for (let i = 0; i < x.length; i++) {
    const c = i % dModel;
    if (!outlierSet.has(c)) {
      // `!==` 而不是 assert.equal：-0 會在這裡出現（見 assertExact 上方那段說明）
      if (got[i] !== want[i]) {
        assert.fail(`${tag}：非離群 channel ${c}（index ${i}）${got[i]} !== ${want[i]}`);
      }
      continue;
    }
    const bound = 2 ** -10 * Math.abs(x[i]) + 2 ** -14;
    const d = Math.abs(got[i] - want[i]);
    assert.ok(d <= bound, `${tag}：離群 channel ${c} 差 ${d} 超過一個 fp16 ULP ${bound}`);
    if (Math.abs(x[i]) > 1e-3) maxRelOutlier = Math.max(maxRelOutlier, d / Math.abs(x[i]));
  }
  assert.ok(maxRelOutlier <= 2 ** -10, `${tag}：離群 channel 最大相對誤差 ${maxRelOutlier}`);
}

/** 每個元素「一個量化階」有多大 —— 用來界定線路與 quant.js 的分歧上限。 */
function quantStep(scheme, x, dModel) {
  const nTok = x.length / dModel;
  const step = new Float64Array(x.length);
  if (scheme === 'per-channel' || scheme === 'per-ch+outlier') {
    const { mag, set } = pickOutliers(x, dModel);
    for (let t = 0; t < nTok; t++) {
      for (let c = 0; c < dModel; c++) {
        const i = t * dModel + c;
        // 離群 channel 走 fp16，沒有量化階；它的分歧上限是一個 fp16 ULP。
        step[i] = (scheme === 'per-ch+outlier' && set.has(c))
          ? Math.max(2 ** -10 * Math.abs(x[i]), 2 ** -14)
          : fl32(mag[c] / QMAX);
      }
    }
    return step;
  }
  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (let g0 = 0; g0 < dModel; g0 += 64) {
      const g1 = Math.min(g0 + 64, dModel);
      let m = 0;
      for (let c = g0; c < g1; c++) m = Math.max(m, Math.abs(x[base + c]));
      for (let c = g0; c < g1; c++) step[base + c] = fl32(m / QMAX);
    }
  }
  return step;
}

// ---------------------------------------------------------------------------
// 1. fp16 位元樣式
// ---------------------------------------------------------------------------

test('fp16 編解碼：已知位元樣式（雙向）', () => {
  // 可以完美往返的正則值：encode 與 decode 都必須對得上。
  const exact = [
    [1, 0x3c00, '1.0'],
    [-1, 0xbc00, '-1.0'],
    [2, 0x4000, '2.0'],
    [-2, 0xc000, '-2.0'],
    [0, 0x0000, '+0'],
    [65504, 0x7bff, '最大正規數'],
    [6.103515625e-5, 0x0400, '最小正規數 2^-14'],
    [6.097555160522461e-5, 0x03ff, '最大非正規數 1023*2^-24'],
    [5.960464477539063e-8, 0x0001, '最小非正規數 2^-24'],
    [-5.960464477539063e-8, 0x8001, '負的最小非正規數'],
    [Infinity, 0x7c00, '+Inf'],
    [-Infinity, 0xfc00, '-Inf'],
    [1.0009765625, 0x3c01, '1 + 2^-10'],
  ];
  for (const [value, bits, label] of exact) {
    assert.equal(
      floatToFp16(value), bits,
      `${label}：floatToFp16(${value}) 應為 0x${bits.toString(16).toUpperCase()}`,
    );
    assert.equal(fp16ToFloat(bits), value, `${label}：fp16ToFloat 沒有還原`);
  }

  // -0 必須保留符號。IEEE 754 裡 -0 的位元樣式是 0x8000，不是 0。
  assert.equal(floatToFp16(-0), 0x8000, '-0 的符號位被吃掉了');
  assert.ok(Object.is(fp16ToFloat(0x8000), -0), 'fp16ToFloat(0x8000) 應該是 -0');

  // NaN 的尾數若被截光就會變成 Inf —— 一個看起來「只是溢位」的災難。
  assert.equal(floatToFp16(NaN), 0x7e00, 'NaN 必須保留 quiet bit');
  assert.ok(Number.isNaN(fp16ToFloat(0x7e00)), '0x7E00 應該解成 NaN');
  assert.ok(Number.isNaN(fp16ToFloat(0x7c01)), '任何非零尾數 + 全 1 指數都是 NaN');
});

test('fp16 編碼：ties-to-even，正是 half-up 會做錯的那些', () => {
  // 只有 encode 方向有意義（這些值不是 fp16 可表示的）。
  const rounding = [
    // [輸入, 正解(RTNE), quant.js 的 half-up 會給的東西, 說明]
    [1 + 2 ** -11, 0x3c00, 0x3c01, '1+2^-11 平手 -> 進到偶數 = 1.0'],
    [-(1 + 2 ** -11), 0xbc00, 0xbc01, '負號不改變平手方向'],
    [1 + 3 * 2 ** -11, 0x3c02, 0x3c02, '平手但偶數在上面：兩者同解'],
    [2049, 0x6800, 0x6801, '2048*(1+2^-11)'],
    [2051, 0x6802, 0x6802, '2048*(1+3*2^-11)'],
    // 65520 恰在 65504 與 65536 中間，RTNE 會選偶數的 65536 —— fp16 表示不了。
    // §4.5.5b 規定這裡飽和到最大有限值，不是 Inf：65520 是個有限的 fp32 值，
    // 編成 Infinity 是靜默的數值毀損（方案 3 的離群本體就是走這條路）。
    [65520, 0x7bff, null, '溢位邊界：飽和到 65504，不准變成 Inf'],
    [65519, 0x7bff, null, '差一點點，必須留在最大正規數'],
    [1e5, 0x7bff, null, '遠超過 fp16 上限也是飽和'],
    [-1e6, 0xfbff, null, '負的溢位飽和到 -65504，符號要留著'],
    [3.4e38, 0x7bff, null, '接近 fp32 上限：仍然是有限值，仍然飽和'],
    [2 ** -25, 0x0000, null, '最小非正規數的一半 -> 平手進到偶數 0'],
    [2 ** -25 * 1.5, 0x0001, null, '超過一半 -> 1'],
    [2 ** -25 * 3, 0x0002, null, '非正規數區間也要 ties-to-even'],
    [2 ** -30, 0x0000, null, '遠小於半個 ULP -> +0'],
    [-(2 ** -30), 0x8000, null, '下溢也要保留符號'],
  ];
  for (const [value, want, , label] of rounding) {
    assert.equal(
      floatToFp16(value), want,
      `${label}：floatToFp16(${value}) 應為 0x${want.toString(16).toUpperCase()}，` +
      `得到 0x${floatToFp16(value).toString(16).toUpperCase()}`,
    );
  }

  // 釘住「為什麼不能直接用 quant.js 的版本」。這不是在批評 quant.js ——
  // 它是 M4 的品質模擬器，半個 ULP 的偏差對 PPL 統計沒有意義；
  // 但線路格式的兩端必須逐位元一致，所以這裡必須有自己的實作。
  assert.notEqual(
    fp16Roundtrip(1 + 2 ** -11), fp16ToFloat(floatToFp16(1 + 2 ** -11)),
    'quant.js 的 half-up 應該在這個平手值上與 RTNE 不同 —— 若相同代表它被改過了，' +
    '請重新檢查這裡的假設',
  );
  assert.equal(fp16Roundtrip(2 ** -24), 0, 'quant.js 會把非正規數沖成 0');
  assert.equal(fp16ToFloat(floatToFp16(2 ** -24)), 2 ** -24, '我們必須保留非正規數');
});

// ---------------------------------------------------------------------------
// 2. round-trip 決定性
// ---------------------------------------------------------------------------

test('round-trip 決定性：同一份輸入必須產生逐位元相同的 frame', () => {
  const dims = [576, 5120]; // 576 = SmolLM2-135M，5120 = 32B 級（§4.1 的表）
  const lens = [1, 2, 8, 16, 32];
  let seed = 20250917;
  for (const scheme of SCHEME_NAMES) {
    for (const dModel of dims) {
      for (const qLen of lens) {
        const x = makeTensor(dModel, qLen, seed++);
        const meta = { msgType: 0, roundId: 7, stageIndex: 2, startPosition: 0 };
        const a = encode(x, dModel, scheme, meta);
        const b = encode(x, dModel, scheme, meta);
        const tag = `${scheme} d=${dModel} K=${qLen}`;

        assert.deepEqual(
          Array.from(new Uint8Array(a)), Array.from(new Uint8Array(b)),
          `${tag}：兩次編碼的位元組不同 —— 編碼器裡有不決定性的東西（迭代順序？Set？）`,
        );
        const da = decode(a);
        const db = decode(b);
        assertExact(da.data, db.data, `${tag}：兩次解碼結果不同`);
        assert.equal(da.dModel, dModel, `${tag}：dModel 沒有往返`);
        assert.equal(da.qLen, qLen, `${tag}：qLen 沒有往返`);
        assert.equal(da.scheme, scheme, `${tag}：scheme 沒有往返`);

        // 再解一次已經解過的 buffer：decode 不能有副作用（例如原地改 payload）
        assertExact(decode(a).data, da.data, `${tag}：decode 有副作用`);
      }
    }
  }
});

// ---------------------------------------------------------------------------
// 3. 與 M4 數值綁定（§4.5.5）
// ---------------------------------------------------------------------------

test('fp32 scale：none / per-channel 與 quant.js 逐位元相等', () => {
  for (const scheme of ['none', 'per-channel']) {
    for (const [dModel, qLen] of [[576, 8], [5120, 16], [576, 1]]) {
      const x = makeTensor(dModel, qLen, 11 + dModel + qLen);
      const got = decode(encode(x, dModel, scheme, { scalePrecision: 'fp32' })).data;
      const want = SCHEMES[QUANT_KEY[scheme]].fn(x, dModel);
      assertExact(got, want, `${scheme} d=${dModel} K=${qLen} 與 quant.js 不相等`);
    }
  }
});

/** 把一個值捨入到線路精度。fp16 那條用的是 wire.js 的編解碼器（它在測試 1 已被位元樣式釘死）。 */
const roundToWire = (v, prec) => (prec === 'fp16' ? fp16ToFloat(floatToFp16(v)) : fl32(v));

const qdqWith = (v, s) => (s === 0 ? 0 : Math.max(-128, Math.min(127, Math.round(v / s))) * s);

/**
 * quant.js 的 perChannel / groupWise，唯一改動是 §4.5.5 要求的那一步：
 * **scale 先捨入到線路精度，再拿它去量化。**
 *
 * 這兩個參考實作是整份測試的支點。少了它們，「§4.5.5 有沒有被遵守」只能靠
 * 統計上的間接證據 —— 而實測過：把 wire.js 的 roundScale 改成直接回傳原值
 * （也就是違反 §4.5.5），所有純比對 quant.js 的測試仍然全綠。
 */
function refPerChannel(x, dModel, prec) {
  const nTok = x.length / dModel;
  const out = new Float32Array(x.length);
  for (let c = 0; c < dModel; c++) {
    let m = 0;
    for (let t = 0; t < nTok; t++) m = Math.max(m, Math.abs(x[t * dModel + c]));
    const s = roundToWire(m / QMAX, prec);
    for (let t = 0; t < nTok; t++) out[t * dModel + c] = qdqWith(x[t * dModel + c], s);
  }
  return out;
}

function refGroup64(x, dModel, prec = 'fp32') {
  const nTok = x.length / dModel;
  const out = new Float32Array(x.length);
  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (let g0 = 0; g0 < dModel; g0 += 64) {
      const g1 = Math.min(g0 + 64, dModel);
      let m = 0;
      for (let c = g0; c < g1; c++) m = Math.max(m, Math.abs(x[base + c]));
      const s = roundToWire(m / QMAX, prec);
      for (let c = g0; c < g1; c++) out[base + c] = qdqWith(x[base + c], s);
    }
  }
  return out;
}

const groupWiseWireScale = (x, dModel) => refGroup64(x, dModel, 'fp32');

test('fp32 scale：group-64 —— 與 quant.js 的分歧被界定在一個量化階內', () => {
  /*
   * ⚠ 這一項**沒有**斷言逐位元相等，而且那不是實作沒寫好。
   *
   * `quant.js` 的 `perChannel` 把 scale 存進 `Float32Array`（quant.js:59），
   * 所以它量化時用的就是 fp32 值 —— 剛好等於 §4.5.5 要求的行為，因此上面那個
   * 測試可以斷言逐位元相等。
   *
   * 但 `groupWise` 的 scale 是個裸的 double（quant.js:92 `const scale = m / qmax`）。
   * `m / 127` 幾乎不可能剛好是 fp32 可表示的值，所以 quant.js 是**用一個永遠送不上線的
   * 數字在量化**。§4.5.5 明令編碼端必須先捨入再量化，兩者因此必然分歧：
   * 實測 d=576/K=8 有約 28% 的值在最後一個 fp32 位元上不同。
   *
   * 所以這裡拆成兩個都能證明的斷言：
   *   1. 與「quant.js 的演算法 + §4.5.5 的 scale 捨入」逐位元相等 —— 證明演算法本身沒走樣
   *   2. 與 quant.js 原樣輸出的差距不超過一個量化階 —— 證明分歧只來自 scale 的最後一個位元
   * 兩邊都對，是 quant.js 的 groupWise 與 perChannel 對 scale 的處理本來就不一致。
   */
  for (const [dModel, qLen] of [[576, 8], [5120, 16], [576, 1]]) {
    const x = makeTensor(dModel, qLen, 700 + dModel + qLen);
    const got = decode(encode(x, dModel, 'group-64', { scalePrecision: 'fp32' })).data;
    const tag = `group-64 d=${dModel} K=${qLen}`;

    assertExact(got, groupWiseWireScale(x, dModel), `${tag}：與 §4.5.5 語意的參考實作不同`);

    const quantjs = groupWise(x, dModel, 8, 64);
    const step = quantStep('group-64', x, dModel);
    let worst = 0;
    for (let i = 0; i < x.length; i++) {
      const d = Math.abs(got[i] - quantjs[i]);
      assert.ok(
        d <= step[i] * (1 + 2 ** -20),
        `${tag}：第 ${i} 個值與 quant.js 差 ${d}，超過一個量化階 ${step[i]} —— ` +
        '這已經不是 scale 捨入造成的，是 int8 碼真的錯了',
      );
      if (step[i] > 0) worst = Math.max(worst, d / step[i]);
    }
    assert.ok(worst <= 1 + 2 ** -20, `${tag}：最大分歧 ${worst} 個量化階`);
  }
});

test('fp32 scale：per-ch+outlier —— 非離群逐位元相等，離群 channel 容忍 2^-10', () => {
  for (const [dModel, qLen] of [[576, 8], [5120, 16], [576, 1]]) {
    const x = makeTensor(dModel, qLen, 900 + dModel + qLen);
    const buf = encode(x, dModel, 'per-ch+outlier', { scalePrecision: 'fp32' });
    const { set, sorted, nOut } = pickOutliers(x, dModel);
    const tag = `per-ch+outlier d=${dModel} K=${qLen}`;

    // 線路挑的離群集合必須與 quant.js 一模一樣，否則「非離群逐位元相等」根本無從談起
    assert.deepEqual(readOutlierIndex(buf), sorted, `${tag}：離群索引與 quant.js 挑的不同`);
    assert.equal(peekHeader(buf).outlierCount, nOut, `${tag}：outlierCount 不對`);

    assertAgainstQuantJs(decode(buf).data, x, dModel, set, tag);
  }
});

test('§4.5.5：必須用「送出去的那個 scale」量化', () => {
  /*
   * 這是整個 frame 格式唯一的規範性語意，也是最容易被無聲違反的一條：
   * 編碼端若用 fp32（或 double）scale 算 int8 碼、卻在線上送 fp16 scale，
   * `decode(encode(x))` 就不等於編碼端自己算出來的東西。差的不是最後一個位元 ——
   * scale 動 2^-11，而 `|x/scale|` 最大到 127，落在 .5 邊界附近的值會直接換一個 int8 碼。
   *
   * 這一項必須直接驗，不能靠「跟 quant.js 比」間接驗：實測把 wire.js 的
   * roundScale 改成直接回傳原值（明確違反 §4.5.5），所有比對 quant.js 的測試
   * 仍然全綠 —— 因為 quant.js 本來就用 fp32 scale，兩種錯法在 fp32 下幾乎重合。
   */
  for (const prec of ['fp16', 'fp32']) {
    for (const [dModel, qLen] of [[576, 8], [5120, 16], [577, 3], [576, 1]]) {
      const x = makeTensor(dModel, qLen, 2200 + dModel + qLen);
      const tag = `${prec} d=${dModel} K=${qLen}`;

      assertExact(
        decode(encode(x, dModel, 'per-channel', { scalePrecision: prec })).data,
        refPerChannel(x, dModel, prec),
        `per-channel ${tag}：沒有用送出去的那個 scale 量化`,
      );
      assertExact(
        decode(encode(x, dModel, 'group-64', { scalePrecision: prec })).data,
        refGroup64(x, dModel, prec),
        `group-64 ${tag}：沒有用送出去的那個 scale 量化`,
      );

      // 更直白的說法：解出來的值除以「線路上那個 scale」必須剛好是最接近
      // x/scale 的整數。per-channel 的 scale 與 channel 一一對應，所以驗得最乾淨。
      const buf = encode(x, dModel, 'per-channel', { scalePrecision: prec });
      const scales = readWireScales(buf);
      const got = decode(buf).data;
      for (let t = 0; t < qLen; t++) {
        for (let c = 0; c < dModel; c++) {
          const i = t * dModel + c;
          const s = scales[c];
          const want = s === 0 ? 0 : Math.max(-128, Math.min(127, Math.round(x[i] / s)));
          const code = s === 0 ? 0 : Math.round(got[i] / s);
          // `!==` 而不是 assert.equal：Math.round(-0.3) 是 -0，而 -0 進 Int8Array 會變 +0
          if (code !== want) {
            assert.fail(
              `per-channel ${tag}：channel ${c} 的 int8 碼是 ${code}，但用線路 scale ` +
              `${s} 量化 x=${x[i]} 應該得到 ${want}`,
            );
          }
        }
      }
    }
  }
});

test('fp16 scale：scale 本身相對誤差 < 2^-10，值的分歧不超過一個量化階', () => {
  /*
   * §4.5.5 說「fp16 scale 時相對誤差 < 2^-10」。那句話講的是 **scale**，
   * 逐元素直接這樣斷言會失敗，而且失敗是對的：
   * §4.5.5 自己要求先捨入 scale 再量化，scale 動了 2^-11，`v/scale` 就跟著動；
   * `|v/scale|` 最大到 127，所以落在 .5 邊界附近的值會換到隔壁的 int8 碼，
   * 兩邊差整整一個量化階（2^-7 量級），不是 2^-10。
   * 實測 d=5120/K=32 的 per-channel 有約 0.95% 的值會這樣翻。
   *
   * 把「相對誤差 < 2^-10」硬套在值上，只會逼人把容差一路調鬆到失去意義。
   * 所以拆成兩個各自成立、而且都可以證明的斷言。
   */
  for (const scheme of ['per-channel', 'group-64', 'per-ch+outlier']) {
    for (const [dModel, qLen] of [[576, 8], [5120, 16], [576, 1]]) {
      const x = makeTensor(dModel, qLen, 1300 + dModel + qLen);
      const buf = encode(x, dModel, scheme, { scalePrecision: 'fp16' });
      const got = decode(buf).data;
      const want = SCHEMES[QUANT_KEY[scheme]].fn(x, dModel);
      const exact = decode(encode(x, dModel, scheme, { scalePrecision: 'fp32' }));
      const tag = `${scheme} d=${dModel} K=${qLen}`;

      // (1) scale 區塊：這才是「相對誤差 < 2^-10」真正的內容（實測上限是 2^-11）
      const wire = readWireScales(buf);
      const ref = readWireScales(encode(x, dModel, scheme, { scalePrecision: 'fp32' }));
      assert.equal(wire.length, ref.length, `${tag}：兩種精度的 scaleCount 不同`);
      let maxScaleRel = 0;
      let subnormalScales = 0;
      for (let i = 0; i < wire.length; i++) {
        const d = Math.abs(wire[i] - ref[i]);
        // fp16 的捨入誤差上限是 max(2^-11 × |v|, 半個非正規階 2^-25)，兩種情形一起寫。
        assert.ok(
          d <= 2 ** -10 * Math.abs(ref[i]) + 2 ** -25,
          `${tag}：第 ${i} 個 scale 的誤差 ${d} 超過一個 fp16 ULP（scale=${ref[i]}）`,
        );
        // 「相對誤差 < 2^-10」只在 fp16 正規數區間（≥ 2^-14）成立。
        // K=1 時只要某個 channel 的那一個值接近 0，scale 就會掉進非正規數區間，
        // 相對誤差可以到 2^-2 —— 這不是 bug，是 fp16 在 6.1e-5 以下就沒有 11 bits 精度了。
        // 實測 d=576/K=1 每組資料大概會出現一兩個。量化上完全無害：
        // scale 那麼小代表整個 channel 的值也那麼小，絕對誤差仍然 ≤ 2^-25。
        if (Math.abs(ref[i]) >= 2 ** -14) {
          maxScaleRel = Math.max(maxScaleRel, d / Math.abs(ref[i]));
        } else if (ref[i] !== 0) {
          subnormalScales++;
        }
      }
      assert.ok(maxScaleRel < 2 ** -10, `${tag}：scale 最大相對誤差 ${maxScaleRel} 不小於 2^-10`);
      assert.ok(
        subnormalScales <= wire.length * 0.01,
        `${tag}：有 ${subnormalScales} 個 scale 掉進 fp16 非正規數區間，` +
        '這組測試資料的量級不對，測不到原本要測的東西',
      );

      // (2) 值：與 quant.js 的分歧不超過一個量化階（int8 碼最多偏 1）
      const step = quantStep(scheme, x, dModel);
      for (let i = 0; i < x.length; i++) {
        const d = Math.abs(got[i] - want[i]);
        assert.ok(
          d <= step[i] * (1 + 2 ** -10) + 2 ** -14,
          `${tag}：第 ${i} 個值與 quant.js 差 ${d}，超過一個量化階 ${step[i]}`,
        );
      }
      // (3) fp16 scale 不該把 int8 碼打爛：換到隔壁碼的比例必須是零星的。
      //     門檻取四分之一個量化階：單純「scale 差 2^-11」最多只能讓值動
      //     2^-11 × 127 ≈ 0.06 個量化階，所以超過 0.25 個階的只可能是 int8 碼真的翻了。
      //     不能直接數「值有沒有變」—— scale 一動，幾乎每個值的最後幾位都會變。
      let flipped = 0;
      for (let i = 0; i < x.length; i++) {
        if (Math.abs(got[i] - exact.data[i]) > 0.25 * step[i]) flipped++;
      }
      assert.ok(
        flipped / x.length < 0.05,
        `${tag}：fp16 scale 讓 ${(flipped / x.length * 100).toFixed(2)}% 的 int8 碼翻掉，太多了`,
      );
    }
  }
});

// ---------------------------------------------------------------------------
// 4. 位元組帳
// ---------------------------------------------------------------------------

test('位元組帳與 bench/wire_size.py 一致', (t) => {
  /*
   * ⚠ d ∈ {1, 2, 4, 8} 不是湊數的，它們是這一項唯一抓得到某個真實分歧的地方。
   *
   * 原本只測 {576, 5120, 100, 577}，而 wire_size.py 的 outlier_index_bytes()
   * 取 `min(nOut*2, ceil(d/8))`（允許 bitmap），wire.js 則永遠寫 u16 索引表。
   * 當時 wire.js 的註解寫著「要 outlierFrac 調到 6.25% 以上兩邊才會對不上，
   * 而位元組帳測試會立刻抓到」—— 兩句都是假的：真正生效的比例是
   * `max(1, floor(d*frac))/d`，d < 34 時就已經超過 6.25%，而這裡的維度全都 ≥ 100，
   * 所以測試永遠看不到。實測 d ∈ {1,2,4,8} × K ∈ {1,8} × 兩種精度共 16 種組合
   * 每則訊息差 1 個位元組。§4.5.4 只定義 u16 表，所以錯的是 wire_size.py。
   */
  const dims = [576, 5120, 100, 577, 1, 2, 4, 8];
  const lens = [1, 8, 16, 32];
  const precs = ['fp16', 'fp32'];
  const combos = [];
  for (const scheme of SCHEME_NAMES) {
    for (const d of dims) for (const k of lens) for (const p of precs) combos.push([scheme, d, k, p]);
  }

  // (a) 與測試自己重寫的 layout() 比。兩份算術獨立寫出來，對不上就代表有一邊算錯了 ——
  //     這一項才是讓 §4.5 那張表可信的東西，不然表只是手打的數字。
  for (const [scheme, d, k, p] of combos) {
    const x = new Float32Array(d * k);
    const got = encode(x, d, scheme, { scalePrecision: p }).byteLength - frameHeaderSize;
    const want = mirrorTotal(scheme, d, k, p);
    assert.equal(
      got, want,
      `${scheme} d=${d} K=${k} ${p}：payload ${got} 位元組，佈局算出來應該是 ${want}`,
    );
  }

  // (b) 與 Python 那一份比。同一張帳有兩個實作（JS 送、Python 報告），
  //     它們分家的那一天，文件裡的 bits/值 就開始騙人了。
  let fromPy;
  try {
    const script = [
      'import sys, json',
      'req = json.load(sys.stdin)',
      'sys.path.insert(0, req["bench"])',
      'import wire_size',
      'print(json.dumps([wire_size.total_bytes(*c) for c in req["combos"]]))',
    ].join('\n');
    const input = JSON.stringify({
      bench: benchDir,
      combos: combos.map(([s, d, k, p]) => [QUANT_KEY[s], d, k, p]),
    });
    fromPy = JSON.parse(execFileSync('python3', ['-c', script], { input, encoding: 'utf8' }));
  } catch (err) {
    t.diagnostic(`跳過 Python 對照：${err.message}`);
    return;
  }

  for (let i = 0; i < combos.length; i++) {
    const [scheme, d, k, p] = combos[i];
    const x = new Float32Array(d * k);
    const got = encode(x, d, scheme, { scalePrecision: p }).byteLength - frameHeaderSize;
    assert.equal(
      got, fromPy[i],
      `${scheme} d=${d} K=${k} ${p}：wire.js 送 ${got} 位元組，` +
      `bench/wire_size.py 說是 ${fromPy[i]}。兩者必有一個是錯的 —— ` +
      '文件 §4.5 的表是 wire_size.py 產生的，所以這個不一致會直接讓文件說謊',
    );
  }

  // 順手釘住 §4.2 的那個修正：K=8、d=576、fp16 scale 時是 10.24 bits/值，不是 8.24
  const bytes = encode(new Float32Array(576 * 8), 576, 'per-ch+outlier').byteLength - frameHeaderSize;
  const bits = (bytes * 8) / (576 * 8);
  assert.ok(
    Math.abs(bits - 10.24) < 0.01,
    `§4.2 修正後的 10.24 bits/值對不上，算出 ${bits.toFixed(2)}`,
  );
});

// ---------------------------------------------------------------------------
// 5. 邊界情況
// ---------------------------------------------------------------------------

test('邊界：dModel 不是 64 的整數倍（group-64 的尾巴組）', () => {
  // quant.js:88-89 記著「早期版本這裡會靜默退回 per-channel，是個真實踩過的 bug」。
  // 100 -> 1 組滿 + 36 的尾巴；577 -> 9 組滿 + 1 的尾巴（最惡劣：尾巴只有一個 channel）
  for (const dModel of [100, 577]) {
    for (const qLen of [1, 3, 8]) {
      const x = makeTensor(dModel, qLen, 31 + dModel + qLen);
      const tag = `d=${dModel} K=${qLen}`;
      const h = peekHeader(encode(x, dModel, 'group-64', { scalePrecision: 'fp32' }));
      assert.equal(
        h.scaleCount, qLen * Math.ceil(dModel / 64),
        `${tag}：尾巴沒有自成一組 —— scaleCount 應是 K × ceil(d/64)`,
      );
      const got = decode(encode(x, dModel, 'group-64', { scalePrecision: 'fp32' })).data;
      assertExact(got, groupWiseWireScale(x, dModel), `${tag}：尾巴組的值不對`);

      // 其餘方案也要能處理非對齊的 dModel
      for (const scheme of SCHEME_NAMES) {
        const r = decode(encode(x, dModel, scheme));
        assert.equal(r.data.length, dModel * qLen, `${tag} ${scheme}：長度不對`);
        for (let i = 0; i < r.data.length; i++) {
          assert.ok(Number.isFinite(r.data[i]), `${tag} ${scheme}：第 ${i} 個值不是有限數`);
        }
      }
    }
  }
});

test('邊界：全零張量 —— scale 為 0 不能解出 NaN', () => {
  // quant.js:17 的 `if (scale === 0) return 0` 擋的就是 0/0。
  // 全零不是假想情境：padding 位置、還沒寫入的 KV 槽都會是整片 0，
  // 而 NaN 一旦進了流水線會沿著 hop 一路傳下去，最後只看得到「輸出壞掉」。
  //
  // 誠實標示這一項的強度：拿掉 wire.js 的那道防線，這個測試**不會**變紅，
  // 因為 Int8Array 的整數轉換也會把 NaN 吃成 0（實際試過）。
  // 所以它驗到的是「整片 0 進去、整片 0 出來」這個端到端性質，
  // 不是「那一行 if 還在」。真正該擋住的是「全零張量解出非零值」那條斷言。
  for (const dModel of [576, 100]) {
    for (const scheme of SCHEME_NAMES) {
      for (const prec of ['fp16', 'fp32']) {
        const x = new Float32Array(dModel * 4);
        const r = decode(encode(x, dModel, scheme, { scalePrecision: prec }));
        const tag = `${scheme} d=${dModel} ${prec}`;
        for (let i = 0; i < r.data.length; i++) {
          assert.ok(!Number.isNaN(r.data[i]), `${tag}：第 ${i} 個值是 NaN`);
          assert.equal(r.data[i], 0, `${tag}：全零張量解出非零值 ${r.data[i]}`);
        }
      }
    }
  }
});

test('邊界：單一巨大離群值', () => {
  // 一個 channel 的一個位置有 1e4 級的值，其餘是 O(1)。
  // per-channel 會被它把整個 channel 的 scale 撐大（§4.2 結論 2 講的就是這件事），
  // per-ch+outlier 則必須把它挑進 fp16 區塊、幾乎不損失精度。
  const dModel = 576;
  const qLen = 8;
  const x = makeTensor(dModel, qLen, 4242);
  const hot = 3 * dModel + 137; // t=3, c=137
  // 12345.6875 落在 fp16 ULP = 8 的那一段，所以它一定要經過捨入，
  // 不會因為「剛好可表示」而讓離群那條路徑被跳過。
  x[hot] = 12345.6875;

  for (const scheme of SCHEME_NAMES) {
    const r = decode(encode(x, dModel, scheme, { scalePrecision: 'fp32' }));
    assert.ok(Number.isFinite(r.data[hot]), `${scheme}：巨值解出 ${r.data[hot]}`);
    const tag = `${scheme}：有巨值時與參考實作不同`;
    if (scheme === 'per-ch+outlier') {
      assertAgainstQuantJs(r.data, x, dModel, pickOutliers(x, dModel).set, tag);
    } else {
      assertExact(
        r.data,
        scheme === 'group-64' ? groupWiseWireScale(x, dModel) : SCHEMES[QUANT_KEY[scheme]].fn(x, dModel),
        tag,
      );
    }
  }

  // 離群方案必須真的把 137 這個 channel 挑進去，否則 §4.2 的 99.32% 是假的
  const buf = encode(x, dModel, 'per-ch+outlier');
  assert.ok(readOutlierIndex(buf).includes(137), '最大的 channel 沒有被挑成離群');
  const rel = Math.abs(decode(buf).data[hot] - x[hot]) / Math.abs(x[hot]);
  assert.ok(rel <= 2 ** -11, `離群值走 fp16 後相對誤差 ${rel} 太大`);
});

test('邊界：qLen = 1（投機解碼被完全拒絕時每輪只剩一個位置）', () => {
  for (const dModel of [576, 5120, 100, 577]) {
    const x = makeTensor(dModel, 1, 5150 + dModel);
    for (const scheme of SCHEME_NAMES) {
      const buf = encode(x, dModel, scheme, { scalePrecision: 'fp32' });
      const h = peekHeader(buf);
      assert.equal(h.qLen, 1, `${scheme} d=${dModel}：qLen 沒有寫對`);
      const r = decode(buf);
      assert.equal(r.data.length, dModel);
      if (scheme === 'group-64') {
        assert.equal(h.scaleCount, Math.ceil(dModel / 64), 'K=1 時 scale 數 = 組數');
      }
    }
  }
});

// ---------------------------------------------------------------------------
// 6. header 往返
// ---------------------------------------------------------------------------

test('header：每個欄位都要活著走完 encode -> peekHeader', () => {
  const dModel = 577;
  const qLen = 5;
  const x = makeTensor(dModel, qLen, 606);
  const meta = {
    msgType: 1,            // logits 回程（§4.5.7 現在就要佔位）
    roundId: 4294967295,   // u32 上限
    stageIndex: 65535,     // u16 上限（§4.5.2 改版後它在位移 36，不再是 u8）
    startPosition: 123456, // M3 恆為 0，但欄位現在就要能帶（§4.5.3）
    scalePrecision: 'fp32',
  };
  const buf = encode(x, dModel, 'per-ch+outlier', meta);
  const h = peekHeader(buf);

  assert.equal(h.magic, FRAME_MAGIC);
  assert.equal(h.version, FRAME_VERSION);
  assert.equal(h.msgType, meta.msgType);
  assert.equal(h.schemeId, SCHEME_IDS['per-ch+outlier']);
  assert.equal(h.scheme, 'per-ch+outlier');
  assert.equal(h.flags, 0, 'fp32 scale 時 bit0 必須是 0');
  assert.equal(h.scalePrecision, 'fp32');
  assert.equal(h.roundId, meta.roundId);
  assert.equal(h.stageIndex, meta.stageIndex);
  assert.equal(h.dModel, dModel);
  assert.equal(h.qLen, qLen);
  assert.equal(h.startPosition, meta.startPosition);
  assert.equal(h.outlierCount, Math.max(1, Math.floor(dModel * OUTLIER_FRAC)));
  assert.equal(h.scaleCount, dModel - h.outlierCount);
  assert.equal(h.payloadLen, buf.byteLength - frameHeaderSize);

  // decode 回傳的 meta 也要帶同一組值 —— 接收端是靠它決定 KV 位置的
  const r = decode(buf);
  assert.deepEqual(r.meta, {
    msgType: 1,
    roundId: 4294967295,
    stageIndex: 65535,
    startPosition: 123456,
    scalePrecision: 'fp32',
  });

  // fp16 是預設值，bit0 要是 1
  assert.equal(peekHeader(encode(x, dModel, 'group-64')).flags, 1, 'fp16 scale 時 bit0 必須是 1');

  // magic 在位元組層面就是 'ECL1'，方便 hex dump 認人
  assert.equal(
    new TextDecoder().decode(new Uint8Array(buf, 0, 4)), 'ECL1',
    "magic 的位元組順序不對 —— little-endian 寫入 0x314C4345 應該看到 'ECL1'",
  );

  // 保留欄位必須是 0，否則之後拿它們擴充時會讀到舊節點留下的垃圾
  const dv = new DataView(buf);
  assert.equal(dv.getUint16(38, true), 0, 'offset 38 是保留欄位（§4.5.2 的最後兩個位元組）');

  // 四個 schemeId 都要能往返
  for (const name of SCHEME_NAMES) {
    assert.equal(peekHeader(encode(x, dModel, name)).scheme, name);
    assert.equal(peekHeader(encode(x, dModel, SCHEME_IDS[name])).schemeId, SCHEME_IDS[name]);
  }
  // quant.js / wire_size.py 的 'wire' 別名要接得上
  assert.equal(peekHeader(encode(x, dModel, 'wire')).scheme, 'per-ch+outlier');
});

// ---------------------------------------------------------------------------
// 7. 拒絕壞輸入
// ---------------------------------------------------------------------------

test('拒絕：壞掉的 frame 必須丟出可行動的錯誤，不能回傳垃圾', () => {
  const dModel = 128;
  const qLen = 4;
  const x = makeTensor(dModel, qLen, 77);
  const good = encode(x, dModel, 'per-channel', { scalePrecision: 'fp32' });

  const corrupt = (mutate) => {
    const copy = good.slice(0);
    mutate(new DataView(copy));
    return copy;
  };

  // magic 不對 —— 最可能的真實成因是 chunk 子標頭沒剝掉（§4.5.6）
  assert.throws(
    () => decode(corrupt((dv) => dv.setUint32(0, 0xdeadbeef, true))),
    /magic|ECL1/,
    'magic 不符必須丟錯',
  );
  // 真的把 8 位元組 chunk 子標頭黏在前面：這是會發生的那種錯
  const withSubHeader = new Uint8Array(8 + good.byteLength);
  withSubHeader.set(new Uint8Array(good), 8);
  assert.throws(() => decode(withSubHeader.buffer), /magic|ECL1/);

  // 版本不認得
  assert.throws(
    () => decode(corrupt((dv) => dv.setUint8(4, 99))),
    /版本/,
    '未知版本必須丟錯',
  );

  // schemeId 超出 §4.5.2 定義的範圍
  assert.throws(() => decode(corrupt((dv) => dv.setUint8(6, 9))), /schemeId/);

  // 被截斷的 payload
  assert.throws(
    () => decode(good.slice(0, good.byteLength - 10)),
    /長度不符|截斷|payload/,
    '截斷的 frame 必須丟錯',
  );
  // 連 header 都不夠長
  assert.throws(() => decode(good.slice(0, 16)), /header|太短/);
  assert.throws(() => decode(new ArrayBuffer(0)), /header|太短/);

  // payloadLen 與 buffer 長度不符（header 說得比實際多 / 少）
  assert.throws(() => decode(corrupt((dv) => dv.setUint32(32, 999999, true))), /長度不符/);
  assert.throws(() => decode(corrupt((dv) => dv.setUint32(32, 4, true))), /長度不符/);

  // payloadLen 對得上 buffer，但與 header 描述的佈局不符 —— 這是最陰險的一種，
  // 因為長度檢查會過，接著 Int8Array 視圖就越界或讀到別的區塊
  assert.throws(
    () => decode(corrupt((dv) => dv.setUint32(24, 1, true))),
    /佈局/,
    'scaleCount 與 dModel 對不上時必須丟錯',
  );
  assert.throws(
    () => decode(corrupt((dv) => dv.setUint32(12, dModel + 1, true))),
    /佈局/,
  );

  // 錯誤訊息必須是繁體中文而且說得出下一步怎麼做
  try {
    decode(corrupt((dv) => dv.setUint32(0, 0, true)));
    assert.fail('應該丟錯');
  } catch (err) {
    assert.match(err.message, /[一-鿿]/, '錯誤訊息必須是中文');
    assert.match(err.message, /binaryType|chunk/, '錯誤訊息要指出常見成因');
  }

  // encode 端的參數檢查
  assert.throws(() => encode(new Float32Array(10), 3, 'none'), /整數倍/);
  assert.throws(() => encode(new Float32Array(10), 5, 'nope'), /未知的量化方案/);
  assert.throws(() => encode(new Float32Array(10), 5, 'none', { scalePrecision: 'fp8' }), /scalePrecision/);
  // dModel 現在是 u32（§4.5.2），70000 是合法的 —— Llama 3 的 vocab 是 128256，
  // 回程 logits 本來就會用到這種寬度。擋的是超過 u32 與非整數。
  assert.equal(peekHeader(encode(new Float32Array(70000), 70000, 'per-channel')).dModel, 70000);
  assert.throws(() => encode(new Float32Array(10), 2 ** 32, 'none'), /dModel/);
  assert.throws(() => encode(new Float32Array(10), 5.5, 'none'), /dModel/);
  // 但方案 3 的離群索引區塊仍然是 u16（§4.5.4），所以它有自己的上限。
  // 不擋的話 setUint16 會把 70000 靜默地寫成 4464。
  assert.throws(
    () => encode(new Float32Array(70000), 70000, 'per-ch+outlier'),
    /離群索引|u16/,
    'dModel 超過 65536 時 per-ch+outlier 必須明確拒絕，不能靜默截斷索引',
  );
  assert.throws(() => encode(new Float32Array(10), 5, 'none', { stageIndex: 65536 }), /stageIndex/);
  assert.throws(() => encode(new Float32Array(10), 5, 'none', { roundId: -1 }), /roundId/);
  assert.throws(() => decode('不是 buffer'), /ArrayBuffer/);
});

test('decode 接受大 buffer 中間的切片（dcframe.js 重組不想再複製一次）', () => {
  const dModel = 64;
  const x = makeTensor(dModel, 3, 88);
  const frame = encode(x, dModel, 'per-ch+outlier', { scalePrecision: 'fp32' });

  // 把 frame 放進一個更大的 buffer 的中間，模擬重組緩衝區
  const big = new Uint8Array(1000 + frame.byteLength);
  big.set(new Uint8Array(frame), 1000);
  const view = new Uint8Array(big.buffer, 1000, frame.byteLength);

  // 早期版本這裡用 `new Int8Array(dv.buffer, bodyOff, n)` 忽略了 byteOffset，
  // 結果 int8 本體整個位移，解出來的張量「只是有點怪」而且不會報錯。
  assertExact(decode(view).data, decode(frame).data, '切片輸入解出來的值不同');
});

// ---------------------------------------------------------------------------
// 8. §4.5.2 的絕對位移 —— 跨實作解析的唯一依據
// ---------------------------------------------------------------------------

test('header：每個欄位都釘在 §4.5.2 表上的那個位元組', () => {
  /*
   * ⚠ 這一項與上面那個「往返」測試**不能互相取代**，而且它才是重點。
   *
   * 往返測試只證明 `encode` 與 `peekHeader` 彼此同意 —— 把 writer 與 reader 裡的
   * dModel 與 qLen 兩個位移**同時**對調，整份測試仍然全綠，而送出去的 frame
   * 沒有任何別的實作解得開。§4.5.2 存在的理由就是跨實作解析，所以至少要有一個
   * 測試站在「別人家的解析器」那一邊：只認表上的數字。
   *
   * 因此下面的位移全部是手寫的字面常數，**不准**從 wire.js 匯入任何東西來算
   * （連 frameHeaderSize 都不用）—— 那樣測試只會跟著 bug 一起搬家。
   * 每個欄位給一個彼此不同的哨兵值，任何兩個欄位互換都會紅。
   */
  const dModel = 300;   // nOut = floor(300 * 0.03) = 9，nIn = 291
  const qLen = 7;
  const x = makeTensor(dModel, qLen, 31337);
  const buf = encode(x, dModel, 'per-ch+outlier', {
    msgType: 2,              // 2 = 控制（§4.5.2），與 version 1 / schemeId 3 都不同
    roundId: 0xa1b2c3d4,     // 2712847316：四個位元組各不相同，順序錯了一定看得出來
    startPosition: 0x11223344, // 287454020
    stageIndex: 0x0102,      // 258：超過 u8，證明它真的是 u16 而且不在位移 12
    scalePrecision: 'fp16',  // flags bit0 = 1
  });
  const dv = new DataView(buf);

  // payloadLen 照 §4.5.4 手算：索引 9×2 + scale 291×2 + fp16 離群 9×7×2 + int8 291×7
  const payloadLen = 9 * 2 + 291 * 2 + 9 * 7 * 2 + 291 * 7; // = 2763
  assert.equal(buf.byteLength, 40 + payloadLen, 'header 是 40 位元組（§4.5.2），payload 照 §4.5.4');
  assert.equal(frameHeaderSize, 40, 'frameHeaderSize 必須等於 §4.5.2 的 40');

  //         位移  大小  欄位
  assert.equal(dv.getUint32(0, true), 0x314c4345, '位移 0 u32 magic');
  assert.equal(dv.getUint8(4), 1, '位移 4 u8 version');
  assert.equal(dv.getUint8(5), 2, '位移 5 u8 msgType');
  assert.equal(dv.getUint8(6), 3, '位移 6 u8 schemeId（per-ch+outlier）');
  assert.equal(dv.getUint8(7), 1, '位移 7 u8 flags（bit0 = scale 用 fp16）');
  assert.equal(dv.getUint32(8, true), 0xa1b2c3d4, '位移 8 u32 roundId');
  assert.equal(dv.getUint32(12, true), 300, '位移 12 u32 dModel');
  assert.equal(dv.getUint32(16, true), 7, '位移 16 u32 qLen');
  assert.equal(dv.getUint32(20, true), 0x11223344, '位移 20 u32 startPosition');
  assert.equal(dv.getUint32(24, true), 291, '位移 24 u32 scaleCount（= dModel - outlierCount）');
  assert.equal(dv.getUint32(28, true), 9, '位移 28 u32 outlierCount');
  assert.equal(dv.getUint32(32, true), payloadLen, '位移 32 u32 payloadLen');
  assert.equal(dv.getUint16(36, true), 258, '位移 36 u16 stageIndex');
  assert.equal(dv.getUint16(38, true), 0, '位移 38 u16 保留，必須是 0');

  // 逐位元組確認 little-endian（§4.5.2 寫死，不跟平台走）。
  assert.deepEqual(
    Array.from(new Uint8Array(buf, 0, 4)), [0x45, 0x43, 0x4c, 0x31],
    "magic 的四個位元組應該是 'E' 'C' 'L' '1'",
  );
  assert.deepEqual(
    Array.from(new Uint8Array(buf, 8, 4)), [0xd4, 0xc3, 0xb2, 0xa1],
    'roundId 必須是 little-endian',
  );
  assert.deepEqual(
    Array.from(new Uint8Array(buf, 36, 2)), [0x02, 0x01],
    'stageIndex 必須是 little-endian 的 u16',
  );

  // version 與 flags 在上面剛好都是 1，單靠那個 frame 分不出兩者有沒有互換。
  // 再編一個 fp32 的：flags 變 0、version 仍然是 1。
  const fp32 = new DataView(encode(x, dModel, 'per-ch+outlier', { scalePrecision: 'fp32' }));
  assert.equal(fp32.getUint8(4), 1, '位移 4 是 version，不會跟著 scalePrecision 變');
  assert.equal(fp32.getUint8(7), 0, '位移 7 是 flags，fp32 時 bit0 = 0');

  /*
   * 寬度也要釘：上面的值全都塞得進 u16，所以光靠它們分不出 u32 與 u16。
   * 這兩個維度正是 §4.5.2 從 32 位元組改成 40 位元組的理由 ——
   * Llama 3 的 vocab 是 128256、Qwen 2.5 是 151936，u16 的 dModel 表示不了
   * §4.5.7 規定要送的 logits。
   */
  const wide = new DataView(encode(new Float32Array(128256), 128256, 'per-channel'));
  assert.equal(wide.getUint32(12, true), 128256, '位移 12 的 dModel 必須是 u32（Llama 3 的 vocab）');
  assert.equal(wide.getUint32(24, true), 128256, '位移 24 的 scaleCount 必須是 u32');

  const longSeq = new DataView(encode(new Float32Array(70000), 1, 'none'));
  assert.equal(longSeq.getUint32(16, true), 70000, '位移 16 的 qLen 必須是 u32');
  assert.equal(longSeq.getUint32(32, true), 280000, '位移 32 的 payloadLen 必須是 u32');
});

// ---------------------------------------------------------------------------
// 9. §4.5.5b 飽和規則：有限的輸入不准變成 Inf 或 NaN
// ---------------------------------------------------------------------------

test('§4.5.5b：fp16 離群本體溢位要飽和到 65504，不准解出 Infinity', () => {
  /*
   * 對抗性審查的原始重現：d=576、K=4、x[137] = 65520，其餘 O(3)。
   * `encode(x, 576, 'per-ch+outlier')` 解回來 `data[137] === Infinity`，兩端都不報錯。
   * 65520 落在 65504 與 65536 正中間，RTNE 進位到 65536 = fp16 的 Inf。
   *
   * 諷刺的地方要寫下來：方案 3 的離群路徑是 §4.2 專門為了「保住巨值」才加的，
   * 而它是四個方案裡唯一會把巨值變成 Infinity 的。同一個值走 int8 路徑毫無問題 ——
   * 下面的 per-channel 對照組就是為了證明這件事。
   */
  const dModel = 576;
  const qLen = 4;
  const hot = 137; // t=0 的 channel 137，magnitude 最大所以一定被挑成離群

  for (const huge of [65520, 1e5, 1e6]) {
    // 離群本體固定 fp16，與 flags bit0 無關 —— 兩種 scalePrecision 都要驗。
    for (const prec of ['fp16', 'fp32']) {
      const x = makeTensor(dModel, qLen, 808);
      for (let i = 0; i < x.length; i++) x[i] = (i % dModel) === hot ? 0 : 3;
      x[hot] = huge;
      const tag = `x=${huge} ${prec}`;

      const buf = encode(x, dModel, 'per-ch+outlier', { scalePrecision: prec });
      assert.ok(readOutlierIndex(buf).includes(hot), `${tag}：巨值的 channel 沒被挑成離群`);
      const got = decode(buf).data;

      assert.ok(
        Number.isFinite(got[hot]),
        `${tag}：離群 channel 解出 ${got[hot]} —— 有限的輸入不准變成 Inf（§4.5.5b）`,
      );
      assert.equal(got[hot], 65504, `${tag}：應該飽和到 fp16 的最大有限值 65504`);
      for (let i = 0; i < got.length; i++) {
        assert.ok(Number.isFinite(got[i]), `${tag}：第 ${i} 個值是 ${got[i]}`);
      }
    }
  }

  // 對照組：同一個 1e5 走 per-channel（int8 路徑）從來都沒壞過。
  // 這就是為什麼不能把這個 bug 說成「fp16 本來就表示不了 1e5」——
  // 表示不了是真的，解出 Infinity 不是必然的。
  const ctrl = new Float32Array(dModel * qLen).fill(3);
  ctrl[hot] = 1e5;
  const gotCtrl = decode(encode(ctrl, dModel, 'per-channel')).data;
  assert.ok(Number.isFinite(gotCtrl[hot]), `per-channel 對照組解出 ${gotCtrl[hot]}`);
  assert.ok(
    Math.abs(gotCtrl[hot] - 1e5) / 1e5 < 0.01,
    `per-channel 對照組應該還原成 1e5 量級，得到 ${gotCtrl[hot]}`,
  );
});

test('§4.5.5b：fp16 scale 溢位要飽和，不准整個 channel / 整組解出 NaN', () => {
  /*
   * 第二條靜默路徑，比上面那條更惡劣：壞掉的不是那一個值，是整個 channel。
   *
   * 重現：d=2、K=2、x = [8.4e6, 1.5, ...]，預設 fp16 scale。
   * scale = max|x| / 127 = 66141 > 65504，捨入成 fp16 的 Inf；
   * 量化時 `x / Inf` = 0，所有 int8 碼變 0；解碼端算 `0 × Inf` = **NaN**。
   * 門檻剛好是 127 × 65520 = 8321040（8321039 還活著）。
   *
   * 舊版的 knownGap 寫著「Inf 會被 clamp 夾成 ±127」—— 實測是錯的：
   * Inf 與巨大的有限值根本走不到 clamp，它們是先被 scale 除成 0、
   * 再在解碼端乘回 Inf 變 NaN 的。NaN 沿著 hop 傳播是這個專案最難查的失敗模式。
   */
  const NAN_THRESHOLD = 127 * 65520; // 8321040：舊實作從這個值開始整個 channel 變 NaN

  {
    const x = Float32Array.from([8.4e6, 1.5, 2.5, 0.5]); // d=2, K=2
    const got = decode(encode(x, 2, 'per-channel')).data; // 預設就是 fp16 scale
    for (let i = 0; i < got.length; i++) {
      assert.ok(!Number.isNaN(got[i]), `d=2 K=2：第 ${i} 個值是 NaN —— scale 溢位成 Inf 了`);
      assert.ok(Number.isFinite(got[i]), `d=2 K=2：第 ${i} 個值是 ${got[i]}`);
    }
    // 飽和之後最大可表示值是 127 × 65504；超過的部分損失大小但保持有限（§4.5.5b）。
    assert.equal(got[0], 127 * 65504, '8.4e6 應該夾在 127 × 65504 = 8319008');
  }

  // 恰好在舊門檻上，以及剛好可表示的上限：兩個值都必須有限，而且上限要精確往返。
  for (const m of [NAN_THRESHOLD - 1, NAN_THRESHOLD, 127 * 65504]) {
    const x = Float32Array.from([m, 1.5]);
    const got = decode(encode(x, 2, 'per-channel')).data;
    assert.ok(Number.isFinite(got[0]), `max|x| = ${m}：解出 ${got[0]}`);
  }
  {
    // scale 剛好是 65504 時沒有任何損失，這是 §4.5.5b 說的「最大可表示值」。
    const x = Float32Array.from([127 * 65504, 1.5]);
    assert.equal(decode(encode(x, 2, 'per-channel')).data[0], 127 * 65504);
  }

  // 一個值毒死一整組：per-channel 壞 K 個、group-64 壞整組 64 個 channel。
  for (const scheme of ['per-channel', 'group-64', 'per-ch+outlier']) {
    const dModel = 128;
    const qLen = 4;
    const x = new Float32Array(dModel * qLen).fill(2);
    x[5] = 9e6;
    const got = decode(encode(x, dModel, scheme)).data; // fp16 scale
    let nan = 0;
    for (let i = 0; i < got.length; i++) if (Number.isNaN(got[i])) nan++;
    assert.equal(
      nan, 0,
      `${scheme} d=128 K=4：x[5]=9e6 讓 ${nan} 個值變成 NaN —— ` +
      '一個值不該毒死整個 channel 或整組 64 個 channel',
    );
  }
});

test('§4.5.5b：encode() 對 Inf / NaN 的輸入直接丟錯，不偷偷吞掉', () => {
  /*
   * 前兩項講的是「有限的輸入不准變成 Inf/NaN」。這一項是反方向：
   * 輸入本身就有 Inf/NaN 時，線路格式**不負責**把它變得好看。
   * 那是呼叫端的 bug（上一段的 softmax 炸了、KV 槽沒初始化…），
   * 而且無聲吞掉的代價極高 —— 下游只會看到「輸出看起來合理、只是慢慢偏掉」。
   * 掃這一遍的成本是零：除了 scheme 0 以外每個方案本來就要走一次 max|x|。
   */
  const dModel = 64;
  const qLen = 2;
  for (const scheme of SCHEME_NAMES) {
    for (const bad of [Infinity, -Infinity, NaN]) {
      const x = new Float32Array(dModel * qLen).fill(1.5);
      x[dModel + 7] = bad; // t=1, c=7
      assert.throws(
        () => encode(x, dModel, scheme),
        (err) => {
          assert.match(err.message, /Inf|NaN/, `${scheme}：錯誤訊息要說出是 Inf 還是 NaN`);
          assert.match(err.message, /71/, `${scheme}：錯誤訊息要說出是第幾個值（71）`);
          assert.match(err.message, /[一-鿿]/, `${scheme}：錯誤訊息必須是中文`);
          return true;
        },
        `${scheme}：輸入含 ${bad} 時 encode 必須丟錯（§4.5.5b）`,
      );
    }
    // 同一份資料把壞值拿掉就必須編得出來 —— 證明擋的是壞值，不是別的東西
    assert.ok(encode(new Float32Array(dModel * qLen).fill(1.5), dModel, scheme).byteLength > 0);
  }
});

// ---------------------------------------------------------------------------
// 10. 重複的離群索引
// ---------------------------------------------------------------------------

test('拒絕：離群索引重複 —— 每個 header 欄位都自洽，值卻整個錯位', () => {
  /*
   * 這是「header 檢查全過、資料照樣爛掉」最乾淨的例子：
   * decode 驗了每個索引 < dModel，也驗了 scaleCount === dModel - outlierCount，
   * 就是沒驗索引互不相同。把第 2 個 u16 改成跟第 1 個一樣，長度、scaleCount、
   * outlierCount、payloadLen 全都沒動，所有既有檢查都會放行。
   *
   * 後果：非離群 channel 的還原迴圈少跳過一個 channel，之後每個 channel 的
   * int8 碼都錯位一格。實測 d=576/K=4 有 2123/2304 個值是錯的、最大偏差 7.4e12，
   * 外加 Int8Array 讀到尾端外面產生的一個 NaN —— 而且完全不報錯。
   */
  const dModel = 576;
  const qLen = 4;
  const x = makeTensor(dModel, qLen, 4242);
  const good = encode(x, dModel, 'per-ch+outlier', { scalePrecision: 'fp32' });
  const h = peekHeader(good);

  const copy = good.slice(0);
  const dv = new DataView(copy);
  const first = dv.getUint16(frameHeaderSize, true);
  dv.setUint16(frameHeaderSize + 2, first, true); // 第 2 個索引 = 第 1 個

  // 先證明「每個 header 欄位都還自洽」，不然這個測試只是在測一個顯然壞掉的 frame
  const hc = peekHeader(copy);
  assert.equal(copy.byteLength, good.byteLength, '長度沒變');
  assert.equal(hc.outlierCount, h.outlierCount, 'outlierCount 沒變');
  assert.equal(hc.scaleCount, dModel - hc.outlierCount, 'scaleCount 仍然自洽');
  assert.equal(hc.payloadLen, copy.byteLength - frameHeaderSize, 'payloadLen 仍然自洽');

  assert.throws(
    () => decode(copy),
    (err) => {
      assert.match(err.message, /重複/, '錯誤訊息要說出是「重複」');
      assert.match(err.message, new RegExp(String(first)), '錯誤訊息要說出是哪個索引');
      return true;
    },
    '重複的離群索引必須丟錯 —— 放行的話 2123/2304 個值是錯的而且沒人會發現',
  );

  // 只有重複要擋，合法的索引集合不能被誤傷
  assert.doesNotThrow(() => decode(good));
});

// ---------------------------------------------------------------------------
// 11. 錯誤訊息要能操作
// ---------------------------------------------------------------------------

test('錯誤訊息：佈局不符時說出是哪個欄位，不准出現 -1 這種哨兵值', () => {
  /*
   * 舊版 `expectedPayloadLen()` 用 -1 當「header 自相矛盾」的哨兵值，
   * 而 decode 直接把它插進訊息裡：
   *   「payload 長度 1024 與 header 描述的佈局不符（應為 -1）」
   * -1 不是一個長度。看到它的人只能來讀 wire.js 才知道發生了什麼事 ——
   * 而「是哪個欄位對不上、正確值是多少」在那個函式裡明明算得出來。
   */
  const dModel = 128;
  const qLen = 4;
  const x = makeTensor(dModel, qLen, 606);

  const bend = (scheme, mutate) => {
    const buf = encode(x, dModel, scheme, { scalePrecision: 'fp32' }).slice(0);
    mutate(new DataView(buf));
    try {
      decode(buf);
    } catch (err) {
      return err.message;
    }
    return assert.fail(`${scheme}：改壞的 header 必須丟錯`);
  };

  const cases = [
    // [說明, 方案, 改哪裡, 訊息裡必須出現的字]
    ['per-channel 的 scaleCount 不等於 dModel', 'per-channel',
      (dv) => dv.setUint32(24, 1, true), [/scaleCount/, /dModel/, /128/]],
    ['per-channel 卻帶了 outlierCount', 'per-channel',
      (dv) => dv.setUint32(28, 3, true), [/outlierCount/]],
    ['group-64 的 scaleCount 不是 K × ceil(d/64)', 'group-64',
      (dv) => dv.setUint32(24, 5, true), [/scaleCount/, /group-64/, /8/]],
    ['none 不該有 scale 區塊', 'none',
      (dv) => dv.setUint32(24, 2, true), [/scaleCount/]],
    ['per-ch+outlier 的 outlierCount 是 0', 'per-ch+outlier',
      (dv) => dv.setUint32(28, 0, true), [/outlierCount/]],
    ['per-ch+outlier 的 scaleCount 對不上 d - nOut', 'per-ch+outlier',
      (dv) => dv.setUint32(24, 7, true), [/scaleCount/, /outlierCount/]],
  ];

  for (const [label, scheme, mutate, wants] of cases) {
    const msg = bend(scheme, mutate);
    assert.ok(!msg.includes('-1'), `${label}：訊息裡不准出現 -1 哨兵值 —— 「${msg}」`);
    assert.match(msg, /[一-鿿]/, `${label}：錯誤訊息必須是中文`);
    assert.match(msg, /佈局/, `${label}：訊息要指出是佈局的問題`);
    for (const want of wants) {
      assert.match(msg, want, `${label}：訊息要說出對不上的是哪個欄位 —— 「${msg}」`);
    }
    // 只說「壞了」不夠，要說下一步：不要解，並指出最可能的成因
    assert.match(msg, /不要嘗試解/, `${label}：訊息要說出下一步怎麼做`);
  }

  // 佈局本身成立、只是 payloadLen 被改小的那條路徑，訊息要給得出真正的長度
  const truncated = encode(x, dModel, 'per-channel', { scalePrecision: 'fp32' }).slice(0);
  new DataView(truncated).setUint32(32, 8, true);
  assert.throws(() => decode(truncated), (err) => {
    assert.ok(!err.message.includes('-1'), `不准出現 -1：「${err.message}」`);
    assert.match(err.message, /長度不符/);
    return true;
  });
});

// ---------------------------------------------------------------------------
// 以下四個測試補的是 mutation testing 找到的漏洞：這四個 mutation 在 38 個
// 測試全綠的情況下存活下來。它們共通的形狀都是「encode 與 decode 彼此同意，
// 但兩邊一起違反規格」—— 和第 17 號測試（header 絕對位移）要防的是同一類，
// 只是當時只釘了 header，沒釘 payload 區塊。

test('decode：payloadLen 與佈局對不上時要丟錯（這條分支先前完全沒被測到）', () => {
  // mutation：把 `if (layout.bytes !== h.payloadLen)` 改成 `if (false)`，
  // 38 個測試依然全綠 —— 而這是整支檔案最吃重的一道防線。
  //
  // 重現方式：一則合法的 per-channel frame，只改 qLen（u32@16）4 -> 3。
  // scaleCount 仍等於 dModel、payloadLen 仍等於真正的位元組數，
  // 所以前面每一道檢查都過；停掉這條分支的話 decode() 會安靜地回傳
  // 384 個值而不是 512 個，而且不報錯。
  const dModel = 128;
  const buf = encode(makeTensor(dModel, 4, 7), dModel, 'per-channel', { scalePrecision: 'fp32' });
  const dv = new DataView(buf);
  assert.equal(dv.getUint32(16, true), 4, '前提：qLen 在位移 16');
  dv.setUint32(16, 3, true);

  assert.throws(() => decode(buf), (err) => {
    assert.match(err.message, /佈局不符/);
    // 順便釘住 E 的修正：這條分支也不准出現 -1 哨兵值。
    // 先前只有 layout.error 那條分支被測到，所以把 -1 塞回這裡也能存活。
    assert.ok(!err.message.includes('-1'), `不准出現 -1：「${err.message}」`);
    assert.match(err.message, /\d+ 位元組/, '要說出正確的位元組數是多少');
    return true;
  });
});

test('scheme 0 的 fp32 本體是 little-endian（§4.5.7 回程走的就是這條）', () => {
  // mutation：把 encodeNone 的 setFloat32 與 decode 的 getFloat32 一起翻成
  // big-endian，38 個測試全綠 —— 因為本體只被往返測試驗過，encoder 與
  // decoder 可以彼此同意卻一起違反規格。
  //
  // 這一條特別要緊：scheme 0 是 §4.5.7 的 logits 回程，是最可能被另一個
  // 實作讀到的訊息。
  const buf = encode(Float32Array.from([1.5, -2.0]), 2, 'none', {});
  const u8 = new Uint8Array(buf, frameHeaderSize);
  // 1.5 的 fp32 是 0x3FC00000，little-endian 就是 00 00 C0 3F
  assert.deepEqual(Array.from(u8.slice(0, 4)), [0x00, 0x00, 0xc0, 0x3f], '1.5 必須是 00 00 C0 3F');
  // -2.0 是 0xC0000000 -> 00 00 00 C0
  assert.deepEqual(Array.from(u8.slice(4, 8)), [0x00, 0x00, 0x00, 0xc0], '-2.0 必須是 00 00 00 C0');
});

test('scheme 3 的 fp16 離群本體是 little-endian', () => {
  // 同一類 mutation：fp16 scale 區塊、fp32 scale 區塊、離群索引區塊都已經
  // 被現有測試殺掉了，唯獨離群「本體」沒有任何絕對位元組斷言。
  //
  // d=4 時 nOut = max(1, floor(4 * 0.03)) = 1，所以 channel 0（最大值）是離群。
  // 佈局（§4.5.4，fp16 scale）：
  //   40  索引    1 x u16 = 2
  //   42  scale   3 x 2   = 6
  //   48  fp16 本體 1 x 1 x 2 = 2   <- 要驗的就是這兩個位元組
  //   50  int8 本體 3 x 1 = 3
  const buf = encode(Float32Array.from([1000, 1, 2, 3]), 4, 'per-ch+outlier', { scalePrecision: 'fp16' });
  const dv = new DataView(buf);
  assert.equal(dv.getUint32(28, true), 1, '前提：outlierCount = 1');
  assert.equal(dv.getUint16(frameHeaderSize, true), 0, '前提：離群的是 channel 0');

  const u8 = new Uint8Array(buf, frameHeaderSize + 2 + 3 * 2, 2);
  // 1000 = 1.953125 x 2^9 -> e=24, mantissa=976 -> 0x63D0 -> LE 是 D0 63
  assert.equal(floatToFp16(1000), 0x63d0, '前提：fp16(1000) = 0x63D0');
  assert.deepEqual(Array.from(u8), [0xd0, 0x63], 'fp16 離群本體必須是 little-endian');
});

test('§4.5.2 的保留位元必須是 0，不准靜默接受', () => {
  const buf = encode(makeTensor(64, 2, 11), 64, 'group-64', {});

  const withFlag = buf.slice(0);
  const dv1 = new DataView(withFlag);
  dv1.setUint8(7, dv1.getUint8(7) | 0x02); // 設一個「未來的」flag bit
  assert.throws(() => decode(withFlag), /保留位元|保留/, 'flags bit1..7 不是 0 就該拒絕');

  const withReserved = buf.slice(0);
  const dv2 = new DataView(withReserved);
  dv2.setUint16(38, 0xbeef, true);
  assert.throws(() => decode(withReserved), /保留欄位|位移 38/, '位移 38 不是 0 就該拒絕');

  // 對照組：原本那一則仍然解得開，證明上面兩個拒絕不是誤殺
  assert.equal(decode(buf).qLen, 2);
});
