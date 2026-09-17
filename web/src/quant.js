/**
 * 激活值量化 —— 這是 **M4 的品質模擬器，不是線路格式的實作**。
 *
 * `SCHEMES.fn` 回傳 `Float32Array`（量化 → 反量化的往返），這個檔案
 * 從頭到尾沒有產生過任何一個位元組。真正把張量變成位元組的是
 * `web/src/wire.js`，格式定義在 `docs/01-architecture.md` §4.5。
 * 要改線路格式請去那裡改，不是改這裡。
 *
 * ⚠ **這段註解原本寫「必須與 spike/quant_schemes.py 逐位元對應」。那是假的。**
 * 三個地方對不上。寫下來是因為「以為逐位元相等」比「知道不相等」危險得多 ——
 * 真的照那句話去寫一個 assert 兩邊相等的測試，會先懷疑測試、懷疑載入路徑、
 * 懷疑浮點累加順序，最後才發現是那句註解在說謊：
 *
 *   1. **捨入方向不同。** JS 的 `Math.round` 是 half-up（.5 一律往 +∞），
 *      `torch.round` 是 half-to-even。
 *      `torch.round([2.5, -2.5, 3.5, 0.5])` 得到 `[2, -2, 4, 0]`，
 *      同樣的輸入在 JS 是 `[3, -2, 4, 1]` —— 四個裡有三個不一樣。
 *      這條吃到每一個 `qdq()`，也就是每一個被量化的值。
 *   2. **fp16 往返不同。** 細節見 `fp16Roundtrip()` 上面的註解。
 *   3. **離群比例的預設值不同。** Python 的 `per_channel_outlier_fp16`
 *      預設 `outlier_frac=0.01`（`spike/quant_schemes.py:96-97`），
 *      JS 這邊是 `0.03`。所以連「哪些 channel 算離群」都不一定一樣。
 *      §4.5 的 frame header 因此必須明確帶 `outlierCount` 欄位，
 *      不能讓接收端拿一個「大家都知道的」比例去反推 —— 沒有那個比例。
 *
 * 兩邊真正該成立的關係是**統計上等價**：同一個方案、同一份語料，
 * PPL 退化與 argmax 一致率要落在同一個結論裡。逐位元相等的那個要求
 * 屬於 `wire.js` 的 encode/decode 往返（§4.5.5），不屬於這裡。
 *
 * M4 實測結論（見 docs/00-feasibility.md §6）：
 *   - per-tensor int8 不可用 —— PPL 暴增 30444%
 *   - scale 必須「逐 token」適應，因為巨值集中在特定 token
 *   - 預設格式：per-channel int8 + 前 3% 通道保留 fp16
 *     （本體 8.24 bits/值；**不含**中繼資料，見下方 `SCHEMES` 的註解）
 */

/**
 * 對稱量化的 quantize -> dequantize 往返。scale 需能廣播。
 *
 * ⚠ `Math.round` 是 half-up，`torch.round` 是 half-to-even ——
 * 恰好落在中點的值兩邊會差一個量化階。這是檔頭第 1 點的來源。
 */
function qdq(value, scale, qmax) {
  if (scale === 0) return 0; // 整片都是 0
  const q = Math.max(-qmax - 1, Math.min(qmax, Math.round(value / scale)));
  return q * scale;
}

/**
 * fp16 往返 —— 模擬用半精度傳輸造成的精度損失。
 *
 * ⚠ **這不是 IEEE 754 的 fp16 轉換，也不等於 `torch.Tensor.to(float16)`。**
 * 兩處偏離列在這裡，免得有人把它當成可用的 fp16 編碼器搬去寫線路
 * （真正要寫進 frame 的 fp16 在 `web/src/wire.js`）：
 *
 *   - 尾數捨入是 **half-up**：`(mant + 0x1000) >>> 13` 對低 13 bits
 *     無條件加半再截斷，中點一律進位。IEEE 與 torch 都是
 *     round-to-nearest-**even**，所以中點的值會差 1 ulp。
 *   - **非正規數（subnormal）直接沖成 ±0**，torch 會如實保留。
 *
 * 對 M4 要量的東西（PPL、argmax 一致率）這兩項的影響淹沒在雜訊裡，
 * 所以沒有動它；但**不能**拿它去支持「和 Python 端逐位元相同」的說法。
 */
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
export function fp16Roundtrip(x) {
  _f32[0] = x;
  const bits = _u32[0];
  const sign = bits >>> 31;
  let exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;

  if (exp === 0xff) return x; // Inf / NaN 原樣通過
  // 轉成 fp16 的指數範圍後再轉回來
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign ? -Infinity : Infinity; // 溢位
  if (e <= 0) return sign ? -0 : 0; // 下溢沖成 0；torch 會保留非正規數，這裡刻意不追
  // fp16 只有 10 bits 尾數：截掉低 13 bits。注意這是 half-up，不是最近偶數
  const round = (mant + 0x1000) >>> 13;
  mant = (round & 0x3ff) << 13;
  if (round > 0x3ff) e += 1;
  _u32[0] = (sign << 31) | ((e - 15 + 127) << 23) | mant;
  return _f32[0];
}

/**
 * 每個 hidden 維度一個 scale，scale 跨所有 token 共用。
 *
 * ⚠ M4 實測這個方案單獨使用時表現不佳（PPL +6.23%），比 group-wise 還差。
 * 原因是 scale 跨 token 共用：一個 token 的巨值會把該 channel 的 scale
 * 對所有 token 都撐大。保留它是為了與 Python 版對齊、以及當作
 * per-channel+outlier 的基底。
 */
export function perChannel(x, dModel, bits = 8) {
  const qmax = (1 << (bits - 1)) - 1;
  const n = x.length;
  const nTok = n / dModel;
  const out = new Float32Array(n);

  const scale = new Float32Array(dModel);
  for (let c = 0; c < dModel; c++) {
    let m = 0;
    for (let t = 0; t < nTok; t++) m = Math.max(m, Math.abs(x[t * dModel + c]));
    scale[c] = m / qmax;
  }
  for (let t = 0; t < nTok; t++) {
    for (let c = 0; c < dModel; c++) {
      const i = t * dModel + c;
      out[i] = qdq(x[i], scale[c], qmax);
    }
  }
  return out;
}

/**
 * 把 channel 切成固定大小的組，每組、每個 token 各一個 scale。
 *
 * M4 實測：這是最好的「純 int8、零額外頻寬」選項（PPL +0.48%）。
 * d_model 不是 group 整數倍時，尾巴自成一組（與 Python 版行為一致 ——
 * 早期版本這裡會靜默退回 per-channel，是個真實踩過的 bug）。
 */
export function groupWise(x, dModel, bits = 8, group = 64) {
  const qmax = (1 << (bits - 1)) - 1;
  const nTok = x.length / dModel;
  const out = new Float32Array(x.length);

  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (let g0 = 0; g0 < dModel; g0 += group) {
      const g1 = Math.min(g0 + group, dModel);
      let m = 0;
      for (let c = g0; c < g1; c++) m = Math.max(m, Math.abs(x[base + c]));
      const scale = m / qmax;
      for (let c = g0; c < g1; c++) out[base + c] = qdq(x[base + c], scale, qmax);
    }
  }
  return out;
}

/**
 * 預設方案：per-channel int8 + 數值最大的一小撮 channel 保留 fp16。
 * （「預設」指的是 M4 選出來的量化方案，不是說這個函式產生線路位元組 ——
 *   它回傳的一樣是 Float32Array，見檔頭。）
 *
 * M4 實測最佳：PPL 退化 0.03%、argmax 一致 99.32%，
 * 本體成本僅比純 int8 多 3%。
 *
 * ⚠ `outlierFrac` 預設 **0.03**，Python 版的 `per_channel_outlier_fp16`
 * 預設是 **0.01**（`spike/quant_schemes.py:96-97`）。兩邊挑出來的離群集合
 * 因此不同，所以 frame header 才要明確帶 `outlierCount`（§4.5.2）。
 */
export function perChannelOutlierFp16(x, dModel, bits = 8, outlierFrac = 0.03) {
  const nTok = x.length / dModel;

  // 先算每個 channel 的最大絕對值，用來挑離群
  const mag = new Float32Array(dModel);
  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (let c = 0; c < dModel; c++) {
      const a = Math.abs(x[base + c]);
      if (a > mag[c]) mag[c] = a;
    }
  }
  const nOut = Math.max(1, Math.floor(dModel * outlierFrac));
  // 取前 nOut 大的 channel（用索引排序，dModel 不大所以直接排）
  const idx = Array.from({ length: dModel }, (_, i) => i)
    .sort((a, b) => mag[b] - mag[a])
    .slice(0, nOut);
  const isOutlier = new Uint8Array(dModel);
  for (const i of idx) isOutlier[i] = 1;

  const out = perChannel(x, dModel, bits);
  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (const c of idx) out[base + c] = fp16Roundtrip(x[base + c]);
  }
  return out;
}

/**
 * 名稱 -> { fn, wireBits }。
 *
 * ⚠ **`wireBits` 是「本體」的位元數，不是真正會上線路的位元數。**
 * 它沒算 scale 表，也沒算離群 channel 的索引表 —— 而這兩樣都必須跟著
 * 每一則訊息走：`scale[c]` 是對「這則訊息裡的這 K 個 token」取
 * `max|x[t][c]|` 算的，接收端推不出來也不能快取；離群集合每則訊息重挑一次。
 *
 * 真實的位元組帳由 `bench/wire_size.py` 算、由 `web/src/wire.js` 產生，
 * 格式見 `docs/01-architecture.md` §4.5。差距不是小數點等級：K=8 時
 * `wire` 方案真正的成本是 **10.24 bits/值**，不是這裡寫的 8.24 —— 差 24%。
 * 而且把中繼資料算進去之後 **group-64 反而比較省**，方案排名會整個反轉。
 *
 * 那為什麼不把數字改成對的？因為這兩個值的用途是 **M4 的品質比較** ——
 * 在「大家都只算本體」這同一把尺下比 PPL 與 argmax 一致率，尺歪得一致就還能比。
 * 拿它去做部署決策或估頻寬才是誤用，那時該去查 §4.5 的表。
 */
export const SCHEMES = {
  none: { fn: (x) => x, wireBits: 32, label: 'fp32（不量化）' },
  'per-channel': {
    fn: (x, d) => perChannel(x, d, 8),
    wireBits: 8,
    label: 'per-channel int8',
  },
  'group-64': {
    fn: (x, d) => groupWise(x, d, 8, 64),
    wireBits: 8,
    label: 'group-64 int8（純 int8 最佳）',
  },
  wire: {
    fn: (x, d) => perChannelOutlierFp16(x, d, 8, 0.03),
    wireBits: 8.24,
    label: 'per-channel + 3% 通道 fp16（預設）',
  },
};

export const DEFAULT_SCHEME = 'wire';

/**
 * 這個張量若用該方案傳輸，**本體**佔多少位元組。
 *
 * ⚠ 不是實際會送出去的位元組數：沒有 frame header（32 B）、沒有 chunk
 * 子標頭（每 16 KiB 一個 8 B）、沒有 scale 表、沒有離群索引表。
 * 要真實數字請跑 `bench/wire_size.py`，或直接量 `wire.js` 編出來的長度。
 */
export function wireBytes(numel, schemeName) {
  const s = SCHEMES[schemeName] ?? SCHEMES[DEFAULT_SCHEME];
  return Math.ceil((numel * s.wireBits) / 8);
}
