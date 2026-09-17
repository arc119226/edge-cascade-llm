/**
 * Frame 編解碼 —— `docs/01-architecture.md` §4.5 制定的線路格式。
 *
 * 這一層只管「一則語意完整的訊息長什麼樣」。分片（§4.5.6）在它底下，
 * 所以這支檔案完全不知道 DataChannel 的存在，可以用純 node 單元測試 ——
 * M1/M2 的教訓是「先確定算得對，再加網路」，格式這一層也一樣。
 *
 * ⚠ 這裡**不是** `quant.js` 的替代品，兩者的職責必須分清楚：
 *   - `quant.js` 是 M4 的**品質模擬器**，`SCHEMES.fn` 回傳 Float32Array，
 *     從來沒有產生過任何位元組。
 *   - 這支檔案產生真正要送上線的位元組。
 * 兩者的綁定關係寫在 §4.5.5，也寫在 test/wire.test.mjs 裡 ——
 * 不是無條件相等，因為 §4.5.5 規定「先把 scale 捨入到線路精度、再量化」，
 * 而 `quant.js` 只有 perChannel 剛好做到（它的 scale 存在 Float32Array 裡），
 * groupWise 的 scale 是個 double（quant.js:92），那個值根本送不上線。
 */

/**
 * header 固定 40 位元組（§4.5.2）。
 *
 * 初版是 32 位元組、`dModel` / `qLen` / `scaleCount` / `outlierCount` 都是 u16，
 * 對抗性審查一行輸入就打穿了：§4.5.7 的回程要送完整 logits，而 logits 的列寬是
 * `vocab_size`。SmolLM2-135M 的 49152 塞得進 u16，所以本機怎麼測都不會紅 ——
 * 但 Llama 3 是 128256、Qwen 2.5 是 151936、Gemma 是 256000，全部塞不下，
 * u16 版的 frame **無法表達 §4.5.7 自己規定的訊息**。
 * `scaleCount` 同理：group-64 是 `K × ceil(d/64)`，d=8192/K=512 就是 65536，剛好溢位。
 * 那四個欄位因此改成 u32、`stageIndex` 移到尾端當 u16，header 從 32 變 40。
 * 多 8 個位元組換掉一個「之後只能靠升版本才能修」的協定缺陷，現在改是免費的。
 */
export const frameHeaderSize = 40;

/** 'ECL1'。little-endian 寫入後位元組依序是 45 43 4C 31，所以用 hex dump 看得懂。 */
export const FRAME_MAGIC = 0x314c4345;

export const FRAME_VERSION = 1;

/** §4.5.2 的 schemeId。名稱是規格用的名稱，不是 quant.js 的 key。 */
export const SCHEME_IDS = Object.freeze({
  none: 0,
  'per-channel': 1,
  'group-64': 2,
  'per-ch+outlier': 3,
});

/** id -> 名稱。索引即 schemeId。 */
export const SCHEME_NAMES = Object.freeze(['none', 'per-channel', 'group-64', 'per-ch+outlier']);

/**
 * `quant.js` 把 schemeId 3 叫做 `wire`、`bench/wire_size.py` 也叫 `wire`，
 * 但 §4.5.2 的欄位名是 `per-ch+outlier`。三個地方三種叫法是既成事實，
 * 所以這裡明確收下別名，而不是讓呼叫端在 `SCHEMES['per-ch+outlier']` 上拿到 undefined。
 */
const SCHEME_ALIASES = Object.freeze({ wire: 'per-ch+outlier', 'per-ch-outlier': 'per-ch+outlier' });

/** §4.5.4 的 S：flags bit0 決定 scale 的位元組數。 */
const SCALE_BYTES = Object.freeze({ fp16: 2, fp32: 4 });

/** group-64 的組大小（§4.5.4 / quant.js:81）。 */
const GROUP = 64;

/** int8 對稱量化，與 quant.js:54 的 `(1 << (bits - 1)) - 1` 同義。 */
const QMAX = 127;

/** 與 quant.js:117 完全一致的離群比例。改這個值會改變線路位元組數，不要偷偷調。 */
const DEFAULT_OUTLIER_FRAC = 0.03;

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/** 把 JS double 夾成 fp32。量化必須在 fp32 上做，否則對不上 quant.js 的 Float32Array。 */
function toF32(v) {
  _f32[0] = v;
  return _f32[0];
}

/**
 * fp32 -> fp16 位元樣式。round-to-nearest-even、保留非正規數、Inf/NaN 正確。
 *
 * **不要改用 `quant.js` 的 `fp16Roundtrip`。** 它用 `(mant + 0x1000) >>> 13`
 * 做 half-up 捨入（平手一律進位，不是進到偶數），而且 `e <= 0` 時直接回 ±0
 * 把非正規數沖掉。那對 M4 的品質模擬夠用，但線路格式的兩端必須逐位元一致，
 * half-up 會讓 1 + 2^-11 變成 0x3C01（正解是 0x3C00），
 * 2^-24 變成 0（正解是 0x0001）。這種差一個 ULP 的錯在流水線上會慢慢累積成
 * 「輸出看起來合理、只是慢慢偏掉」，正是最難查的那一類。
 *
 * **溢位一律飽和到 ±65504，有限的輸入永遠不會編成 Inf（§4.5.5b）。**
 * 初版照 IEEE 的作法讓 |x| ≥ 65520 進位成 0x7C00，兩條路徑因此靜默壞掉：
 *   1. 方案 3 的離群本體是 fp16，`x[137] = 65520` 編出 Inf，
 *      `decode().data[137]` 就是 Infinity —— 而諷刺的是這條路徑存在的理由
 *      （§4.2：巨值集中在少數 channel）正好就是它會爆的場景，
 *      同一個值走 int8 路徑毫無問題。
 *   2. scale 走 `roundScale()` 也是 fp16：channel 的 max|x| 到
 *      `127 × 65520 = 8321040` 時 scale 變 Inf，`x / Inf` 量化成 0 碼，
 *      解碼端算 `0 × Inf` = **NaN**，整個 channel（group-64 是整組 64 個）
 *      全變 NaN，而且兩端都不報錯。
 * 飽和之後最大可表示的 scale 是 65504，也就是 `127 × 65504 = 8319008`；
 * 再大的值 int8 碼夾在 ±127，會損失數值大小，但保持有限 —— 這是刻意的取捨：
 * 偏小的數字看得出來，NaN 沿著 hop 傳播則是這個專案最難查的失敗模式。
 *
 * Inf / NaN 的**輸入**仍然編成 0x7C00 / 0x7E00（那是它們的 IEEE 位元樣式，
 * 解碼端要認得），但它們永遠不會從 `encode()` 走到這裡：§4.5.5b 要求
 * `encode()` 在掃 max|x| 時就把非有限輸入丟回給呼叫端。
 */
export function floatToFp16(value) {
  _f32[0] = value; // 先落到 fp32：線路上的來源本來就是 fp32，雙重捨入要從這裡開始
  const bits = _u32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exp = (bits >>> 23) & 0xff;
  const mant = bits & 0x7fffff;

  if (exp === 0xff) {
    // NaN 的尾數若被截光就會變成 Inf，所以強制保留一個 quiet bit。
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }

  let e = exp - 127 + 15;

  // 指數大到連捨入都救不回來（>= 2^16）：飽和到最大有限值，不是 Inf（§4.5.5b）。
  if (e >= 0x1f) return sign | 0x7bff;

  if (e > 0) {
    // 正規數：23 bits 尾數捨到 10 bits。
    let m = mant >>> 13;
    const rem = mant & 0x1fff;
    const half = 0x1000;
    // 平手時進到偶數 —— 這一行就是與 quant.js 唯一但致命的差別。
    if (rem > half || (rem === half && (m & 1) === 1)) m += 1;
    if (m === 0x400) {
      // 尾數進位溢出到指數。65520 走的就是這條路：它剛好在 65504 與 65536 中間，
      // ties-to-even 選偶數的 65536 —— 也就是 fp16 表示不了的地方。
      // 這裡不給 Inf 而是飽和到 0x7BFF（§4.5.5b）：65520 是有限的 fp32 值，
      // 送上線變成 Infinity 是靜默的數值毀損。65504 本身（rem=0）走不到這裡。
      m = 0;
      e += 1;
      if (e >= 0x1f) return sign | 0x7bff;
    }
    return sign | (e << 10) | m;
  }

  // e <= 0：落在 fp16 的非正規數區間，或小到連最小非正規數的一半都不到。
  // 值 = (mant | 0x800000) * 2^(exp-150)，而 fp16 非正規數的階是 2^-24，
  // 所以要右移 (126 - exp) 位。
  const shift = 126 - exp;
  if (shift > 24) return sign; // 連 2^-25 都不到：捨成 ±0，但符號要留著（-0 不是 0）
  const full = mant | 0x800000;
  let m = full >>> shift;
  const rem = full & ((1 << shift) - 1);
  const half = 1 << (shift - 1);
  if (rem > half || (rem === half && (m & 1) === 1)) m += 1;
  // m 進位成 0x400 時，`sign | 0x400` 剛好就是最小正規數的位元樣式（exp=1, mant=0），
  // 不需要特判 —— IEEE 754 的編碼刻意讓非正規數與正規數在這裡接得上。
  return sign | m;
}

/**
 * fp16 位元樣式 -> JS number。
 *
 * 與 `runner.js:376` 的 `fp16ToFloat` 同一套算式（那支是私有的，不想為了共用
 * 去動 M1 已經驗過的檔案），但這裡多處理了 frac=0 的非正規數要給出 ±0。
 */
export function fp16ToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -24 * frac; // frac=0 時是 ±0，符號靠 sign 帶出來
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

/** 名稱或 id -> { id, name }。錯的輸入要在這裡就炸掉，不要等到 payload 寫壞。 */
function resolveScheme(scheme) {
  if (typeof scheme === 'number') {
    const name = SCHEME_NAMES[scheme];
    if (name === undefined) {
      throw new Error(
        `未知的 schemeId ${scheme}。§4.5.2 只定義 0..3：` +
        `${SCHEME_NAMES.map((n, i) => `${i}=${n}`).join(' ')}`,
      );
    }
    return { id: scheme, name };
  }
  const name = SCHEME_ALIASES[scheme] ?? scheme;
  const id = SCHEME_IDS[name];
  if (id === undefined) {
    throw new Error(
      `未知的量化方案 '${scheme}'。可用：${SCHEME_NAMES.join(' / ')}` +
      `（quant.js 與 bench/wire_size.py 把 per-ch+outlier 叫作 'wire'，這裡兩種都收）`,
    );
  }
  return { id, name };
}

function resolvePrecision(p) {
  const prec = p ?? 'fp16';
  if (SCALE_BYTES[prec] === undefined) {
    throw new Error(`scalePrecision 只能是 'fp16' 或 'fp32'，收到 '${p}'`);
  }
  return prec;
}

/** 與 quant.js:117 / wire_size.py:46 一致：max(1, floor(d * frac))，再夾住不超過 d。 */
function outlierCountFor(dModel, frac) {
  return Math.min(dModel, Math.max(1, Math.floor(dModel * frac)));
}

/**
 * §4.5.5 的規範性語意：**先把 scale 捨入到要送出去的精度，再拿它去量化。**
 *
 * 少了這一步，編碼端用 fp32 scale 算 q、解碼端用 fp16 scale 還原，
 * 兩邊差一個 scale 的捨入誤差，而且那個誤差會被 q（最大 127）放大 ——
 * `decode(encode(x))` 就不等於編碼端自己算出來的東西。
 *
 * fp16 那條順帶吃下 §4.5.5b 的飽和規則：`floatToFp16` 不再產生 Inf，
 * 所以 scale 最多到 65504。少了它，max|x| ≥ 127 × 65520 的 channel 會拿到
 * Inf scale，解碼端 `0 × Inf` 解出整個 channel 的 NaN（實測 d=2、x=[8.4e6,1.5]
 * 就重現得出來，兩端都不報錯）。
 */
function roundScale(value, precision) {
  return precision === 'fp16' ? fp16ToFloat(floatToFp16(value)) : toF32(value);
}

function writeScale(dv, off, value, precision) {
  if (precision === 'fp16') dv.setUint16(off, floatToFp16(value), true);
  else dv.setFloat32(off, value, true);
}

function readScale(dv, off, precision) {
  return precision === 'fp16' ? fp16ToFloat(dv.getUint16(off, true)) : dv.getFloat32(off, true);
}

/**
 * 對稱量化的 quantize 半邊，與 quant.js:16-20 的 `qdq` 前半完全同義。
 *
 * `scale === 0` 的來源是整片都是 0 的張量（padding、還沒寫入的 KV 位置），
 * 0/0 會給出 NaN。誠實地說：拿掉這一行**目前也不會有 NaN 流出去**，
 * 因為 `Int8Array` 的整數轉換會把 NaN 變成 0 —— 實際試過。
 * 留著它有兩個理由：一是與 quant.js:17 的寫法對齊（兩邊要看得出是同一個決定），
 * 二是哪天本體改成 bit-packing 到普通陣列（4-bit 方案），那層意外的保護就沒了。
 * 不要因為「反正 Int8Array 會擋」就把它刪掉。
 */
function quantize(value, scale) {
  if (scale === 0) return 0;
  const q = Math.round(value / scale);
  return q < -128 ? -128 : q > 127 ? 127 : q;
}

/** payload 的區塊尺寸。與 bench/wire_size.py 的 layout() 是同一張帳，兩邊由測試釘死。 */
function planPayload(id, dModel, qLen, sBytes, outlierFrac) {
  const numel = dModel * qLen;
  if (id === 0) return { scaleCount: 0, outlierCount: 0, bytes: 4 * numel };
  if (id === 1) return { scaleCount: dModel, outlierCount: 0, bytes: dModel * sBytes + numel };
  if (id === 2) {
    // d 不是 64 的整數倍時尾巴自成一組（quant.js:88-89 記過這個真實 bug）。
    const groups = Math.ceil(dModel / GROUP);
    const scaleCount = qLen * groups;
    return { scaleCount, outlierCount: 0, bytes: scaleCount * sBytes + numel };
  }
  const nOut = outlierCountFor(dModel, outlierFrac);
  const nIn = dModel - nOut;
  return {
    scaleCount: nIn,
    outlierCount: nOut,
    // 索引一律用 u16 表，因為 §4.5.4 只定義了這一種。
    //
    // 這裡原本寫著「wire_size.py 取 min(nOut*2, ceil(d/8)) 允許改用 bitmap，
    // 但 3% 的離群比例下 u16 表永遠勝出，若哪天 outlierFrac 調到 6.25% 以上
    // 兩邊才會對不上，而位元組帳測試會立刻抓到」。**那兩句都是假的**：
    //   1. 真正生效的比例不是 outlierFrac，而是 `max(1, floor(d*frac)) / d` ——
    //      那個 max(1) 讓小 d 的實際比例遠高於 3%，d < 34 時就已經超過 6.25%。
    //      實測 d ∈ {1,2,4,8} 兩邊每則訊息就差 1 個位元組（16 種組合全中）。
    //   2. 位元組帳測試當時只測 d ∈ {576, 5120, 100, 577}，永遠看不到這件事。
    // 現在的結論是 wire_size.py 錯（規格沒定義 bitmap），它已改成無條件 nOut*2，
    // 而位元組帳測試加上了 d ∈ {1,2,4,8}。要改成 bitmap 得先改 §4.5.4。
    bytes: nOut * 2 + nIn * sBytes + nOut * qLen * 2 + nIn * qLen,
  };
}

function checkU16(name, value) {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff) {
    throw new Error(`${name} 必須是 0..65535 的整數，收到 ${value}（frame header 只有 2 位元組）`);
  }
}

function checkU32(name, value) {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`${name} 必須是 0..4294967295 的整數，收到 ${value}`);
  }
}

/**
 * 把一個 `[qLen, dModel]` 的激活值張量編成一則 frame。
 *
 * 本體一律 token-major（t 外層、c 內層），對應 `x[t * dModel + c]`（§4.5.4）。
 *
 * @param {Float32Array|ArrayLike<number>} x
 * @param {number} dModel
 * @param {string|number} scheme  名稱或 schemeId
 * @param {{msgType?:number, roundId?:number, stageIndex?:number, startPosition?:number,
 *          scalePrecision?:'fp16'|'fp32', outlierFrac?:number}} [meta]
 * @returns {ArrayBuffer}
 */
export function encode(x, dModel, scheme, meta = {}) {
  const src = x instanceof Float32Array ? x : Float32Array.from(x);
  const { id, name } = resolveScheme(scheme);
  const precision = resolvePrecision(meta.scalePrecision);
  const sBytes = SCALE_BYTES[precision];
  const outlierFrac = meta.outlierFrac ?? DEFAULT_OUTLIER_FRAC;

  checkU32('dModel', dModel);
  if (dModel === 0) throw new Error('dModel 不能是 0');
  // 方案 3 的離群索引區塊是 u16（§4.5.4），dModel 卻已經是 u32 —— 這個縫必須明講。
  // 不擋的話 `setUint16` 會靜默地把 70000 寫成 4464，解碼端讀到一個合法但完全
  // 不相干的 channel，張量只是「有點怪」。u32 的是 header 的計數欄位，不是索引表。
  if (id === 3 && dModel > 0x10000) {
    throw new Error(
      `dModel ${dModel} 超過 65536，per-ch+outlier 送不上線：` +
      '§4.5.4 的離群索引區塊是 u16，表示不了這麼大的 channel 索引。' +
      '這種寬度（logits 回程）請改用 group-64 或 none',
    );
  }
  if (src.length % dModel !== 0) {
    throw new Error(
      `張量長度 ${src.length} 不是 dModel (${dModel}) 的整數倍 —— ` +
      '本體是 token-major 的 [qLen, dModel]，請確認沒有把 batch 維度一起攤平進來',
    );
  }
  const qLen = src.length / dModel;
  checkU32('qLen', qLen);
  if (qLen === 0) throw new Error('qLen 不能是 0：一則 frame 至少要帶一個 token 位置');

  // §4.5.5b 的第三條：輸入含 Inf/NaN 一律丟回去，不要讓線路格式偷偷吞掉。
  // 那是呼叫端的 bug（上一段的 softmax 炸了、KV 槽沒初始化…），而 NaN 沿著 hop
  // 傳播正是這個專案最難查的失敗模式 —— 輸出看起來合理，只是慢慢偏掉。
  // 掃這一遍不是額外成本：除了 scheme 0 以外的每個方案本來就要走一次 max|x|。
  for (let i = 0; i < src.length; i++) {
    if (!Number.isFinite(src[i])) {
      throw new Error(
        `輸入第 ${i} 個值是 ${src[i]}（t=${Math.floor(i / dModel)}, c=${i % dModel}），` +
        'frame 不接受 Inf/NaN（§4.5.5b）。線路格式不會幫忙吞掉它 —— ' +
        '請往上游找是誰產生的，NaN 傳過一個 hop 之後就只看得到「輸出壞掉」',
      );
    }
  }

  const msgType = meta.msgType ?? 0;
  const roundId = meta.roundId ?? 0;
  const stageIndex = meta.stageIndex ?? 0;
  const startPosition = meta.startPosition ?? 0;
  if (!Number.isInteger(msgType) || msgType < 0 || msgType > 255) {
    throw new Error(`msgType 必須是 0..255（0=激活值 1=logits 2=控制），收到 ${msgType}`);
  }
  checkU16('stageIndex', stageIndex);
  checkU32('roundId', roundId);
  checkU32('startPosition', startPosition);

  const plan = planPayload(id, dModel, qLen, sBytes, outlierFrac);
  checkU32('scaleCount', plan.scaleCount);
  checkU32('outlierCount', plan.outlierCount);
  checkU32('payloadLen', plan.bytes);

  const buf = new ArrayBuffer(frameHeaderSize + plan.bytes);
  const dv = new DataView(buf);

  // 端序一律 little-endian 且明確寫死（§4.5.2）。這就是為什麼全程用 DataView 而不是
  // TypedArray 視圖 —— TypedArray 跟著平台端序走，在 big-endian 上會靜默地送出垃圾。
  dv.setUint32(0, FRAME_MAGIC, true);
  dv.setUint8(4, FRAME_VERSION);
  dv.setUint8(5, msgType);
  dv.setUint8(6, id);
  dv.setUint8(7, precision === 'fp16' ? 1 : 0);
  dv.setUint32(8, roundId, true);
  dv.setUint32(12, dModel, true);
  dv.setUint32(16, qLen, true);
  dv.setUint32(20, startPosition, true);
  dv.setUint32(24, plan.scaleCount, true);
  dv.setUint32(28, plan.outlierCount, true);
  dv.setUint32(32, plan.bytes, true);
  dv.setUint16(36, stageIndex, true);
  dv.setUint16(38, 0, true);

  if (id === 0) encodeNone(dv, src, dModel, qLen);
  else if (id === 1) encodePerChannel(buf, dv, src, dModel, qLen, precision, sBytes);
  else if (id === 2) encodeGroup(buf, dv, src, dModel, qLen, precision, sBytes);
  else encodeOutlier(buf, dv, src, dModel, qLen, precision, sBytes, plan.outlierCount);

  return buf;
}

function encodeNone(dv, src, dModel, qLen) {
  // fp32 本體。用 DataView 逐值寫看起來很慢，但這是唯一能保證 little-endian 的寫法，
  // 而 scheme 0 本來就只在「不量化的對照組」用，不是熱路徑。
  for (let i = 0, n = dModel * qLen; i < n; i++) dv.setFloat32(frameHeaderSize + i * 4, src[i], true);
}

function encodePerChannel(buf, dv, src, dModel, qLen, precision, sBytes) {
  // scale 跨所有 token 共用（§4.2 實測這反而輸給 group-64，保留是為了與 quant.js 對齊）
  const scale = new Float64Array(dModel);
  for (let c = 0; c < dModel; c++) {
    let m = 0;
    for (let t = 0; t < qLen; t++) m = Math.max(m, Math.abs(src[t * dModel + c]));
    scale[c] = roundScale(m / QMAX, precision);
    writeScale(dv, frameHeaderSize + c * sBytes, scale[c], precision);
  }
  const bodyOff = frameHeaderSize + dModel * sBytes;
  const body = new Int8Array(buf, bodyOff, dModel * qLen);
  for (let t = 0; t < qLen; t++) {
    const base = t * dModel;
    for (let c = 0; c < dModel; c++) body[base + c] = quantize(src[base + c], scale[c]);
  }
}

function encodeGroup(buf, dv, src, dModel, qLen, precision, sBytes) {
  const groups = Math.ceil(dModel / GROUP);
  const bodyOff = frameHeaderSize + qLen * groups * sBytes;
  const body = new Int8Array(buf, bodyOff, dModel * qLen);
  for (let t = 0; t < qLen; t++) {
    const base = t * dModel;
    for (let g = 0; g < groups; g++) {
      const g0 = g * GROUP;
      const g1 = Math.min(g0 + GROUP, dModel);
      let m = 0;
      for (let c = g0; c < g1; c++) m = Math.max(m, Math.abs(src[base + c]));
      const s = roundScale(m / QMAX, precision);
      writeScale(dv, frameHeaderSize + (t * groups + g) * sBytes, s, precision);
      for (let c = g0; c < g1; c++) body[base + c] = quantize(src[base + c], s);
    }
  }
}

function encodeOutlier(buf, dv, src, dModel, qLen, precision, sBytes, nOut) {
  const mag = new Float32Array(dModel);
  for (let t = 0; t < qLen; t++) {
    const base = t * dModel;
    for (let c = 0; c < dModel; c++) {
      const a = Math.abs(src[base + c]);
      if (a > mag[c]) mag[c] = a;
    }
  }
  // 「挑哪些 channel」必須與 quant.js:119-121 用一模一樣的比較器與 slice。
  // 平手時 Array.prototype.sort 的穩定性決定了誰入選，換個寫法就會在少數 channel 上
  // 與 M4 模擬器分歧 —— 那種不一致查起來像鬼打牆，所以這裡刻意照抄。
  const picked = Array.from({ length: dModel }, (_, i) => i)
    .sort((a, b) => mag[b] - mag[a])
    .slice(0, nOut);
  // 但送上線的索引是遞增的（§4.5.4），因為接收端要照 channel 順序重建。
  const outIdx = picked.slice().sort((a, b) => a - b);

  const isOutlier = new Uint8Array(dModel);
  for (const c of outIdx) isOutlier[c] = 1;

  for (let j = 0; j < nOut; j++) dv.setUint16(frameHeaderSize + j * 2, outIdx[j], true);

  const nIn = dModel - nOut;
  const scaleOff = frameHeaderSize + nOut * 2;
  const scale = new Float64Array(dModel);
  let rank = 0;
  for (let c = 0; c < dModel; c++) {
    if (isOutlier[c]) continue;
    scale[c] = roundScale(mag[c] / QMAX, precision);
    writeScale(dv, scaleOff + rank * sBytes, scale[c], precision);
    rank++;
  }

  // 離群本體固定 fp16，與 flags bit0 無關（§4.5.4 寫死 `nOut × K × 2`）。
  // 它是「保精度」用的，不是 scale；把它跟著 scalePrecision 走會讓
  // wire_size.py 的帳算不出來。
  const fp16Off = scaleOff + nIn * sBytes;
  const bodyOff = fp16Off + nOut * qLen * 2;
  const body = new Int8Array(buf, bodyOff, nIn * qLen);
  for (let t = 0; t < qLen; t++) {
    const base = t * dModel;
    for (let j = 0; j < nOut; j++) {
      dv.setUint16(fp16Off + (t * nOut + j) * 2, floatToFp16(src[base + outIdx[j]]), true);
    }
    let k = 0;
    for (let c = 0; c < dModel; c++) {
      if (isOutlier[c]) continue;
      body[t * nIn + k] = quantize(src[base + c], scale[c]);
      k++;
    }
  }
}

/** 接受 ArrayBuffer 或 TypedArray/DataView —— dcframe.js 重組出來的是 Uint8Array 切片。 */
function asDataView(buffer, who) {
  if (buffer instanceof ArrayBuffer) return new DataView(buffer);
  if (ArrayBuffer.isView(buffer)) return new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  throw new Error(`${who}() 需要 ArrayBuffer 或 TypedArray，收到 ${Object.prototype.toString.call(buffer)}`);
}

/**
 * 只讀 header，不碰 payload。
 *
 * 路由用得到：中繼節點要看 `stageIndex` / `roundId` 決定往哪送，
 * 但它沒必要（也不該）反量化一次再重新量化 —— 那會多疊一層量化誤差。
 */
export function peekHeader(buffer) {
  const dv = asDataView(buffer, 'peekHeader');
  if (dv.byteLength < frameHeaderSize) {
    throw new Error(
      `frame 太短：只有 ${dv.byteLength} 位元組，光 header 就要 ${frameHeaderSize}（§4.5.2）`,
    );
  }
  const magic = dv.getUint32(0, true);
  if (magic !== FRAME_MAGIC) {
    throw new Error(
      `不是 EdgeCascadeLLM frame：magic 應為 0x${FRAME_MAGIC.toString(16).toUpperCase()}（'ECL1'），` +
      `收到 0x${magic.toString(16).toUpperCase().padStart(8, '0')}。` +
      '常見原因：chunk 的 8 位元組子標頭沒有先剝掉（§4.5.6），' +
      "或是 DataChannel 的 binaryType 沒設成 'arraybuffer'（§4.4）",
    );
  }
  const version = dv.getUint8(4);
  if (version !== FRAME_VERSION) {
    throw new Error(
      `frame 版本 ${version} 不支援，這個節點只懂版本 ${FRAME_VERSION}。` +
      '兩端的 web/src/wire.js 必須是同一版',
    );
  }
  const schemeId = dv.getUint8(6);
  const scheme = SCHEME_NAMES[schemeId];
  if (scheme === undefined) {
    throw new Error(
      `未知的 schemeId ${schemeId}。§4.5.2 只定義 0..3，` +
      '收到別的值代表對方跑的是更新的格式，或這段位元組根本不是 header',
    );
  }
  const flags = dv.getUint8(7);
  // §4.5.2 把 flags 的 bit1..7 與位移 38 的 u16 都標成「保留，0」。
  // 不驗的話，對方設了一個未來的 flag bit 我們會靜默當成沒設 ——
  // 那是 §4.5.5b 規則 3 要防的同一種失敗：看起來解得出來，只是解錯了。
  // 寧可在這裡明確拒絕，讓對方知道版本對不上。
  if (flags & 0xfe) {
    throw new Error(
      `flags 的 bit1..7 是保留位元，§4.5.2 規定必須為 0，收到 0x${flags.toString(16)}。` +
      '對方可能跑的是更新的格式，帶了這一版不認得的旗標 —— 不要猜它的意思。',
    );
  }
  const reserved = dv.getUint16(38, true);
  if (reserved !== 0) {
    throw new Error(
      `位移 38 的 u16 是保留欄位，§4.5.2 規定必須為 0，收到 ${reserved}。` +
      '最可能的原因是 header 位移對不上（例如兩邊的 wire.js 版本不同），不要嘗試解。',
    );
  }
  return {
    magic,
    version,
    msgType: dv.getUint8(5),
    schemeId,
    scheme,
    flags,
    scalePrecision: (flags & 1) ? 'fp16' : 'fp32',
    roundId: dv.getUint32(8, true),
    dModel: dv.getUint32(12, true),
    qLen: dv.getUint32(16, true),
    startPosition: dv.getUint32(20, true),
    scaleCount: dv.getUint32(24, true),
    outlierCount: dv.getUint32(28, true),
    payloadLen: dv.getUint32(32, true),
    stageIndex: dv.getUint16(36, true),
  };
}

/**
 * 解一則 frame。
 *
 * @returns {{data: Float32Array, dModel: number, qLen: number, scheme: string,
 *            schemeId: number, meta: object}}
 */
export function decode(buffer) {
  const dv = asDataView(buffer, 'decode');
  const h = peekHeader(dv);

  const avail = dv.byteLength - frameHeaderSize;
  if (h.payloadLen !== avail) {
    throw new Error(
      `frame 長度不符：header 說 payload 有 ${h.payloadLen} 位元組，實際是 ${avail}。` +
      (h.payloadLen > avail
        ? '多半是 dcframe.js 的重組少收了 chunk（§4.5.6 的 chunkCount 要對得上）'
        : '多半是把多則 frame 黏在同一個 buffer 裡了 —— frame 層不做自我分界，請按 chunk 邊界切開'),
    );
  }

  const precision = h.scalePrecision;
  const sBytes = SCALE_BYTES[precision];
  const { dModel, qLen, schemeId } = h;
  // 拿 header 自己的欄位重算一次佈局。header 可能被截斷後補零、也可能是別的協定的
  // 位元組剛好撞上 magic；不驗的話後面的 Int8Array 視圖會越界或讀到垃圾。
  const layout = describeLayout(h, sBytes);
  const where =
    `（scheme=${h.scheme}, dModel=${dModel}, qLen=${qLen}, ` +
    `scaleCount=${h.scaleCount}, outlierCount=${h.outlierCount}, payloadLen=${h.payloadLen}）`;
  if (layout.error) {
    throw new Error(
      `header 自己描述的佈局就不成立，這則 frame 不能解：${layout.error}${where}。` +
      '常見成因：兩端的 web/src/wire.js 不同版，或這段位元組根本不是 frame 開頭' +
      '（chunk 子標頭沒剝掉、多則 frame 黏在一起）。header 已經壞掉了，不要嘗試解',
    );
  }
  if (layout.bytes !== h.payloadLen) {
    throw new Error(
      `payload 長度 ${h.payloadLen} 與 header 描述的佈局不符：` +
      `照 §4.5.4 用 header 自己的欄位算出來應該是 ${layout.bytes} 位元組${where}。` +
      'header 已經壞掉了，不要嘗試解',
    );
  }

  const data = new Float32Array(dModel * qLen);
  if (schemeId === 0) {
    for (let i = 0; i < data.length; i++) data[i] = dv.getFloat32(frameHeaderSize + i * 4, true);
  } else if (schemeId === 1) {
    const bodyOff = frameHeaderSize + dModel * sBytes;
    const body = bodyView(dv, bodyOff, dModel * qLen);
    for (let c = 0; c < dModel; c++) {
      const s = readScale(dv, frameHeaderSize + c * sBytes, precision);
      for (let t = 0; t < qLen; t++) data[t * dModel + c] = body[t * dModel + c] * s;
    }
  } else if (schemeId === 2) {
    const groups = Math.ceil(dModel / GROUP);
    const bodyOff = frameHeaderSize + qLen * groups * sBytes;
    const body = bodyView(dv, bodyOff, dModel * qLen);
    for (let t = 0; t < qLen; t++) {
      const base = t * dModel;
      for (let g = 0; g < groups; g++) {
        const s = readScale(dv, frameHeaderSize + (t * groups + g) * sBytes, precision);
        const g1 = Math.min(g * GROUP + GROUP, dModel);
        for (let c = g * GROUP; c < g1; c++) data[base + c] = body[base + c] * s;
      }
    }
  } else {
    const nOut = h.outlierCount;
    const nIn = dModel - nOut;
    const scaleOff = frameHeaderSize + nOut * 2;
    const fp16Off = scaleOff + nIn * sBytes;
    const bodyOff = fp16Off + nOut * qLen * 2;
    const body = bodyView(dv, bodyOff, nIn * qLen);

    // 用索引區塊重建 isOutlier 表，再逐 channel 還原（§4.5.4）。
    const isOutlier = new Uint8Array(dModel);
    const outIdx = new Uint16Array(nOut);
    for (let j = 0; j < nOut; j++) {
      const c = dv.getUint16(frameHeaderSize + j * 2, true);
      if (c >= dModel) {
        throw new Error(`離群索引 ${c} 超出 dModel ${dModel} 的範圍，索引區塊已損壞`);
      }
      // 索引必須**互不相同**。只驗範圍是不夠的：把第 2 個 u16 改成跟第 1 個一樣，
      // 每個 header 欄位都還自洽（nOut 沒變、scaleCount 仍等於 d-nOut、payloadLen 也對），
      // 所以長度檢查與佈局檢查全部會過。但重複的索引會讓非離群 channel 的還原迴圈
      // 少跳過一個 channel，之後每個 channel 的 int8 碼都錯位一格 ——
      // 實測 d=576/K=4 有 2123/2304 個值是錯的、最大偏差 7.4e12，
      // 再加上 Int8Array 讀到尾端外面產生的一個 NaN，而且完全不報錯。
      if (isOutlier[c]) {
        throw new Error(
          `離群索引區塊有重複的索引 ${c}（第 ${j} 個），§4.5.4 規定它們必須互不相同。` +
          '重複會讓非離群 channel 整個錯位，解出來的值看起來像雜訊但不會報錯 —— ' +
          '不要放行。多半是送端的挑選邏輯壞了，或這段位元組被截斷後補過',
        );
      }
      outIdx[j] = c;
      isOutlier[c] = 1;
    }
    const scale = new Float64Array(dModel);
    let rank = 0;
    for (let c = 0; c < dModel; c++) {
      if (isOutlier[c]) continue;
      scale[c] = readScale(dv, scaleOff + rank * sBytes, precision);
      rank++;
    }
    for (let t = 0; t < qLen; t++) {
      const base = t * dModel;
      for (let j = 0; j < nOut; j++) {
        data[base + outIdx[j]] = fp16ToFloat(dv.getUint16(fp16Off + (t * nOut + j) * 2, true));
      }
      let k = 0;
      for (let c = 0; c < dModel; c++) {
        if (isOutlier[c]) continue;
        data[base + c] = body[t * nIn + k] * scale[c];
        k++;
      }
    }
  }

  return {
    data,
    dModel,
    qLen,
    scheme: h.scheme,
    schemeId,
    meta: {
      msgType: h.msgType,
      roundId: h.roundId,
      stageIndex: h.stageIndex,
      startPosition: h.startPosition,
      scalePrecision: precision,
    },
  };
}

/**
 * int8 本體的零複製視圖。
 *
 * 之所以要 byteOffset 相加：`decode()` 可能拿到的是一個大 buffer 中間的切片
 * （dcframe.js 重組時不想再複製一次），這時 `dv.byteOffset` 不是 0。
 * 早期直接用 `new Int8Array(dv.buffer, bodyOff, n)` 會在切片輸入下讀到整個 frame 前面的
 * header 位元組，而且不會報錯 —— 解出來的張量只是「有點怪」，非常難查。
 */
function bodyView(dv, bodyOff, n) {
  return new Int8Array(dv.buffer, dv.byteOffset + bodyOff, n);
}

/**
 * 拿 header 自己的欄位重算一次 §4.5.4 的佈局。
 *
 * 回傳 `{ bytes }` 或 `{ error }`，**不再回傳 -1 當哨兵值**。
 * 舊版回 -1，而 `decode()` 直接把它插進訊息裡：
 * 「payload 長度 1024 與 header 描述的佈局不符（應為 -1）」。
 * -1 不是一個長度，操作的人看到它只能來讀這支檔案才知道發生什麼事 ——
 * 而真正的資訊（是哪個欄位對不上、正確值應該是多少）在這個函式裡明明算得出來。
 */
function describeLayout(h, sBytes) {
  const { dModel, qLen, schemeId, scaleCount, outlierCount } = h;
  const numel = dModel * qLen;
  const noCounts = (scheme) => {
    if (scaleCount !== 0) {
      return `${scheme} 沒有 scale 區塊，scaleCount 必須是 0，header 卻寫 ${scaleCount}`;
    }
    if (outlierCount !== 0) {
      return `${scheme} 沒有離群區塊，outlierCount 必須是 0，header 卻寫 ${outlierCount}`;
    }
    return null;
  };

  if (schemeId === 0) {
    const bad = noCounts('none');
    return bad ? { error: bad } : { bytes: 4 * numel };
  }
  if (schemeId === 1) {
    if (scaleCount !== dModel) {
      return {
        error: `per-channel 每個 channel 一個 scale，scaleCount 必須等於 dModel ${dModel}，` +
          `header 卻寫 ${scaleCount}`,
      };
    }
    if (outlierCount !== 0) {
      return { error: `per-channel 沒有離群區塊，outlierCount 必須是 0，header 卻寫 ${outlierCount}` };
    }
    return { bytes: dModel * sBytes + numel };
  }
  if (schemeId === 2) {
    const want = qLen * Math.ceil(dModel / GROUP);
    if (scaleCount !== want) {
      return {
        error: `group-64 的 scale 數是 K × ceil(d/64) = ${qLen} × ${Math.ceil(dModel / GROUP)} = ` +
          `${want}，header 的 scaleCount 卻寫 ${scaleCount}`,
      };
    }
    if (outlierCount !== 0) {
      return { error: `group-64 沒有離群區塊，outlierCount 必須是 0，header 卻寫 ${outlierCount}` };
    }
    return { bytes: want * sBytes + numel };
  }

  const nOut = outlierCount;
  const nIn = dModel - nOut;
  if (nOut < 1 || nOut > dModel) {
    return {
      error: `per-ch+outlier 的 outlierCount 必須落在 1..dModel（1..${dModel}），header 卻寫 ${nOut}`,
    };
  }
  if (scaleCount !== nIn) {
    return {
      error: `per-ch+outlier 只有非離群 channel 需要 scale，scaleCount 必須等於 ` +
        `dModel - outlierCount = ${dModel} - ${nOut} = ${nIn}，header 卻寫 ${scaleCount}`,
    };
  }
  return { bytes: nOut * 2 + nIn * sBytes + nOut * qLen * 2 + nIn * qLen };
}
