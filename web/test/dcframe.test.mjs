/**
 * Chunk 層單元測試（docs/01-architecture.md §4.5.6）。
 *
 * 這支測試**不開瀏覽器**：dcframe.js 不碰 RTCPeerConnection，搬的是不透明的
 * ArrayBuffer，所以假 channel 就足夠。browser.test.mjs 那種「起 http-server +
 * 兩個 browser context」的成本，這一層不該付 —— 付了就會有人為了 CI 跑得快
 * 而把它關掉。
 *
 * FakeChannel 模型的是 §4.4 **實測到的**行為，不是規格書寫的行為。
 * 最關鍵的一條：send() 一則超過上限的訊息**不會丟例外**，它正常返回、
 * 然後非同步關掉 channel、0 位元組送達。所以這裡不能用
 * `assert.throws(() => channel.send(huge))` 來驗 —— 那個測試在真實環境下
 * 永遠是綠的，卻什麼都沒保證。要驗的是「違規次數為 0」。
 *
 *   node --test test/dcframe.test.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSender, createReceiver, CHUNK_SIZE, CHUNK_HEADER_SIZE } from '../src/dcframe.js';

const MAX_PAYLOAD = CHUNK_SIZE - CHUNK_HEADER_SIZE; // 16376
const HIGH_WATER = 524288;
const LOW_WATER = 262144;
/** §4.5.7 的壓力測試尺寸：vocab 49152 × seq 16 × 4 位元組的 fp32 logits。 */
const LOGITS_BYTES = 3145728;

/**
 * 假 DataChannel。每一條行為都對應 §4.4 的一項實測結果。
 *
 * 刻意不繼承 EventTarget：要能數「監聽器註冊了幾次」與「bufferedAmount 被讀了幾次」，
 * 後者是判斷「有沒有在空轉輪詢」的唯一可靠指標。
 */
class FakeChannel {
  constructor({ maxMessageSize = 262144, drainPerTick = 131072, onDeliver = null } = {}) {
    this.maxMessageSize = maxMessageSize;
    this.drainPerTick = drainPerTick;
    this.onDeliver = onDeliver;
    this.readyState = 'open';

    this.sentSizes = [];
    /** 超過 maxMessageSize 的 send()。這個陣列必須永遠是空的。 */
    this.violations = [];
    this.bytesDelivered = 0;
    this.sendsWhileClosed = 0;
    this.peakBufferedAmount = 0;
    this.bufferedAmountReads = 0;
    this.thresholdWrites = 0;
    this.listenerAdds = new Map();

    this._lowThreshold = 0;
    this._buffered = 0;
    this._queue = [];
    this._headRemaining = 0;
    this._timer = null;
    this._listeners = new Map();
  }

  get bufferedAmountLowThreshold() { return this._lowThreshold; }

  set bufferedAmountLowThreshold(v) {
    // 數寫入次數：這個門檻只該設一次（事件是邊緣觸發的，重設會吃掉邊緣）。
    this.thresholdWrites++;
    this._lowThreshold = v;
  }

  get bufferedAmount() {
    this.bufferedAmountReads++;
    return this._buffered;
  }

  addEventListener(type, fn, opts) {
    this.listenerAdds.set(type, (this.listenerAdds.get(type) ?? 0) + 1);
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type).push({ fn, once: !!(opts && opts.once) });
  }

  removeEventListener(type, fn) {
    const list = this._listeners.get(type);
    if (!list) return;
    const i = list.findIndex((l) => l.fn === fn);
    if (i >= 0) list.splice(i, 1);
  }

  _emit(type, ev) {
    const list = this._listeners.get(type);
    if (!list) return;
    for (const l of [...list]) {
      if (l.once) this.removeEventListener(type, l.fn);
      l.fn(ev);
    }
  }

  send(buffer) {
    if (this.readyState !== 'open') {
      this.sendsWhileClosed++;
      return;
    }
    const size = buffer.byteLength;
    this.sentSizes.push(size);
    if (size > this.maxMessageSize) {
      // 實測：這裡**沒有** TypeError（與 W3C 的 send() 演算法描述不符）。
      // 正常返回，然後非同步 error + close，0 位元組送達。
      this.violations.push(size);
      queueMicrotask(() => {
        this.readyState = 'closed';
        this._queue.length = 0;
        this._buffered = 0;
        this._emit('error', { name: 'OperationError', errorDetail: 'data-channel-failure' });
        this._emit('close', {});
      });
      return;
    }
    this._buffered += size;
    if (this._buffered > this.peakBufferedAmount) this.peakBufferedAmount = this._buffered;
    this._queue.push(buffer);
    this._schedule();
  }

  _schedule() {
    if (this._timer === null) this._timer = setTimeout(() => this._tick(), 0);
  }

  _tick() {
    this._timer = null;
    if (this.readyState !== 'open') return;
    let budget = this.drainPerTick;
    while (budget > 0 && this._queue.length) {
      const head = this._queue[0];
      const remaining = this._headRemaining || head.byteLength;
      const take = Math.min(budget, remaining);
      budget -= take;
      this._setBuffered(this._buffered - take);
      if (take === remaining) {
        this._queue.shift();
        this._headRemaining = 0;
        this.bytesDelivered += head.byteLength;
        if (this.onDeliver) this.onDeliver(head);
      } else {
        this._headRemaining = remaining - take;
      }
    }
    if (this._queue.length) this._schedule();
  }

  _setBuffered(next) {
    const prev = this._buffered;
    this._buffered = next;
    // 邊緣觸發：只在「從高於門檻掉到門檻以下」那一刻發事件，和真實實作一致。
    if (prev > this._lowThreshold && next <= this._lowThreshold) {
      this._emit('bufferedamountlow', {});
    }
  }

  /** 等到全部送出佇列排空（測試用，不是被測程式的一部分）。 */
  async flush() {
    while (this._queue.length) await new Promise((r) => setTimeout(r, 0));
  }
}

/** 決定性的偽隨機酬載：測試失敗時要能重現，不能用 Math.random()。 */
function payload(n, seed = 0x9e3779b9) {
  const out = new Uint8Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = s & 0xff;
  }
  return out;
}

/** 手工組一個 chunk，用來測畸形輸入。 */
function mkChunk(messageId, chunkIndex, chunkCount, payloadLen = 4) {
  const buf = new ArrayBuffer(CHUNK_HEADER_SIZE + payloadLen);
  const v = new DataView(buf);
  v.setUint32(0, messageId, true);
  v.setUint16(4, chunkIndex, true);
  v.setUint16(6, chunkCount, true);
  return buf;
}

function expectedChunks(n) {
  return Math.max(1, Math.ceil(n / MAX_PAYLOAD));
}

test('3 MiB 的 fp32 logits 逐位元組往返（§4.5.7 的壓力路徑）', async () => {
  const received = [];
  const receiver = createReceiver((buf) => received.push(buf));
  const channel = new FakeChannel({ onDeliver: (buf) => receiver.handle({ data: buf }) });
  const sender = createSender(channel);

  const src = payload(LOGITS_BYTES);
  await sender.send(src.buffer);
  await channel.flush();

  assert.equal(received.length, 1, '應該只重組出一則訊息');
  assert.equal(received[0].byteLength, LOGITS_BYTES);
  assert.equal(
    Buffer.compare(Buffer.from(received[0]), Buffer.from(src.buffer)), 0,
    '重組結果與來源不是逐位元組相同 —— chunk 順序或偏移算錯了',
  );
  // 3145728 / 16376 = 192.1...，所以最後一塊是不滿的 1536 位元組。
  assert.equal(sender.stats().chunksSent, 193);
  assert.equal(channel.bytesDelivered, LOGITS_BYTES + 193 * CHUNK_HEADER_SIZE);
  assert.deepEqual(channel.violations, [], '出現了超過 maxMessageSize 的 send()');
});

test('零次超大 send()，含子標頭算術的邊界值', async () => {
  // 16375 / 16376 / 16377 圍住的是 CHUNK_SIZE − CHUNK_HEADER_SIZE 這個減法：
  // 漏掉減 8 的話，16376 會變成一個 16384+8 的 chunk。
  const sizes = [1, MAX_PAYLOAD - 1, MAX_PAYLOAD, MAX_PAYLOAD + 1, MAX_PAYLOAD * 2, LOGITS_BYTES];
  for (const n of sizes) {
    const received = [];
    const receiver = createReceiver((buf) => received.push(buf));
    const channel = new FakeChannel({ onDeliver: (buf) => receiver.handle({ data: buf }) });
    const sender = createSender(channel);

    const src = payload(n, n + 1);
    await sender.send(src.buffer);
    await channel.flush();

    assert.deepEqual(channel.violations, [], `${n} 位元組的訊息送出了超大 chunk`);
    assert.equal(channel.sendsWhileClosed, 0, `${n} 位元組：channel 被送死了`);
    const over = channel.sentSizes.filter((s) => s > CHUNK_SIZE);
    assert.deepEqual(over, [], `${n} 位元組：有 chunk 超過 CHUNK_SIZE，實際 ${over[0]}`);
    assert.equal(sender.stats().chunksSent, expectedChunks(n), `${n} 位元組的分塊數不對`);
    assert.equal(received.length, 1);
    assert.equal(
      Buffer.compare(Buffer.from(received[0]), Buffer.from(src.buffer)), 0,
      `${n} 位元組的往返內容不一致`,
    );
  }
  // 邊界值本身也順便釘住：改 CHUNK_SIZE 而忘了改測試的話會在這裡先爆。
  assert.equal(CHUNK_SIZE, 16384);
  assert.equal(CHUNK_HEADER_SIZE, 8);
  assert.equal(MAX_PAYLOAD, 16376);
});

test('背壓把 bufferedAmount 壓在高水位加一塊之內', async () => {
  const receiver = createReceiver(() => {});
  // 排空速度刻意慢於送出速度，否則永遠碰不到高水位，這個測試會變成空測。
  const channel = new FakeChannel({
    drainPerTick: 64 * 1024,
    onDeliver: (buf) => receiver.handle(buf),
  });
  const sender = createSender(channel);

  await sender.send(payload(LOGITS_BYTES).buffer);
  await channel.flush();

  const s = sender.stats();
  assert.ok(s.backpressureWaits > 0, '從頭到尾沒等過背壓，這組參數量不到東西');
  assert.ok(
    channel.peakBufferedAmount <= HIGH_WATER + CHUNK_SIZE,
    `bufferedAmount 峰值 ${channel.peakBufferedAmount} 超過高水位 ${HIGH_WATER} + 一塊 ${CHUNK_SIZE}`,
  );
  assert.ok(
    channel.peakBufferedAmount > LOW_WATER,
    '峰值連低水位都沒到，代表 drainPerTick 太快、這個測試沒在測背壓',
  );
});

test('背壓等的是 bufferedamountlow 事件，不是輪詢', async () => {
  const receiver = createReceiver(() => {});
  const channel = new FakeChannel({
    drainPerTick: 64 * 1024,
    onDeliver: (buf) => receiver.handle(buf),
  });
  const sender = createSender(channel);

  await sender.send(payload(LOGITS_BYTES).buffer);
  await channel.flush();

  const s = sender.stats();
  assert.equal(
    channel.listenerAdds.get('bufferedamountlow'), s.backpressureWaits,
    '每次等待應該剛好註冊一個一次性監聽器',
  );
  assert.equal(channel.thresholdWrites, 1, 'bufferedAmountLowThreshold 只該設一次');
  assert.equal(channel.bufferedAmountLowThreshold, LOW_WATER);

  // 真正的反輪詢證據：bufferedAmount 的讀取次數必須和 chunk 數同階。
  // 每塊最多讀兩次（送前檢查 + 送後記峰值），每次等待再多一次補救讀取。
  // 空轉迴圈在這個尺寸下會讀到數萬次，差好幾個數量級。
  const budget = 2 * s.chunksSent + s.backpressureWaits + 2;
  assert.ok(
    channel.bufferedAmountReads <= budget,
    `bufferedAmount 被讀了 ${channel.bufferedAmountReads} 次，超過上限 ${budget} —— 有人改成輪詢了`,
  );
});

test('兩則訊息交錯也能各自重組', async () => {
  // 兩個 sender 各自從不同的 messageId 起跳，模擬「同一條 channel 上兩則訊息同時在途」。
  const capA = [];
  const capB = [];
  const chA = new FakeChannel({ onDeliver: (b) => capA.push(b) });
  const chB = new FakeChannel({ onDeliver: (b) => capB.push(b) });
  const sa = createSender(chA, { firstMessageId: 7 });
  const sb = createSender(chB, { firstMessageId: 1000 });

  const srcA = payload(MAX_PAYLOAD * 3 + 11, 1);
  const srcB = payload(MAX_PAYLOAD + 5, 2);
  await Promise.all([sa.send(srcA.buffer), sb.send(srcB.buffer)]);
  await chA.flush();
  await chB.flush();
  assert.equal(capA.length, 4);
  assert.equal(capB.length, 2);

  const received = [];
  const receiver = createReceiver((buf) => received.push(buf));
  // 交錯順序刻意讓 B 夾在 A 中間、而且 B 先送完 —— 完成順序不等於開始順序。
  receiver.handle(capA[0]);
  receiver.handle(capB[0]);
  receiver.handle(capA[1]);
  receiver.handle(capB[1]);
  assert.equal(received.length, 1, 'B 的最後一塊到了就該立刻交付，不必等 A');
  receiver.handle(capA[2]);
  receiver.handle(capA[3]);
  assert.equal(received.length, 2);

  assert.equal(Buffer.compare(Buffer.from(received[0]), Buffer.from(srcB.buffer)), 0, 'B 內容不對');
  assert.equal(Buffer.compare(Buffer.from(received[1]), Buffer.from(srcA.buffer)), 0, 'A 內容不對');
  assert.equal(receiver.stats().inFlight, 0, '全部交付後不該還留著半成品');
});

test('畸形 chunk 一律明確報錯，不靜默丟棄', () => {
  const receiver = createReceiver(() => {});

  // 1. 比子標頭還短
  assert.throws(
    () => receiver.handle(new ArrayBuffer(CHUNK_HEADER_SIZE - 1)),
    /子標頭/,
    '短於 8 位元組的 chunk 必須被拒絕',
  );
  // 2. chunkIndex 超出 chunkCount
  assert.throws(() => receiver.handle(mkChunk(1, 3, 3)), /超出 chunkCount/);
  // 3. chunkCount 是 0
  assert.throws(() => receiver.handle(mkChunk(1, 0, 0)), /chunkCount 是 0/);
  // 4. 重複的 chunkIndex
  receiver.handle(mkChunk(2, 0, 3));
  assert.throws(() => receiver.handle(mkChunk(2, 0, 3)), /重複/);
  // 5. 同一個 messageId 的 chunkCount 前後不一致
  assert.throws(() => receiver.handle(mkChunk(2, 1, 4)), /前後不一致/);
  // 6. 字串（binaryType 沒設成 arraybuffer 的典型症狀）
  assert.throws(() => receiver.handle({ data: 'hello' }), /binaryType/);
  // 7. 完全不是位元組的東西
  assert.throws(() => receiver.handle({ data: 42 }), /必須是 ArrayBuffer/);

  // 被拒絕的 chunk 不該汙染狀態：messageId 2 仍只有 chunk 0。
  assert.equal(receiver.stats().inFlight, 1);
});

test('重組中的訊息數有上限，擋掉「一百萬個 chunk 0」', () => {
  const receiver = createReceiver(() => {}, { maxInFlight: 8 });
  for (let id = 0; id < 8; id++) receiver.handle(mkChunk(id, 0, 2));
  assert.equal(receiver.stats().inFlight, 8);
  assert.throws(
    () => receiver.handle(mkChunk(8, 0, 2)),
    /同時重組中的訊息已達上限 8 則/,
    '超過上限必須丟錯，不能無限長大',
  );
  // 上限之內的既有訊息仍要能收尾，不能被上限誤傷。
  const done = [];
  const r2 = createReceiver((b) => done.push(b), { maxInFlight: 1 });
  r2.handle(mkChunk(5, 0, 2));
  assert.throws(() => r2.handle(mkChunk(6, 0, 2)), /已達上限 1 則/);
  r2.handle(mkChunk(5, 1, 2));
  assert.equal(done.length, 1);
  assert.equal(r2.stats().inFlight, 0);
  r2.handle(mkChunk(6, 0, 1)); // 位置空出來了就該收得下
  assert.equal(done.length, 2);
});

test('messageId 逐則遞增並在 u32 繞回', async () => {
  const seen = [];
  const channel = new FakeChannel({
    onDeliver: (buf) => seen.push(new DataView(buf).getUint32(0, true)),
  });
  const sender = createSender(channel, { firstMessageId: 0xfffffffe });
  await sender.send(payload(4).buffer);
  await sender.send(payload(4).buffer);
  await sender.send(payload(4).buffer);
  await channel.flush();

  assert.deepEqual(seen, [0xfffffffe, 0xffffffff, 0]);
  // 計數器本身也要是 u32：長成 2^32 的話下一則就和收端對不上了。
  assert.equal(sender.stats().nextMessageId, 1);
  assert.equal(sender.stats().messagesSent, 3);
});

test('send() 要等到全部 chunk 都交給 channel 才 resolve', async () => {
  const channel = new FakeChannel({ drainPerTick: 32 * 1024 });
  const sender = createSender(channel);
  const n = LOGITS_BYTES;
  const p = sender.send(payload(n).buffer);
  // 還沒 await 之前一定送不完：193 塊裡至少有一塊要等背壓。
  assert.ok(channel.sentSizes.length < expectedChunks(n), 'send() 不該同步送完全部 chunk');
  await p;
  assert.equal(channel.sentSizes.length, expectedChunks(n), 'resolve 時必須每一塊都交出去了');
});
