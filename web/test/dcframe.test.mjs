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
    /**
     * onDeliver（實務上就是 receiver.handle）丟出來的例外。
     *
     * 這裡不能讓例外直接從 _tick 往外噴：那是在 setTimeout 回呼裡，噴出去之後
     * 「重排下一次 tick」那行就不會執行，佇列從此不再排空，而 flush() 的
     * `while (this._queue.length)` 會永遠輪詢下去 —— node --test **整個掛住**，
     * CI 只會看到逾時，連哪個測試壞了都不知道。掛住的測試比紅的測試更糟。
     * 所以收集起來，由 flush() / assertDelivered() 明確丟給測試。
     */
    this.deliverErrors = [];
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
    // 有界：每一圈不是把隊頭送完（佇列變短）就是把 budget 用完（迴圈結束），
    // 所以圈數不會超過目前的佇列長度。寫成上界而不是靠 `budget > 0`，
    // 是因為一個 byteLength 為 0 的隊頭會讓 take 也是 0 —— budget 不減、
    // 佇列不縮，就地空轉。假 channel 卡住的代價是整個測試檔掛住。
    let guard = this._queue.length + 1;
    try {
      while (budget > 0 && this._queue.length && guard-- > 0) {
        const head = this._queue[0];
        const remaining = this._headRemaining || head.byteLength;
        const take = Math.min(budget, remaining);
        budget -= take;
        this._setBuffered(this._buffered - take);
        if (take >= remaining) {
          this._queue.shift();
          this._headRemaining = 0;
          this.bytesDelivered += head.byteLength;
          if (this.onDeliver) {
            try {
              this.onDeliver(head);
            } catch (err) {
              // 接收端拒絕一塊 chunk 是被測行為之一（畸形、超過上限、重播……），
              // 不是假 channel 壞了。記下來繼續排空，讓測試自己決定怎麼斷言。
              this.deliverErrors.push(err);
            }
          }
        } else {
          this._headRemaining = remaining - take;
        }
      }
    } finally {
      // finally：上面任何一個意料外的例外都不該讓排空永久停擺，
      // 否則下一個等 flush() 的人會等到天荒地老。
      if (this._queue.length && this.readyState === 'open') this._schedule();
    }
  }

  _setBuffered(next) {
    const prev = this._buffered;
    this._buffered = next;
    // 邊緣觸發：只在「從高於門檻掉到門檻以下」那一刻發事件，和真實實作一致。
    if (prev > this._lowThreshold && next <= this._lowThreshold) {
      this._emit('bufferedamountlow', {});
    }
  }

  /**
   * 等到全部送出佇列排空（測試用，不是被測程式的一部分）。
   *
   * 有界 + 偵測停滯：原本是 `while (this._queue.length) await sleep(0)`，
   * 只要排空停下來就是無限輪詢。現在只要連續 maxStallPolls 次都沒有任何位元組
   * 送達就丟錯，測試會**紅**而不是**掛**。
   * 最後再把接收端丟出來的例外轉交給測試 —— 吞掉它等於讓 receiver 的錯誤
   * 變成靜默失敗，而這個模組整個存在的理由就是不要靜默失敗。
   */
  async flush({ maxStallPolls = 50 } = {}) {
    let stalled = 0;
    let lastDelivered = -1;
    while (this._queue.length) {
      if (this.readyState !== 'open') break; // 被送死的 channel 不會再排空了
      if (this.bytesDelivered === lastDelivered) {
        if (++stalled > maxStallPolls) {
          throw new Error(
            `FakeChannel.flush()：佇列還剩 ${this._queue.length} 塊，但連續 ${maxStallPolls} ` +
            '次輪詢都沒有任何位元組送達 —— 排空停擺了。',
          );
        }
      } else {
        stalled = 0;
        lastDelivered = this.bytesDelivered;
      }
      await new Promise((r) => setTimeout(r, 0));
    }
    this.assertDelivered();
  }

  /** 接收端在 onDeliver 裡丟過例外的話，在這裡原封不動地重新丟給測試。 */
  assertDelivered() {
    if (this.deliverErrors.length) throw this.deliverErrors[0];
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

/**
 * 合法送出端對這一塊會用的酬載長度：非最後一塊一定是滿的 16376，
 * 最後一塊才可以短。預設值要合規，否則「手工組的畸形 chunk」會混進
 * 那些只想測別的東西的測試裡，把酬載長度的檢查誤報成別的錯。
 */
function conformingPayloadLen(chunkIndex, chunkCount) {
  return chunkIndex === chunkCount - 1 ? 4 : MAX_PAYLOAD;
}

/** 手工組一個 chunk，用來測畸形輸入。payloadLen 不給就是合規的長度。 */
function mkChunk(
  messageId,
  chunkIndex,
  chunkCount,
  payloadLen = conformingPayloadLen(chunkIndex, chunkCount),
) {
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
  // 6. Blob —— 這才是「忘了設 binaryType」的症狀（§4.4：預設值是 'blob'），
  //    所以可以照著做的那句提示要掛在這裡。
  assert.throws(() => receiver.handle({ data: new Blob([new Uint8Array(16)]) }), /binaryType/);
  // 7. 字串：對端真的送了文字。和 binaryType 無關，提示掛在這裡等於誤導 ——
  //    兩句原本是反的，這個斷言就是用來釘住「不准再換回去」。
  assert.throws(
    () => receiver.handle({ data: 'hello' }),
    (err) => /字串/.test(err.message) && !/binaryType/.test(err.message),
    '字串分支不該提 binaryType：對端送字串跟 binaryType 沒有關係',
  );
  // 8. 完全不是位元組的東西
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

test('receiver 丟出來的例外要浮到測試面前，不能把假 channel 卡死', async () => {
  // 這個測試同時是 FakeChannel 自己的回歸測試：舊版的 _tick 沒有 try，
  // 第一塊被拒之後例外從 setTimeout 回呼噴出去、排空永久停擺，
  // flush() 的無界輪詢就讓整個 node --test 掛住。掛住 = CI 逾時 = 零訊號。
  const receiver = createReceiver(() => {}, { maxInFlight: 1 });
  receiver.handle(mkChunk(999, 0, 2)); // 唯一的名額先佔走

  const channel = new FakeChannel({ onDeliver: (buf) => receiver.handle({ data: buf }) });
  const sender = createSender(channel);
  await sender.send(payload(MAX_PAYLOAD * 2, 5).buffer);

  await assert.rejects(
    () => channel.flush(),
    /已達上限 1 則/,
    'flush() 必須把接收端的例外交出來，不能吞掉也不能卡住',
  );
  assert.equal(channel.deliverErrors.length, 2, '兩塊都該被拒，而且兩個例外都要留著');
  assert.equal(channel.bytesDelivered, 2 * CHUNK_SIZE, '被拒之後仍然要把佇列排空');
});

test('長度 0 的訊息也要真的送出一塊（Math.max(1, …) 的那個 1）', async () => {
  const received = [];
  const receiver = createReceiver((buf) => received.push(buf));
  const channel = new FakeChannel({ onDeliver: (buf) => receiver.handle({ data: buf }) });
  const sender = createSender(channel);

  await sender.send(new ArrayBuffer(0));
  await channel.flush();

  // Math.ceil(0 / 16376) 是 0：少了 Math.max(1, …) 這則訊息會一塊都不上線，
  // 送出端一切正常返回，對面則永遠等不到 —— 看起來像對方沒回應。
  assert.equal(sender.stats().chunksSent, 1, '長度 0 的訊息也必須送出剛好一塊');
  assert.deepEqual(channel.sentSizes, [CHUNK_HEADER_SIZE], '那一塊就是純子標頭');
  assert.equal(received.length, 1, '對面必須收到一則空訊息，而不是什麼都收不到');
  assert.equal(received[0].byteLength, 0);
  assert.equal(receiver.stats().inFlight, 0);
});

test('超過 chunkCount u16 上限的訊息，在送出任何一塊之前就丟錯', async () => {
  // 65535 * 16376 = 1,073,086,360 是還塞得進 u16 的最大訊息。再多一個位元組
  // 就要 65536 塊（寫進 u16 變成 0），65537 塊則變成 1 ——
  // 後者最陰險：收端會把第 0 塊當成一則完整訊息交付，靜默截斷成 16376 位元組。
  // 兩個尺寸都只配置不寫入，所以沒有真的吃掉 1 GB 實體記憶體。
  for (const chunks of [65536, 65537]) {
    const channel = new FakeChannel();
    // 任何一塊上線都算失敗：這不是「送到一半才發現」，是根本不該開始送。
    channel.send = () => { throw new Error('超過上限的訊息不該有任何一塊上線'); };
    const sender = createSender(channel);

    const tooBig = new ArrayBuffer(MAX_PAYLOAD * (chunks - 1) + 1);
    await assert.rejects(
      () => sender.send(tooBig),
      /超過 chunkCount u16 的上限 65535/,
      `${chunks} 塊的訊息必須被擋下來`,
    );
    assert.deepEqual(channel.sentSizes, [], `${chunks} 塊：不該有任何 chunk 交給 channel`);
    assert.equal(sender.stats().chunksSent, 0);
    assert.equal(sender.stats().messagesSent, 0);
  }
});

test('每一塊的酬載長度都要驗，截斷的 chunk 不准靜默重組成短訊息', () => {
  const received = [];
  const receiver = createReceiver((buf) => received.push(buf));

  // 沒有這道檢查時：idx0 給 100 位元組、idx1 給 4 位元組，
  // onMessage 會收到一則 104 位元組的訊息、inFlight 歸零、全程無錯。
  // 但 chunkCount=2 的合法送出端一定送了 16376 + N。
  assert.throws(
    () => receiver.handle(mkChunk(1, 0, 2, 100)),
    /非最後一塊必須剛好是 16376 位元組/,
    '被截斷的非最後一塊必須被拒絕',
  );
  assert.equal(received.length, 0);
  assert.equal(receiver.stats().inFlight, 0, '畸形的 chunk 不該佔掉 in-flight 名額');

  // 合規的版本仍然要收：最後一塊才可以短。
  receiver.handle(mkChunk(1, 0, 2));
  receiver.handle(mkChunk(1, 1, 2, 4));
  assert.equal(received.length, 1);
  assert.equal(received[0].byteLength, MAX_PAYLOAD + 4);

  // 多塊訊息的最後一塊不准是空的（那代表送出端多切了一塊）。
  assert.throws(() => receiver.handle(mkChunk(2, 1, 2, 0)), /合法範圍是 1\.\.16376/);
  // 單塊訊息的 0 才是合法的空訊息。
  receiver.handle(mkChunk(3, 0, 1, 0));
  assert.equal(received.length, 2);
  assert.equal(received[1].byteLength, 0);

  // 另一個方向：單一塊 200000 位元組（總長 200008）原本也照單全收。
  assert.throws(
    () => receiver.handle(mkChunk(4, 0, 1, 200000)),
    /超過 CHUNK_SIZE 16384/,
    '沒有任何合法送出端能產生大於 CHUNK_SIZE 的 chunk',
  );
  assert.equal(receiver.stats().inFlight, 0);
});

test('重播一則已經交付的訊息要丟錯，而且記憶是有界的', () => {
  const received = [];
  const receiver = createReceiver((buf) => received.push(buf), { maxInFlight: 2 });

  // 沒有這張表時：同一塊餵兩次，onMessage 觸發**兩次**、一個錯都沒有，
  // 下游把同一個 frame 算兩遍 —— 而模組自己的訊息說重送不該發生。
  receiver.handle(mkChunk(1, 0, 1));
  assert.equal(received.length, 1);
  assert.throws(() => receiver.handle(mkChunk(1, 0, 1)), /是重播/);
  assert.equal(received.length, 1, 'onMessage 不該為同一則訊息觸發第二次');

  // 多塊版本更糟：重播 chunk 0 會開一個永遠補不滿的新 entry，
  // 白佔一個 in-flight 名額到 channel 重建為止。
  receiver.handle(mkChunk(2, 0, 2));
  receiver.handle(mkChunk(2, 1, 2));
  assert.equal(received.length, 2);
  assert.throws(() => receiver.handle(mkChunk(2, 0, 2)), /是重播/);
  assert.equal(receiver.stats().inFlight, 0, '重播不該佔住 in-flight 名額');

  // 有界是刻意的：全部記下來才是真的無上限成長。界線寫死在測試裡，
  // 這樣「安靜地把環改小」會被抓到，而不是變成一條測不到的行為。
  const memory = receiver.stats().completedMemory;
  assert.equal(memory, 64);
  for (let id = 100; id < 100 + memory; id++) receiver.handle(mkChunk(id, 0, 1));
  receiver.handle(mkChunk(1, 0, 1)); // 訊息 1 已經被擠出環外，這一層抓不到了
  assert.equal(received.length, 2 + memory + 1, '超過記憶長度的重播會被當成新訊息收下');
});

test('重組一定要複製：對端把每塊都搬進同一塊暫存區時也要正確', async () => {
  const received = [];
  const receiver = createReceiver((buf) => received.push(buf));

  // node-datachannel / ws 這類 Node 端綁定就是這樣餵資料的：一塊重複使用的
  // scratch buffer，回呼一返回就被下一塊覆寫。handle() 收 TypedArray 是公開
  // 契約的一部分，所以這個用法合法。
  // entry.parts 若是 subarray（視圖）而不是 slice（複製），重組出來的長度
  // 完全正確、一個錯都不會報，內容卻是「最後一塊重複 N 次」——
  // 而 FakeChannel 每塊都給新 buffer，所以這條路徑沒有這個測試就永遠測不到。
  const scratch = new Uint8Array(CHUNK_SIZE);
  const channel = new FakeChannel({
    onDeliver: (buf) => {
      const src = new Uint8Array(buf);
      scratch.fill(0);
      scratch.set(src);
      receiver.handle(scratch.subarray(0, src.byteLength));
    },
  });
  const sender = createSender(channel);

  const src = payload(MAX_PAYLOAD * 2 + 7, 99);
  await sender.send(src.buffer);
  await channel.flush();

  assert.equal(received.length, 1);
  assert.equal(received[0].byteLength, src.byteLength);
  assert.equal(
    Buffer.compare(Buffer.from(received[0]), Buffer.from(src.buffer)), 0,
    '重組結果不對 —— parts 存的是共用暫存區的視圖，不是複製',
  );
});

test('門檻與上限用 Number.isFinite / isInteger 驗，不是用比較', () => {
  // NaN 的任何比較都是 false，所以 `lowWater >= highWater` 放行了 NaN，
  // 之後 `bufferedAmount > NaN` 也永遠是 false：背壓整個消失。
  // 實測 highWater: NaN 送 3 MiB —— backpressureWaits 是 0、
  // bufferedAmount 一路堆到 10,000,000，沒有任何錯誤，只是「看起來比較快」。
  const ch = new FakeChannel();
  assert.throws(() => createSender(ch, { highWater: NaN }), /highWater 必須是非負的有限數/);
  assert.throws(() => createSender(ch, { lowWater: NaN }), /lowWater 必須是非負的有限數/);
  assert.throws(() => createSender(ch, { highWater: Infinity }), /highWater 必須是非負的有限數/);
  assert.throws(() => createSender(ch, { lowWater: -1 }), /lowWater 必須是非負的有限數/);
  assert.throws(() => createSender(ch, { lowWater: 100, highWater: 100 }), /設反了/);
  // 合法的一組仍然要能建起來，否則上面那些斷言可能只是「全部都丟錯」。
  assert.equal(typeof createSender(ch, { lowWater: 1, highWater: 2 }).send, 'function');

  // maxInFlight 同一個洞：NaN 會讓 `partials.size >= maxInFlight` 永遠是 false，
  // 「送一百萬個 chunk 0」就真的能把記憶體堆爆。
  assert.throws(() => createReceiver(() => {}, { maxInFlight: NaN }), /maxInFlight/);
  assert.throws(() => createReceiver(() => {}, { maxInFlight: 0 }), /maxInFlight/);
  assert.throws(() => createReceiver(() => {}, { maxInFlight: 1.5 }), /maxInFlight/);
  assert.throws(() => createReceiver(() => {}, { maxInFlight: Infinity }), /maxInFlight/);
  assert.equal(createReceiver(() => {}, { maxInFlight: 1 }).stats().maxInFlight, 1);
});
