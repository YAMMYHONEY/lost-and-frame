import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreClip, verdict } from './score.mjs';

const cup = (kind, frame) => ({ concept: 'cup', kind, frame });

test('a prediction near the labeled frame is a hit, others are false alarms', () => {
  const r = scoreClip([cup('vanish', 98), cup('vanish', 40)], [cup('vanish', 96)], { frameCount: 150 });
  assert.deepEqual([r.hit, r.miss, r.falseAlarm], [1, 0, 1]);
});

test('wrong kind, wrong object or too far away does not count', () => {
  const truth = [cup('vanish', 96)];
  for (const p of [cup('appear', 96), { concept: 'cat', kind: 'vanish', frame: 96 }, cup('vanish', 112)]) {
    assert.deepEqual(scoreClip([p], truth, { frameCount: 150 }).hit, 0);
  }
});

test('one label matches only one prediction, the nearest one', () => {
  const r = scoreClip([cup('vanish', 100), cup('vanish', 97)], [cup('vanish', 96), cup('vanish', 101)],
    { frameCount: 150 });
  assert.equal(r.hit, 2);
  assert.equal(r.miss, 0);
});

test('no predictions on a clean clip: nothing to watch', () => {
  assert.deepEqual(scoreClip([], [], { frameCount: 150 }), { hit: 0, miss: 0, falseAlarm: 0, burden: 0 });
});

test('success needs half the false alarms and at most 10 points more misses', () => {
  const A = { miss: 2, falseAlarm: 20 };
  assert.deepEqual(verdict({ A, C: { miss: 3, falseAlarm: 10 } }, 20), { fewerFalseAlarms: true, missesHeld: true });
  assert.deepEqual(verdict({ A, C: { miss: 5, falseAlarm: 11 } }, 20), { fewerFalseAlarms: false, missesHeld: false });
});

test('review windows that overlap are not counted twice', () => {
  const r = scoreClip([cup('vanish', 50), cup('appear', 55)], [], { frameCount: 100, pad: 10 });
  assert.equal(r.burden, (65 - 40 + 1) / 100);
});
