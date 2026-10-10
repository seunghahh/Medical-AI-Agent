import assert from "node:assert/strict";
import { test } from "node:test";
import { ClinicMotion, homes, stations } from "../web/motion.mjs";
const arrive = (motion) => {
  for (let i = 0; i < 100; i++) motion.step(0.1);
};
test("question moves doctor to bed and resolver to records", () => {
  const motion = new ClinicMotion();
  motion.handle({ type: "resolving", action: { action: "SAY" } });
  motion.step(0.1);
  assert(motion.positions[0][1] < homes[0][1]);
  assert(motion.positions[0][1] > stations.bed[1]);
  arrive(motion);
  assert.deepEqual(motion.positions[0], stations.bed);
  assert.deepEqual(motion.positions[1], stations.records);
});
test("thinking and repeated questions keep staff at their workstations", () => {
  const motion = new ClinicMotion();
  motion.handle({ type: "resolving", action: { action: "EXAM" } });
  motion.step(0.1);
  const before = [...motion.positions[0]];
  motion.handle({ type: "thinking" });
  assert.deepEqual(motion.positions[0], before);
  arrive(motion);
  assert.deepEqual(motion.positions[0], stations.bed);
  const route = motion.routes[1];
  motion.handle({ type: "resolving", action: { action: "SAY" } });
  assert.equal(motion.routes[1], route);
});
test("catch-up and reduced motion snap; case reset removes stale paths", () => {
  const motion = new ClinicMotion();
  motion.handle({ type: "review_start" }, true);
  assert.deepEqual(motion.positions[4], stations.review);
  motion.handle({ type: "evaluation_start" });
  motion.step(0.016, true);
  assert.deepEqual(motion.positions[5], stations.evaluation);
  motion.handle({ type: "case_start" });
  assert.deepEqual(motion.positions, homes);
  assert(motion.routes.every((r) => !r.length));
  assert.deepEqual(motion.positions[2], homes[2]);
});
