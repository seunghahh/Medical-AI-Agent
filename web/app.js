import { ClinicMotion } from "/motion.mjs";
("use strict");
const $ = (id) => document.getElementById(id);
const names = [
  "의사",
  "기록 조회",
  "메모리",
  "실행 검증",
  "진단 검토",
  "진단 평가",
];
const motion = new ClinicMotion();
const roles = names.map((name, i) => {
  const el = document.createElement("div");
  el.className = "role";
  el.innerHTML = `<b>${String(i + 1).padStart(2, "0")}</b><span>${name}</span><small>${[2, 3].includes(i) ? "CODE" : "LLM"}</small>`;
  $("roles").append(el);
  return el;
});
let run = "",
  seq = 0,
  active = -1,
  busy = false,
  stageSince = 0,
  follow = true,
  ended = false,
  entries = 0,
  caseIndex = null;
let previous = -1,
  changed = 0;
const transfers = [];
let transfer = null,
  catchingUp = false;
const sceneLabels = names.map((name, i) => {
  const el = document.createElement("span");
  el.className = "scene-label";
  el.textContent = name;
  $("scene-labels").append(el);
  return el;
});
function stage(role, title, detail, waiting = false) {
  previous = active;
  active = role;
  changed = performance.now();
  if (!catchingUp && previous >= 0 && role >= 0 && previous !== role) {
    transfers.push({ from: previous, to: role });
    if (transfers.length > 6) transfers.shift();
  }
  sceneLabels.forEach((el, i) => el.classList.toggle("active", i === role));
  $("scene-stage").textContent = (role < 0 ? "" : names[role] + " · ") + title;
  busy = waiting;
  stageSince = Date.now();
  $("stage").textContent = title;
  $("stage-detail").textContent = detail;
  $("active-role").textContent = role < 0 ? "LIVE WORKFLOW" : names[role];
  $("stage-icon").textContent =
    role < 0 ? "✚" : String(role + 1).padStart(2, "0");
  roles.forEach((el, i) => el.classList.toggle("active", i === role));
  $("elapsed").textContent = "";
}
function showConnection(text, ok) {
  $("connection").textContent = text;
  $("connection").className = "chip " + (ok ? "live" : "offline");
}
function text(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
function add(e, who, message, kind = "", evidence = null, details = null) {
  if (!entries) $("feed").replaceChildren();
  entries++;
  $("count").textContent = entries;
  const article = document.createElement("article");
  article.className = "entry " + kind;
  const head = document.createElement("div");
  head.className = "entry-head";
  const label = document.createElement("b");
  label.textContent = who;
  head.append(label);
  const turn = document.createElement("span");
  turn.textContent = e.turn === undefined ? "" : `TURN ${e.turn}`;
  head.append(turn);
  const time = document.createElement("time");
  time.textContent = new Date(e.time * 1000).toLocaleTimeString("ko-KR", {
    hour12: false,
  });
  head.append(time);
  article.append(head);
  const body = document.createElement("div");
  body.className = "entry-body";
  if (message) body.textContent = message;
  for (const fact of evidence || []) {
    const source = document.createElement("span");
    source.className = "source";
    source.textContent = `${fact.id} · ${fact.status} · ${fact.source}`;
    body.append(source, document.createTextNode(text(fact.value)));
  }
  article.append(body);
  if (details) {
    const disclosure = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "상세 데이터";
    const pre = document.createElement("pre");
    pre.textContent = text(details);
    disclosure.append(summary, pre);
    article.append(disclosure);
  }
  $("feed").append(article);
  if (follow) $("feed").scrollTop = $("feed").scrollHeight;
}
function receive(e) {
  motion.handle(e, catchingUp || reduced.matches);
  if (e.case_index !== undefined) caseIndex = e.case_index;
  if (e.turn !== undefined) $("turn").textContent = `${e.turn} TURN`;
  switch (e.type) {
    case "run_start":
      $("model").textContent =
        `${e.mode === "model" ? e.model : "SCRIPTED DEMO"} · ${e.engine}`;
      $("run").textContent = e.run_label || e.run_id;
      roles[2].classList.toggle("disabled", !e.memory_enabled);
      roles[4].classList.toggle("disabled", !e.review_enabled);
      roles[5].querySelector("small").textContent =
        e.judge_mode === "llm" ? "LLM / RULE" : "RULE";
      stage(
        -1,
        "실행 연결됨",
        `${e.case_indices.length}개 증례 · ${e.mode === "model" ? "실제 모델 실행" : "스크립트 데모"}`,
      );
      break;
    case "case_start":
      $("case").textContent = `증례 ${e.case_index} · 진료실`;
      $("turn").textContent = "0 TURN";
      stage(0, "새 증례 접수", "초기 증상과 활력징후를 확인합니다.");
      add(e, `증례 ${e.case_index} · 접수`, "", "answer", e.evidence);
      break;
    case "thinking":
      stage(
        0,
        "다음 행동을 생각하는 중",
        `증례 ${caseIndex} · 시도 ${e.attempt} · 모델의 응답을 기다립니다.`,
        true,
      );
      break;
    case "action":
      stage(
        3,
        "생성된 행동 검증",
        "행동 형식, 반복 질문, 근거 ID를 확인합니다.",
      );
      if (e.action?.action !== "DIAGNOSE")
        add(
          e,
          "의사 · " + (e.action?.action || "INVALID"),
          e.action?.text || text(e.action),
        );
      break;
    case "resolving":
      stage(1, "환자 기록에서 답변 확인", e.action.text, true);
      break;
    case "observation":
      stage(
        1,
        e.new_information ? "새 관찰을 받았어요" : "기록 확인 완료",
        e.new_information
          ? "확인된 근거를 의사에게 전달합니다."
          : "새 기록 없음 · UNKNOWN은 정상 소견이 아닙니다.",
      );
      add(
        e,
        "기록 조회 · 응답",
        e.evidence.length ? "" : "추가 관찰 없음",
        "answer",
        e.evidence,
      );
      break;
    case "memory":
      stage(
        2,
        "작업 메모리 갱신",
        "관찰된 사실과 확인되지 않은 내용을 구분해 저장합니다.",
      );
      break;
    case "rejected":
      stage(3, "행동을 다시 선택합니다", e.reason);
      add(e, "실행 검증", e.reason, "notice");
      break;
    case "review_start":
      stage(4, "진단 후보를 검토하는 중", e.candidates.join(" · "), true);
      add(
        e,
        "진단 검토",
        "검토 초안: " + e.draft + "\n후보: " + e.candidates.join(" / "),
        "notice",
      );
      break;
    case "review_end":
      add(
        e,
        "진단 검토",
        e.accepted
          ? "근거 검증을 통과한 검토 결과를 반영했습니다."
          : "초안을 유지합니다. " + (e.error || ""),
        "notice",
      );
      break;
    case "diagnosis":
      stage(0, "최종 진단 제출", e.action.diagnosis);
      add(
        e,
        "의사 · 최종 진단",
        e.action.diagnosis + "\n근거: " + e.action.basis.join(", "),
        "result",
        null,
        e.action,
      );
      break;
    case "case_end":
      if (e.error) {
        stage(3, "증례 실행 오류", e.error);
        add(e, "실행 오류", e.error, "notice");
      }
      break;
    case "evaluation_start":
      stage(
        5,
        "종료된 증례 평가",
        e.method === "llm"
          ? "일치 규칙 확인 후 필요한 경우 평가 모델을 호출합니다."
          : "동의어와 일치 규칙을 확인합니다.",
        true,
      );
      break;
    case "evaluation":
      add(
        e,
        "진단 평가",
        `예측: ${e.diagnosis || "진단 없음"}\n정답: ${e.gold}\n판정: ${e.assessment.match === true ? "일치" : e.assessment.match === false ? "불일치" : "판정 보류"}`,
        "result",
        null,
        e.assessment,
      );
      break;
    case "run_end":
      ended = true;
      stage(
        -1,
        e.status === "finished"
          ? "실행이 종료되었어요"
          : e.status === "interrupted"
            ? "CLI 실행이 중단되었어요"
            : "실행 오류로 종료",
        e.error ||
          "증례별 진단·평가를 오른쪽에서 확인하세요. 다음 CLI 실행은 자동으로 연결됩니다.",
      );
      break;
  }
}
$("follow").onclick = () => {
  follow = !follow;
  $("follow").textContent = follow ? "최신 따라가기 ✓" : "최신 따라가기";
  $("follow").setAttribute("aria-pressed", follow);
  if (follow) $("feed").scrollTop = $("feed").scrollHeight;
};
$("feed").addEventListener(
  "wheel",
  (e) => {
    if (e.deltaY < 0 && follow) $("follow").click();
  },
  { passive: true },
);
async function poll() {
  try {
    const response = await fetch(
      `/api/events?run=${encodeURIComponent(run)}&after=${seq}`,
      { signal: AbortSignal.timeout(5000) },
    );
    if (!response.ok) throw Error(response.status);
    const data = await response.json();
    catchingUp = data.run_id !== run;
    if (data.run_id && data.run_id !== run) {
      run = data.run_id;
      seq = 0;
      ended = false;
      entries = 0;
      caseIndex = null;
      transfers.length = 0;
      transfer = null;
      $("feed").replaceChildren();
      $("count").textContent = 0;
    }
    for (const event of data.events) {
      receive(event);
      seq = event.seq;
    }
    catchingUp = false;
    showConnection(
      !data.run_id
        ? "● CLI 대기"
        : ended
          ? "● 실행 종료"
          : data.alive
            ? "● LIVE 연결됨"
            : "● 프로세스 종료",
      true,
    );
    if (data.run_id && !data.alive && !ended) {
      ended = true;
      stage(
        -1,
        "실행 프로세스가 종료되었어요",
        "완료 이벤트가 없습니다. CLI 터미널의 오류를 확인하세요.",
      );
    }
  } catch (error) {
    showConnection("● 연결 끊김 · 재연결 중", false);
    busy = false;
    $("elapsed").textContent = "";
  } finally {
    setTimeout(poll, 500);
  }
}
poll();
const canvas = $("room"),
  ctx = canvas.getContext("2d"),
  room = new Image(),
  sprites = new Image();
let boxes = [];
room.src = "/room.jpg";
sprites.src = "/sprites.png";
sprites.onload = () => {
  const sheet = document.createElement("canvas");
  sheet.width = sprites.naturalWidth;
  sheet.height = sprites.naturalHeight;
  const s = sheet.getContext("2d");
  s.drawImage(sprites, 0, 0);
  const pixels = s.getImageData(0, 0, sheet.width, sheet.height).data;
  for (let row = 0; row < 2; row++)
    for (let col = 0; col < 5; col++) {
      const left = Math.round((col * sheet.width) / 5),
        right = Math.round(((col + 1) * sheet.width) / 5),
        top = Math.round((row * sheet.height) / 2),
        bottom = Math.round(((row + 1) * sheet.height) / 2);
      let x1 = right,
        y1 = bottom,
        x2 = left,
        y2 = top;
      for (let y = top; y < bottom; y++)
        for (let x = left; x < right; x++)
          if (pixels[(y * sheet.width + x) * 4 + 3] > 190) {
            x1 = Math.min(x1, x);
            x2 = Math.max(x2, x);
            y1 = Math.min(y1, y);
            y2 = Math.max(y2, y);
          }
      boxes.push({ x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 });
    }
};
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
let lastFrame;
function draw(now) {
  motion.step(
    lastFrame === undefined ? 0 : (now - lastFrame) / 1000,
    reduced.matches,
  );
  lastFrame = now;
  const positions = motion.positions;
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  ctx.imageSmoothingEnabled = false;
  if (room.complete && room.naturalWidth) ctx.drawImage(room, 0, 0, 384, 256);
  positions.forEach(([x, y], i) => {
    const walking = motion.routes[i].length > 0;
    sceneLabels[i].style.left =
      (i === 2 ? 15 : ((x - (i === 1 ? 12 : 0)) / 384) * 100) + "%";
    sceneLabels[i].style.top = (i === 2 ? 86 : ((y + 19) / 256) * 100) + "%";
    sceneLabels[i].classList.toggle("walking", walking);
    sceneLabels[i].title = walking ? names[i] + " · 이동 중" : names[i];
    if (i === active) {
      ctx.fillStyle = "#ecffe788";
      ctx.fillRect(x - 12, y + 13, 24, 5);
      ctx.strokeStyle = "#396d4e";
      ctx.strokeRect(x - 12, y + 13, 24, 5);
    }
    if (i === 2) return;
    const col = { 0: 0, 1: 1, 3: 4, 4: 2, 5: 3 }[i];
    const pose = !reduced.matches && walking ? Math.floor(now / 220) % 2 : 0;
    const b = boxes[pose * 5 + col];
    if (b) {
      const h = i === 3 ? 31 : 35,
        w = (b.w * h) / b.h;
      ctx.drawImage(
        sprites,
        b.x,
        b.y,
        b.w,
        b.h,
        Math.round(x - w / 2),
        Math.round(
          y +
            16 -
            h -
            (walking && !reduced.matches ? Math.sin(now / 90) * 0.8 : 0),
        ),
        Math.round(w),
        h,
      );
    }
  });
  if (!reduced.matches) {
    if (!transfer && transfers.length) {
      const next = transfers.shift();
      transfer = { ...next, origin: [...positions[next.from]], start: now };
    }
    if (transfer) {
      const t = Math.min(1, (now - transfer.start) / 1050),
        a = transfer.origin,
        b = positions[transfer.to];
      const x = a[0] + (b[0] - a[0]) * t,
        y = a[1] + (b[1] - a[1]) * t - 18 * Math.sin(Math.PI * t);
      ctx.fillStyle = "#416a5866";
      ctx.fillRect(Math.round(x - 3), Math.round(y + 10), 9, 3);
      ctx.fillStyle = "#345743";
      ctx.fillRect(Math.round(x - 5), Math.round(y - 6), 10, 12);
      ctx.fillStyle = "#fff9df";
      ctx.fillRect(Math.round(x - 4), Math.round(y - 5), 8, 10);
      ctx.fillStyle = "#83a78a";
      ctx.fillRect(Math.round(x - 2), Math.round(y - 2), 4, 1);
      ctx.fillRect(Math.round(x - 2), Math.round(y + 1), 4, 1);
      if (t === 1) transfer = null;
    }
  }

  if (busy)
    $("elapsed").textContent =
      Math.floor((Date.now() - stageSince) / 1000) + "s";
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
