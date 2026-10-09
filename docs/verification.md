# 검증 기록 — 2026-10-06

## 새 정보 없는 문진의 전환·종료 정책 — 2026-10-09

- 원문 path/value가 이미 OBSERVED에 있으면, 응답/ID가 새로 생성되어도
  new_information=false로 기록. UNKNOWN도 새 정보 확보로 계산하지 않음.
- 수락한 정보 요청이 3회 연속 새 정보를 못 얻으면 SAY를 제한하고
  EXAM/DIAGNOSE(본선 TEST 포함)로 전환. 새 관찰을 얻으면 문진을 다시 허용.
- 5회 연속 새 정보를 못 얻으면 획득한 근거와 불확실성/추가 평가 계획으로
  DIAGNOSE를 요구. 이는 로컬 연습용 휴리스틱이며 임상적 충분성 판정이 아님.
- 중복 질문 거절 이후에도 다른 형식 오류를 내면 진찰 전환 정책을 유지.
  의사 프롬프트와 실행 검증에 같은 allowed_actions를 적용.
- hsh 총 36개 테스트 통과. 기존 원문 재조회, UNKNOWN 연속 전환, 새 관찰 뒤
  정책 초기화, 예선/본선 허용 행동, LangGraph에서 5회 정체 후 제출 확인.
- 실제 gpt-oss:20b + LangGraph, case 0, 최대 50턴 설정에서 2턴/거절 0회 완료.
  outputs/20261009T093342663748Z-run. 정보 부족 경로와 임상 품질은 별도 검토해야 함.
- 공개 상태 재현에서 새 정보 없는 요청 3회 후 실제 모델이 허용된 EXAM을 선택함.
  같은 실행 폴더의 progress-replay.json. 단일 재선택 확인이며 전체 증례 평가가 아님.

## 반복 문진 차단 — 2026-10-09

- 사용자 화면에서 같은 SAY 질문이 UNKNOWN 후 반복됨. 이전 프롬프트 지시만으로
  차단되지 않았고, 기존 실행 검증에 중복 질문 검사가 없었음. history는 이미 전달되고 있었음.
- 수락한 SAY/question과 비교해 공백·문장부호·대소문자를 제외한 동일 문장을
  시뮬레이터 호출 전에 거절. 기존 누적 10회 거절 제한은 유지하고, 턴/새 근거는 추가하지 않음.
- 획득한 이력만 사용한 asked_questions/unavailable_requests 목록을 모델 입력에 추가.
  UNKNOWN을 데이터셋에 답이 없는 상태로 설명하고, 반복 거절 뒤 다른 질문/진찰을 선택하도록 지시.
- hsh 총 33개 테스트 통과. OBSERVED/UNKNOWN 뒤 반복 차단, 표기 차이 처리,
  시뮬레이터 호출/턴/근거 불변, plain/LangGraph에서 재선택 확인.
- 실제 gpt-oss:20b의 공개 3턴 상태 재현: 반복 질문 거절 후 EXAM/눈꺼풀 처짐 확인 선택.
  outputs/20261009T092256056001Z-repeat-replay/replay.json. 단일 다음 행동 선택 검증이며
  새 증례 전체 완주/32증례 성능 검증은 아님. 정답/미획득 원자료를 의사 입력에 넣지 않음.
- 의미가 같은 다른 표현은 문자열 검사만으로 모두 차단할 수 없음.
  기존 실행 프로세스에는 적용되지 않고 다음 실행부터 적용됨.

## 턴별 출력 옵션 — 2026-10-09

- run/demo/eval CLI에 --verbose 추가. plain/LangGraph 공통 Encounter에서 출력하며,
  기존 호출자는 기본 verbose=false로 같은 결과와 출력 동작을 유지.
- 의사 행동을 시뮬레이터 호출 전에 flush하고, 획득한 근거만 원문으로 표시.
  거절 사유/누적 수, UNKNOWN, 진단 제출을 확인할 수 있음.
- hsh에서 총 31개 테스트 통과. 새 검사는 호출 전 행동 출력, UNKNOWN/거절,
  숨은 정답/미획득 검사 제외, 기본 모드 무출력, LangGraph 관찰 출력 확인.
- 실제 CLI scripted demo --engine langgraph --verbose --max-turns 1 출력 확인.
  모델 추론은 출력 기능 검증을 위해 다시 실행하지 않았음.

## 누적 거절 원인 재현 및 수정 — 2026-10-09

- outputs/20261009T085934462200Z-run: case 0, 32턴, 누적 거절 10회,
  completed=false. SAY 30자 초과 4회, 빈 최종 응답 1회, 예상 밖 도구 포장 5회.
  `--limit 5`는 증례 수이며 이 실행의 증례당 최대 턴은 기본값 50.
- 종료 직전 공개 evidence/history로 응답 재현: 단일 `ACTION` function call의
  arguments에 유효한 SAY JSON이 들어 있음. 정답/미획득 원자료는 재현 입력에서 제외.
- 같은 공개 입력에서 json_object와 json_schema 응답 옵션을 각각 시험했으나,
  이 Ollama/gpt-oss 실행에서는 동일한 ACTION 포장이 반환됨. 기본 요청 옵션은 유지.
- 확인된 단일 assistant/ACTION 포장만 JSON 데이터로 파싱. 외부 함수 실행 없이
  기존 SAY 길이·행동·근거 검증을 유지. 임의 이름/다중 호출은 계속 거절.
- 프롬프트에 짧은 SAY, 답변/UNKNOWN 질문 반복 금지, 최대 턴 이전 진단 지시 추가.
  거절 10회 제한이나 임상 규칙을 완화하지 않음.
- hsh에서 29개 테스트 통과. ACTION 포장과 포장 내부 31자 SAY 거절을 추가 확인.
- 수정 후 실제 Ollama gpt-oss:20b + LangGraph, 기본 최대 50턴으로 case 0 완료:
  16턴, 거절 2회(빈 응답 1회, 예선 TEST 요청 1회), 107.01초, 체크포인트 40개.
  결과: outputs/20261009T091023412269Z-run. 이 실행은 최종 content로 응답했고,
  ACTION 포장은 앞선 실패 상태 재현과 단위 테스트로 확인했다.
- 반복 문진이 실제 실행에 여전히 남음. 프롬프트만으로 반복 금지나 임상 적절성을
  보장하지 않으며, 5증례 연속 실행/공식 성능은 이번 수정 후 확인하지 않음.

## 실제 모델 응답 처리 수정 — 2026-10-09

- 최초 실행 오류: doctor 첫 응답의 `content`가 비었으며 finish_reason=tool_calls.
  정상 JSON이 단일 assistant function call의 arguments에 들어 있음을 원응답으로 확인.
- 이후 최종 content 없이 종료하거나 예상 밖 도구 호출을 내는 후속 응답도 확인.
- assistant arguments의 단일 JSON만 데이터로 파싱하며, 외부 함수는 실행하지 않음.
  다른 도구 호출·다중 호출은 거절. reasoning/thinking을 답으로 사용하지 않음.
- 의사 출력 형식 오류는 기존 거절/피드백 루프로 처리. 10회에서 중단.
  모델 연결·시간·토큰 예산 오류는 재시도하지 않고 기존처럼 중단.
- 의사 프롬프트에 final 채널의 JSON 제출 지시와 현재 근거/이력에서 다음 행동 선택 지시 추가.
- hsh에서 총 29개 테스트 통과. 형식 오류 후 재선택, 10회 제한,
  연결 오류 즉시 중단, 다중/외부 도구 호출 거절, reasoning 제외 확인.
- 최종 코드 실제 Ollama gpt-oss:20b + LangGraph: case 0, 5턴 제한,
  completed=true, rejections=0, doctor 6회/simulator 5회, 22.50초.
  결과: outputs/20261009T085443371540Z-run.
- 앞선 중간 코드의 50턴 실행은 33턴에서 예상 밖 호출로 중단.
  해당 시점에서는 50턴 예산 실행과 여러 증례 성능을 검증하지 않았음.
  이후 50턴 예산 실행 결과는 위의 누적 거절 수정 항목을 참고.
- 5턴 검증은 진료/제출 경로 확인이며 임상 성능이나 공식 대회 점수 검증이 아님.

## 추가 검증 — 2026-10-09

- LangGraph 환경에서 총 27개 테스트 통과: 기존 20개 + 평가 관련 7개.
- 고정 seed로 dev 20개/test 30개 생성. 자료 checksum, 분할 중복·교집합·범위 검증.
- 미실행 증례를 요청 분모에 유지하고, usage 미보고를 0 대신 null로 표시하는 지표 확인.
- 모델/증례/예산이 다른 실행과 누락 증례가 있는 실행의 전후 비교를 거절하는 테스트 통과.
- 로컬 HTTP mock으로 eval CLI의 2증례 선택 순서·증례별 독립 호출/토큰 예산·CSV 저장 확인.
  이 mock 테스트는 실제 모델 성능 검증이 아니다.
- plain/LangGraph 데모에서 cases.csv/summary.json 생성 및 report/compare CLI 확인.
  두 데모의 진단 정확도는 null이다.
- Ollama 서버 0.35.1 정상 응답, 설치 모델 목록은 비어 있음. 실제 모델 check는 HTTP 오류.
- GitHub CI에 upstream 지정 commit 체크아웃을 추가. GitHub 원격 생성·push·Actions 실행은 하지 않음.
- 실제 gpt-oss 추론, 진단/안전 성능, 공식 SDK·점수는 검증하지 않음.

## 추가 검증 — 2026-10-08

- LangGraph 1.2.14 전용 환경에서 기존 14개 + 새 6개, 총 20개 테스트 통과.
- 일반 Python 환경에서도 기존 14개 통과; 선택 의존성이 없는 경우 새 6개는 skip.
- plain/LangGraph의 예선·본선 데모 결과가 시간/엔진 메타데이터를 제외하고 동일.
- 예선 LangGraph demo: 5턴, 체크포인트 14개, `accuracy: null`.
- 본선 LangGraph demo: 6턴, 체크포인트 16개, `accuracy: null`.
- 숨은 자료 제외, 과거 체크포인트 보존, 증례 간 메모리 격리, UNKNOWN 유지,
  잘못된 행동의 피드백·재선택, 미획득 근거 인용 거절, 턴·시간 제한 확인.
- 로컬 Ollama 연결은 현재 불가. 실제 모델 추론과 공식 SDK 실행은 하지 않음.
- GitHub 자동 검사 파일은 로컬 준비 상태이며 원격 업로드/Actions 실행은 하지 않음.

## 최초 검증

- `python3 -m unittest discover -s tests -v`: 14개 테스트 통과.
- MedQA_Ext 공개 사례 215개 로딩과 필수 필드 구조 확인.
- 예선 scripted demo: case 0, 5턴, SOAP·원문 근거·턴 번호·실행 manifest 저장.
- 본선 scripted demo: case 0, 6턴, TEST 근거와 SOAP 저장.
- SAY 데모 발화의 30자 제한 확인.
- upstream checkout HEAD: `b6fbe22300e99a267a7ac94eaa465ab552eef741`.
- HTTP transport: 로컬 mock Chat Completions 서버로 최종 content 선택·usage 기록·모델 인자 확인.
- 실제 Ollama: `/api/tags`에서 설치된 모델 없음 확인. `run.py check`는 HTTP 오류로 실패.

실제 gpt-oss-20b 추론, 임상적 성능, 의미적 단일 질문/진찰 준수,
엘리스 대회 모델 API, 공식 가상 환자 및 ZIP 제출은 검증하지 않았다.
데모 진단은 스크립트에 정해져 있으며 모델 성능으로 볼 수 없다.
