# NOVA 에이전트 실행·평가·개선 실습

목표는 프레임워크를 설치하는 데서 끝나지 않고, 같은 증례에서 변경의 효과와 실패를 설명하는 것이다.
이 프로젝트는 공개 AgentClinic 자료를 사용하는 로컬 연습 환경이다. 공식 NOVA SDK와 점수는 구현하지 않는다.

## 1. 현재 코드부터 이해하기

`clinic/langgraph_runner.py`의 흐름:

```text
START → choose_action → execute_action → choose_action → ...
                   오류 ↓         진단 완료 / 오류 ↓
                       END                       END
```

| 기능 | 코드 | 직접 확인할 내용 |
|---|---|---|
| 모델 기반 행동 선택 | `clinic/agent.py:Doctor.act` | 시작 정보와 획득 근거를 입력하고 SAY/EXAM/DIAGNOSE JSON을 선택 |
| 도구 실행 | `clinic/runner.py:Encounter.execute` | 허용 행동·입력·근거 ID를 검증한 뒤 Simulator로 전달 |
| 상태 갱신 | `Encounter.choose/execute` | evidence/history, 남은 턴, 거절 사유를 다음 선택에 전달 |
| 메모리 | `build_graph`, `InMemorySaver` | 증례마다 새 thread, 정답/미획득 자료는 제외, 재시작 시 소실 |
| 평가 | `clinic/evaluation.py` | 분할 검증, 증례별 지표, 요약, 비교 설정 확인 |

현재 구현은 구조화 JSON으로 행동을 호출한다. 제공자 API의 native function-calling 구현이라고 쓰면 안 된다.
한 번에 다음 행동을 고르는 정책이며, 장기 계획·명시적 감별 가설 갱신은 추가 실험이 필요하다.
현재는 새 정보가 없는 요청 3회 후 진찰/검사로 전환하고 5회 후 진단을 요구한다.
같은 원자료 재조회도 새 정보가 없는 것으로 기록한다. 이 종료 기준은 로컬 연습용이며
임상적 정보 충분성이나 대회 점수 향상을 보장하지 않는다.
시뮬레이터도 모델을 호출하므로 doctor와 simulator의 사용량을 따로 기록한다.

## 2. 모델 없이 흐름 검증

프로젝트 폴더의 터미널에서:

```bash
python3 -m venv .venv-langgraph
.venv-langgraph/bin/python -m pip install -r requirements-langgraph.txt
.venv-langgraph/bin/python -m unittest discover -s tests -v
.venv-langgraph/bin/python run.py demo --engine langgraph
```

데모는 case 0의 고정 스크립트이다. 검증용이며 LLM 진단 정확도를 뜻하지 않는다.

## 3. 실제 모델 한 증례 실행

Ollama가 실행 중이지 않다면 별도 터미널에서 `ollama serve`를 실행한다.
다음 다운로드는 사용자가 직접 실행한다. 모델 파일 공간과 실행용 메모리가 필요하다.

```bash
ollama pull gpt-oss:20b
.venv-langgraph/bin/python run.py check
.venv-langgraph/bin/python run.py run --engine langgraph --limit 1 --max-turns 5
```

`check`가 실패하면 평가를 시작하지 않는다. endpoint, 설치 모델 이름, JSON 출력부터 확인한다.
첫 진료는 5턴으로 확인하고, 통과한 뒤 필요에 따라 `--max-turns 50`으로 늘린다.
로컬 개발 모델은 공식 대회 서버의 모델 버전과 동일하다고 보장할 수 없다.
호환 API 서버가 있다면 다음처럼 직접 지정한다:

```bash
.venv-langgraph/bin/python run.py check --base-url http://localhost:8000/v1 --model openai/gpt-oss-20b
```

인증은 `CLINIC_API_KEY` 환경변수로 설정한다. 실제 key를 저장소·로그·문서에 쓰지 않는다.
현재 공식 SDK와 가상 환자 통신은 연결하지 않았다.

## 4. 고정 dev/test 만들기

```bash
python3 eval.py split --dev 20 --test 30 --seed 20261009 --output eval/split.json
```

50개의 공개 증례를 무작위 비중복 분할한다. 이 파일에는 증례 인덱스·seed·자료 checksum만 있다.
동일 파일을 계속 사용한다. 개발은 dev에서 하고 test 결과를 보고 반복적으로 프롬프트를 고치지 않는다.
분할 파일을 다시 생성해 기존 결과를 덮어쓰지 않는다. 이 CLI는 기존 파일 덮어쓰기를 거절한다.
공개 증례는 모델 사전학습에 포함되었을 가능성이 있어 새 임상 일반화 검증으로 해석하지 않는다.

```bash
# 비용·응답 형식 먼저 확인
.venv-langgraph/bin/python run.py eval --engine langgraph \
  --split-file eval/split.json --split dev --limit 1

# 한 증례를 확인한 뒤 개발용 전체 평가
.venv-langgraph/bin/python run.py eval --engine langgraph \
  --split-file eval/split.json --split dev --limit 20 --output outputs/baseline-dev
```

전체 평가에는 최대 20분 × 20증례가 걸릴 수 있으며 doctor와 simulator를 모두 호출한다.
시간·턴·최대 출력 토큰을 줄일 수 있지만 비교하는 두 실행에서는 같은 값을 사용한다.
API/시간/예산 오류가 나면 이후 증례는 중단하고, 요청/실행 증례 수를 모두 기록한다.
`eval`의 세션은 증례 하나다. 공식 SDK의 세션 정의는 공개 규격에 맞춰 별도로 확인한다.

## 5. 결과에서 실패를 찾기

| 파일 | 용도 |
|---|---|
| `manifest.json` | 자료 checksum, 실제 증례 목록, 모델·예산·프롬프트가 포함된 코드 SHA-256 |
| `cases.csv` | 증례별 완주, exact match, 거절·반복·UNKNOWN, 시간·토큰 |
| `results.jsonl` | 행동과 거절 피드백, 실제 받은 근거, SOAP 원문 |
| `summary.json` | 요청/실행 분모, 실행 커버리지와 전체 요약 |

먼저 `cases.csv`에서 미완주·거절·반복 요청이 많은 증례를 찾고 `results.jsonl`의 해당 history를 읽는다.
실패 유형을 "30자 초과", "예선 TEST", "근거 없는 진단 인용", "정보를 못 얻는 요청 반복",
"일찍 진단", "JSON 잘림", "응답 지연" 등으로 나눈다. 자동 분류되지 않는 항목은 직접 판단한다.

지표 정의:
- 완주율: 진단 제출 증례 / 요청 증례. 임상적 정답을 뜻하지 않는다.
- 실행 커버리지: 실제 시도 증례 / 요청 증례. API 오류로 빠진 증례를 숨기지 않는다.
- exact match: 정규화한 진단 문자열의 일치. 동의어와 임상적 적절성을 판정하지 않는다.
- schema 준수율: 형식상 허용된 행동 / history의 행동 시도. 복합 질문·진찰의 의미적 준수는 별도 검토한다.
- 거절률: 거절 행동 / history의 행동 시도. JSON 파싱 전에 실패한 호출은 error에 기록된다.
- 반복 요청: 동일 JSON 행동을 다시 수락한 횟수. 필요한 반복 진찰도 있으므로 무조건 나쁜 행동은 아니다.
- UNKNOWN: 요청했으나 기록된 결과가 없는 횟수. 정상·음성으로 바꾸면 안 된다.
- 토큰: 서버가 usage를 보고한 호출만 집계한다. 누락이 있으면 역할별 합계는 null이다.

임상적 진단·안전·교육의 질은 별도 수동 검토표가 필요하다. 자동 지표를 공식 NOVA 점수로 환산하지 않는다.
서버의 모델 revision/digest와 소프트웨어 버전도 실험 노트에 남긴다. 모델 이름만으로 버전은 고정되지 않는다.

## 6. 본인이 바꿔볼 작은 실험

첫 변경은 `clinic/agent.py`의 `DOCTOR_PROMPT`에 다음 정책을 추가하는 것이다:

```text
Review prior actions and rejection feedback before choosing the next action.
Do not repeat an identical request with no new information unless you have a specific reason.
Prefer a single question or maneuver that distinguishes plausible diagnoses.
Use only acquired observations. UNKNOWN never means normal or absent.
```

프롬프트에 정답 진단·test 증례·숨은 검사 결과를 넣지 않는다. 모델이 실제로 이 정책을 따르는지는
로그와 지표로 확인한다. baseline 파일은 그대로 두고 동일 dev에서 다시 실행한다:

```bash
.venv-langgraph/bin/python run.py eval --engine langgraph \
  --split-file eval/split.json --split dev --limit 20 --output outputs/changed-dev
python3 eval.py compare outputs/baseline-dev outputs/changed-dev --output outputs/dev-comparison
```

`paired.csv`로 같은 증례의 전후 변화를 확인한다. `comparison.json`은 차이를 after−before로 표시한다.
자료, 증례, 모델 이름, 예산, round가 다르면 비교를 거절한다. 누락 증례가 있어도 거절한다.
plain과 LangGraph는 같은 선택·실행 코드를 쓰므로 프레임워크 교체만으로 진단 개선을 기대하지 않는다.
LLM·시뮬레이터의 변동이 있어 한 번의 상승만으로 개선을 확정하지 말고 dev 반복 실행으로 확인한다.
최종 정책을 정한 뒤 test를 실행한다:

```bash
.venv-langgraph/bin/python run.py eval --engine langgraph \
  --split-file eval/split.json --split test --limit 30 --output outputs/final-test
```

## 7. GitHub 저장소 구성

저장소는 [Medical-AI-Agent](https://github.com/seunghahh/Medical-AI-Agent)이다.
코드·테스트·고정 개발/평가 분할·사용법을 관리한다. `.gitignore`로 `.DS_Store`,
원본 vendor, 실행 outputs, 환경 파일, 로그, 모델 자산을 제외한다.
API key·공식 비공개 대회 자료를 넣지 않는다.

GitHub Actions는 push와 PR에서 AgentClinic를 지정 commit으로 별도 체크아웃하고 테스트한다.
새 로컬 clone에도 원본 자료가 필요하다:

```bash
git clone https://github.com/seunghahh/Medical-AI-Agent.git
cd Medical-AI-Agent
git clone https://github.com/SamuelSchmidgall/AgentClinic.git vendor/AgentClinic
git -C vendor/AgentClinic checkout b6fbe22300e99a267a7ac94eaa465ab552eef741
```

두 번째 변경부터 `git switch -c improve-action-policy`로 작업하고 PR을 연다.
PR에는 발견한 실패, 바꾼 정책, 같은 dev 증례의 전후 결과, 남은 한계를 적는다.
동료에게 리뷰를 받고 실제 피드백을 반영하면 PR 리뷰 경험의 근거가 된다. 혼자 merge한 PR은 동료 리뷰 근거가 아니다.
README에는 "로컬 구현·테스트", "실제 모델 실행", "공식 제출" 상태를 구분해 적는다.

참고: [LangGraph quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart),
[Ollama Chat Completions 호환 API](https://docs.ollama.com/api/openai-compatibility).
