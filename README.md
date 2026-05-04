# 쿵로그 AI 모듈 (MVP)

이 저장소는 층간소음 중재 시스템의 AI 파트를 빠르게 개발하기 위한 MVP 코드베이스입니다.

현재 목표:
- 규칙 기반 이벤트 분류(event_type + severity)
- 7일 반복 패턴 분석(야간/10분 클러스터/escalation 후보)
- OpenAI 선택 호출 + fallback 메시지 생성
- 단위테스트 기반 안정화

---

## 1) 폴더 구조

```text
ai/
  __init__.py
  config.py
  schemas.py
  event_classifier.py
  pattern_analyzer.py
  message_generator.py
  openai_client.py
  fallback_templates.py
  tests/
    test_event_classifier.py
    test_pattern_analyzer.py
    test_message_generator.py

.env.example
README.md
```

원칙:
- 불필요하게 파일 구조를 늘리지 않고, 현재 구조 내에서 기능을 확장합니다.

---

## 2) 핵심 설정 파일

### `ai/config.py`
AI 전역 설정을 관리합니다.

- 소음 임계값
  - `AI_DB_BACKGROUND`, `AI_DB_DAILY`, `AI_DB_IMPACT`
  - `AI_VIBRATION_LOW`, `AI_VIBRATION_MID`, `AI_VIBRATION_HIGH`
  - `AI_DURATION_SHORT_MS`, `AI_DURATION_MEDIUM_MS`, `AI_DURATION_LONG_MS`
- 야간 시간대
  - `AI_NIGHT_START_HOUR`, `AI_NIGHT_END_HOUR`
- 패턴 분석
  - `AI_ANALYSIS_DAYS_DEFAULT`, `AI_ANALYSIS_DAYS_EXTENDED`
  - `AI_CLUSTER_WINDOW_MINUTES`
  - `AI_REPEATED_DAYS_WARNING`, `AI_REPEATED_DAYS_ESCALATION`
- OpenAI
  - `ENABLE_OPENAI`, `OPENAI_API_KEY`, `OPENAI_MODEL`
  - `OPENAI_TIMEOUT_SEC`, `OPENAI_MAX_RETRIES`, `OPENAI_MAX_OUTPUT_TOKENS`
  - `OPENAI_REASONING_EFFORT` (기본 `minimal`)
  - `LLM_MIN_SEVERITY`

`get_settings()`로 설정을 캐시해 공통 사용합니다.

---

## 3) 환경변수 설정

1. `.env.example`를 복사해 `.env` 생성
2. `OPENAI_API_KEY`는 서버 환경변수로만 주입
3. 프론트엔드에 키를 절대 노출하지 않음

예시:

```powershell
Copy-Item .env.example .env
```

---

## 4) 모듈 요약

- `event_classifier.py`
  - 이벤트 분류: `background_noise`, `daily_noise`, `impact_noise`, `repeated_vibration`, `unknown`
  - 심각도 분류: `low`, `medium`, `high`, `critical`

- `pattern_analyzer.py`
  - 최근 N일(기본 7일) 이벤트 분석
  - 야간 이벤트 수, 반복 발생 일수, 10분 클러스터 최대치 계산
  - 중재 후 재발 여부(`post_mediation_recurrence`) 및 escalation 후보 판단

- `message_generator.py`
  - 조건부 OpenAI 호출
  - 실패/위험 톤 감지 시 fallback 메시지 반환

- `fallback_templates.py`
  - OpenAI 실패 시에도 항상 중립 메시지 생성

---

## 5) 테스트 실행

```bash
python -m pytest -q
```

---

## 6) 최근 작업 로그

### 2026-05-04

1) `event_classifier.py` 1차 구현 완료
- 분류 우선순위:
  - `background_noise -> repeated_vibration -> impact_noise -> daily_noise -> unknown`
- 입력 반영:
  - `sound_level`, `vibration_value`, `duration_ms`, `acceleration`, `recent_events_10min`, 야간 여부
- 심각도:
  - dB/진동/지속시간/야간/반복 가중치 기반 `low~critical`

2) `pattern_analyzer.py` 구현 완료
- 최근 7일 반복 분석
- 야간 이벤트 집계
- 10분 클러스터(`max_events_in_10min`) 계산
- 중재 이후 재발(`post_mediation_recurrence`) 반영
- escalation 판단(`needs_escalation`) 추가

3) 단위테스트 강화
- `event_classifier`: 배경/일상/충격/반복진동
- `pattern_analyzer`: 7일 반복, 야간 집계, 10분 클러스터, escalation, 무이벤트 케이스

현재 테스트 결과:
- `10 passed`

4) `openai_client.py` 고도화
- Responses API 출력 형식을 `json_schema(strict=true)`로 지정
- 응답 JSON 파싱/필수 키 검증 추가
- 재시도(`OPENAI_MAX_RETRIES`) + 지수형 대기(짧은 backoff) 추가
- `reasoning.effort=minimal` 기본 적용(불필요한 추론 토큰 과다 사용 방지)
- 빈 응답/불완전 응답 상태 감지 시 재시도 후 예외 처리
- 실패 시 상위 레이어(`message_generator`)에서 fallback 처리 가능하도록 예외 표준화

5) OpenAI 클라이언트 테스트 추가
- `ai/tests/test_openai_client.py`
  - 정상 JSON 응답
  - 1회 실패 후 재시도 성공
  - 재시도 소진 후 예외 발생
  - SDK output 아이템(dict shape) 파싱 케이스

6) 실 API 스모크 테스트(환경변수 주입) 완료
- `OPENAI_API_KEY`를 코드에 저장하지 않고 세션 환경변수로만 주입
- `OpenAIMessageClient.generate_message()` 호출 성공 확인
- 확인 항목:
  - `recommended_action` 정상 반환
  - `tone_check.is_neutral=True`
  - `resident_message` 생성 길이 확인

현재 테스트 결과:
- `14 passed`

---

## 7) 운영 주의사항

- API Key 하드코딩 금지
- `OPENAI_API_KEY`는 백엔드 환경변수로만 사용
- MVP 단계에서는 모든 이벤트에 OpenAI 호출하지 않음
- 호출 실패 시 fallback으로 서비스 연속성 보장
