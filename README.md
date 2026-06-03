# KoongLog AI 모듈

## 프로젝트 소개

KoongLog AI 모듈은 층간소음 센서 데이터를 분석해 이벤트 유형과 위험도를 판단하는 Python 기반 AI API 서버입니다. FastAPI로 API를 제공하며, 규칙 기반 분류와 LightGBM 추론을 함께 사용할 수 있도록 구성되어 있습니다.

센서 데이터의 소음, 진동, 지속 시간, 시간대, 최근 발생 맥락을 바탕으로 `impact_noise`, `repeated_vibration`, `daily_noise`, `background_noise`, `unknown` 유형을 분류합니다. 또한 반복 발생 패턴을 분석하고, 필요 시 중재 메시지 생성을 지원합니다.

## 문제 정의

층간소음 이벤트는 단순히 소리 크기만으로 판단하기 어렵습니다. 같은 소음 수치라도 야간 발생 여부, 진동 강도, 반복 횟수, 지속 시간에 따라 위험도가 달라질 수 있습니다.

이 프로젝트는 센서 raw 데이터를 이벤트 단위 feature로 정리하고, 규칙 기반 판단과 모델 기반 추론을 조합해 더 구조화된 분석 결과를 제공하는 것을 목표로 합니다.

## 주요 기능

- 센서 이벤트 분류
  - `ai/event_classifier.py`
  - 이벤트 유형, 심각도, 의미 있는 이벤트 여부를 판단합니다.

- 진동 위험도 계산
  - `ai/vibration.py`
  - raw 진동값을 추정 가속도, dB(V), 위험 카테고리로 변환합니다.

- 반복 패턴 분석
  - `ai/pattern_analyzer.py`
  - 최근 이벤트의 반복 일수, 야간 발생, 10분 내 집중 발생 여부를 분석합니다.

- LightGBM 기반 추론
  - `ai/lgbm_runtime.py`, `ai/ml_features.py`
  - 모델 artifact를 로드해 위험도 또는 이벤트 판단을 보조하고, 실패 시 규칙 기반 결과로 fallback합니다.

- 중재 메시지 생성
  - `ai/message_generator.py`, `ai/openai_client.py`, `ai/fallback_templates.py`
  - OpenAI 호출 또는 fallback 템플릿을 통해 안내 메시지를 생성합니다.

- API 및 대시보드 데이터 제공
  - `ai/dashboard_api.py`
  - 이벤트 분석, 패턴 분석, 대시보드용 통계 API를 제공합니다.

## 기술 스택

- Language: Python 3.12
- API Server: FastAPI, Uvicorn
- Data Validation: Pydantic, dataclass
- Database: SQLAlchemy
- Machine Learning: LightGBM, NumPy, pandas, scikit-learn, joblib
- LLM Integration: OpenAI Python SDK
- Deployment: Railway, Railpack
- Test: pytest 기반 테스트 코드 확인

## 폴더 구조

```text
.
├── ai/
│   ├── dashboard_api.py          # FastAPI API 및 대시보드 엔드포인트
│   ├── serve.py                  # 서버 실행 진입점
│   ├── config.py                 # 환경변수 기반 설정
│   ├── schemas.py                # 이벤트 및 분석 결과 데이터 구조
│   ├── time_utils.py             # 시간 정규화
│   ├── vibration.py              # 진동 위험도 계산
│   ├── event_classifier.py       # 이벤트 분류
│   ├── lgbm_runtime.py           # LightGBM 추론 런타임
│   ├── ml_features.py            # 모델 feature 생성
│   ├── pattern_analyzer.py       # 반복 패턴 분석
│   ├── message_generator.py      # 메시지 생성 분기
│   ├── openai_client.py          # OpenAI API 래퍼
│   ├── fallback_templates.py     # fallback 메시지 템플릿
│   ├── training/                 # 학습 및 평가 스크립트
│   ├── tests/                    # 테스트 코드
│   └── artifacts/                # 모델 및 평가 산출물
├── deploy/                       # 배포 보조 스크립트
├── requirements.txt              # 실행 의존성
├── requirements-train.txt        # 학습 의존성
├── railway.json                  # Railway 배포 설정
├── railpack.json                 # Railpack 설정
├── runtime.txt                   # Python 런타임 버전
├── .env.example                  # 환경변수 예시
└── README.md
```

## 핵심 구현 포인트

### 1. 이벤트 단위 입력 계약 설계

`EventFeatures`는 센서 데이터를 이벤트 분석에 필요한 형태로 정리합니다. 특히 `recent_count_10min`은 raw 샘플 수가 아니라 최근 10분 동안의 의미 있는 이벤트 수로 제한되어 있어, 단순 샘플 증가로 인한 오분류를 줄이도록 설계되어 있습니다.

### 2. 진동 데이터의 위험도 feature 변환

`ai/vibration.py`는 raw 진동값을 추정 가속도와 dB(V)로 변환하고, 시간대와 반복 충격 횟수를 함께 고려해 위험도를 계산합니다. 이를 통해 raw 값보다 해석 가능한 feature를 분류 로직에 제공합니다.

### 3. 규칙 기반 분류와 LightGBM fallback 구조

`classify_event()`는 `rule`, `hybrid`, `lightgbm` 백엔드를 설정으로 선택합니다. 모델 파일이 없거나 confidence가 낮은 경우 규칙 기반 결과로 fallback해 API 응답 안정성을 유지합니다.

### 4. feature spec 기반 모델 관리

`ai/ml_features.py`는 `FEATURE_SPEC_VERSION`과 `FEATURE_COLUMNS`를 정의합니다. 모델 학습과 추론에서 동일한 feature 구조를 사용하도록 관리해, 학습 시점과 추론 시점의 불일치를 줄입니다.

### 5. OpenAI 응답 검증 및 fallback

OpenAI를 이용한 메시지 생성은 JSON Schema와 필수 필드 검증을 거칩니다. API 호출 실패나 응답 검증 실패 시 fallback 템플릿을 사용해 메시지를 반환합니다.
