# DIFF–CLS: 산업 분류의 의미 변화 표현 실험

공유 BERT로 두 입력을 별도 인코딩하고, **수식된 개념의 `[DIFF]` hidden state − 기본 개념의 `[CLS]` hidden state**를 MLP로 분류합니다. `[JUST]`는 사용하지 않습니다.

```text
A: [CLS] 플라스틱 [DIFF] 팔렛 [SEP] → h_diff
B: [CLS] 팔렛 [SEP]                → h_base
z = h_diff - h_base → MLP → 세분화 기준
```

이 프로젝트는 실행 가능한 **연구용 실험 파이프라인**입니다. 검증된 ECLASS 매핑 시스템이나 학습 완료 모델은 아닙니다. 기본 라벨은 재질/형상/용도 같은 분류 기준이며, 이 예측만으로 IOF의 subClassOf·part-of 공리를 생성하지 않습니다. 차이 벡터가 순수한 수식어 의미라는 보장도 없습니다.

## 설치

Python 3.10 이상. 프로젝트 폴더에서:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

최초 pretrained 실행은 Hugging Face 모델 다운로드가 필요합니다. 기본 모델은 `google-bert/bert-base-multilingual-cased`이며, 영어 벤치마크에는 `google-bert/bert-base-cased`를 명시할 수 있습니다. 외부 모델 코드는 실행하지 않습니다. CPU/CUDA/Apple MPS를 지원하며 자동 선택합니다. 재현성 기준 테스트는 CPU를 권장합니다.

### Conda로 설치

프로젝트 루트 폴더에서 실행합니다. 기존 `.venv`를 사용 중이면 먼저 `deactivate`하세요.

```bash
conda env create -f environment.yml
conda activate diff-cls-research
diffcls --help
pytest -q
```

Conda는 Python 3.12와 pip를 준비하고, pip가 `pyproject.toml`의 실행·테스트 의존성과 프로젝트를 editable 모드로 설치합니다. 이 YAML은 플랫폼별 완전 고정 lockfile이 아닙니다. 기존 검증 환경의 정확한 패키지 버전은 `requirements-tested.txt`를 참고하세요. CUDA 버전은 지정하지 않습니다.

## 바로 실행: 외부 모델 다운로드 없는 smoke test

```bash
diffcls validate data/demo.jsonl
diffcls split data/demo.jsonl --mode family --output data/splits/demo
diffcls train --train data/splits/demo/train.jsonl --dev data/splits/demo/dev.jsonl --output runs/smoke --tiny --epochs 2 --lr 0.001 --device cpu
diffcls evaluate --checkpoint runs/smoke/best --data data/splits/demo/test.jsonl --output runs/smoke/test --device cpu
```

`--tiny`는 **랜덤 초기화된 작은 BERT**를 사용합니다. 데모는 90개 영어 합성 예시이며, ECLASS나 Tratz 원본이 아닙니다. 이 결과를 연구 성능으로 보고하면 안 됩니다. 데모의 용도 라벨도 단순한 엔지니어링 테스트용입니다.

## 실제 데이터 형식

UTF-8 CSV 또는 JSONL. 각 행은 아래 문자열 필드를 포함합니다.

```json
{"id":"sample-001","parent":"팔렛","child":"플라스틱 팔렛","child_prefix":"플라스틱","child_suffix":"팔렛","modifier":"플라스틱","label":"material","family":"pallet","source":"expert-reviewed source reference"}
```

- `child_prefix + child_suffix`는 공백 정규화 후 `child`와 같아야 합니다. 마커는 두 필드 사이에 삽입됩니다.
- `[DIFF]`/`[CLS]`를 데이터에 직접 쓰지 않습니다. `[DIFF]`는 등록된 단일 special token입니다.
- 접미 수식 구조나 괄호 표현도 검토 후 명시적으로 정규화해야 합니다. 원문과 출처를 추가 필드에 보존할 수 있습니다. 부분 문자열로 중심어를 자동 추정하지 않습니다.
- `family`는 평가에서 격리하려는 제품군입니다. 서로 다른 표기/동의어를 같은 제품군으로 묶으세요.
- `modifier`는 수식어 단독 기준 모델 및 누수 점검에 사용합니다.
- 추론(`predict`)에서는 `label`을 생략할 수 있습니다. 나머지 필드는 동일합니다.
- `source`, `source_version`, `evidence`, `ontology_relation` 등 추가 필드를 보존할 수 있으나 현재 모델 입력에는 사용하지 않습니다.
- 길이 초과는 오류로 처리합니다. 마커나 수식어가 잘리는 silent truncation을 하지 않습니다.

Tratz의 modifier–head 관계 라벨을 compound–head 대조 입력으로 변환할 경우 **파생 평가 설정**이라고 명시해야 합니다. 원본 릴리스마다 형식·라벨이 달라 자동 다운로드나 임의 라벨 변환을 하지 않습니다. CSV를 위 표준 필드에 맞춰 준비하면 바로 사용할 수 있습니다. ECLASS 원본은 포함하지 않습니다.

## 학습 / 평가 / 추론

```bash
diffcls split data/private/eclass.jsonl --mode family --seed 42 --output data/splits/eclass
diffcls train --train data/splits/eclass/train.jsonl --dev data/splits/eclass/dev.jsonl --output runs/eclass-diff --mode diff_cls --epochs 5
diffcls evaluate --checkpoint runs/eclass-diff/best --data data/splits/eclass/test.jsonl --output runs/eclass-diff/test
diffcls predict --checkpoint runs/eclass-diff/best --data data/private/unlabeled.jsonl --output runs/predictions
```

출력: `history.json`, `data-audit.json`, `best/`(토크나이저·구조·safetensors 가중치·실험 설정), `metrics.json`, `predictions.jsonl`. 최적 epoch는 **dev macro-F1만**으로 선택합니다. 클래스 순서는 train에서만 결정하며 dev/test의 미등록 라벨은 오류입니다. softmax 값은 보정되지 않은 점수이며, 매핑의 근거·정확도 보장이 아닙니다. 재실행 시 기존 출력 폴더 덮어쓰기를 막습니다. 자동 학습 재개 기능은 제공하지 않습니다.

## 비교 실험

| mode | 입력 표현 |
|---|---|
| `diff_cls` | A의 DIFF − B의 CLS (제안) |
| `cls_cls` | 마커 없는 A의 CLS − B의 CLS |
| `diff_only` | A의 DIFF만 (뺄셈 효과 검사) |
| `modifier_only` | 수식어만 인코딩한 CLS (단어 암기 기준 모델) |

```bash
python scripts/ablation.py --splits data/splits/eclass --output runs/ablation --epochs 5 --seeds 13 42 77
```

모든 mode에서 동일한 split과 seed 목록을 사용합니다. 각 seed에서 dev로 체크포인트를 고른 후 test를 평가하고 `comparison.json`에 저장합니다. test 결과를 보고 모델/프롬프트를 계속 조정하면 최종 test가 아니게 됩니다. 모드별 결과 차이는 초기화와 최적화에도 영향을 받으므로 여러 seed를 보고하세요.

## 누수 방지 분할

- `random`: 예시 단위, 낙관적 참고값.
- `family`: 제품군 단위.
- `head`: 정확히 같은 부모 표현 격리.
- `modifier`: 정확히 같은 수식어 표현 격리.
- `lexical`: 부모 또는 수식어가 공유되면 연결요소로 묶어 함께 배치. 둘 다 겹치지 않도록 하지만 거대 연결요소가 있으면 실패할 수 있습니다.

대소문자/양끝 공백은 정규화하지만 동의어·활용형까지 자동 차단하지 않습니다. `manifest.json`에 실제 크기, 라벨 분포, 분할 간 중복을 기록합니다. 그룹 수 기준 약 70/15/15 분할이므로 행 비율은 달라질 수 있습니다. 그룹 보존을 위해 층화 분할을 강제로 하지 않으며, 학습에 없는 라벨이 발생하면 중단합니다. 실제 데이터에서는 라벨 커버리지를 검토하세요. 사용자 제공 train/dev도 중복 예시를 검사하지만 제품군 격리는 audit를 확인해야 합니다.

## 연구 자료

- [DiffVec](https://github.com/ivri/DiffVec): 벡터 차이 기반 관계 학습.
- [Tratz & Hovy 2010](https://aclanthology.org/P10-1070/): 복합명사 관계 분류.
- [Matching the Blanks](https://aclanthology.org/P19-1279/): 마커 hidden state 기반 관계 표현.
- [Lexical memorization](https://aclanthology.org/N15-1098/): 단어 암기와 관계 학습 구분.

## 다음 연구 단계

실제 데이터 표본 검토 → 전문가 라벨 정의/합의 → 제품군 격리 실험 → 마커/뺄셈 효과 비교 → IOF 공리 선택과 독립적으로 검증. 현재 구현에는 OWL 공리 자동 생성, 전문가 gold 제작, pretrained 모델 전체 학습 결과가 포함되지 않습니다.
