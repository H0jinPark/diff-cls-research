# DIFF–CLS: 산업 분류의 의미 변화 표현 실험

공유 BERT로 두 입력을 별도 인코딩하고, **수식된 개념의 `[DIFF]` hidden state − 기본 개념의 `[CLS]` hidden state**를 MLP로 분류합니다. `[JUST]`는 사용하지 않습니다.

```text
A: [CLS] 플라스틱 [DIFF] 팔렛 [SEP] → h_diff
B: [CLS] 팔렛 [SEP]                → h_base
z = h_diff - h_base → MLP → 세분화 기준
```

이 프로젝트는 실행 가능한 **연구용 실험 파이프라인**입니다. 검증된 ECLASS 매핑 시스템이나 학습 완료 모델은 아닙니다. 기본 라벨은 재질/형상/용도 같은 분류 기준이며, 이 예측만으로 IOF의 subClassOf·part-of 공리를 생성하지 않습니다. 차이 벡터가 순수한 수식어 의미라는 보장도 없습니다.

## 설치

Linux RTX 5080 서버 설정 및 실행 방법은 [docs/RTX_5080.md](docs/RTX_5080.md)를 참고하세요.

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

### PANIC의 Tratz 벤치마크 실행

받아둔 `data/benchmarks/panic` 폴더의 Tratz 기반 TSV를 변환합니다. PANIC이 제공한 train/val/test 분할과 라벨을 그대로 보존하고, 각 입력 파일의 SHA-256과 분할 간 단어 중복을 `manifest.json`에 남깁니다. 분할 폴더는 기존 파일을 덮어쓰지 않습니다.

```bash
python -m diffcls.cli prepare-tratz \
  --source data/benchmarks/panic \
  --granularity coarse --split-type lexical \
  --output data/splits/tratz-coarse-lexical

python -m diffcls.cli train \
  --train data/splits/tratz-coarse-lexical/train.jsonl \
  --dev data/splits/tratz-coarse-lexical/dev.jsonl \
  --output runs/tratz-bert-experiment \
  --model prajjwal1/bert-tiny --mode diff_cls \
  --epochs 5 --batch-size 32 --lr 0.00005 --max-length 32

python -m diffcls.cli evaluate \
  --checkpoint runs/tratz-bert-experiment/best \
  --data data/splits/tratz-coarse-lexical/test.jsonl \
  --output runs/tratz-bert-experiment/test
```

`--model prajjwal1/bert-tiny`는 영어 사전학습 BERT입니다. 최초 실행에는 모델 다운로드가 필요합니다. 고정 모델 대신 `google-bert/bert-base-cased`로도 실험할 수 있지만 훨씬 많은 시간과 메모리가 듭니다. `--device auto`가 기본입니다. coarse/fine 및 random/lexical을 각각 지정할 수 있습니다. PANIC lexical 분할은 random보다 작으며, 실제 중복 여부는 `manifest.json`으로 확인하세요.

중요: Tratz 라벨은 복합명사의 modifier–head 사이 어휘 관계입니다. 위 변환에서 `parent=head`, `child=modifier + head`는 **실험용 텍스트 구성**이며 해당 관계가 OWL `subClassOf`라는 의미가 아닙니다. 이 실험 결과만으로 ECLASS 또는 IOF 매핑 성능을 주장할 수 없습니다.

## 학습 / 평가 / 추론

```bash
diffcls split data/private/eclass.jsonl --mode family --seed 42 --output data/splits/eclass
diffcls train --train data/splits/eclass/train.jsonl --dev data/splits/eclass/dev.jsonl --output runs/eclass-diff --mode diff_cls --epochs 5
diffcls evaluate --checkpoint runs/eclass-diff/best --data data/splits/eclass/test.jsonl --output runs/eclass-diff/test
diffcls predict --checkpoint runs/eclass-diff/best --data data/private/unlabeled.jsonl --output runs/predictions
```

출력: `history.json`, `data-audit.json`, `best/`(토크나이저·구조·safetensors 가중치·실험 설정), `metrics.json`, `predictions.jsonl`. 최적 epoch는 **dev macro-F1만**으로 선택합니다. 클래스 순서는 train에서만 결정하며 dev/test의 미등록 라벨은 오류입니다. softmax 값은 보정되지 않은 점수이며, 매핑의 근거·정확도 보장이 아닙니다. 재실행 시 기존 출력 폴더 덮어쓰기를 막습니다. 자동 학습 재개 기능은 제공하지 않습니다.

## 비교 실험

### 2단계 대조학습 실험

`--contrastive-epochs`를 지정하면 `diff_cls`와 `concat_cls` 모델이 1단계 alignment를 실행합니다. 학습 데이터의 `[DIFF]`와 기본어 `[CLS]` hidden state를 L2 정규화해 코사인 유사도를 구하고, 같은 head 쌍은 positive, 다른 head 쌍은 in-batch negative인 대칭 multi-positive InfoNCE를 적용합니다. 같은 head가 배치에 반복될 때 false negative로 취급하지 않습니다. negative가 없는 배치는 건너뜁니다. 이 단계는 train 데이터만 이용합니다.

2단계는 1단계 BERT 가중치를 이어받아 정규화 **전** hidden state를 차이(`diff_cls`) 또는 이어 붙인 표현(`concat_cls`)으로 분류합니다. 별도 projection head는 없으며, 분류 단계에서도 BERT와 분류기를 함께 업데이트합니다. 따라서 수식어 정보 보존이나 최종 시점에서 alignment가 유지되는 것은 보장하지 않습니다. 대조 손실과 별도로 관계 분류 성능 및 임베딩 변화 분석이 필요합니다. `contrastive-history.json`, `contrastive-encoder.safetensors`, 그리고 분류 단계의 `history.json`과 `best/`가 저장됩니다. 1단계 선택 시 2단계의 학습 데이터 섞는 순서가 달라질 수 있고, 각 baseline 대비 총 훈련량이 늘어납니다.

```bash
export HF_HOME="$PWD/.hf-cache"
python scripts/compare_dev.py \
  --output "runs/contrastive-comparison-$(date +%Y%m%d-%H%M%S)" \
  --contrastive-epochs 3 --contrastive-lr 0.00002 \
  --contrastive-temperature 0.1 --epochs 20 --seeds 13 42 77 --device cpu
```

이 비교에서 1단계는 `diff_cls`와 `concat_cls`에 적용되며 나머지 모델은 기존의 단일 분류 단계로 학습합니다. 동일 조건의 `--contrastive-epochs 0` 결과도 비교해야 alignment 효과를 확인할 수 있습니다. `--tiny`는 무작위 초기화 모델이라 개발용입니다. 자동 이어학습 기능은 없습니다.

테스트셋을 사용하지 않고 여러 seed의 검증 결과만 비교하려면:

```bash
export HF_HOME="$PWD/.hf-cache"
python scripts/compare_dev.py --output "runs/dev-comparison-$(date +%Y%m%d-%H%M%S)"
```

기본값은 coarse lexical, 사전학습 `prajjwal1/bert-tiny`, `diff_cls`와 `concat_cls`, seed 13/42/77, 각 10 epoch, CPU입니다. 각 실행에서 가장 좋은 dev Macro-F1의 epoch를 선택하고 `summary.json`에 평균, 표본 표준편차, seed별 점수, 대응 seed 간 차이를 기록합니다. 조기 종료는 하지 않습니다. 결과를 보며 설정을 반복 변경하면 검증셋에 과적합될 수 있으므로 최종 test는 실험 설정 확정 후 평가하세요. 실행 도중 중단되면 완료 결과가 남지만 자동 재개는 지원하지 않습니다.

| mode | 입력 표현 |
|---|---|
| `diff_cls` | A의 DIFF − B의 CLS |
| `concat_cls` | [A의 DIFF; B의 CLS], 256차원(`bert-tiny`) |
| `pair_interaction` | [A의 DIFF; B의 CLS; 절대 차이; 원소곱], 512차원(`bert-tiny`) |
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
