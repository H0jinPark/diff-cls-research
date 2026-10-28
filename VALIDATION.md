# 검증 기록 — 2026-10-28

- 프로젝트 전용 `.venv`에 의존성을 설치했습니다. 실제 설치 버전은 `requirements-tested.txt`에 기록했습니다.
- `pytest -q`: **11 passed**. 데이터 중복, 제품군 분리, lexical 연결요소, 공리와 별개인 분류 지표, 마커 단일 토큰/위치, 길이 초과, 실제 hidden state 선택, 뺄셈 방향, 학습 및 safetensors 저장·복원 후 예측 동일성을 검사했습니다.
- `scripts/ablation.py`: 네 모드(`diff_cls`, `cls_cls`, `diff_only`, `modifier_only`) 모두 seed 42, 1 epoch로 CLI 학습 및 평가 완료.
- 테스트 데이터: 90개 합성 예시, train/dev/test = 63/18/9. 부모/제품군은 분리되고 수식어는 공유됩니다. 이는 수식어 비중복 실험이 아닙니다.
- 실행 산출물: `runs/smoke-ablation/` 아래 체크포인트, 지표, 예측 및 비교 JSON.

**이것은 동작 검증입니다.** 작은 무작위 초기화 BERT의 점수는 성능 주장에 사용할 수 없습니다. 실제 pretrained BERT 다운로드/학습, Tratz 및 ECLASS 실데이터 검증, 한국어 모델 정확도 평가는 아직 수행하지 않았습니다.
