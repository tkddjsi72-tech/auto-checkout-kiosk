# 키오스크 상품 판별

한 장의 키오스크 사진에서 상품을 가리거나, 다시 넣으라고 돌려보낸다. 단서는 바코드, 포장 문구(OCR), 무게 세 가지이고, 서로 독립적으로 모은 뒤 아래 규칙으로 합친다.

결과는 둘 중 하나다.

- **CONFIRMED** — 상품 이름이 하나 정해짐
- **REINSERT** — 재투입. 상품을 확정하지 않음

상품이 하나인 사진의 구현은 `src/run_db_input_matching.py`, 여러 상품이 한 트레이에 있는 사진의 구현은 `src/run_multiple_matching.py`다. OCR 점수 계산은 같고, 바코드가 한 상품으로 좁혀졌을 때 다음 단계가 다르다.

## 상품 DB

대상은 11종이다. 공칭 무게의 허용 오차는 **1%** (`tolerance_g = weight_g × 0.01`). 바코드·무게는 `DB/DB_WEIGHT&BARCODE/products.json`.

| id | 상품 | 무게 (g) | 허용 (g) | 바코드 |
|---|---|---:|---:|---|
| 01 | 참크래커 | 280 | 2.8 | 8801111614566 |
| 02 | 새우깡 | 90 | 0.9 | 8801043035989 |
| 03 | 칸타타 | 170 | 1.7 | 8801056102036 |
| 04 | 오레오 | 100 | 1.0 | 8801037088168 |
| 05 | 프링글스 | 110 | 1.1 | 8886467100017 |
| 06 | 신라면 | 120 | 1.2 | 8801043014809 |
| 07 | 후라보노 | 26 | 0.26 | 8801062323159 |
| 08 | 마이구미 | 66 | 0.66 | 8801117342104 |
| 09 | 오뚜기컵밥 | 315 | 3.15 | 8801045892238 |
| 10 | 자유시간 | 36 | 0.36 | 8801019206818 |
| 11 | 마이쮸 | 44 | 0.44 | 8801111187893 |

문구 DB는 뷰마다 따로 둔다.

- 정면: `DB/DB_정면TEXT/**/*_text_set.json`
- 멀티뷰: `DB/DB_멀티뷰TEXT/**/*_text_set.json`

한 상품에 뷰가 여러 개여도 토큰을 한 덩어리로 합치지 않는다. 뷰마다 coverage를 구한 뒤 **그 상품의 최댓값**을 쓴다.

## 1. 단서

세 단서는 판정 전에 각각 만든다. OCR이 바코드 단계를 대신하지 않고, 무게가 OCR 점수를 바꾸지 않는다.

### 바코드

검출은 Laura YOLOv5, 디코드는 검출 박스를 잘라 OpenCV `BarcodeDetector`로 읽는다. pyzbar(`libzbar`)가 있으면 같은 크롭에 함께 시도한다.

검출 설정 (정면·멀티뷰 INPUT과 동일):

- `--barcode-model-family laura_yolov5`
- `--yolo-only`
- `--conf 0.25`
- `--yolo-imgsz 640`
- `--max-long-edge 640`
- GPU

가중치: `models/laura_yolov5_barcode/barcode_model.pt`

디코드는 크롭을 색/회색/CLAHE/샤픈으로 만들고, 배율 1–3배, 0/90/180/270도 회전을 돌린다. 채택 조건은 EAN-13 체크섬이 맞거나, 숫자 12–14자다. 짧은 오검출은 버린다. DB에 없는 번호는 상품 매칭에 넣지 않는다.

트레이 판별에서만, Laura 박스와 겹치는 OCR 숫자 상자에서 DB에 있는 바코드를 복구한다 (`apply_ocr_digit_barcode_fallback`). 상품 1장 판별에는 이 복구를 쓰지 않는다.

### OCR

PaddleOCR, GPU (`paddlepaddle-gpu`, `device="gpu"`). 검출 `PP-OCRv6_medium_det`, 인식 `korean_PP-OCRv5_mobile_rec`.

같은 사진에 세 결과를 만들고, 상품마다 coverage가 더 높은 쪽을 고른다.

| 소스 | 스크립트 | 하는 일 |
|---|---|---|
| `paddleocr` | `src/run_db_input_paddleocr.py` | 원본 이미지 1회 |
| `paddleocr_preproc` | `src/run_db_input_paddleocr_preproc.py` | 상품 크롭, CLAHE, 짧은 변 업스케일, 4방향 회전 중 점수 최고 |
| `paddleocr_boost` | `src/run_db_input_paddleocr_boost.py` | loose/tight 크롭, 2×2 타일, 한글+영문 인식 합집합, 상위 박스 2차 OCR |

### 무게

조건은 `|측정값 − 공칭무게| ≤ tolerance_g` 이다.

지금 실험 코드의 측정값은 저울 출력이 아니다. 상품 1장이면 `공칭 + 허용×0.25`라서, OCR(또는 트레이의 바코드)로 상품이 정해지면 무게 관문은 항상 통과한다. 트레이는 매니페스트에 적힌 상품들의 공칭 합에 같은 비율을 더한다. 실제 저울을 붙이면 같은 부등식을 측정값에 적용하면 된다.

## 2. OCR coverage

한 뷰의 coverage는 그 뷰의 DB 토큰만 분모로 쓴다.

```
coverage = (맞춘 DB 토큰 credit 합) / (그 뷰에서 채점한 DB 토큰 수)
```

채점에서 빼는 토큰은 빈 문자열과, 숫자 아닌 라틴 문자 한 글자다.

비교 전에 공백·기호를 지우고 NFKC, 소문자로 맞춘다. DB 파일 원문은 바꾸지 않는다.

DB 토큰 하나의 credit:

| 방법 | 조건 | credit |
|---|---|---|
| exact | 정규화 후 동일 | 1.0 |
| fuzzy | `SequenceMatcher` 비율 ≥ **0.70** | 1.0 |
| contain_db_in_query | DB 토큰(길이≥2)이 OCR 토큰 안에 있고 길이비 ≥ **0.40** | 길이비 |
| contain_query_in_db | OCR 토큰이 DB 토큰 안에 있고 길이비 ≥ 0.40 | 길이비 |
| contain_query_spans | OCR 조각들이 DB 토큰 글자를 덮은 비율 ≥ 0.40. 예: `500`+`ml` → `500ml` | 덮인 글자 비율 |
| contain_joined | OCR 토큰을 이어 붙인 문자열에 DB 토큰이 포함 | 0.9 |

0.70 미만 fuzzy, 0.40 미만 포함은 credit 0이다.

상품 점수는 그 상품의 모든 뷰(정면+멀티뷰) coverage 중 최댓값이다. 뷰 토큰을 합쳐 분모를 키우지 않는다.

상품이 매칭된 것으로 세려면 그 점수 ≥ **0.60** 이다. 0.60 이상인 상품이 정확히 하나일 때만 다음 단계로 간다. 없거나 둘 이상이면 REINSERT.

## 3. 상품이 하나일 때

`src/run_db_input_matching.py`

1. 바코드가 DB 상품 **둘 이상** → `BARCODE_MULTIPLE` → REINSERT.
2. 바코드가 하나이거나 없으면 OCR로 간다.
3. OCR이 0.60 이상인 상품이 없거나 둘 이상 → `OCR_NO_MATCH` / `OCR_MULTIPLE` → REINSERT.
4. OCR 상품이 하나이고, 바코드도 하나인데 서로 다르면 → `BARCODE_OCR_CONFLICT` → REINSERT.
5. 그 OCR 상품의 무게가 허용 안이면 → **CONFIRMED**. 밖이면 → `WEIGHT_OUT_OF_TOLERANCE` → REINSERT.

바코드만 맞고 OCR이 비면 확정하지 않는다. 바코드가 한 상품을 가리키면 OCR도 그 상품이어야 한다.

## 4. 트레이에 여러 상품이 있을 때

`src/run_multiple_matching.py`

1. 바코드가 DB 상품 **둘 이상** → REINSERT.
2. 바코드가 **하나**면 OCR을 기다리지 않고 그 상품 무게와 비교한다. 허용 안이면 CONFIRMED, 밖이면 `BARCODE_WEIGHT_MISMATCH_MULTIPLE` → REINSERT.
3. 바코드가 없으면 OCR로 간다. 0.60 이상이 없거나 둘 이상이면 REINSERT. 하나면 그 상품 무게와 비교해 통과 시 CONFIRMED, 실패 시 REINSERT.

트레이 측정값은 담긴 상품 무게의 합으로 두기 때문에, 바코드나 OCR이 상품 하나만 가리키면 합산 무게와 어긋나 REINSERT가 된다.

## 재현

OCR과 Laura 검출은 GPU에서 돌린다.

```bash
# 바코드 검출
CUDA_VISIBLE_DEVICES=0 python3 src/inspect_barcode_instances.py \
  --dataset-root Kiosk_experiment/정면INPUT이미지 \
  --barcode-model-family laura_yolov5 \
  --yolo-only --conf 0.25 --yolo-imgsz 640 --max-long-edge 640 \
  --output-dir output/db_input_matching/laura

# OCR 세 갈래 (각각 GPU)
python3 src/run_db_input_paddleocr.py \
  --input-dir Kiosk_experiment/정면INPUT이미지 \
  --output-dir output/db_input_matching/paddleocr
python3 src/run_db_input_paddleocr_preproc.py \
  --input-dir Kiosk_experiment/정면INPUT이미지 \
  --output-dir output/db_input_matching/paddleocr_preproc
python3 src/run_db_input_paddleocr_boost.py \
  --input-dir Kiosk_experiment/정면INPUT이미지 \
  --output-dir output/db_input_matching/paddleocr_boost

# 판별
python3 src/run_db_input_matching.py \
  --input-dir Kiosk_experiment/정면INPUT이미지 \
  --laura-results output/db_input_matching/laura/results.jsonl \
  --paddle-dir output/db_input_matching/paddleocr \
  --paddle-dir output/db_input_matching/paddleocr_preproc \
  --paddle-dir output/db_input_matching/paddleocr_boost \
  --db-text-dir DB/DB_정면TEXT \
  --db-text-dir DB/DB_멀티뷰TEXT \
  --output-dir output/db_input_matching
```

산출은 `matching_results.json`, `matching_summary.csv`다.
