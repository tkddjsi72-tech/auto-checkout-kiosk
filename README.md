# 키오스크 상품 판별

무인 키오스크 사진에서 상품 11종을 가리거나 재투입을 요구한다. 실행 진입점은 `run_identification.py`다.

## 환경

NVIDIA GPU가 필요하다. OCR 스크립트는 CPU용 Paddle을 감지하면 종료하고, `run_identification.py`의 바코드 검출은 CUDA로 고정되어 있다.

- Python 3.10
- CUDA 11.8용 Paddle GPU 휠 (`paddlepaddle-gpu==3.3.1`)

확인한 환경은 Ubuntu 22.04.5, Python 3.10.12, 드라이버 535.309.01, CUDA 12.2, NVIDIA RTX A6000이다. PyPI 기본 torch는 이 드라이버에서 열리지 않는다. Paddle CUDA 11.8 휠과 맞추기 위해 `requirements.txt`는 `torch==2.3.1+cu118`, `torchvision==0.18.1+cu118`, `nvidia-cudnn-cu11==8.9.6.50`을 고정한다. YOLOv5 v6.2가 `pkg_resources`를 쓰므로 `setuptools<81`도 고정한다.
- 바코드 번호 읽기에 시스템 라이브러리 `libzbar0`가 있으면 pyzbar를 함께 쓴다.

첫 실행 때 PaddleOCR 인식 모델과 YOLOv5 코드(`ultralytics/yolov5:v6.2`)를 인터넷으로 받는다. Laura 가중치는 옛 YOLOv5 체크포인트라 `ultralytics`만으로는 열리지 않을 수 있다. 그때 코드가 torch.hub로 YOLOv5를 받으며, 이 코드가 `pandas`, `seaborn`을 사용한다.

## 설치

```bash
sudo apt-get install -y libzbar0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

정면 사진 11장:

```bash
python3 run_identification.py \
  --input-dir dataset/single_front \
  --output-dir output/full_identification
```

멀티뷰 사진 29장:

```bash
python3 run_identification.py \
  --input-dir dataset/single_multiview \
  --output-dir output/multiview_identification
```

결과는 각 출력 폴더의 `matching_results.json`, `matching_summary.csv`다.

여러 상품이 한 사진에 있는 `dataset/multiple`도 같은 명령으로 돌린다.

```bash
python3 run_identification.py \
  --input-dir dataset/multiple \
  --output-dir output/multiple_identification
```

매니페스트의 `expected_product`나 `expected_products`는 채점용이다. 판별은 그 키 없이 사진과 DB만으로 한다. 저울값은 항목의 `measured_weight_g`다. 지금 매니페스트 세 개 모두 이 값이 없으므로 무게 비교는 실행되지 않는다. 따라서 이 폴더들을 지금 상태로 돌렸을 때의 CONFIRMED는 바코드와 OCR만으로 나온 결과다.

`dataset/multiple`도 같은 판정이다. 바코드와 OCR이 상품 하나만 읽으면, 저울값이 없는 동안에는 그 상품으로 CONFIRMED가 된다. 트레이 전체 무게가 `measured_weight_g`에 있으면, 그 무게가 읽힌 상품 하나의 공칭과 1% 넘게 다를 때 재투입이 된다.

코드는 저울 장치를 읽지 않는다. 측정값은 매니페스트 항목의 `measured_weight_g`로 들어온다. `db/products.json`의 `weight_g`는 포장지에 적힌 내용량이다. 실제 저울은 포장을 포함한 총중량을 재므로, 저울을 붙일 때는 그 총중량을 `measured_weight_g`에 넣는다.

다른 사진 한 장을 보려면 폴더에 이미지와 `manifest.json`을 둔다. `expected_product`는 채점용이라 없어도 판별된다.

```text
dataset/custom/test.jpg
dataset/custom/manifest.json
```

```json
{ "items": [ { "input_image": "test.jpg" } ] }
```

```bash
python3 run_identification.py \
  --input-dir dataset/custom \
  --output-dir output/custom
```

## 기록된 판정

아래는 위 환경에서 세 폴더를 끝까지 돌린 결과다. 매니페스트에 `measured_weight_g`가 없어 무게 비교는 하지 않았다. 맞는 확정은 `final_product`가 `expected_product`와 같은 경우다. 원본은 `expected_results/`의 `matching_summary.csv`다.

| 입력 | 장수 | CONFIRMED | 재투입 | 정답 일치 |
|---|---:|---:|---:|---:|
| 정면 | 11 | 11 | 0 | 11 |
| 멀티뷰 | 29 | 16 | 13 | 16 |
| 여러 상품 | 7 | 2 | 5 | — |

확정된 상품이 정답과 다른 경우는 없다. 여러 상품은 `expected_products` 목록이라 정답 일치를 매기지 않았다. 그 2장의 CONFIRMED는 저울값 없이 상품 하나만 읽힌 경우다.

같은 실행의 바코드는 정면 박스 3장·번호 해독 0장, 멀티뷰 박스 15장·번호 해독 10장·해독 번호가 정답 상품 9장, 여러 상품 박스 5장·번호 해독 3장이다. 정면 11장의 CONFIRMED는 OCR만으로 나왔다.

## 폴더

| 경로 | 내용 |
|---|---|
| `run_identification.py` | 검출, OCR 세 갈래, 판정을 순서대로 실행 |
| `dataset/single_front` | 상품 1개, 정면 11장 |
| `dataset/single_multiview` | 상품 1개, 윗면·아랫면·옆면 29장 |
| `dataset/multiple` | 한 사진에 상품이 여러 개인 입력 |
| `db/products.json` | 바코드, 공칭 무게, 허용 오차 1% |
| `db/text_front`, `db/text_multiview` | 뷰별 문구. 뷰 토큰은 합치지 않는다 |
| `models/laura_yolov5_barcode/barcode_model.pt` | Laura YOLOv5 바코드 검출 가중치 |
| `expected_results/` | 위 판정의 `matching_summary.csv` |
| `src/` | 검출, OCR, 판정 구현 |

## 판별 규칙

한 장의 키오스크 사진에서 상품을 가리거나, 다시 넣으라고 돌려보낸다. 단서는 바코드, 포장 문구(OCR), 무게 세 가지이고, 서로 독립적으로 모은 뒤 아래 규칙으로 합친다.

결과는 둘 중 하나다.

- **CONFIRMED** — 상품 이름이 하나 정해짐
- **REINSERT** — 재투입. 상품을 확정하지 않음

판정 구현은 `src/run_db_input_matching.py`다.

## 상품 DB

대상은 11종이다. 공칭 무게의 허용 오차는 **1%** (`tolerance_g = weight_g × 0.01`). 바코드·무게는 `db/products.json`.

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

- 정면: `db/text_front/**/*_text_set.json`
- 멀티뷰: `db/text_multiview/**/*_text_set.json`

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

가중치: `models/laura_yolov5_barcode/barcode_model.pt`. 직접 학습한 파일이 아니라 [lauraAriasFdez/barcodeDetector](https://github.com/lauraAriasFdez/barcodeDetector)의 YOLOv5 `barcode_model.pt`다. 그 저장소에는 LICENSE 파일이 없다.

디코드는 크롭을 색/회색/CLAHE/샤픈으로 만들고, 배율 1–3배, 0/90/180/270도 회전을 돌린다. 채택 조건은 EAN-13 체크섬이 맞거나, 숫자 12–14자다. 짧은 오검출은 버린다. DB에 없는 번호는 상품 매칭에 넣지 않는다.

### OCR

PaddleOCR, GPU (`paddlepaddle-gpu`, `device="gpu"`). 검출 `PP-OCRv6_medium_det`, 인식 `korean_PP-OCRv5_mobile_rec`.

같은 사진에 세 결과를 만들고, 상품마다 coverage가 더 높은 쪽을 고른다.

| 소스 | 스크립트 | 하는 일 |
|---|---|---|
| `paddleocr` | `src/run_db_input_paddleocr.py` | 원본 이미지 1회 |
| `paddleocr_preproc` | `src/run_db_input_paddleocr_preproc.py` | 상품 크롭, CLAHE, 짧은 변 업스케일, 4방향 회전 중 점수 최고 |
| `paddleocr_boost` | `src/run_db_input_paddleocr_boost.py` | loose/tight 크롭, 2×2 타일, 한글+영문 인식 합집합, 상위 박스 2차 OCR |

### 무게

저울 장치를 읽는 코드는 없다. 측정값은 매니페스트 항목의 `measured_weight_g`로 넣는다. 그 값이 있으면 `|측정값 − 공칭무게| ≤ tolerance_g` 이다. 없으면 무게 비교를 하지 않고, 바코드와 OCR이 상품 하나를 가리키면 CONFIRMED가 된다. 공칭무게 `weight_g`는 포장지 내용량이다. 저울의 총중량을 넣을 때는 포장 무게가 1% 허용 안에 드는지도 본다.

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

## 3. 판정

`src/run_db_input_matching.py`

1. 바코드가 DB 상품 **둘 이상** → `BARCODE_MULTIPLE` → REINSERT.
2. 바코드가 하나이거나 없으면 OCR로 간다.
3. OCR이 0.60 이상인 상품이 없거나 둘 이상 → `OCR_NO_MATCH` / `OCR_MULTIPLE` → REINSERT.
4. OCR 상품이 하나이고, 바코드도 하나인데 서로 다르면 → `BARCODE_OCR_CONFLICT` → REINSERT.
5. `measured_weight_g`가 없으면 그 상품으로 **CONFIRMED**. 있으면 `|측정 − 공칭| ≤ 공칭의 1%`일 때 CONFIRMED, 밖이면 `WEIGHT_OUT_OF_TOLERANCE` → REINSERT.

바코드만 맞고 OCR이 비면 확정하지 않는다. 바코드가 한 상품을 가리키면 OCR도 그 상품이어야 한다.

## 출처

이 저장소의 코드는 [MIT](LICENSE)다. 아래 가중치와 라이브러리는 각자 라이선스를 따른다.

| 구성 | 출처 | 라이선스 |
|---|---|---|
| 바코드 검출 가중치 | [lauraAriasFdez/barcodeDetector](https://github.com/lauraAriasFdez/barcodeDetector) `barcode_model.pt` | 원 저장소에 LICENSE 없음 |
| 바코드 검출 코드 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) `v6.2`. 첫 실행 때 torch.hub로 받는다 | AGPL-3.0 |
| OCR | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). 검출 `PP-OCRv6_medium_det`, 인식 `korean_PP-OCRv5_mobile_rec`, boost의 `PP-OCRv6_medium_rec` | Apache-2.0 |
