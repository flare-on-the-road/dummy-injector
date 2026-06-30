"""
injector/main.py — 로컬 이미지를 파이프라인에 수동 주입하는 CLI 스크립트.

사용법:
    python main.py <이미지경로> --cctv <cctv_id> [--cctv-name <이름>] [--location <위치명>]

모드:
    --mode full (기본): 원본 이미지 → ONNX(1차) → bbox → VLM(2차) → detection/ → 백엔드 → 프론트
    --mode vlm:         bbox 이미지 → VLM만 → detection/ → 백엔드 → 프론트 (VLM 성능 단독 검증)

예시:
    python main.py fire.jpg --cctv goduck_tunnel --cctv-name "[세종] 고덕터널" --location 고덕터널
    python main.py bbox_fire.jpg --cctv goduck_tunnel --mode vlm
"""

import argparse
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
import requests
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 환경변수 ─────────────────────────────────────────────────────────────────

R2_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.getenv("CF_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("CF_SECRET_KEY", "")
R2_BUCKET_NAME = os.getenv("CF_BUCKET_NAME", "")

VISION_API_URL = os.getenv("VISION_API_URL", "")
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "30"))

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "")
BACKEND_TIMEOUT = int(os.getenv("BACKEND_TIMEOUT", "10"))

KST = timezone(timedelta(hours=9))


# ── R2 ───────────────────────────────────────────────────────────────────────

def _init_r2() -> Optional[object]:
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY and R2_SECRET_KEY):
        return None
    try:
        return boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
            region_name="auto",
        )
    except Exception as e:
        logger.error(f"❌ R2 초기화 실패: {e}")
        return None


def upload_to_r2(s3, image_bytes: bytes, key: str) -> bool:
    if not s3:
        logger.warning("⚠ R2 미설정 → 업로드 생략")
        return False
    try:
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
        logger.info(f"✓ R2 업로드: {key}")
        return True
    except Exception as e:
        logger.error(f"❌ R2 업로드 실패: {e}")
        return False


# ── Vision API ────────────────────────────────────────────────────────────────

def _parse_detections(spec: str) -> list:
    """'fire:0.65,carlight:0.55' → [{'class_name':'fire','confidence':0.65}, ...]

    신뢰도를 생략하면(예: 'fire') 0.5로 채운다. vlm 모드에서 VLM에 무엇을 판정시킬지 지정한다.
    """
    items = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, conf = token.split(":", 1)
            items.append({"class_name": name.strip(), "confidence": float(conf)})
        else:
            items.append({"class_name": token, "confidence": 0.5})
    return items


def call_vlm_only(image_bytes: bytes, detections: list) -> Optional[dict]:
    """bbox 이미지 + 탐지목록을 vision-model /vlm 엔드포인트에 직접 전달해 VLM 결과만 받는다."""
    if not VISION_API_URL:
        logger.error("❌ VISION_API_URL 미설정")
        return None
    url = f"{VISION_API_URL.rstrip('/')}/vlm"
    try:
        resp = requests.post(
            url,
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={"detections": json.dumps(detections)},
            timeout=VISION_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()  # 신 형식: [{class_name, is_false_positive, reason}, ...]
        items = result if isinstance(result, list) else []
        confirmed = [v for v in items if not v.get("is_false_positive")]
        logger.info(f"✓ /vlm 응답: {len(items)}건 판정, 화재확정 {len(confirmed)}건")
        return result
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Vision API 연결 실패: {url}")
    except requests.exceptions.Timeout:
        logger.error(f"❌ Vision API 타임아웃 ({VISION_TIMEOUT}s)")
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Vision API HTTP 오류: {e}")
    except Exception as e:
        logger.error(f"❌ Vision API 예외: {e}")
    return None


def call_predict(image_bytes: bytes, display_name: str) -> Optional[dict]:
    if not VISION_API_URL:
        logger.error("❌ VISION_API_URL 미설정")
        return None
    url = f"{VISION_API_URL.rstrip('/')}/predict"
    try:
        resp = requests.post(
            url,
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={"confidence": 0.25, "max_detections": 100},
            timeout=VISION_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        summary = result.get("summary", {})
        logger.info(
            f"✓ /predict 응답: total={summary.get('total_detections')} "
            f"max_conf={summary.get('max_confidence', 0.0):.2f}"
        )
        vlm = result.get("vlm")
        if vlm:
            confirmed = [v for v in vlm if not v.get("is_false_positive")]
            logger.info(f"  VLM: {len(vlm)}건 판정, 화재확정 {len(confirmed)}건")
        return result
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Vision API 연결 실패: {url}")
    except requests.exceptions.Timeout:
        logger.error(f"❌ Vision API 타임아웃 ({VISION_TIMEOUT}s)")
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Vision API HTTP 오류: {e}")
    except Exception as e:
        logger.error(f"❌ Vision API 예외: {e}")
    return None


# ── Backend ───────────────────────────────────────────────────────────────────

def post_event(location: dict, vision_result: dict, snapshot_key: str) -> Optional[int]:
    if not BACKEND_API_URL:
        logger.error("❌ BACKEND_API_URL 미설정")
        return None
    url = f"{BACKEND_API_URL.rstrip('/')}/api/events"
    vlm_results = vision_result.get("vlm") or []
    detections = [
        {
            "label": d["class_name"],
            "confidence": round(d["confidence"], 4),
            "bbox": d.get("bbox", []),
        }
        for d in vision_result.get("detections", [])
    ]
    payload = {
        "cctv_id": location["id"],
        "cctv_name": location["display_name"],
        "location_name": location["location_name"],
        "detected_at": datetime.now(timezone.utc).astimezone(KST).isoformat(),
        "vlm_results": vlm_results,
        "detections": detections,
        "snapshot_key": snapshot_key,
    }
    try:
        resp = requests.post(url, json=payload, timeout=BACKEND_TIMEOUT)
        resp.raise_for_status()
        event_id = resp.json().get("id")
        logger.info(f"✓ 이벤트 저장 완료 (id={event_id})")
        return event_id
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Backend 연결 실패: {url}")
    except requests.exceptions.Timeout:
        logger.error(f"❌ Backend 타임아웃 ({BACKEND_TIMEOUT}s)")
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Backend HTTP 오류: {e}")
    except Exception as e:
        logger.error(f"❌ Backend 예외: {e}")
    return None


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="로컬 이미지를 flare 파이프라인(R2 → AI → Backend)에 수동 주입합니다."
    )
    parser.add_argument("image", help="주입할 이미지 파일 경로")
    parser.add_argument("--cctv", required=True, help="CCTV ID (예: goduck_tunnel)")
    parser.add_argument("--cctv-name", default=None, help="CCTV 표시명 (기본값: --cctv 값)")
    parser.add_argument("--location", default=None, help="위치명 (기본값: --cctv 값)")
    parser.add_argument(
        "--skip-r2", action="store_true", help="R2 업로드를 건너뛰고 AI 호출만 실행"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "vlm"],
        default="full",
        help=(
            "full(기본): 원본→ONNX→bbox→VLM→detection/→백엔드 전체 파이프라인 | "
            "vlm: bbox 이미지→VLM만→detection/→백엔드 (VLM 단독 검증)"
        ),
    )
    parser.add_argument(
        "--detections",
        default="fire:0.6",
        help=(
            "vlm 모드에서 VLM에 판정시킬 탐지목록. "
            "형식: 'fire:0.65,carlight:0.55' (신뢰도 생략 시 0.5). full 모드에선 무시됨"
        ),
    )
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        logger.error(f"❌ 이미지 파일 없음: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    logger.info(f"✓ 이미지 로드: {image_path} ({len(image_bytes) // 1024} KB)")

    location = {
        "id": args.cctv,
        "display_name": args.cctv_name or args.cctv,
        "location_name": args.location or args.cctv,
    }

    timestamp = datetime.now(timezone.utc).astimezone(KST).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{location['location_name']}.jpg"
    detection_key = f"detection/{filename}"

    s3 = _init_r2() if not args.skip_r2 else None

    if args.mode == "full":
        # ── 모드 1: 전체 파이프라인 ──────────────────────────────────────────
        # 원본 이미지 → ONNX(1차) → bbox → VLM(2차) → detection/ → 백엔드
        logger.info("[ 모드: 전체 파이프라인 ]")

        vision_result = call_predict(image_bytes, location["display_name"])
        if vision_result is None:
            logger.error("❌ Vision API 실패 → 파이프라인 중단")
            sys.exit(1)

        # worker와 동일: vlm 결과가 비면(carlight 단독/저신뢰도/무시 케이스) 저장 생략
        # → detection/ 업로드와 백엔드 저장 모두 건너뜀
        if not (vision_result.get("vlm") or []):
            logger.info("⏭ 위험 탐지 없음 (carlight 단독/저신뢰도) → 저장 생략 (worker와 동일)")
            sys.exit(0)

        if not args.skip_r2:
            annotated_b64 = vision_result.get("annotated_image_b64")
            detection_img = base64.b64decode(annotated_b64) if annotated_b64 else image_bytes
            upload_to_r2(s3, detection_img, detection_key)
        else:
            logger.info("⚠ --skip-r2 옵션으로 R2 업로드 생략")

        event_id = post_event(location, vision_result, detection_key)

    else:
        # ── 모드 2: VLM 단독 파이프라인 ─────────────────────────────────────
        # bbox 이미지 → VLM만 → detection/ → 백엔드 (ONNX 생략)
        logger.info("[ 모드: VLM 단독 파이프라인 ]")

        det_list = _parse_detections(args.detections)
        if not det_list:
            logger.error("❌ --detections 가 비어있습니다. 예: --detections 'fire:0.65'")
            sys.exit(1)
        logger.info(f"  VLM 입력 탐지목록: {det_list}")

        vlm_result = call_vlm_only(image_bytes, det_list)
        if vlm_result is None:
            logger.error("❌ VLM 호출 실패 → 파이프라인 중단")
            sys.exit(1)

        if not args.skip_r2:
            upload_to_r2(s3, image_bytes, detection_key)
        else:
            logger.info("⚠ --skip-r2 옵션으로 R2 업로드 생략")

        # VLM 결과를 /predict 응답 형식으로 래핑해 post_event 재사용
        synthetic_result = {
            "summary": {"total_detections": 0, "max_confidence": 0.0},
            "detections": [],
            "vlm": vlm_result,
        }
        event_id = post_event(location, synthetic_result, detection_key)

    if event_id is None:
        logger.error("❌ 이벤트 저장 실패")
        sys.exit(1)

    logger.info(f"\n✅ 완료 — mode={args.mode}, event_id={event_id}, snapshot_key={detection_key}")


if __name__ == "__main__":
    main()
