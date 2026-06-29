"""
R2 detection/ 디렉토리 내 파일 전체 삭제 스크립트.
injector/.env 기준으로 실행.
"""

import os
import sys
import logging
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

R2_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
R2_ACCESS_KEY  = os.getenv("CF_ACCESS_KEY", "")
R2_SECRET_KEY  = os.getenv("CF_SECRET_KEY", "")
R2_BUCKET_NAME = os.getenv("CF_BUCKET_NAME", "")

if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET_NAME]):
    logger.error("❌ R2 환경변수 미설정 — .env 확인")
    sys.exit(1)

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

PREFIX = "detection/"

paginator = s3.get_paginator("list_objects_v2")
keys_to_delete = []

for page in paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        keys_to_delete.append(obj["Key"])

if not keys_to_delete:
    logger.info("detection/ 에 파일 없음 — 이미 비어있음")
    sys.exit(0)

logger.info(f"삭제 대상: {len(keys_to_delete)}개")
for k in keys_to_delete:
    logger.info(f"  - {k}")

# 최대 1000개씩 배치 삭제
BATCH = 1000
deleted = 0
for i in range(0, len(keys_to_delete), BATCH):
    batch = keys_to_delete[i:i + BATCH]
    resp = s3.delete_objects(
        Bucket=R2_BUCKET_NAME,
        Delete={"Objects": [{"Key": k} for k in batch]},
    )
    deleted += len(resp.get("Deleted", []))
    errors = resp.get("Errors", [])
    if errors:
        for err in errors:
            logger.error(f"❌ 삭제 실패: {err['Key']} — {err['Message']}")

logger.info(f"\n✅ 완료 — {deleted}개 삭제")
