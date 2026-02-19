"""
NSE F&O Bhavcopy Downloader — Phase B3

Downloads daily F&O Bhavcopy files from NSE archives,
stores raw ZIPs in S3, and registers metadata via ODAL.

TDR-ORACLE-008: S3 Data Lake + DynamoDB metadata index
Architecture: §5.0 (Ingestion Layer)

URL Pattern (post-2024):
  https://nsearchives.nseindia.com/content/fo/
  BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip

Rate limiting: NSE enforces aggressive anti-bot protection.
  - 1 request per 2 seconds (conservative)
  - Session cookies required (acquire from market-data page)
  - Retry with backoff on 403/503
"""

import gzip
import hashlib
import io
import json
import logging
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import boto3
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oracle.ingestion.bhavcopy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NSE_BASE_URL = "https://nsearchives.nseindia.com/content/fo"
NSE_COOKIE_URL = "https://www.nseindia.com/market-data/live-equity-market"
AWS_REGION = "ap-south-1"
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_RAW_PREFIX = "raw/bhavcopy/fo"

# Rate limiting
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0
SESSION_REFRESH_INTERVAL = 50  # Re-acquire cookies every N requests

# Headers mimicking a real browser
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DownloadResult:
    """Result of a single Bhavcopy download attempt."""
    trade_date: date
    success: bool
    s3_key: Optional[str] = None
    file_checksum: Optional[str] = None
    raw_size_bytes: int = 0
    csv_row_count: int = 0
    nifty_option_count: int = 0
    banknifty_option_count: int = 0
    error_message: Optional[str] = None


@dataclass
class DownloadSummary:
    """Summary of a batch download run."""
    start_date: date
    end_date: date
    total_trading_days: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    failed_dates: list = field(default_factory=list)
    total_bytes: int = 0


# ---------------------------------------------------------------------------
# Bhavcopy Downloader
# ---------------------------------------------------------------------------
class BhavcopyDownloader:
    """Downloads NSE F&O Bhavcopy files and stores them in S3."""

    def __init__(self, region: str = AWS_REGION, bucket: str = S3_BUCKET):
        self._s3 = boto3.client("s3", region_name=region)
        self._session = requests.Session()
        self._bucket = bucket
        self._request_count = 0
        self._last_cookie_refresh = 0
        self._refresh_session()

    def _refresh_session(self) -> None:
        """Acquire NSE session cookies."""
        try:
            r = self._session.get(
                NSE_COOKIE_URL,
                headers=BROWSER_HEADERS,
                timeout=15,
            )
            if r.status_code == 200:
                logger.info(
                    f"NSE session refreshed — {len(self._session.cookies)} cookies"
                )
            else:
                logger.warning(f"NSE session refresh returned {r.status_code}")
            self._last_cookie_refresh = self._request_count
        except Exception as e:
            logger.error(f"NSE session refresh failed: {e}")

    def _ensure_session(self) -> None:
        """Refresh session cookies periodically."""
        if (self._request_count - self._last_cookie_refresh
                >= SESSION_REFRESH_INTERVAL):
            self._refresh_session()

    @staticmethod
    def bhavcopy_url(trade_date: date) -> str:
        """Construct NSE F&O Bhavcopy URL for a given date."""
        return (
            f"{NSE_BASE_URL}/BhavCopy_NSE_FO_0_0_0_"
            f"{trade_date.strftime('%Y%m%d')}_F_0000.csv.zip"
        )

    @staticmethod
    def s3_raw_key(trade_date: date) -> str:
        """Construct S3 key per TDR-ORACLE-008 prefix structure."""
        return (
            f"{S3_RAW_PREFIX}/year={trade_date.year}/"
            f"month={trade_date.month:02d}/"
            f"fo_bhavcopy_{trade_date.strftime('%Y%m%d')}.csv.gz"
        )

    def _s3_key_exists(self, key: str) -> bool:
        """Check if an S3 object already exists (skip re-download)."""
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._s3.exceptions.ClientError:
            return False

    def download_date(
        self,
        trade_date: date,
        skip_existing: bool = True,
    ) -> DownloadResult:
        """Download Bhavcopy for a single date.

        Pipeline:
          1. Check if already in S3 (skip if exists)
          2. Download ZIP from NSE
          3. Extract CSV from ZIP
          4. Re-compress as gzip (smaller, standard)
          5. Compute SHA-256 checksum
          6. Upload to S3
          7. Return metadata for ODAL registration

        Args:
            trade_date: The trading date to download
            skip_existing: If True, skip dates already in S3

        Returns:
            DownloadResult with metadata
        """
        s3_key = self.s3_raw_key(trade_date)

        # Step 1: Check existing
        if skip_existing and self._s3_key_exists(s3_key):
            logger.info(f"{trade_date}: Already in S3, skipping")
            return DownloadResult(
                trade_date=trade_date,
                success=True,
                s3_key=s3_key,
                error_message="skipped_existing",
            )

        # Step 2: Download from NSE with retry
        self._ensure_session()
        url = self.bhavcopy_url(trade_date)
        content = None

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY_SECONDS)
                self._request_count += 1
                r = self._session.get(
                    url, headers=BROWSER_HEADERS, timeout=30,
                )
                if r.status_code == 200 and len(r.content) > 1000:
                    content = r.content
                    break
                elif r.status_code == 404:
                    # Legitimate missing date (holiday / not yet available)
                    return DownloadResult(
                        trade_date=trade_date,
                        success=False,
                        error_message=f"HTTP 404 — not a trading day or not in archive",
                    )
                else:
                    logger.warning(
                        f"{trade_date}: Attempt {attempt+1} got "
                        f"{r.status_code} ({len(r.content)} bytes)"
                    )
            except Exception as e:
                logger.warning(
                    f"{trade_date}: Attempt {attempt+1} failed: {e}"
                )

            # Backoff before retry
            if attempt < MAX_RETRIES - 1:
                backoff = REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** (attempt + 1))
                logger.info(f"Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                self._refresh_session()

        if content is None:
            return DownloadResult(
                trade_date=trade_date,
                success=False,
                error_message="All download attempts failed",
            )

        # Step 3: Extract CSV from ZIP
        try:
            z = zipfile.ZipFile(io.BytesIO(content))
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                csv_bytes = f.read()
        except Exception as e:
            return DownloadResult(
                trade_date=trade_date,
                success=False,
                error_message=f"ZIP extraction failed: {e}",
            )

        # Step 4: Count rows and NIFTY/BANKNIFTY instruments
        csv_text = csv_bytes.decode("utf-8", errors="replace")
        lines = csv_text.strip().split("\n")
        row_count = len(lines) - 1  # Exclude header

        nifty_opts = sum(
            1 for l in lines[1:]
            if ",NIFTY," in l and (",CE," in l or ",PE," in l)
            and ",BANKNIFTY," not in l
        )
        bnifty_opts = sum(
            1 for l in lines[1:]
            if ",BANKNIFTY," in l and (",CE," in l or ",PE," in l)
        )

        # Step 5: Compress as gzip
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buffer, mode="wb") as gz:
            gz.write(csv_bytes)
        gz_bytes = gz_buffer.getvalue()

        # Step 6: Compute checksum
        checksum = hashlib.sha256(gz_bytes).hexdigest()

        # Step 7: Upload to S3
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=gz_bytes,
                ContentType="application/gzip",
                Metadata={
                    "trade-date": trade_date.isoformat(),
                    "source": "NSE_FO_BHAVCOPY",
                    "original-zip-size": str(len(content)),
                    "csv-rows": str(row_count),
                    "nifty-options": str(nifty_opts),
                    "banknifty-options": str(bnifty_opts),
                    "sha256": checksum,
                },
            )
        except Exception as e:
            return DownloadResult(
                trade_date=trade_date,
                success=False,
                error_message=f"S3 upload failed: {e}",
            )

        logger.info(
            f"{trade_date}: ✓ {len(gz_bytes):,} bytes → s3://{self._bucket}/{s3_key} "
            f"({row_count} rows, NIFTY:{nifty_opts}, BNIFTY:{bnifty_opts})"
        )

        return DownloadResult(
            trade_date=trade_date,
            success=True,
            s3_key=s3_key,
            file_checksum=checksum,
            raw_size_bytes=len(gz_bytes),
            csv_row_count=row_count,
            nifty_option_count=nifty_opts,
            banknifty_option_count=bnifty_opts,
        )

    def download_range(
        self,
        start_date: date,
        end_date: date,
        skip_existing: bool = True,
    ) -> DownloadSummary:
        """Download Bhavcopy for a date range.

        Iterates through calendar dates, skipping weekends.
        Holidays are detected by 404 responses from NSE.
        """
        summary = DownloadSummary(start_date=start_date, end_date=end_date)
        current = start_date

        while current <= end_date:
            # Skip weekends
            if current.weekday() >= 5:  # Saturday=5, Sunday=6
                current += timedelta(days=1)
                continue

            summary.total_trading_days += 1
            result = self.download_date(current, skip_existing=skip_existing)

            if result.success:
                if result.error_message == "skipped_existing":
                    summary.skipped_existing += 1
                else:
                    summary.downloaded += 1
                    summary.total_bytes += result.raw_size_bytes
            else:
                if "404" in (result.error_message or ""):
                    # Not a trading day — don't count as failure
                    summary.total_trading_days -= 1
                else:
                    summary.failed += 1
                    summary.failed_dates.append(current)
                    logger.error(
                        f"{current}: FAILED — {result.error_message}"
                    )

            current += timedelta(days=1)

        # Print summary
        logger.info(
            f"\n{'='*60}\n"
            f"DOWNLOAD SUMMARY: {start_date} to {end_date}\n"
            f"{'='*60}\n"
            f"  Trading days attempted: {summary.total_trading_days}\n"
            f"  Downloaded:             {summary.downloaded}\n"
            f"  Skipped (existing):     {summary.skipped_existing}\n"
            f"  Failed:                 {summary.failed}\n"
            f"  Total bytes:            {summary.total_bytes:,}\n"
            f"  Failed dates:           {summary.failed_dates}\n"
        )
        return summary
