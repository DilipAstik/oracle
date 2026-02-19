"""
NSE F&O Bhavcopy Parser — Phase B4

Reads raw gzipped Bhavcopy CSVs from S3, extracts NIFTY and BANKNIFTY
option chains and futures, normalises to Oracle's canonical schema,
and writes Parquet files back to S3.

TDR-ORACLE-008 §2.2: raw/ → canonical/
Data Contract §1.1: Observation-based state snapshots

Input:  s3://oracle-data-lake-.../raw/bhavcopy/fo/year=YYYY/month=MM/fo_bhavcopy_YYYYMMDD.csv.gz
Output: s3://oracle-data-lake-.../canonical/option_chains/underlying=NIFTY/year=YYYY/month=MM/NIFTY_YYYYMMDD.parquet
        s3://oracle-data-lake-.../canonical/option_chains/underlying=BANKNIFTY/year=YYYY/month=MM/BANKNIFTY_YYYYMMDD.parquet
        s3://oracle-data-lake-.../canonical/futures/underlying=NIFTY/year=YYYY/month=MM/NIFTY_FUT_YYYYMMDD.parquet
        s3://oracle-data-lake-.../canonical/futures/underlying=BANKNIFTY/year=YYYY/month=MM/BANKNIFTY_FUT_YYYYMMDD.parquet

Bhavcopy Schema (new NSE format, post-2024):
  TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId, ISIN, TckrSymb,
  SctySrs, XpryDt, FininstrmActlXpryDt, StrkPric, OptnTp, FinInstrmNm,
  OpnPric, HghPric, LwPric, ClsPric, LastPric, PrvsClsgPric, UndrlygPric,
  SttlmPric, OpnIntrst, ChngInOpnIntrst, TtlTradgVol, TtlTrfVal,
  TtlNbOfTxsExctd, SsnId, NewBrdLotQty, Rmks, Rsvd1-4

Canonical Option Schema:
  trade_date, underlying, expiry_date, strike, option_type, open, high,
  low, close, last, settlement_price, underlying_price, previous_close,
  open_interest, oi_change, volume, turnover, num_trades, lot_size,
  instrument_name

Canonical Futures Schema:
  trade_date, underlying, expiry_date, open, high, low, close, last,
  settlement_price, underlying_price, previous_close, open_interest,
  oi_change, volume, turnover, num_trades, lot_size, instrument_name
"""

import gzip
import io
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oracle.ingestion.parser")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AWS_REGION = "ap-south-1"
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
UNDERLYINGS = ("NIFTY", "BANKNIFTY")

# Bhavcopy instrument type codes
OPTION_TYPE_CODE = "IDO"  # Index Derivative Option
FUTURES_TYPE_CODE = "IDF"  # Index Derivative Future

# Column mapping: Bhavcopy → Canonical
OPTION_COLUMN_MAP = {
    "TradDt": "trade_date",
    "TckrSymb": "underlying",
    "XpryDt": "expiry_date",
    "StrkPric": "strike",
    "OptnTp": "option_type",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "SttlmPric": "settlement_price",
    "UndrlygPric": "underlying_price",
    "PrvsClsgPric": "previous_close",
    "OpnIntrst": "open_interest",
    "ChngInOpnIntrst": "oi_change",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "num_trades",
    "NewBrdLotQty": "lot_size",
    "FinInstrmNm": "instrument_name",
}

FUTURES_COLUMN_MAP = {
    "TradDt": "trade_date",
    "TckrSymb": "underlying",
    "XpryDt": "expiry_date",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "SttlmPric": "settlement_price",
    "UndrlygPric": "underlying_price",
    "PrvsClsgPric": "previous_close",
    "OpnIntrst": "open_interest",
    "ChngInOpnIntrst": "oi_change",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "num_trades",
    "NewBrdLotQty": "lot_size",
    "FinInstrmNm": "instrument_name",
}

# Parquet data types for canonical schema
OPTION_SCHEMA = pa.schema([
    ("trade_date", pa.date32()),
    ("underlying", pa.string()),
    ("expiry_date", pa.date32()),
    ("strike", pa.float64()),
    ("option_type", pa.string()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("last", pa.float64()),
    ("settlement_price", pa.float64()),
    ("underlying_price", pa.float64()),
    ("previous_close", pa.float64()),
    ("open_interest", pa.int64()),
    ("oi_change", pa.int64()),
    ("volume", pa.int64()),
    ("turnover", pa.float64()),
    ("num_trades", pa.int64()),
    ("lot_size", pa.int32()),
    ("instrument_name", pa.string()),
])

FUTURES_SCHEMA = pa.schema([
    ("trade_date", pa.date32()),
    ("underlying", pa.string()),
    ("expiry_date", pa.date32()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("last", pa.float64()),
    ("settlement_price", pa.float64()),
    ("underlying_price", pa.float64()),
    ("previous_close", pa.float64()),
    ("open_interest", pa.int64()),
    ("oi_change", pa.int64()),
    ("volume", pa.int64()),
    ("turnover", pa.float64()),
    ("num_trades", pa.int64()),
    ("lot_size", pa.int32()),
    ("instrument_name", pa.string()),
])


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a single Bhavcopy file."""
    trade_date: date
    success: bool
    nifty_options: int = 0
    banknifty_options: int = 0
    nifty_futures: int = 0
    banknifty_futures: int = 0
    nifty_expiries: int = 0
    banknifty_expiries: int = 0
    nifty_strikes_near: int = 0
    output_keys: list = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ParseSummary:
    """Summary of a batch parse run."""
    total_files: int = 0
    parsed: int = 0
    skipped_existing: int = 0
    failed: int = 0
    failed_dates: list = field(default_factory=list)
    total_nifty_options: int = 0
    total_banknifty_options: int = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class BhavcopyParser:
    """Parses raw Bhavcopy CSVs into canonical Parquet files."""

    def __init__(self, region: str = AWS_REGION, bucket: str = S3_BUCKET):
        self._s3 = boto3.client("s3", region_name=region)
        self._bucket = bucket

    @staticmethod
    def canonical_option_key(underlying: str, trade_date: date) -> str:
        return (
            f"canonical/option_chains/underlying={underlying}/"
            f"year={trade_date.year}/month={trade_date.month:02d}/"
            f"{underlying}_{trade_date.strftime('%Y%m%d')}.parquet"
        )

    @staticmethod
    def canonical_futures_key(underlying: str, trade_date: date) -> str:
        return (
            f"canonical/futures/underlying={underlying}/"
            f"year={trade_date.year}/month={trade_date.month:02d}/"
            f"{underlying}_FUT_{trade_date.strftime('%Y%m%d')}.parquet"
        )

    def _s3_key_exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._s3.exceptions.ClientError:
            return False

    def _read_raw_csv(self, trade_date: date) -> Optional[pd.DataFrame]:
        """Read a raw gzipped Bhavcopy CSV from S3 into a DataFrame."""
        s3_key = (
            f"raw/bhavcopy/fo/year={trade_date.year}/"
            f"month={trade_date.month:02d}/"
            f"fo_bhavcopy_{trade_date.strftime('%Y%m%d')}.csv.gz"
        )
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=s3_key)
            gz_bytes = response["Body"].read()
            csv_bytes = gzip.decompress(gz_bytes)
            df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            # Strip whitespace from string columns
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
            return df
        except Exception as e:
            logger.error(f"Failed to read raw CSV for {trade_date}: {e}")
            return None

    def _extract_options(
        self, df: pd.DataFrame, underlying: str,
    ) -> pd.DataFrame:
        """Extract and normalise option rows for a given underlying."""
        mask = (
            (df["TckrSymb"] == underlying)
            & (df["FinInstrmTp"] == OPTION_TYPE_CODE)
            & (df["OptnTp"].isin(["CE", "PE"]))
        )
        opts = df.loc[mask].copy()
        if opts.empty:
            return pd.DataFrame()

        # Rename columns
        opts = opts.rename(columns=OPTION_COLUMN_MAP)
        opts = opts[list(OPTION_COLUMN_MAP.values())]

        # Type conversions
        opts["trade_date"] = pd.to_datetime(opts["trade_date"]).dt.date
        opts["expiry_date"] = pd.to_datetime(opts["expiry_date"]).dt.date
        opts["strike"] = pd.to_numeric(opts["strike"], errors="coerce")

        for col in ["open", "high", "low", "close", "last",
                     "settlement_price", "underlying_price", "previous_close",
                     "turnover"]:
            opts[col] = pd.to_numeric(opts[col], errors="coerce")

        for col in ["open_interest", "oi_change", "volume", "num_trades"]:
            opts[col] = pd.to_numeric(opts[col], errors="coerce").fillna(0).astype("int64")

        opts["lot_size"] = pd.to_numeric(opts["lot_size"], errors="coerce").fillna(0).astype("int32")

        # Sort by expiry, strike, option_type
        opts = opts.sort_values(
            ["expiry_date", "strike", "option_type"]
        ).reset_index(drop=True)

        return opts

    def _extract_futures(
        self, df: pd.DataFrame, underlying: str,
    ) -> pd.DataFrame:
        """Extract and normalise futures rows for a given underlying."""
        mask = (
            (df["TckrSymb"] == underlying)
            & (df["FinInstrmTp"] == FUTURES_TYPE_CODE)
        )
        futs = df.loc[mask].copy()
        if futs.empty:
            return pd.DataFrame()

        futs = futs.rename(columns=FUTURES_COLUMN_MAP)
        futs = futs[list(FUTURES_COLUMN_MAP.values())]

        futs["trade_date"] = pd.to_datetime(futs["trade_date"]).dt.date
        futs["expiry_date"] = pd.to_datetime(futs["expiry_date"]).dt.date

        for col in ["open", "high", "low", "close", "last",
                     "settlement_price", "underlying_price", "previous_close",
                     "turnover"]:
            futs[col] = pd.to_numeric(futs[col], errors="coerce")

        for col in ["open_interest", "oi_change", "volume", "num_trades"]:
            futs[col] = pd.to_numeric(futs[col], errors="coerce").fillna(0).astype("int64")

        futs["lot_size"] = pd.to_numeric(futs["lot_size"], errors="coerce").fillna(0).astype("int32")

        futs = futs.sort_values("expiry_date").reset_index(drop=True)
        return futs

    def _write_parquet(
        self, df: pd.DataFrame, s3_key: str, schema: pa.Schema,
    ) -> None:
        """Write a DataFrame to S3 as Parquet."""
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )

    def parse_date(
        self, trade_date: date, skip_existing: bool = True,
    ) -> ParseResult:
        """Parse a single day's Bhavcopy into canonical Parquet files."""

        # Check if already parsed (check NIFTY options as proxy)
        nifty_key = self.canonical_option_key("NIFTY", trade_date)
        if skip_existing and self._s3_key_exists(nifty_key):
            return ParseResult(
                trade_date=trade_date, success=True,
                error_message="skipped_existing",
            )

        # Read raw CSV
        df = self._read_raw_csv(trade_date)
        if df is None:
            return ParseResult(
                trade_date=trade_date, success=False,
                error_message="raw_csv_read_failed",
            )

        output_keys = []
        nifty_opts_count = 0
        bnifty_opts_count = 0
        nifty_futs_count = 0
        bnifty_futs_count = 0
        nifty_expiries = 0
        bnifty_expiries = 0
        nifty_strikes_near = 0

        for underlying in UNDERLYINGS:
            # Options
            opts = self._extract_options(df, underlying)
            if not opts.empty:
                key = self.canonical_option_key(underlying, trade_date)
                self._write_parquet(opts, key, OPTION_SCHEMA)
                output_keys.append(key)

                if underlying == "NIFTY":
                    nifty_opts_count = len(opts)
                    nifty_expiries = opts["expiry_date"].nunique()
                    nearest_exp = opts["expiry_date"].min()
                    nifty_strikes_near = opts.loc[
                        opts["expiry_date"] == nearest_exp, "strike"
                    ].nunique()
                else:
                    bnifty_opts_count = len(opts)
                    bnifty_expiries = opts["expiry_date"].nunique()

            # Futures
            futs = self._extract_futures(df, underlying)
            if not futs.empty:
                key = self.canonical_futures_key(underlying, trade_date)
                self._write_parquet(futs, key, FUTURES_SCHEMA)
                output_keys.append(key)

                if underlying == "NIFTY":
                    nifty_futs_count = len(futs)
                else:
                    bnifty_futs_count = len(futs)

        logger.info(
            f"{trade_date}: ✓ NIFTY {nifty_opts_count} opts "
            f"({nifty_expiries} exp, {nifty_strikes_near} strikes near) + "
            f"{nifty_futs_count} futs | "
            f"BNIFTY {bnifty_opts_count} opts ({bnifty_expiries} exp) + "
            f"{bnifty_futs_count} futs"
        )

        return ParseResult(
            trade_date=trade_date,
            success=True,
            nifty_options=nifty_opts_count,
            banknifty_options=bnifty_opts_count,
            nifty_futures=nifty_futs_count,
            banknifty_futures=bnifty_futs_count,
            nifty_expiries=nifty_expiries,
            banknifty_expiries=bnifty_expiries,
            nifty_strikes_near=nifty_strikes_near,
            output_keys=output_keys,
        )

    def parse_range(
        self,
        start_date: date,
        end_date: date,
        skip_existing: bool = True,
    ) -> ParseSummary:
        """Parse all Bhavcopy files in a date range."""
        summary = ParseSummary()
        current = start_date

        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # Check if raw file exists
            raw_key = (
                f"raw/bhavcopy/fo/year={current.year}/"
                f"month={current.month:02d}/"
                f"fo_bhavcopy_{current.strftime('%Y%m%d')}.csv.gz"
            )
            if not self._s3_key_exists(raw_key):
                current += timedelta(days=1)
                continue

            summary.total_files += 1
            result = self.parse_date(current, skip_existing=skip_existing)

            if result.success:
                if result.error_message == "skipped_existing":
                    summary.skipped_existing += 1
                else:
                    summary.parsed += 1
                    summary.total_nifty_options += result.nifty_options
                    summary.total_banknifty_options += result.banknifty_options
            else:
                summary.failed += 1
                summary.failed_dates.append(current)
                logger.error(f"{current}: FAILED — {result.error_message}")

            current += timedelta(days=1)

        logger.info(
            f"\n{'='*60}\n"
            f"PARSE SUMMARY: {start_date} to {end_date}\n"
            f"{'='*60}\n"
            f"  Raw files found:   {summary.total_files}\n"
            f"  Parsed:            {summary.parsed}\n"
            f"  Skipped (existing):{summary.skipped_existing}\n"
            f"  Failed:            {summary.failed}\n"
            f"  NIFTY options:     {summary.total_nifty_options:,}\n"
            f"  BANKNIFTY options: {summary.total_banknifty_options:,}\n"
            f"  Failed dates:      {summary.failed_dates}\n"
        )
        return summary
