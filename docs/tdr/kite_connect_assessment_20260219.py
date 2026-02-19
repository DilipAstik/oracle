"""
Kite Connect API Assessment for Oracle Phase B
================================================
Read-only exploration — no writes, no state changes.
Assesses: instruments, historical data, rate limits, coverage.
"""

import json
import time
from datetime import datetime, date, timedelta
from kiteconnect import KiteConnect

# --- Load credentials ---
with open("/tmp/kite_token.json") as f:
    creds = json.load(f)

kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])

print("=" * 70)
print("KITE CONNECT API ASSESSMENT — Oracle Phase B")
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Token expires: {creds['expires_at']}")
print("=" * 70)

# =====================================================================
# 1. INSTRUMENT METADATA — What instruments are available?
# =====================================================================
print("\n--- 1. INSTRUMENT METADATA ---\n")

instruments = kite.instruments("NFO")
print(f"Total NFO instruments: {len(instruments)}")

# Filter NIFTY and BANKNIFTY options
nifty_opts = [i for i in instruments if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
bnifty_opts = [i for i in instruments if i["name"] == "BANKNIFTY" and i["instrument_type"] in ("CE", "PE")]

print(f"NIFTY options (CE+PE): {len(nifty_opts)}")
print(f"BANKNIFTY options (CE+PE): {len(bnifty_opts)}")

# Unique expiries
nifty_expiries = sorted(set(i["expiry"] for i in nifty_opts))
bnifty_expiries = sorted(set(i["expiry"] for i in bnifty_opts))
print(f"\nNIFTY expiries available: {len(nifty_expiries)}")
for exp in nifty_expiries[:5]:
    print(f"  {exp}")
if len(nifty_expiries) > 5:
    print(f"  ... ({len(nifty_expiries) - 5} more)")
    for exp in nifty_expiries[-3:]:
        print(f"  {exp}")

print(f"\nBANKNIFTY expiries available: {len(bnifty_expiries)}")
for exp in bnifty_expiries[:5]:
    print(f"  {exp}")
if len(bnifty_expiries) > 5:
    print(f"  ... ({len(bnifty_expiries) - 5} more)")
    for exp in bnifty_expiries[-3:]:
        print(f"  {exp}")

# Strikes for nearest expiry
nearest_nifty_exp = nifty_expiries[0] if nifty_expiries else None
if nearest_nifty_exp:
    nearest_opts = [i for i in nifty_opts if i["expiry"] == nearest_nifty_exp]
    strikes = sorted(set(i["strike"] for i in nearest_opts))
    print(f"\nNIFTY strikes for nearest expiry ({nearest_nifty_exp}): {len(strikes)}")
    print(f"  Range: {strikes[0]} to {strikes[-1]}")
    print(f"  Step: {strikes[1] - strikes[0]} (near ATM)")

# Sample instrument record
print(f"\nSample NIFTY option instrument record:")
sample = nifty_opts[0] if nifty_opts else None
if sample:
    for k, v in sample.items():
        print(f"  {k}: {v}")

# =====================================================================
# 2. NIFTY/BANKNIFTY FUTURES — needed for realized vol
# =====================================================================
print("\n--- 2. FUTURES INSTRUMENTS ---\n")

nifty_futs = [i for i in instruments if i["name"] == "NIFTY" and i["instrument_type"] == "FUT"]
bnifty_futs = [i for i in instruments if i["name"] == "BANKNIFTY" and i["instrument_type"] == "FUT"]
print(f"NIFTY futures: {len(nifty_futs)}")
for f in nifty_futs:
    print(f"  {f['tradingsymbol']} — expiry: {f['expiry']}, token: {f['instrument_token']}")
print(f"BANKNIFTY futures: {len(bnifty_futs)}")
for f in bnifty_futs:
    print(f"  {f['tradingsymbol']} — expiry: {f['expiry']}, token: {f['instrument_token']}")

# =====================================================================
# 3. INDIA VIX — check if available
# =====================================================================
print("\n--- 3. INDIA VIX ---\n")

nse_instruments = kite.instruments("NSE")
vix = [i for i in nse_instruments if "VIX" in i.get("tradingsymbol", "").upper() or "VIX" in i.get("name", "").upper()]
print(f"VIX-related instruments on NSE: {len(vix)}")
for v in vix:
    print(f"  {v['tradingsymbol']} — token: {v['instrument_token']}, type: {v['instrument_type']}")

# Also check indices
indices = [i for i in nse_instruments if i["instrument_type"] == "EQ" and "VIX" in i.get("tradingsymbol", "").upper()]
print(f"VIX equity instruments: {len(indices)}")

# =====================================================================
# 4. HISTORICAL DATA — Test what's available
# =====================================================================
print("\n--- 4. HISTORICAL DATA CAPABILITIES ---\n")

# 4a. NIFTY 50 index — try to get historical
nifty_idx = [i for i in nse_instruments if i["tradingsymbol"] == "NIFTY 50"]
print(f"NIFTY 50 index instruments: {len(nifty_idx)}")
if nifty_idx:
    print(f"  Token: {nifty_idx[0]['instrument_token']}")

# 4b. Test historical candles for a NIFTY future
if nifty_futs:
    token = nifty_futs[0]["instrument_token"]
    symbol = nifty_futs[0]["tradingsymbol"]
    print(f"\nHistorical test — {symbol} (token {token}):")
    
    # Try different intervals
    for interval in ["day", "60minute", "15minute", "5minute", "minute"]:
        try:
            time.sleep(0.4)  # rate limit safety
            from_date = date.today() - timedelta(days=5)
            to_date = date.today()
            data = kite.historical_data(token, from_date, to_date, interval)
            print(f"  {interval:>10}: {len(data)} candles | fields: {list(data[0].keys()) if data else 'empty'}")
            if data:
                print(f"              sample: {data[-1]}")
        except Exception as e:
            print(f"  {interval:>10}: ERROR — {e}")

# 4c. Test how far back historical data goes for futures
if nifty_futs:
    token = nifty_futs[0]["instrument_token"]
    symbol = nifty_futs[0]["tradingsymbol"]
    print(f"\nHistorical depth test — {symbol}:")
    for years_back in [1, 2, 3, 5]:
        try:
            time.sleep(0.4)
            from_date = date.today() - timedelta(days=365 * years_back)
            to_date = from_date + timedelta(days=30)
            data = kite.historical_data(token, from_date, to_date, "day")
            print(f"  {years_back}Y back: {len(data)} candles (from {from_date})")
        except Exception as e:
            print(f"  {years_back}Y back: ERROR — {e}")

# 4d. Test historical for an OPTION
print("\nHistorical test — NIFTY OPTION:")
if nifty_opts:
    # Pick an ATM-ish option from nearest expiry
    test_opt = nifty_opts[len(nifty_opts)//2]  # mid-list, near ATM
    token = test_opt["instrument_token"]
    symbol = test_opt["tradingsymbol"]
    print(f"  Instrument: {symbol}, token: {token}")
    print(f"  Expiry: {test_opt['expiry']}, strike: {test_opt['strike']}, type: {test_opt['instrument_type']}")
    
    for interval in ["day", "60minute", "15minute", "5minute"]:
        try:
            time.sleep(0.4)
            from_date = date.today() - timedelta(days=30)
            to_date = date.today()
            data = kite.historical_data(token, from_date, to_date, interval)
            print(f"  {interval:>10}: {len(data)} candles")
            if data and len(data) > 0:
                print(f"              first: {data[0]}")
                print(f"              last:  {data[-1]}")
        except Exception as e:
            print(f"  {interval:>10}: ERROR — {e}")

# 4e. CRITICAL: Can we get historical for EXPIRED options?
print("\nEXPIRED OPTIONS — Can we get historical data?")
# Try to get instrument tokens for expired options (they won't be in current list)
# This is the key question: does Kite provide historical for instruments
# that are no longer active?
print("  Current instruments() only returns ACTIVE instruments.")
print("  Expired option tokens are NOT available via instruments() API.")
print("  Kite historical API requires instrument_token — which is only")
print("  available for currently listed instruments.")

# =====================================================================
# 5. RATE LIMITS & API INFO
# =====================================================================
print("\n--- 5. API INFORMATION ---\n")

profile = kite.profile()
print(f"User: {profile.get('user_name', 'N/A')}")
print(f"Broker: {profile.get('broker', 'N/A')}")
print(f"Exchanges: {profile.get('exchanges', 'N/A')}")
print(f"Products: {profile.get('products', 'N/A')}")

# Rate limit info (from documentation, not API)
print(f"\nKite Connect API rate limits (from docs):")
print(f"  Historical data: 3 requests/second")
print(f"  Other APIs: 10 requests/second")
print(f"  instruments(): No limit but heavy payload (~50MB for NFO)")

# =====================================================================
# 6. SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("ASSESSMENT SUMMARY")
print("=" * 70)
print(f"""
Instruments available:
  NIFTY options:    {len(nifty_opts)} (CE+PE)
  BANKNIFTY options:{len(bnifty_opts)} (CE+PE)  
  NIFTY futures:    {len(nifty_futs)}
  BANKNIFTY futures:{len(bnifty_futs)}
  NIFTY expiries:   {len(nifty_expiries)}
  BANKNIFTY expiries:{len(bnifty_expiries)}
  India VIX:        {len(vix)} instruments found

Key limitation:
  instruments() only returns CURRENTLY LISTED instruments.
  Historical data requires instrument_token.
  Expired options are NOT accessible via this API.
  
Implication for Oracle:
  Kite Connect can provide FORWARD data (going-concern snapshots)
  but CANNOT backfill historical option chains for expired contracts.
  Alternative sources needed for historical backfill.
""")
