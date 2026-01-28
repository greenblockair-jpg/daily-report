!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orderbookからfoot-traffic(実需)スコアを計算し、
「一定時間 False(= demand_score < THRESH) が一度も出なければロング」
というシグナルを可視化・テストトレードします。

出力:
  - demand_features.csv
  - long_signal_<N>min_rule.csv
  - long_true_rows.csv ← long_ok_windowがTrueの行だけ抽出
  - best_bid_price.png ← シグナル印付き
"""

import os, glob, json, io, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ========= 設定 =========
FILENAME = "solusdt_orderbook.data"
N_ROWS = 5000

LONG_WINDOW_MINUTES = 2.0
THRESH = 0.3

ROLL_WIN, MINP = 300, 50
K_DELTA, REFILL_EPS = 0.5, 1e-9
W_MAG, W_PRO, W_PER, W_REF = 0.35, 0.30, 0.20, 0.15
SIGNAL_Q = 0.80  # 現状未使用だが残しておく

# ========= ヘルパ =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, FILENAME)

def is_gzip_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except FileNotFoundError:
        return False

# ========= データ読込（ファイル選択） =========
def pick_data_file(path: str) -> str:
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        folder = path
    else:
        folder = os.path.dirname(path) if os.path.dirname(path) else SCRIPT_DIR

    exact = os.path.join(folder, FILENAME)
    if os.path.isfile(exact):
        return exact

    patterns = [
        "*2025-10-10*SOLUSDT*ob200*.data",
        "*2025-10-10*SOLUSDT*ob200*.csv.gz",
        "*2025-10-10*SOLUSDT*ob200*.csv",
        "*SOLUSDT*ob200*.data",
        "*SOLUSDT*ob200*.csv.gz",
        "*SOLUSDT*ob200*.csv",
    ]
    candidates = []
    for pat in patterns:
        candidates += glob.glob(os.path.join(folder, pat))

    if candidates:
        return sorted(candidates)[0]

    listing = "\n".join(" - " + name for name in os.listdir(folder))
    raise FileNotFoundError(
        "ファイルが見つかりません。\n"
        f"探索フォルダ: {os.path.abspath(folder)}\n"
        f"探したファイル名（最優先）: {FILENAME}\n"
        "フォルダ内一覧:\n" + listing
    )

def iter_jsonl_lines(file_path: str):
    try:
        if is_gzip_file(file_path):
            with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    except PermissionError:
        raise PermissionError(
            "PermissionError: ファイルを開けませんでした。\n"
            "OneDriveの同期でロックされている可能性があります。\n"
            "→ ファイルを C:\\temp にコピーして、FILENAME をそのパスに変更してください。"
        )

# ========= ローダー（両フォーマット対応） =========
def _coerce_ts_to_utc(ts_raw):
    """msエポック or ISO文字列などを UTC の pandas.Timestamp に変換"""
    if ts_raw is None:
        return None
    # msエポック数値
    if isinstance(ts_raw, (int, float)):
        try:
            return pd.to_datetime(int(ts_raw), unit="ms", utc=True)
        except Exception:
            pass
    # 文字列（ISO等）
    if isinstance(ts_raw, str):
        try:
            return pd.to_datetime(ts_raw, utc=True)
        except Exception:
            pass
    return None

def _extract_bids_and_ts(obj):
    """
    保存フォーマットの違いを吸収して (timestamp_utc, bids_list) を返す。
    対応1) { "timestamp": "...", "bids": [[p,q], ...], "asks": ... }
    対応2) { "ts": 1730..., "data": { "b": [[p,q],...], "a": ... } }
    """
    # 1) こちら側の保存形式
    if "bids" in obj or "asks" in obj:
        bids = obj.get("bids") or []
        ts_raw = obj.get("timestamp") or obj.get("ts") or obj.get("T")
        ts_utc = _coerce_ts_to_utc(ts_raw)
        return ts_utc, bids

    # 2) Bybit生のような形式
    data = obj.get("data") or {}
    bids = data.get("b") or data.get("bids") or []
    ts_raw = obj.get("ts") or obj.get("timestamp") or obj.get("T") or data.get("ts")
    ts_utc = _coerce_ts_to_utc(ts_raw)
    return ts_utc, bids

def load_first_n_snapshots(path: str, n_rows: int) -> pd.DataFrame:
    fp = pick_data_file(path)
    print("読み込むファイル:", fp)

    rows = []
    for line in iter_jsonl_lines(fp):
        try:
            obj = json.loads(line)
        except Exception:
            # 壊れた行・半端行はスキップ
            continue

        ts_utc, bids = _extract_bids_and_ts(obj)
        if ts_utc is None or not bids:
            continue

        # [["price","size",...], ...] を float に変換
        pairs = []
        for item in bids:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    p = float(item[0]); s = float(item[1])
                    if np.isfinite(p) and np.isfinite(s):
                        pairs.append((p, s))
                except Exception:
                    pass
        if not pairs:
            continue

        # 価格降順で上位100
        pairs.sort(key=lambda x: x[0], reverse=True)
        top_price, top_size = pairs[0]
        top100 = pairs[:100]
        max_size, price_max = max((s, p) for p, s in top100)

        rows.append({
            "timestamp_utc": ts_utc,
            "top_bid_price": top_price,
            "top_bid_size": top_size,
            "max_bid_size_in_top100": max_size,
            "price_of_max_bid_size_in_top100": price_max,
        })
        if len(rows) >= n_rows:
            break

    if not rows:
        raise RuntimeError("有効なスナップショットがありません。")

    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)

# ========= デバッグ用プローブ（任意） =========
def quick_probe(path, n=5):
    """先頭数行のキー構造を表示してフォーマットを確認（任意で使用）"""
    try:
        fp = pick_data_file(path)
        print("probe:", fp)
    except Exception as e:
        print("probe失敗:", e)
        return
    c = 0
    for line in iter_jsonl_lines(fp):
        try:
            obj = json.loads(line)
            keys = list(obj.keys())
            sub = list((obj.get("data") or {}).keys()) if isinstance(obj.get("data"), dict) else []
            print(f"keys={keys}  data.keys={sub}")
        except Exception as e:
            print("parse error:", e)
        c += 1
        if c >= n: break

# ========= スコア計算 =========
def zscore(s, win=300, minp=50):
    m = s.rolling(win, min_periods=minp).mean()
    v = s.rolling(win, min_periods=minp).std()
    return (s - m) / v.replace(0, np.nan)

def build_demand_features(df):
    d = df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    d["delta"] = (d["top_bid_price"] - d["price_of_max_bid_size_in_top100"]).clip(lower=0)
    d["mag_z"] = zscore(d["max_bid_size_in_top100"], ROLL_WIN, MINP).clip(-3, 3)
    rng = d["mag_z"].max() - d["mag_z"].min()
    d["mag_score"] = 0 if rng == 0 else (d["mag_z"] - d["mag_z"].min()) / (rng + 1e-12)
    d["pro_score"] = 1.0 / (1.0 + (d["delta"] / K_DELTA))

    same = (d["price_of_max_bid_size_in_top100"].diff().abs() < 1e-12)
    streak, run = [], 0
    for f in same:
        run = run + 1 if f else 0
        streak.append(run)
    d["persist_raw"] = streak
    pr_max = d["persist_raw"].rolling(ROLL_WIN, min_periods=MINP).max()
    d["per_score"] = (d["persist_raw"] / pr_max.replace(0, np.nan)).fillna(0).clip(0, 1)

    same_best = (d["top_bid_price"].diff().abs() < 1e-12)
    inc = (d["top_bid_size"].diff() > REFILL_EPS) & same_best
    d["refill_freq"] = inc.rolling(ROLL_WIN, min_periods=MINP).mean().fillna(0)
    pos_diff = d["top_bid_size"].diff().clip(lower=0)
    d["refill_strength"] = (pos_diff.where(same_best, 0)).rolling(ROLL_WIN, min_periods=MINP).mean().fillna(0)
    for col in ["refill_freq", "refill_strength"]:
        r = d[col].max() - d[col].min()
        d[col + "_n"] = 0 if r == 0 else (d[col] - d[col].min()) / (r + 1e-12)
    d["ref_score"] = 0.5 * d["refill_freq_n"] + 0.5 * d["refill_strength_n"]

    w = np.array([W_MAG, W_PRO, W_PER, W_REF]); w /= w.sum()
    d["demand_score"] = (
        w[0] * d["mag_score"]
        + w[1] * d["pro_score"]
        + w[2] * d["per_score"]
        + w[3] * d["ref_score"]
    )
    return d

# ========= グラフ描画 =========
def plot_best_bid_with_signals(df, fname="best_bid_price.png"):
    d = df.sort_values("timestamp_utc")
    plt.figure(figsize=(10, 4))
    plt.plot(d["timestamp_utc"], d["top_bid_price"], label="Top Bid Price")

    sig = d[d["long_ok_window"]]
    if not sig.empty:
        plt.scatter(
            sig["timestamp_utc"],
            sig["top_bid_price"],
            s=35,
            color="green",          # シグナルに緑
            label="long_ok_window=True",
            zorder=3,
        )

    plt.title("Best Bid Price with long_ok_window=True markers")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Top Bid Price")
    ax = plt.gca()
    loc = mdates.AutoDateLocator()
    fmt = mdates.ConciseDateFormatter(loc)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# ========= メイン =========
def main():
    # 必要なら最初にフォーマット確認
    # quick_probe(INPUT_PATH, n=5)

    df = load_first_n_snapshots(INPUT_PATH, N_ROWS)
    d = build_demand_features(df)

    window_td = pd.Timedelta(minutes=LONG_WINDOW_MINUTES)
    d["is_ok"] = d["demand_score"] >= THRESH
    d["bad"] = ~d["is_ok"]
    d = d.set_index("timestamp_utc")
    d["long_ok_window"] = (d["bad"].rolling(window_td).sum() == 0)
    d["long_entry"] = d["long_ok_window"] & (~d["long_ok_window"].shift(fill_value=False))
    d = d.reset_index()

    d.to_csv(f"long_signal_{LONG_WINDOW_MINUTES}min_rule.csv", index=False, encoding="utf-8")
    print(f"Saved: long_signal_{LONG_WINDOW_MINUTES}min_rule.csv")

    out_cols = [
        "timestamp_utc","top_bid_price","top_bid_size",
        "max_bid_size_in_top100","price_of_max_bid_size_in_top100","delta",
        "mag_score","pro_score","per_score","ref_score","demand_score",
        "is_ok","long_ok_window","long_entry"
    ]
    d[out_cols].to_csv("demand_features.csv", index=False, encoding="utf-8")
    print("Saved: demand_features.csv")

    long_true = d[d["long_ok_window"]].copy()
    long_true.to_csv("long_true_rows.csv", index=False, encoding="utf-8")
    print(f"Saved: long_true_rows.csv ({len(long_true)} rows)")

    plot_best_bid_with_signals(d, "best_bid_price.png")

if __name__ == "__main__":
    main()
