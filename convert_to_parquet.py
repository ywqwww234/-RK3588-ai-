from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
EVENT_DIR = ROOT / "event_replays"
OUT_DIR = ROOT / "dataset"


def ensure_dirs():
    (OUT_DIR / "raw_events_jsonl").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "features_parquet").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exports_csv").mkdir(parents=True, exist_ok=True)


def convert_event_replays_to_jsonl_and_parquet():
    rows = []
    jsonl_path = OUT_DIR / "raw_events_jsonl" / "event_replays.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for p in sorted(EVENT_DIR.glob("replay_*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                event_id = payload.get("event_id", p.stem)
                for s in payload.get("samples", []):
                    rec = {
                        "event_id": event_id,
                        "ts": s.get("ts"),
                        "risk": s.get("risk"),
                        **{f"visual_{k}": v for k, v in (s.get("visual") or {}).items()},
                        **{f"physio_{k}": v for k, v in (s.get("physio") or {}).items()},
                    }
                    rows.append(rec)
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                continue

    if rows:
        df = pd.DataFrame(rows)
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        df.to_parquet(OUT_DIR / "features_parquet" / "event_replays.parquet", index=False)
        df.to_csv(OUT_DIR / "exports_csv" / "event_replays.csv", index=False, encoding="utf-8-sig")


def convert_aligned_features_to_parquet():
    src = ROOT / "nn" / "aligned_features.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    df.to_parquet(OUT_DIR / "features_parquet" / "aligned_features.parquet", index=False)
    df.to_csv(OUT_DIR / "exports_csv" / "aligned_features.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ensure_dirs()
    convert_event_replays_to_jsonl_and_parquet()
    convert_aligned_features_to_parquet()
    print("done: dataset/raw_events_jsonl + dataset/features_parquet + dataset/exports_csv")
