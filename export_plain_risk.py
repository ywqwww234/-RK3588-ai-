from recorder import Recorder

OUT_PATH = r"D:\Anti_depression\data\risk_log_plain.csv"


def main():
    rec = Recorder(data_dir=r"D:\Anti_depression\data")
    df = rec.get_decrypted_history()
    if df.empty:
        print("没有可导出的记录，请先开始采集。")
        return
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已导出明文CSV: {OUT_PATH}")
    print(f"共 {len(df)} 条")


if __name__ == "__main__":
    main()
