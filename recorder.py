"""
风险记录与历史数据读取模块。

负责加密保存实时风险值，并为家长端/图表回放提供统一的数据读取入口。
"""

import csv
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet


class Recorder:
    """负责风险持久化、历史导入与统一读取。"""
    def __init__(self, filename='risk_log_encrypted.csv', key_file='secret.key', data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else (Path(__file__).resolve().parent / 'data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.filename = str(self.data_dir / filename)
        self.key_file = str(self.data_dir / key_file)
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'encrypted_risk'])

    def _load_or_generate_key(self):
        if not os.path.exists(self.key_file):
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key
        with open(self.key_file, 'rb') as f:
            return f.read()

    def add_record(self, risk):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        risk_str = str(risk).encode('utf-8')
        encrypted_risk = self.cipher.encrypt(risk_str).decode('utf-8')

        # Windows 下如果文件被 Excel/WPS 占用会抛 PermissionError。
        # 这里做短重试，避免采集线程直接崩溃。
        import time
        last_err = None
        for _ in range(6):
            try:
                with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, encrypted_risk])
                return
            except PermissionError as exc:
                last_err = exc
                time.sleep(0.2)

        # 若持续占用，写入备用文件，保证数据不丢。
        fallback = str(self.data_dir / 'risk_log_encrypted_fallback.csv')
        if not os.path.exists(fallback):
            with open(fallback, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'encrypted_risk'])
        with open(fallback, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, encrypted_risk])

    def start_new_session(self):
        pass

    def _excel_sort_key(self, path_obj):
        name = path_obj.stem
        nums = re.findall(r'\d+', name)
        if nums:
            return (0, int(nums[-1]), name)
        return (1, path_obj.stat().st_mtime, name)

    def _is_valid_excel_file(self, path_obj):
        return path_obj.is_file() and not path_obj.name.startswith('~$')

    def import_excel_folder(self, source_dir, clear_existing=True):
        source_dir = Path(source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            return 0

        self.data_dir.mkdir(parents=True, exist_ok=True)

        if clear_existing and source_dir.resolve() != self.data_dir.resolve():
            for old_file in list(self.data_dir.glob('*.xlsx')) + list(self.data_dir.glob('*.xls')):
                try:
                    old_file.unlink()
                except Exception:
                    pass

        copied = 0
        excel_files = [p for p in (list(source_dir.glob('*.xlsx')) + list(source_dir.glob('*.xls'))) if self._is_valid_excel_file(p)]
        excel_files = sorted(excel_files, key=self._excel_sort_key)
        for src in excel_files:
            try:
                dst = self.data_dir / src.name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                copied += 1
            except Exception:
                pass
        return copied

    def _load_from_excel_files(self):
        if not self.data_dir.exists():
            return pd.DataFrame(columns=['timestamp', 'risk'])

        excel_files = [p for p in (list(self.data_dir.glob('*.xlsx')) + list(self.data_dir.glob('*.xls'))) if self._is_valid_excel_file(p)]
        excel_files = sorted(excel_files, key=self._excel_sort_key)
        if not excel_files:
            return pd.DataFrame(columns=['timestamp', 'risk'])

        target_dates = [
            pd.Timestamp('2026-05-04').date(),
            pd.Timestamp('2026-05-05').date(),
            pd.Timestamp('2026-05-06').date(),
            pd.Timestamp('2026-05-07').date(),
            pd.Timestamp('2026-05-08').date(),
            pd.Timestamp('2026-05-09').date(),
            pd.Timestamp('2026-05-10').date(),
        ]

        all_parts = []
        for idx, file_path in enumerate(excel_files[:7]):
            try:
                part = pd.read_excel(file_path)
                if part is None or part.empty:
                    continue
                part = part.head(144).copy()
                day_start = pd.Timestamp(target_dates[idx])
                part['timestamp'] = [day_start + pd.Timedelta(minutes=10 * i) for i in range(len(part))]
                part['_source_file'] = file_path.name
                all_parts.append(part)
            except Exception:
                continue

        if not all_parts:
            return pd.DataFrame(columns=['timestamp', 'risk'])

        df = pd.concat(all_parts, ignore_index=True)

        rename_map = {
            '时间': 'timestamp', '日期时间': 'timestamp', 'datetime': 'timestamp', 'time': 'timestamp',
            '风险': 'risk', '风险值': 'risk', '风险指数': 'risk', 'risk_index': 'risk',
            '视觉风险': 'visual_risk', 'visual': 'visual_risk',
            '生理风险': 'hrv_risk', 'hrv': 'hrv_risk',
            '脑电风险': 'eeg_risk', 'eeg': 'eeg_risk'
        }
        df = df.rename(columns={c: rename_map.get(str(c).strip(), str(c).strip()) for c in df.columns})
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='last')]

        if 'risk' not in df.columns:
            fuzzy_risk_cols = [c for c in df.columns if ('风险' in str(c)) or ('risk' in str(c).lower())]
            fuzzy_risk_cols = [c for c in fuzzy_risk_cols if c not in ['visual_risk', 'hrv_risk', 'eeg_risk']]
            if fuzzy_risk_cols:
                df['risk'] = pd.to_numeric(df[fuzzy_risk_cols[0]], errors='coerce')

        if 'risk' not in df.columns:
            for col in ['visual_risk', 'hrv_risk', 'eeg_risk']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            modal_cols = [c for c in ['visual_risk', 'hrv_risk', 'eeg_risk'] if c in df.columns]
            if modal_cols:
                df['risk'] = df[modal_cols].mean(axis=1)
            else:
                exclude_cols = {'timestamp', 'date', '_source_file'}
                numeric_candidates = []
                for col in df.columns:
                    if col in exclude_cols:
                        continue
                    s = pd.to_numeric(df[col], errors='coerce')
                    if s.notna().sum() > 0:
                        numeric_candidates.append(s)
                if not numeric_candidates:
                    return pd.DataFrame(columns=['timestamp', 'risk'])
                if len(numeric_candidates) >= 3:
                    df['risk'] = pd.concat(numeric_candidates[:3], axis=1).mean(axis=1)
                else:
                    raw = numeric_candidates[0]
                    vmin, vmax = raw.min(), raw.max()
                    df['risk'] = 0.5 if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin else (raw - vmin) / (vmax - vmin)

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['risk'] = pd.to_numeric(df['risk'], errors='coerce')
        for col in ['visual_risk', 'hrv_risk', 'eeg_risk']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['timestamp', 'risk']).sort_values('timestamp').reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'risk'])

        for col in ['risk', 'visual_risk', 'hrv_risk', 'eeg_risk']:
            if col in df.columns:
                df[col] = df[col].clip(0.0, 1.0)

        keep_cols = ['timestamp', 'risk'] + [c for c in ['visual_risk', 'hrv_risk', 'eeg_risk'] if c in df.columns]
        return df[keep_cols]

    def _load_from_encrypted_csv(self):
        if not os.path.exists(self.filename):
            return pd.DataFrame(columns=['timestamp', 'risk'])
        df = pd.read_csv(self.filename)
        if df.empty or 'encrypted_risk' not in df.columns:
            return pd.DataFrame(columns=['timestamp', 'risk'])

        df = df.tail(1000).copy()
        decrypted_risks = []
        for enc_val in df['encrypted_risk']:
            try:
                dec_str = self.cipher.decrypt(enc_val.encode('utf-8')).decode('utf-8')
                decrypted_risks.append(float(dec_str))
            except Exception:
                decrypted_risks.append(0.0)

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['risk'] = pd.to_numeric(decrypted_risks, errors='coerce')
        df = df.dropna(subset=['timestamp', 'risk']).copy()
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'risk'])
        df['risk'] = df['risk'].clip(0.0, 1.0)
        return df[['timestamp', 'risk']]

    def get_decrypted_history(self):
        excel_df = self._load_from_excel_files()
        csv_df = self._load_from_encrypted_csv()

        if excel_df.empty and csv_df.empty:
            return pd.DataFrame(columns=['timestamp', 'risk'])
        if excel_df.empty:
            return csv_df.sort_values('timestamp').reset_index(drop=True)
        if csv_df.empty:
            return excel_df.sort_values('timestamp').reset_index(drop=True)

        merged = pd.concat([excel_df, csv_df], ignore_index=True, sort=False)
        merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
        merged['risk'] = pd.to_numeric(merged['risk'], errors='coerce')
        merged = merged.dropna(subset=['timestamp', 'risk']).sort_values('timestamp').reset_index(drop=True)
        return merged
