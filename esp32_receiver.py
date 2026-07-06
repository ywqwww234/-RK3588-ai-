from flask import Flask, request, jsonify
from datetime import datetime
import threading
from collections import deque

app = Flask(__name__)

_state_lock = threading.Lock()
_hr_wave = deque(maxlen=800)
_hr_display = {
    "ppg": [],
    "ecg": [],
    "hr": [],
    "hrv": [],
    "sdnn_trend": [],
}
_eeg_wave = deque(maxlen=15000)
_latest = {
    "device_id": None,
    "bpm": None,
    "rmssd": None,
    "sdnn": None,
    "lfhf": None,
    "ir_value": None,
    "hand_present": None,
    "updated_at": None,
    "attention": None,
    "meditation": None,
    "poor_signal": None,
    "eeg_power": None,
    "raw_value": None,
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "esp32_receiver"})


@app.route("/esp32/bpm", methods=["POST"])
def ingest_bpm():
    data = request.get_json(silent=True) or {}

    bpm = data.get("bpm")
    if bpm is None:
        return jsonify({"ok": False, "msg": "missing field: bpm"}), 400

    try:
        bpm = float(bpm)
    except Exception:
        return jsonify({"ok": False, "msg": "invalid bpm"}), 400

    if bpm <= 0:
        return jsonify({"ok": False, "msg": "bpm must be > 0"}), 400

    device_id = data.get("device_id", "esp32_01")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _state_lock:
        _latest["device_id"] = device_id
        _latest["bpm"] = bpm
        _latest["updated_at"] = now

    return jsonify({"ok": True, "device_id": device_id, "bpm": bpm, "updated_at": now})


@app.route("/esp32/latest", methods=["GET"])
def latest_bpm():
    with _state_lock:
        payload = dict(_latest)

    return jsonify(payload)


@app.route("/a2b/telemetry", methods=["POST"])
def ingest_telemetry():
    """
    A电脑 -> B电脑 统一推送接口（Wi-Fi）
    兼容字段：
      - 生理: bpm
      - 脑电: attention, meditation, poor_signal, eeg_power, raw_value
      - 标识: device_id
    """
    data = request.get_json(silent=True) or {}

    device_id = data.get("device_id", "a_host_01")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bpm = data.get("bpm")
    rmssd = data.get("rmssd")
    sdnn = data.get("sdnn")
    lfhf = data.get("lfhf")
    ir_value = data.get("ir_value")
    hand_present = data.get("hand_present")
    attention = data.get("attention")
    meditation = data.get("meditation")
    poor_signal = data.get("poor_signal")
    eeg_power = data.get("eeg_power")
    raw_value = data.get("raw_value")
    ir_values = data.get("ir_values")
    hr_display = data.get("hr_display")
    raw_values = data.get("raw_values")

    def _to_float_or_none(v):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _to_int_or_none(v):
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    bpm = _to_float_or_none(bpm)
    rmssd = _to_float_or_none(rmssd)
    sdnn = _to_float_or_none(sdnn)
    lfhf = _to_float_or_none(lfhf)
    ir_value = _to_int_or_none(ir_value)
    hand_present = _to_int_or_none(hand_present)
    attention = _to_int_or_none(attention)
    meditation = _to_int_or_none(meditation)
    poor_signal = _to_int_or_none(poor_signal)
    raw_value = _to_int_or_none(raw_value)

    # 至少包含一个核心字段才认为有效
    if bpm is None and attention is None and meditation is None:
        return jsonify({"ok": False, "msg": "missing core fields: bpm/attention/meditation"}), 400

    with _state_lock:
        _latest["device_id"] = device_id
        _latest["bpm"] = bpm if bpm is not None and bpm > 0 else _latest.get("bpm")
        _latest["rmssd"] = rmssd if rmssd is not None and rmssd >= 0 else _latest.get("rmssd")
        _latest["sdnn"] = sdnn if sdnn is not None and sdnn >= 0 else _latest.get("sdnn")
        _latest["lfhf"] = lfhf if lfhf is not None and lfhf >= 0 else _latest.get("lfhf")
        _latest["ir_value"] = ir_value if ir_value is not None else _latest.get("ir_value")
        _latest["hand_present"] = hand_present if hand_present is not None else _latest.get("hand_present")
        _latest["attention"] = attention if attention is not None else _latest.get("attention")
        _latest["meditation"] = meditation if meditation is not None else _latest.get("meditation")
        _latest["poor_signal"] = poor_signal if poor_signal is not None else _latest.get("poor_signal")
        _latest["eeg_power"] = eeg_power if eeg_power is not None else _latest.get("eeg_power")
        _latest["raw_value"] = raw_value if raw_value is not None else _latest.get("raw_value")
        _latest["updated_at"] = now
        if isinstance(ir_values, list):
            for v in ir_values[-400:]:
                iv = _to_int_or_none(v)
                if iv is not None:
                    _hr_wave.append(iv)
        elif ir_value is not None:
            _hr_wave.append(ir_value)
        if isinstance(hr_display, dict):
            for key in ("ppg", "ecg", "hr", "hrv", "sdnn_trend"):
                vals = hr_display.get(key)
                if isinstance(vals, list):
                    _hr_display[key] = vals[-1000:]
        if isinstance(raw_values, list):
            for v in raw_values[-1000:]:
                rv = _to_int_or_none(v)
                if rv is not None:
                    _eeg_wave.append(rv)
        elif raw_value is not None:
            _eeg_wave.append(raw_value)

    return jsonify({"ok": True, "updated_at": now, "device_id": device_id})


@app.route("/eeg/latest", methods=["GET"])
def latest_eeg():
    with _state_lock:
        payload = {
            "device_id": _latest.get("device_id"),
            "attention": _latest.get("attention"),
            "meditation": _latest.get("meditation"),
            "poor_signal": _latest.get("poor_signal"),
            "eeg_power": _latest.get("eeg_power"),
            "raw_value": _latest.get("raw_value"),
            "updated_at": _latest.get("updated_at"),
        }

    return jsonify(payload)


@app.route("/esp32/wave", methods=["GET"])
def latest_hr_wave():
    with _state_lock:
        return jsonify({
            "ok": True,
            "ir_values": list(_hr_wave),
            "display": dict(_hr_display),
            "count": len(_hr_wave),
            "updated_at": _latest.get("updated_at"),
        })


@app.route("/eeg/wave", methods=["GET"])
def latest_eeg_wave():
    with _state_lock:
        return jsonify({
            "ok": True,
            "raw_values": list(_eeg_wave),
            "count": len(_eeg_wave),
            "updated_at": _latest.get("updated_at"),
        })


if __name__ == "__main__":
    # 监听全部网卡，供局域网其他电脑访问
    app.run(host="0.0.0.0", port=5001, debug=False)
