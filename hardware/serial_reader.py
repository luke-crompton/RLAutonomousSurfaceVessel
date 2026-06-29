from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import time
import serial


@dataclass
class SensorPacket:
    rx_time: float
    ranges_m: Dict[str, float]


def parse_csv_line(line: str) -> Optional[Dict[str, float]]:
    # expected format:
    # US1,US2,US3,US4,US5
    # example:
    # 123.4,98.2,-1.0,201.5,87.9

    line = line.strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None

    keys = ["US1", "US2", "US3", "US4", "US5"]

    out: Dict[str, float] = {}
    for k, p in zip(keys, parts):
        try:
            out[k] = float(p)
        except ValueError:
            return None

    return out


class SerialSensorReader:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        unit_scale_to_m: float = 0.01,   # cm to m
        min_m: float = 0.02,
        max_m: float = 3.5,
    ):
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.25)
        self.unit_scale_to_m = unit_scale_to_m
        self.allowed_keys = {"US1", "US2", "US3", "US4", "US5"}
        self.min_m = min_m
        self.max_m = max_m

    def read_packet(self) -> Optional[SensorPacket]:
        try:
            raw = self.ser.readline()
        except serial.SerialException:
            return None

        if not raw:
            return None

        line = raw.decode("utf-8", errors="ignore")
        data_raw = parse_csv_line(line)
        if data_raw is None:
            return None

        ranges_m: Dict[str, Optional[float]] = {k: None for k in self.allowed_keys}

        for key, raw_value in data_raw.items():
            raw_m = raw_value * self.unit_scale_to_m

            if self.min_m <= raw_m <= self.max_m:
                ranges_m[key] = raw_m
            else:
                ranges_m[key] = None

        return SensorPacket(
            rx_time=time.time(),
            ranges_m=ranges_m,
        )