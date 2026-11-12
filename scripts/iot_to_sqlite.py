"""Run the TPCx-IoT data generator and store its records in SQLite.

The TPCx-IoT kit (not in git: its licence forbids redistribution) is a modified YCSB. Its dummy
database `BasicDB` prints every record it would insert; this script runs it once per client
(= one substation = one future edge worker) and parses that output into a table:

  records(client, sensor, location, measurement, ts_ms, reading, value_ts_ms, raw_key, raw_value, raw_bytes)

Observed record format (kit v2.1.2):
  key   = "<client>:<sensor>:<ts_ms>"   e.g. client1:ang_30_Humidity:1794501961787
          ts_ms advances exactly 100 ms per record; each record is ONE randomly chosen sensor
  value = field0, exactly 1000 chars: "<sensor>:<sensor>_value:<reading>:timestamp:<ms>:<hash>:..."
          repeated as filler; <reading> is a uniform random number in [0, 1) with 4 decimals

Example:
  python scripts/iot_to_sqlite.py --clients 10 --records-per-client 5000
"""
import argparse
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^INSERT (\S+) (\S+) \[ field0=(.*) \]\s*$")
READING = re.compile(r"_value:(-?[0-9.]+):timestamp:(\d+)")

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    client       TEXT    NOT NULL,   -- generator client = substation = edge worker
    sensor       TEXT    NOT NULL,   -- e.g. ang_30_Humidity
    location     TEXT    NOT NULL,   -- e.g. ang_30
    measurement  TEXT    NOT NULL,   -- e.g. Humidity
    ts_ms        INTEGER NOT NULL,   -- timestamp from the key
    reading      REAL,               -- the actual measurement
    value_ts_ms  INTEGER,            -- wall-clock timestamp inside the value
    raw_key      TEXT    NOT NULL,
    raw_value    TEXT    NOT NULL,   -- the full 1000-char field, exactly as generated
    raw_bytes    INTEGER NOT NULL,   -- len(key) + len(value): what the benchmark would store
    PRIMARY KEY (client, sensor, ts_ms)
);
CREATE INDEX IF NOT EXISTS idx_client_ts ON records(client, ts_ms);
CREATE INDEX IF NOT EXISTS idx_sensor_ts ON records(sensor, ts_ms);
CREATE TABLE IF NOT EXISTS runs (
    client TEXT, records INTEGER, started_at TEXT, seconds REAL, generator_cmd TEXT
);
"""


def split_sensor(sensor):
    """ang_30_Humidity -> (ang_30, Humidity); aft_1120_level_admittance -> (aft_1120, level_admittance)."""
    m = re.match(r"^([a-z]+_\d+)_(.+)$", sensor)
    return (m.group(1), m.group(2)) if m else (sensor, "")


def generator_cmd(kit, client, n):
    lib = kit / "tpcx-iot" / "couchbase2-binding" / "lib" / "*"  # smallest lib dir that contains core-*.jar
    return ["java", "-cp", str(lib), "com.yahoo.ycsb.Client", "-t",
            "-db", "com.yahoo.ycsb.BasicDB", "-P", str(kit / "tpcx-iot" / "workloads" / "workloadiot"),
            "-p", "basicdb.verbose=true",
            "-p", "insertproportion=1", "-p", "readproportion=0", "-p", "updateproportion=0",
            "-p", f"recordcount={n}", "-p", f"operationcount={n}", "-p", "threadcount=1",
            "-p", f"client={client}"]


def import_client(con, kit, client, n, batch=2000):
    cmd = generator_cmd(kit, client, n)
    started, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=kit)
    rows, count, unparsed = [], 0, 0
    for line in proc.stdout:
        if not line.startswith("INSERT"):
            continue
        m = LINE.match(line.rstrip("\n"))
        if not m:
            unparsed += 1
            continue
        _table, key, value = m.groups()
        key_client, rest = key.split(":", 1)
        sensor, ts = rest.rsplit(":", 1)
        r = READING.search(value)
        location, measurement = split_sensor(sensor)
        rows.append((key_client, sensor, location, measurement, int(ts),
                     float(r.group(1)) if r else None, int(r.group(2)) if r else None,
                     key, value, len(key) + len(value)))
        if len(rows) >= batch:
            con.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            count += len(rows)
            rows = []
    if rows:
        con.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        count += len(rows)
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise SystemExit(f"generator failed for {client} (exit {proc.returncode}):\n{err[-2000:]}")
    seconds = time.time() - t0
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (client, count, started, seconds, " ".join(cmd)))
    con.commit()
    return count, unparsed, seconds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kit", default=str(ROOT / "TPCx-IoT_Tools_v2.1.2"), help="path to the TPCx-IoT kit")
    p.add_argument("--clients", type=int, default=10, help="clients = substations = edge workers")
    p.add_argument("--records-per-client", type=int, default=5000)
    p.add_argument("--db", default=str(ROOT / "data" / "iot" / "tpcx_iot.db"))
    p.add_argument("--fresh", action="store_true", help="delete the database first")
    a = p.parse_args()

    kit = Path(a.kit)
    if not (kit / "tpcx-iot" / "workloads" / "workloadiot").exists():
        raise SystemExit(f"TPCx-IoT kit not found at {kit}")
    db = Path(a.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    if a.fresh and db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    total = 0
    for i in range(1, a.clients + 1):
        n, unparsed, secs = import_client(con, kit, f"client{i}", a.records_per_client)
        total += n
        print(f"client{i}: {n} records in {secs:.1f}s" + (f"  ({unparsed} UNPARSED lines)" if unparsed else ""),
              flush=True)
    con.close()
    print(f"done: {total} records -> {db} ({db.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
