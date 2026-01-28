#!/usr/bin/env python3
import os, subprocess, time, signal

TARGET_PID = 92633
CHECK_INTERVAL = 30  # Sekunden

def pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # existiert, gehört aber jemand anderem
        return True

print(f"Warte, bis Prozess {TARGET_PID} beendet ist...")
while pid_exists(TARGET_PID):
    print(f"{time.strftime('%H:%M:%S')}: Prozess {TARGET_PID} läuft noch – warte ...")
    time.sleep(CHECK_INTERVAL)

print(f"✅ Prozess {TARGET_PID} ist beendet – starte Training!")
subprocess.run(["python3", "seed_run.py"])
