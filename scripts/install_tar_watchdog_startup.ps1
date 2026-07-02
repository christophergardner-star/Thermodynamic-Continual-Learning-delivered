$ErrorActionPreference = "Stop"
throw "RETIRED. This script auto-started EXECUTION services on boot (unsafe). Use scripts/install_tar_platform_task.ps1 instead: it supervises only the watchdog (resurrecting it after a crash/reboot) while execution stays human-gated by the fail-closed autonomy ramp + RAIL-3 manifest. See scripts/tar_platform_supervisor.py."
