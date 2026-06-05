"""Keystone #2 (K2.4) — one supervised --platform entrypoint.

run_platform() plans/launches the watchdog as a single supervised unit (the watchdog
then starts + supervises the daemon, queue-maintainer, and dashboard). Tests use
dry_run=True so nothing is ever launched; they verify the planned command, the
supervised set, and that the kill-switch flags are surfaced.
"""
import tar_living_research as tlr


def test_platform_dry_run_plans_watchdog(tmp_path):
    status = tlr.run_platform(tmp_path, dry_run=True)
    assert status["entrypoint"] == "platform"
    assert status["launched"] is False
    assert "error" not in status  # the real tar_watchdog.py exists in the repo
    assert status["watchdog_script"].endswith("tar_watchdog.py")
    cmd = status["command"]
    assert any(str(c).endswith("tar_watchdog.py") for c in cmd)
    assert "--poll-interval-s" in cmd
    assert status["supervises"] == ["living_research_daemon", "queue_maintainer", "dashboard"]
    assert "watchdog_pid" not in status  # dry_run launches nothing


def test_platform_once_flag_in_command(tmp_path):
    status = tlr.run_platform(tmp_path, dry_run=True, once=True)
    assert "--once" in status["command"]


def test_platform_surfaces_kill_switch_flags(tmp_path):
    (tmp_path / "tar_state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tar_state" / "daemon_paused.flag").write_text("x", encoding="utf-8")
    status = tlr.run_platform(tmp_path, dry_run=True)
    assert status["daemon_paused"] is True
    assert status["execution_enabled"] is False
