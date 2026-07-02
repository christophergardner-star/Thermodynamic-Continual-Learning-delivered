"""
SSH host-key handling for the RunPod bridge (TOFU pinning).

Replaces blanket paramiko.AutoAddPolicy() (which silently trusts EVERY host on
EVERY connection — a standing MITM exposure) with trust-on-first-use:

- host keys are pinned to a persistent known_hosts file on first contact;
- a CHANGED key for a known host makes paramiko raise BadHostKeyException
  (paramiko checks loaded host keys in connect()), aborting the connection.

Honest caveat: RunPod pods are ephemeral — most connections are first contact
with a brand-new host, where TOFU cannot authenticate the endpoint. The value
is (a) re-connections to the same pod/volume host are verified, and (b) key
material is durably logged for audit. This is strictly stronger than AutoAdd
and never weaker.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_KNOWN_HOSTS = Path.home() / ".tar" / "runpod_known_hosts"


class TofuAddPolicy:
    """paramiko MissingHostKeyPolicy: pin unknown hosts to a known_hosts file."""

    def __init__(self, known_hosts_path: Path) -> None:
        self._path = Path(known_hosts_path)

    def missing_host_key(self, client, hostname, key):  # paramiko interface
        client.get_host_keys().add(hostname, key.get_name(), key)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            client.save_host_keys(str(self._path))
        except OSError:
            pass  # pin in-memory even if the file cannot be written


def apply_tofu_policy(client, known_hosts_path: "Path | None" = None) -> None:
    """Load pinned host keys into *client* and install the TOFU policy.

    Call in place of client.set_missing_host_key_policy(AutoAddPolicy()).
    """
    path = Path(known_hosts_path) if known_hosts_path else _DEFAULT_KNOWN_HOSTS
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        client.load_host_keys(str(path))
    except OSError:
        pass  # no pins to load; TOFU still pins from here on
    client.set_missing_host_key_policy(TofuAddPolicy(path))
