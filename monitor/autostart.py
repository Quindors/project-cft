# monitor/autostart.py
"""Windows integration for running the frozen exe "whenever the computer is in use".

`cft.exe --install-autostart` registers a per-user Task Scheduler task that
launches this exe at logon and on every workstation unlock, so the monitor
comes back automatically whenever a session of computer use starts. The task
points at sys.executable, so it survives dist moves as long as it is
re-installed from the new location, and a rebuild in place needs nothing.

No admin rights are required: the task runs with the user's own interactive
token (the exe must run in the interactive session to see foreground windows).
`--uninstall-autostart` removes the task again.

The task also re-fires every ten minutes as a self-heal: if the monitor was
closed or crashed, it is back within ten minutes without anyone noticing. The
single-instance mutex makes those launches free while it IS running.

The single-instance mutex complements the task: Task Scheduler's IgnoreNew
policy stops the task from starting twice, but not a manual double-click racing
a task launch. The mutex is session-local, which still covers elevated +
non-elevated instances in the same session.

cft:// is the third way in: the exe registers itself as the handler for that
URL scheme (per-user, HKCU, no admin), so the hosted dashboard's "Start CFT"
button — a plain link to cft://start — makes the browser ask "Open CFT
Monitor?" and launch the exe on a PC where it isn't running.
"""
import ctypes
import os
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape

TASK_NAME = "CFT Monitor"
PROTOCOL_SCHEME = "cft"       # cft://start — the dashboard's "Start CFT" button
SELF_HEAL_INTERVAL = "PT10M"  # ISO 8601 duration for the task's repetition

_MUTEX_NAME = "cft-monitor-single-instance"
_ERROR_ACCESS_DENIED = 5     # mutex exists but is held by an elevated instance
_ERROR_ALREADY_EXISTS = 183
_mutex_handle = None


def strip_protocol_args(args):
    """Drop cft:// URLs from argv. A protocol launch is just "start": the URL
    is whatever web page linked it, so it is never treated as a command."""
    return [a for a in args if not a.lower().startswith(PROTOCOL_SCHEME + ":")]


def register_protocol():
    """Register cft:// under HKCU so a browser can launch this exe.

    Per-user, no admin. Idempotent and cheap: it rewrites the keys only when
    the stored command differs from this exe's path, so a moved dist re-points
    the scheme at itself on its next start. Source mode never registers —
    a python.exe with the wrong args is not what a click should open.
    """
    if not getattr(sys, "frozen", False) or sys.platform != "win32":
        return False
    import winreg
    exe = sys.executable
    command = f'"{exe}" "%1"'
    root = rf"Software\Classes\{PROTOCOL_SCHEME}"
    try:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, root + r"\shell\open\command") as k:
                current, _ = winreg.QueryValueEx(k, "")
                if current == command:
                    return True
        except OSError:
            pass
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, root) as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "URL:CFT Monitor")
            winreg.SetValueEx(k, "URL Protocol", 0, winreg.REG_SZ, "")
            winreg.SetValueEx(k, "FriendlyAppName", 0, winreg.REG_SZ, "CFT Monitor")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, root + r"\DefaultIcon") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{exe}",0')
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, root + r"\shell\open\command") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, command)
            winreg.SetValueEx(k, "FriendlyAppName", 0, winreg.REG_SZ, "CFT Monitor")
        print(f"[autostart] registered {PROTOCOL_SCHEME}:// -> {exe}")
        return True
    except OSError as e:
        print(f"[autostart] could not register {PROTOCOL_SCHEME}:// ({e})")
        return False


def acquire_single_instance():
    """Try to become the only cft instance in this session. False = already running."""
    global _mutex_handle
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    err = ctypes.get_last_error()
    if not handle or err in (_ERROR_ALREADY_EXISTS, _ERROR_ACCESS_DENIED):
        return False
    _mutex_handle = handle  # hold for process lifetime; released by the OS on exit
    return True


def _task_xml(exe, workdir, user):
    exe, workdir, user = escape(exe), escape(workdir), escape(user)
    # Logon is delayed 20s (and failed launches retried) because the exe lives
    # on \\wsl.localhost, which may still be spinning up right after boot.
    # Priority 5 = Normal: scheduled tasks default to BelowNormal, too slow for
    # an app serving the dashboard. PT0S = no execution time limit.
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>project-cft</Author>
    <Description>Start the CFT focus monitor at logon and on unlock, so it runs whenever the computer is in use.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
      <Delay>PT20S</Delay>
    </LogonTrigger>
    <SessionStateChangeTrigger>
      <Enabled>true</Enabled>
      <StateChange>SessionUnlock</StateChange>
      <UserId>{user}</UserId>
    </SessionStateChangeTrigger>
    <TimeTrigger>
      <Enabled>true</Enabled>
      <StartBoundary>2026-01-01T00:00:00</StartBoundary>
      <Repetition>
        <Interval>{SELF_HEAL_INTERVAL}</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{exe}</Command>
      <WorkingDirectory>{workdir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _schtasks(*args):
    exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "schtasks.exe")
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        [exe, *args], capture_output=True, text=True, timeout=60, creationflags=no_window,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def install_task():
    if not getattr(sys, "frozen", False):
        print("[autostart] refusing: source mode is trace mode; only the frozen exe should auto-start.")
        return False
    exe = sys.executable
    user = os.environ.get("USERDOMAIN", "") + "\\" + os.environ.get("USERNAME", "")
    xml = _task_xml(exe, os.path.dirname(exe), user)
    fd, xml_path = tempfile.mkstemp(suffix=".xml", prefix="cft-task-")
    try:
        # schtasks expects task XML in UTF-16 (matching the declaration above).
        with os.fdopen(fd, "w", encoding="utf-16") as f:
            f.write(xml)
        rc, out = _schtasks("/Create", "/TN", TASK_NAME, "/XML", xml_path, "/F")
    finally:
        try:
            os.unlink(xml_path)
        except OSError:
            pass
    if rc != 0:
        print(f"[autostart] schtasks /Create failed (rc={rc}): {out}")
        return False
    print(f"[autostart] installed task '{TASK_NAME}' -> {exe}")
    print(f'[autostart] start it now with: schtasks /Run /TN "{TASK_NAME}"')
    return True


def uninstall_task():
    rc, _ = _schtasks("/Query", "/TN", TASK_NAME)
    if rc != 0:
        print(f"[autostart] task '{TASK_NAME}' is not installed; nothing to do.")
        return True
    rc, out = _schtasks("/Delete", "/TN", TASK_NAME, "/F")
    if rc != 0:
        print(f"[autostart] schtasks /Delete failed (rc={rc}): {out}")
        return False
    print(f"[autostart] removed task '{TASK_NAME}'")
    return True
