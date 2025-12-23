# monitor/alerts.py
import ctypes

def show_popup(message: str, title: str = "Productivity Alert") -> None:
    MB_ICONINFORMATION = 0x40
    MB_SETFOREGROUND = 0x00010000
    MB_TOPMOST = 0x00040000
    flags = MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)