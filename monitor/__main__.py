# monitor/__main__.py
from monitor.app import run

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[info] stopped by user")