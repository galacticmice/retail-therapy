import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_PATH = ROOT / "src/retail-therapy"


def launch_script(filename, flags=()):
    """run a stage with the current Python environment"""
    command = [sys.executable, str(SCRIPTS_PATH / filename), *flags]
    print()
    print(shlex.join(command), flush=True)
    try:
        process = subprocess.Popen(command, cwd=ROOT, shell=False)
    except OSError as error:
        print(f"Could not start stage: {error}")
        return 1

    # parent and child receive terminal Ctrl-C, let the child finish cleanup
    interrupted = False
    while True:
        try:
            result = process.wait()
            break
        except KeyboardInterrupt:
            interrupted = True
            print("\nWaiting for stage to finish stopping...", flush=True)
    # leave the launcher after stopping the simulator, skip help-only calls
    if interrupted and filename == "purchase_simulator.py":
        if "--help" not in flags and "-h" not in flags:
            raise KeyboardInterrupt
    if result != 0:
        print(f"Stage exited with code {result}")
    return result


def launch_options(filename, defaults):
    """choose defaults or enter flags for one stage"""
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"\n        {filename}")
        print(f"        Default flags: {shlex.join(defaults) or '(none)'}")
        print("""
        [1] Launch with defaults
        [2] Enter custom flags

        [x] Back
            """)
        cin = input("\t>> ").strip().lower()
        match cin:
            case "1":
                launch_script(filename, defaults)
                input("\033[3mPress Enter to continue...\033[0m")
                return
            case "2":
                # use the script's parser to show its current flag options
                if launch_script(filename, ["--help"]) != 0:
                    input("\033[3mPress Enter to continue...\033[0m")
                    return
                print(f"\nDefault flags: {shlex.join(defaults) or '(none)'}")
                print(
                    "Enter flags only. Blank or x goes back. Quote paths with spaces."
                )
                while True:
                    entered = input("\tflags >> ").strip()
                    if not entered or entered.lower() == "x":
                        break
                    try:
                        flags = shlex.split(entered)
                    except ValueError as error:
                        print(f"Invalid flags: {error}")
                        continue
                    launch_script(filename, flags)
                    input("\033[3mPress Enter to continue...\033[0m")
                    return
            case "x":
                return
            case _:
                print("Invalid input, try again.")
                input("\033[3mPress Enter to continue...\033[0m")


def main():
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print("""
        [1] Data preparation
        [2] Classify products
        [3] Train prediction model

        [4] Launch simulator

        [x] Exit
                """)
            cin = input("\t>> ").strip().lower()
            match cin:
                case "1":
                    # data preparation already has its own submenu
                    if launch_script("dataset.py") != 0:
                        input("\033[3mPress Enter to continue...\033[0m")
                case "2":
                    launch_options("categorical_ml.py", ["--full"])
                case "3":
                    launch_options("next_category.py", ["--full", "--epochs", "3"])
                case "4":
                    launch_options("purchase_simulator.py", [])
                case "x":
                    return
                case _:
                    print("Invalid input, try again.")
                    input("\033[3mPress Enter to continue...\033[0m")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
