from typing import Any


def print_red(string: Any) -> None:
    """
    print string in ANSI escape code for colored (red) visual
    """
    print(f"\033[91m {string}\033[00m")


def print_green(string: Any) -> None:
    """
    print string in ANSI escape code for colored (green) visual
    """
    print(f"\033[92m {string}\033[00m")


def print_yellow(string: Any) -> None:
    """
    print string in ANSI escape code for colored (yellow) visual
    """
    print(f"\033[93m {string}\033[00m")
