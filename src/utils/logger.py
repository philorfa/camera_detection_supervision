from colorama import Fore
import os
from .timestamps import get_utc_time

MUTED_ENVS = ["test"]


class Logger:

    def info(self, info):
        if os.getenv("ENV") not in MUTED_ENVS:
            print(Fore.LIGHTGREEN_EX + "\nTIME" + Fore.WHITE + ":" + 5 * " " +
                  f"{get_utc_time()}")
            print(Fore.GREEN + "INFO" + Fore.WHITE + ":" + 4 * " ", info)

    def error(self, error):
        if os.getenv("ENV") not in MUTED_ENVS:
            print(Fore.LIGHTGREEN_EX + "\nTIME" + Fore.WHITE + ":" + 5 * " " +
                  f"{get_utc_time()}")
            print(Fore.LIGHTRED_EX + "ERROR" + Fore.WHITE + ":" + 3 * " ",
                  error)

    def success(self, msg):
        if os.getenv("ENV") not in MUTED_ENVS:
            print(Fore.LIGHTGREEN_EX + "\nTIME" + Fore.WHITE + ":" + 5 * " " +
                  f"{get_utc_time()}")
            print(Fore.LIGHTGREEN_EX + "SUCCESS" + Fore.WHITE + ":" + 1 * " ",
                  msg)


log = Logger()
