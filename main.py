from asyncio.windows_events import NULL
import os
import torch
from owasp_chain import owasp_print
from nist_chain import nist_print

# =========================
# BASIC SETUP
# =========================

torch.set_num_threads(4)


def get_choice():
    while True:
        choice_str = input('Select one from this 1) owasp  2) nist  (anything else to quit): ').strip()
        if not choice_str:
            # empty input -> quit
            return None
        try:
            choice = int(choice_str)
            return choice
        except ValueError:
            print('Please enter 1, 2, or press Enter/other key to exit.')


a = get_choice()
while a in (1, 2):
    question = input('Enter the question related to selected part: ')
    if a == 1:
        print(owasp_print(question))
    elif a == 2:
        print(nist_print(question))
    a = get_choice()

print('Thank you for using the application')
