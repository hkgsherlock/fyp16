from threading import Thread

import time

from gpiozero import Buzzer

from PinPortDefinition import PinPortDefinition


class Alarming:
    __INSTANCE = None
    __PIN_BCM = PinPortDefinition.GPIO_BCM_BUZZER  # gpiozero uses BCM code
    buzzer = Buzzer(__PIN_BCM)

    def __init__(self):
        pass

    def toggle(self):
        self.buzzer.value = not self.buzzer.value

    def buzz(self):
        self.buzzer.on()

    def set_buzzing(self, value):
        self.buzzer.value = value

    def silent(self):
        self.buzzer.off()
