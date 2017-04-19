from threading import Thread

import time

from gpiozero import Buzzer

from PinPortDefinition import PinPortDefinition


class Alarming:
    __INSTANCE = None
    __PIN_BCM = PinPortDefinition.GPIO_BCM_BUZZER  # gpiozero uses BCM code

    def __init__(self):
        self.__buzzing = False
        self.__thread = Thread(target=self.__thread_run())
        self.__thread.daemon = True
        if self.__PIN_BCM == -1:
            print('buzzer port = ?')
            return
        self.__thread.start()

    @classmethod
    def get_instance(cls):
        if cls.__INSTANCE is None:
            cls.__INSTANCE = Alarming()
        return cls.__INSTANCE

    def __thread_run(self):
        buzzer = Buzzer(self.__PIN_BCM)
        while True:
            if self.__buzzing:
                buzzer.on()
                time.sleep(.2)
                buzzer.off()
                time.sleep(.2)

    def toggle(self):
        self.__buzzing = not self.__buzzing

    def buzz(self):
        self.__buzzing = True

    def set_buzzing(self, value):
        self.__buzzing = value

    def silent(self):
        self.__buzzing = False
