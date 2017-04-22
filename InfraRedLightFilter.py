from PinPortDefinition import PinPortDefinition
from gpiozero import LED


class InfraRedLightFilter:
    FILTER = LED(PinPortDefinition.GPIO_BCM_IR_FILTER, initial_value=True)

    def __init__(self):
        pass

    @classmethod
    def on(cls):
        cls.FILTER.on()

    @classmethod
    def off(cls):
        cls.FILTER.off()

    @classmethod
    def set_state(cls, value):
        cls.FILTER.value = value

    @classmethod
    def get_state(cls):
        return cls.FILTER.value
