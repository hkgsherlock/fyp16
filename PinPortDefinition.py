class PinPortDefinition:
    # wiring pi = 0
    GPIO_BOARD_IR_FILTER = 11
    GPIO_BCM_IR_FILTER = 17

    # wiring pi = ?
    GPIO_BOARD_LIGHT_SENSOR = -1
    GPIO_BCM_LIGHT_SENSOR = -1

    # wiring pi = 6
    GPIO_BOARD_BUZZER = 22
    GPIO_BCM_BUZZER = 25

    def __init__(self):
        pass