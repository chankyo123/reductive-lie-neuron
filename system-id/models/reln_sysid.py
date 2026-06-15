from models.sysid_common import BaseLieNet


class ReLNNet(BaseLieNet):
    def __init__(self, hid_c=16):
        super().__init__(hid_c=hid_c, algebra_type='gl3')
