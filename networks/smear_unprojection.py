

class SmearUnprojectionConfig(BaseConfig):
    pass

class SmearUnprojection(nn.Module):
    def __init__(self, config: SmearUnprojectionConfig):
        super().__init__()
        self.config = config

    def forward(self, data):
        pass