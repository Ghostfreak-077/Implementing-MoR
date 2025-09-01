
class LinearRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, out_dim, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
    def forward(self, x):
        return self.router(x)