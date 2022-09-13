class Config:
    def __init__(self):
        self.max_len = 512
        self.batch_size = 8
        self.epoch = 50
        self.train_path = 'train.json'
        self.test_path = 'testA.json'
        self.lr = 5e-6
        self.eps = 1e-8
        self.weight_decay = 1e-6
        self.hidden_size = 768