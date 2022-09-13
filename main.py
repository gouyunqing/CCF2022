from train import train
from config import Config
from baseline import Baseline
import pandas as pd

if __name__ == '__main__':
    config = Config()
    model = Baseline()
    train(model, config)

    # submit = pd.DataFrame({'id': id_list, 'label':  res_list})
    # submit.to_csv('submit_baseline.csv', index=None, encoding='utf-8')