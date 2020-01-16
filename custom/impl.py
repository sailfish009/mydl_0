import torch
from .. utils.logger import setup_logger
from .. utils.collect_env import collect_env_info
from . ml_stratifiers import MultilabelStratifiedShuffleSplit

def get_cuda_count():
    return torch.cuda.device.count()

def get_logger(cfg, save_path="", distributed_rank=0):
    logger = setup_logger("model", save_path, distributed_rank=distributed_rank)
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    logger.info(cfg)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    return logger

def train_test_split(df, n_splits=4):
    df_backup = df.copy()
    X = df['Id'].tolist()
    y = df['target_vec'].tolist()
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=42)
    train_dfs, valid_dfs = [], []
    for train_index, valid_index in msss.split(X,y):
        train_df = df_backup.iloc[train_index]
        train_dfs.append(train_df[['Id', 'Target']])
        valid_df = df_backup.iloc[valid_index]
        valid_dfs.append(valid_df[['Id', 'Target']])
    return train_dfs, valid_dfs
