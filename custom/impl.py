import torch
from .. utils.logger import setup_logger
from .. utils.collect_env import collect_env_info

def get_cuda_count():
    return torch.cuda.device.count()

def get_logger(cfg, save_path="", distributed_rank=0):
    logger = setup_logger("model", save_path, distributed_rank=distributed_rank)
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    logger.info(cfg)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    return logger

