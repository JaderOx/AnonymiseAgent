import os
import sys
import time
import logging

output_dir = '/path/to/data/example/agent/output/'

def setup_logger(output_dir, log_name, level='INFO'):

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 环境变量统一日志路径：所有子进程写同一个文件
    env_log = os.environ.get('ANONYMISE_LOG_FILE')
    if env_log:
        log_file = env_log
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(logs_dir, f'{log_name}_{timestamp}.log')

    # 创建logger
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除已有 handler，避免重复写入
    logger.handlers.clear()

    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file

def get_logger(logger_file, log_name, level='INFO'):

    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, level.upper()))

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件handler
    file_handler = logging.FileHandler(logger_file)
    file_handler.setFormatter(formatter)

    # 将handler添加到logger
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    # 示例用法
    logger, logger_file = setup_logger(output_dir, 'testLog', "DEBUG")

    logger.info(f"模型名称: ")
    logger.debug(f"Batch size: ")
    logger.warning(f"学习率: ")
    logger.critical(f"Epochs: ")
    logger.error(f"随机种子: ")

    logger1 = get_logger(logger_file, 'Test')
    logger1.info(f"测试")