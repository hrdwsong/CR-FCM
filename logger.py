import logging


def get_logger(file_path):
    #
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)  #

    #
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)

    #
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    #
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    #
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # test
    # logger.debug('This is a debug message')
    # logger.info('This is an info message')
    # logger.warning('This is a warning message')
    # logger.error('This is an error message')
    # logger.critical('This is a critical message')
    return logger
