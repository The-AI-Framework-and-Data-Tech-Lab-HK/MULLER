import logging
from muller.client.config import LOG_LEVEL, LOG_FORMAT

class LoggerHandler(logging.Logger):

    def __init__(self, level="DEBUG", file=None,
                 out_format=None):
        super().__init__("gtn_f")
        self.setLevel(level)
        fmt = logging.Formatter(out_format)
        if file:
            file_handler = logging.FileHandler(file,encoding='utf-8')
            file_handler.setLevel(level)
            self.addHandler(file_handler)
            file_handler.setFormatter(fmt)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        self.addHandler(stream_handler)
        stream_handler.setFormatter(fmt)

logger = LoggerHandler(level=LOG_LEVEL,file=None,out_format=LOG_FORMAT)


