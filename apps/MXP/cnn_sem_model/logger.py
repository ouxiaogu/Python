'''
-*- coding: utf-8 -*-
Created: ouxiaogu, 2017-12-11 17:52:08
Last Modified by: ouxiaogu
logger: Logging stream control module
'''

import logging
import sys

FORMAT = logging.Formatter('%(name)-8s: %(message)s')
DEBUGFORMAT = logging.Formatter('%(name)-8s %(levelname)-8s: %(message)s')

def setup(name=__name__, level=logging.INFO, logpath=None):
    # change type
    if type(level) == str:
        types = ['info', 'debug', 'error']
        if level.lower() not in types:
            raise ValueError("log level should be in: {}\n".format(', '.join(types) ) )
        else:
            if level.lower() == 'info':
                level = logging.INFO
            else:
                level = logging.DEBUG
    fmt = FORMAT
    if level == logging.DEBUG:
        fmt = DEBUGFORMAT

    # check file handler
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logpath is not None:
        fh = logging.FileHandler(logpath, mode='a')
        fh.setLevel(level=level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if logger.handlers:
        return logger

    # if without file handler, use stdout
    handler = logging.StreamHandler()
    handler.setLevel(level=level)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':
    log = setup('test', 'info')
    log.info('info')
    log.debug('debug')