'''
-*- coding: utf-8 -*-
Created: hshi, 2017-12-11 17:52:08

Last Modified by: ouxiaogu
logger: Logging stream control module
'''

'''
import logging
import sys

def initlogging(logpath=None, debug=False):
    if debug:
        LOGLEVEL = logging.DEBUG
    else:
        LOGLEVEL = logging.INFO
    logging.basicConfig(stream=sys.stdout, level=LOGLEVEL)
    log = logging.getLogger(LOGNAME)
    log.setLevel(LOGLEVEL)

    DEBUGFORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    FORMAT = logging.Formatter('%(asctime)s - %(message)s')

    # shstd = logging.StreamHandler(sys.stdout)
    # shstd.setLevel(LOGLEVEL)
    # shstd.setFormatter(FORMAT)
    # log.addHandler(shstd)

    # sherror = logging.StreamHandler(sys.stderr)
    # sherror.setLevel(logging.WARNING)
    # sherror.setFormatter(FORMAT)
    # log.addHandler(sherror)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOGLEVEL)
    stream_handler.setFormatter(FORMAT)

    if logpath is not None:
        fh = logging.FileHandler(logpath, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(DEBUGFORMAT)
        log.addHandler(fh)
        initlogging.fh = fh

def getLogger(name):
    return logging.getLogger('{}.{}'.format(LOGNAME, name))


def closelogging():
    initlogging.fh.close()

'''

'''
-*- coding: utf-8 -*-
Created: ouxiaogu, 2017-12-11 17:52:08
Refer to [github Alephbet gimel](https://github.com/Alephbet/gimel/blob/e807e8819e632ca018aac20a8c621207a219e799/gimel/logger.py
)
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
    log = setup('test')
    log.info('test')
    log.info('debug')