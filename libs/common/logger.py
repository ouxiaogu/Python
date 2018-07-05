'''
-*- coding: utf-8 -*-
Created: hshi, 2017-12-11 17:52:08

Last Modified by: ouxiaogu
logger: Logging stream control module
'''

import logging
import sys

LOGNAME = 'PY'

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
