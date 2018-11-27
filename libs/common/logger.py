# -*- coding: utf-8 -*-
'''
Created: hshi, 2017-12-11 17:52:08

Reference: https://docs.python.org/3/howto/logging-cookbook.html

Last Modified by:  ouxiaogu
logger: Logging stream control module
'''

import logging
import sys

class logger(object):
    LOGNAME = 'MXP'

    # configure once only in main module
    @staticmethod
    def initlogging(level=logging.INFO, logpath=None):

        if type(level) == str:
            types = ['info', 'debug', 'error']
            if level.lower() not in types:
                raise ValueError("log level should be in: {}\n".format(
                    ', '.join(types)))
            else:
                if level.lower() == 'info':
                    level = logging.INFO
                else:
                    level = logging.DEBUG

        LOGLEVEL = level
        log = logging.getLogger(logger.LOGNAME)
        # Why are there two setLevel() methods?
        # The level set in the logger determines which severity of messages it will pass to its handlers.
        # The level set in each handler determines which messages that handler will send on.
        log.setLevel(logging.DEBUG)

        DEBUGFORMAT = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        FORMAT = logging.Formatter('%(name)s - %(message)s')

        if logpath is not None:
            fh = logging.FileHandler(logpath, mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(DEBUGFORMAT)
            log.addHandler(fh)

        shstd = logging.StreamHandler(sys.stdout)
        shstd.setLevel(LOGLEVEL)
        shstd.setFormatter(FORMAT)
        log.addHandler(shstd)

        sherror = logging.StreamHandler(sys.stderr)
        sherror.setLevel(logging.WARNING)
        sherror.setFormatter(DEBUGFORMAT)
        log.addHandler(sherror)

    # create (but not configure) a child logger in a separate module
    @staticmethod
    def getLogger(name):
        return logging.getLogger(logger.LOGNAME + '.' + name)

    @staticmethod
    def closelogging(name):
        log = logger.getLogger(logger.LOGNAME + '.' + name)
        for i, fh in enumerate(log.handlers):
            print(i)
            fh.close()


if __name__ == '__main__':
    logger.initlogging(level='debug')
    log = logger.getLogger('D2DB')
    log.info('info test')
    log.debug('debug test')
    logger.closelogging('D2DB')

    logger.LOGNAME = "MXP"
    logger.initlogging(level='debug')
    log = logger.getLogger('D2DB')
    log.info('info test')
    log.debug('debug test')
    logger.closelogging('D2DB')