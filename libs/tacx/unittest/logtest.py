import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mylog")
formatter = logging.Formatter('%(asctime)s, %(name)s, %(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# logFilePath = "my.log"
# file_handler = logging.handlers.TimedRotatingFileHandler(filename = logFilePath, when = 'midnight', backupCount = 30)
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.DEBUG)

# logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("Started");
try:
    x = 14
    y = 0
    z = x / y
except Exception as ex:
    logger.error("Operation failed.")
    logger.debug("Encountered {0} when trying to perform calculation.".format(ex))

logger.info("Ended");