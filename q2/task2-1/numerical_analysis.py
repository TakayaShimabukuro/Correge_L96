from logging import getLogger, config
import json
logger = getLogger(__name__)

class Analysis_Methods:
    # オイラー法
    def Euler(self, n_step):
        logger.info('Euler()')

    # ルンゲクッタ2次
    def RK2(self, n_step):
        logger.info('RK2()')

    # ルンゲクッタ4次
    def RK4(self, n_step):
        logger.info('RK4()')
