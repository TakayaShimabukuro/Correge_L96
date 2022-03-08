import math
import random
import sys
import traceback


class RndnumBoxMuller:
    M = 10        # 平均
    S = 2.5       # 標準偏差
    N = 10000     # 生成する個数
    SCALE = N // 100  # ヒストグラムのスケール

    def __init__(self):
        self.hist = [0 for _ in range(self.M * 5)]

    def generate_rndnum(self):
        try:
            for _ in range(self.N):
                res = self.__rnd()
                self.hist[res[0]] += 1
                self.hist[res[1]] += 1
                
            print(len(self.hist))
        except Exception as e:
            raise

    def __rnd(self):
        try:
            r_1 = random.random()
            r_2 = random.random()
            x = self.S \
                * math.sqrt(-2 * math.log(r_1)) \
                * math.cos(2 * math.pi * r_2) \
                + self.M
            y = self.S \
                * math.sqrt(-2 * math.log(r_1)) \
                * math.sin(2 * math.pi * r_2) \
                + self.M
            return [math.floor(x), math.floor(y)]
        except Exception as e:
            raise


box_muller = RndnumBoxMuller()
box_muller.generate_rndnum()
