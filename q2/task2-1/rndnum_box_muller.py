import math
import random
import sys
import traceback

class RndnumBoxMuller:
    M     = 20        # 平均
    S     = 2.5       # 標準偏差
    N     = 100     # 生成する個数
    SCALE = N // 100  # ヒストグラムのスケール

    def __init__(self):
        self.hist = [0 for _ in range(self.M * 5)]
    
    def generate_rndnum(self):
        try:
            for _ in range(self.N):
                res = self.__rnd()
                self.hist[res[0]] += 1
                self.hist[res[1]] += 1
                
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
            print(x)
            return [math.floor(x), math.floor(y)]
        except Exception as e:
            raise
    def display(self):
        """ Display """
        try:
            for i in range(0, self.M * 2 + 1):
                print("{:>3}:{:>4} | ".format(i, self.hist[i]), end="")
                for j in range(1, self.hist[i] // self.SCALE + 1):
                    print("*", end="")
                print()
        except Exception as e:
            raise

box_muller = RndnumBoxMuller()
box_muller.generate_rndnum()
box_muller.display()