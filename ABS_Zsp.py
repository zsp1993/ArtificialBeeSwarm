#-*-coding:utf-8 -*-
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import operator

class ABSIndividual:
    '''
    individual of artificial bee swarm algorithm
    '''
    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.   #适应度
        self.trials = 0     #在同一蜜源附近搜索次数

    def generate(self):
        '''
        generate a random chromsome for artificial bee swarm algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in xrange(0, len):
            self.chrom[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = self.objfunction(self.chrom)

    def objfunction(self, x):
        value = 0
        for j in range(self.vardim / 4):
            i = j + 1
            value = value + (x[4 * i - 3 - 1] + 10 * x[4 * i - 2 - 1]) ** 2
            value = value + 5 * ((x[4 * i - 1 - 1] - x[4 * i - 1]) ** 2)
            value = value + (x[4 * i - 2 - 1] - 2 * x[4 * i - 1 - 1]) ** 4
            value = value + 10 * ((x[4 * i - 3 - 1] - x[4 * i - 1]) ** 4)
        return value

class ArtificialBeeSwarm:
    '''
    the class for artificial bee swarm algorithm
    '''
    def __init__(self, sizepop, vardim, bound, MAXGEN, limit):
        '''
        sizepop: 蜂群大小规模
        vardim: 变量维度
        bound: 边界
        MAXGEN: 最大循环次数
        params: 最大探索次数
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.foodSource = self.sizepop / 2        #采蜜蜂个数选择为蜜蜂个数的一半
        self.MAXGEN = MAXGEN
        self.limit = limit
        self.population = []

    def initialize(self):
        #初始化蜂群
        for i in xrange(0, self.sizepop):
            ind = ABSIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluation(self):
        #计算蜂群内各蜜蜂的适应度
        for i in xrange(0, self.sizepop):
            self.population[i].calculateFitness()

    def bee_one(self):
        #采蜜蜂活动
        for i in xrange(0, self.foodSource):
            k = np.random.random_integers(0, self.vardim - 1)  # 随机选择一个维度
            j = np.random.random_integers(0, self.foodSource - 1)  # 随机选择一只采蜜蜂
            while j == i:
                j = np.random.random_integers(0, self.foodSource - 1)
            vi = copy.deepcopy(self.population[i])
            vi.chrom[k] += np.random.uniform(low=-1, high=1.0, size=1) * (vi.chrom[k] - self.population[j].chrom[k])
            if vi.chrom[k] < self.bound[0, k]:
                vi.chrom[k] = self.bound[0, k]
            if vi.chrom[k] > self.bound[1, k]:
                vi.chrom[k] = self.bound[1, k]
            vi.calculateFitness()
            if vi.fitness > self.population[i].fitness:
                self.population[i] = vi
                self.population[i].trials = 0
                if vi.fitness > self.best.fitness:
                    self.best = vi
            else:
                self.population[i].trials += 1

    def bee_two(self):
        #跟随蜂的活动
        sumfit = 0  #采蜜蜂总适应度
        sum_fit = np.zeros(self.foodSource)
        for i in xrange(0, self.foodSource):    #生成一个转盘，用来随机选择将要跟随的采蜜群
            sumfit = sumfit + self.population[i].fitness
            sum_fit[i] = sumfit
        for i in xrange(0,self.sizepop-self.foodSource):
            self.population[i + self.foodSource].trials = 0
            rand_value = np.random.uniform(low=0,high=sumfit,size=1)
            for index in xrange(0,self.foodSource):
                if rand_value < sum_fit[index]:
                    j1 = index                                      #选择要跟随的采蜜蜂
                    break
            k = np.random.random_integers(0, self.vardim - 1)      #随机选择一维
            j = np.random.random_integers(0, self.foodSource - 1)  # 随机选择一只采蜜蜂
            while j == j1:
                j = np.random.random_integers(0, self.foodSource - 1)
            vi = copy.deepcopy(self.population[j1])
            vj = copy.deepcopy(self.population[j1])
            vi.chrom[k] += np.random.uniform(low=-1, high=1.0, size=1) * (vi.chrom[k] - self.population[j].chrom[k])
            if vi.chrom[k] < self.bound[0, k]:
                vi.chrom[k] = self.bound[0, k]
            if vi.chrom[k] > self.bound[1, k]:
                vi.chrom[k] = self.bound[1, k]
            vi.calculateFitness()
            if vi.fitness > self.population[j1].fitness:
                self.population[i+self.foodSource] = vi
                self.population[i+self.foodSource].trials = 0
                if vi.fitness > self.best.fitness:
                    self.best = vi
            else:
                self.population[i+self.foodSource] = vj
                self.population[i+self.foodSource].trials += 1

    def scoutBeePhase(self):
        #计算各蜜蜂在同一个地点搜索次数
        for i in xrange(0, self.sizepop):
            if self.population[i].trials > self.limit:
                self.population[i].generate()
                self.population[i].trials = 0
                self.population[i].calculateFitness()

    def solve(self):
        #人工蜂群计算过程
        self.t = 0         #循环次数
        self.initialize()  #初始化种群
        self.evaluation()  #计算适应度
        cmpfun = operator.attrgetter("fitness")
        self.population.sort(key=cmpfun, reverse=True)  #按照适应度排序，前一半为采蜜蜂，后一半为跟随蜂
        self.best = copy.deepcopy(self.population[0])   #最优适应度位置
        print("Generation %d: optimal function value is: %f;" % (
            self.t, self.best.fitness))
        while self.t < self.MAXGEN :
            self.t += 1
            self.bee_one()  #population 前一半为采蜜蜂，采蜜粉活动
            self.bee_two()  #population 后一半为跟随蜂，跟随蜂活动
            self.scoutBeePhase()
            self.population.sort(key=cmpfun, reverse=True)  # 按照适应度排序，前一半为采蜜蜂，后一半为跟随蜂
            if self.population[0].fitness > self.best.fitness:
                self.best = copy.deepcopy(self.population[0])  # 最优适应度位置
            print("Generation %d: optimal function value is: %f;" % (
                self.t, self.best.fitness))
        print "Optimal solution is:"
        print self.best.chrom





if __name__ == "__main__":
    bound = np.tile([[-1], [1]], 20)
    abs = ArtificialBeeSwarm(60, 20, bound, 1000, 100)
    abs.solve()
