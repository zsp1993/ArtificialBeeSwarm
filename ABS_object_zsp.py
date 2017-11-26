#coding=utf-8
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
import operator

class ABSIndividual:
    '''
    individual of artificial bee swarm algorithm
    '''
    def __init__(self,  vardim, bound, object):
        '''
        vardim: 变量维度，包括椭圆中心横、纵坐标，长轴长度，短轴长度，旋转角度，共五个
        bound: 每一个变量的上下边界
        '''
        self.vardim = vardim
        self.bound = bound
        self.object = object  #待匹配的边缘二值矩阵
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

    def objfunction(self, x):   #计算椭圆轮廓上存在的边缘点对椭圆中心的边缘势场
        value = 0
        sp = self.object.shape
        m = sp[0]
        n = sp[1]
        cenx = x[0]
        ceny = x[1]
        a = x[2]
        b = x[3]
        angle = x[4]
        num = int(2 * math.pi / 0.001)
        x0 = 0
        y0 = 0
        for i in range(num):
            x1 = int(a * math.sin(0.001 * i))  #相对于椭圆中心的坐标
            y1 = int(b * math.cos(0.001 * i))
            x2 = int(cenx + int(x1 * math.cos(angle) + y1 * math.sin(angle)))  #相对于椭圆中心的旋转
            y2 = int(ceny + int(-x1 * math.sin(angle) + y1 * math.cos(angle)))
            if (x2 != x0 or y2 != y0) and (x2 < m and x2>=0 and y2<n and y2>=0):#保证和上次算出的边缘点不一样且没有出界

                if self.object[x2,y2] > 0:
                    value = value + math.sqrt(np.square(x1) + np.square(y1))#累加一次对中心的边缘势场,忽略了常量
                    #value = value +1
            x0 = x2
            y0 = y2
        return value


class ArtificialBeeSwarm:
    '''
    the class for artificial bee swarm algorithm椭圆识别
    '''
    def __init__(self, sizepop, vardim, bound, MAXGEN, limit,object, object2):
        '''
        sizepop: 蜂群大小规模
        vardim: 变量维度
        bound: 边界
        MAXGEN: 最大循环次数
        limit: 最大探索次数
        object:待匹配图片canny二值化矩阵
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.foodSource = self.sizepop / 2        #采蜜蜂个数选择为蜜蜂个数的一半
        self.MAXGEN = MAXGEN
        self.limit = limit
        self.object = object
        self.object2 = object2  # 原来的彩色图
        self.population = []
        self.best_fit = []  # 最优适应值迭代过程
    def initialize(self):
        #初始化蜂群
        for i in xrange(0, self.sizepop):
            ind = ABSIndividual(self.vardim, self.bound, self.object)
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
        self.best_fit.append(self.best.fitness)
        print("Generation %d: optimal function value is: %f;" % (
            self.t, self.best.fitness))
        while self.t < self.MAXGEN-1 :
            self.t += 1
            self.bee_one()  #population 前一半为采蜜蜂，采蜜粉活动
            self.bee_two()  #population 后一半为跟随蜂，跟随蜂活动
            self.scoutBeePhase()
            self.population.sort(key=cmpfun, reverse=True)  # 按照适应度排序，前一半为采蜜蜂，后一半为跟随蜂
            if self.population[0].fitness > self.best.fitness:
                self.best = copy.deepcopy(self.population[0])  # 最优适应度位置
                self.best_fit.append(self.best.fitness)
            else:
                self.best_fit.append(self.best.fitness)
            print("Generation %d: optimal function value is: %f;" % (
                self.t, self.best.fitness))
        print "Optimal solution is:"
        print self.best.chrom
        self.printResult()
    def printResult(self):
        '''
        plot the result of abs algorithm
        '''
        color_b, color_g, color_r = cv2.split(self.object2)
        sp = self.object.shape
        m = sp[0]
        n = sp[1]
        cenx = self.best.chrom[0]
        ceny = self.best.chrom[1]
        a = self.best.chrom[2]
        b = self.best.chrom[3]
        num = int(2 * math.pi / 0.001)
        angle = self.best.chrom[4]
        '''value = 0
        x0 = 0
        y0 = 0'''
        for i in xrange(num):
            x1 = int(a * math.sin(0.001 * i))
            y1 = int(b * math.cos(0.001 * i))
            x2 = int(cenx + int(x1 * math.cos(angle) + y1 * math.sin(angle)))
            y2 = int(ceny + int(-x1 * math.sin(angle) + y1 * math.cos(angle)))
            '''if (x2 != x0 or y2 != y0) and (x2 < m and x2 >= 0 and y2 < n and y2 >= 0):  # 保证和上次算出的边缘点不一样且没有出界

                if canny1[x2, y2] > 0:
                    # value = value + math.sqrt(np.square(x1) + np.square(y1))#累加一次对中心的边缘势场,忽略了常量
                    value = value + 1
            x0 = x2
            y0 = y2'''
            if x2 < m and y2 + 1 < n:
                color_r[x2][y2] = 255
                color_r[x2][y2 + 1] = 255
                color_r[x2][y2 - 1] = 255
                color_g[x2][y2] = 0
                color_g[x2][y2 + 1] = 0
                color_g[x2][y2 - 1] = 0
                color_b[x2][y2] = 0
                color_b[x2][y2 + 1] = 0
                color_b[x2][y2 - 1] = 0
        img2 = cv2.merge([color_r, color_g, color_b])
        plt.subplot(1, 2, 1)
        plt.imshow(self.object, 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.show()


if __name__ == "__main__":
    img_color = cv2.imread('circle.jpeg', cv2.IMREAD_COLOR)  #彩色图
    img = cv2.imread('circle.jpeg', 0)  # 直接读为灰度图像
    sp = img.shape
    m = sp[0]                        #获取图片的维度
    n = sp[1]
    blur = cv2.medianBlur(img, 5)  # 中值滤波
    canny = cv2.Canny(img, 50, 100)  #canny边缘二值化
    canny1 = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if canny[i][j] == 255:
                min1 = np.max([0, i - 5])
                max1 = np.min([m - 1, i + 5])
                min2 = np.max([0, j - 5])
                max2 = np.min([n - 1, j + 5])
                for k1 in xrange(min1, max1):
                    for k2 in xrange(min2, max2):
                        canny1[k1][k2] = 255

    bound1 =np.array([[200,150,20,20,0],[300,250,40,40,0.6*math.pi]])         #原图
    bound = np.array([[300,200,300,200,0],[m-300,n-200,500,400,0.6*math.pi]]) #鸡蛋图，由于椭圆的对称性，我们只考虑旋转角度在【0，pi】
    abs = ArtificialBeeSwarm(50, 5, bound, 200, 20,canny1,img_color)
    abs.solve()
