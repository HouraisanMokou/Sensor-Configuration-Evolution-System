# add support for geatpy

import numpy as np
import geatpy as ea

import time

from geatpy.algorithms.soeas.DE.DE_currentToBest_1_L.soea_DE_currentToBest_1_L_templet import \
    soea_DE_currentToBest_1_L_templet


class SensorConfigurationProblem(ea.Problem):
    def __init__(self, dim, varTypes, lb, ub):
        name = "sensor_configuration"
        M = 1
        maxormins = [-1]
        super().__init__(
            name,
            M,
            maxormins,
            dim,
            varTypes,
            lb,
            ub
        )
        self.fitness_dict = {}
        # self.manager = solver
        self.pop_buffer = None
        self.fitness_buffer = None
        self.Fields = None
        self.kth = 6

    def set_kth(self, kth):
        self.kth = 6

    def set_Fields(self, Fields):
        self.Fields = Fields

    def update_buffer(self, pops, fitness):
        for pop, f in zip(pops, fitness):
            pop = np.array(pop).astype('float')
            pop = (pop - self.Fields[1, :]) / (self.Fields[0, :] - self.Fields[1, :])
            self.fitness_buffer = np.array([f]) if self.fitness_buffer is None else \
                np.hstack([self.fitness_buffer, np.array([f])])
            self.pop_buffer = pop if self.pop_buffer is None else \
                np.vstack([self.pop_buffer, pop])
        if len(self.fitness_buffer) > 200:# 500:
            mu = np.mean(self.fitness_buffer)
            sigma = np.std(self.fitness_buffer)
            lb = mu - 3 * sigma
            ub = mu + 3 * sigma
            idx = np.argwhere((self.fitness_buffer > lb) & (self.fitness_buffer < ub)).squeeze()
            self.fitness_buffer = self.fitness_buffer[idx]
            self.pop_buffer = self.pop_buffer[idx, :]

    def gaussian_distribution(self, x, sigma=1 / 3):
        exp = np.exp(-1 / (2 * (sigma ** 2)) * (x ** 2))
        return exp

    def evalVars(self, Vars):
        kth = self.kth
        fs = []
        for pop in Vars:
            pop = np.array(pop).squeeze()
            norm_pop = (pop - self.Fields[1, :]) / (self.Fields[0, :] - self.Fields[1, :])
            dis = np.linalg.norm(self.pop_buffer - norm_pop, 2, axis=1)
            idxes = np.argpartition(dis, kth=kth)[:kth]
            neigh_fitness = self.fitness_buffer[idxes]
            neigh_dis = dis[idxes]
            distributions = self.gaussian_distribution(neigh_dis)
            final_fitness = np.sum(neigh_fitness * distributions) / np.sum(distributions)
            self.fitness_dict[tuple(list(pop))] = final_fitness
            fs.append(final_fitness)
        res = np.array(fs)
        res = np.expand_dims(res, 1)
        return res


class DE_currentToBest_1_L_online(soea_DE_currentToBest_1_L_templet):
    def __init__(self,
                 problem,
                 population,
                 F=None,
                 CR=None,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/current-to-best/1/L'
        if population.Encoding == 'RI':
            F = F if F is not None else 0.3
            CR = CR if CR is not None else 0.6
            self.mutOper = ea.Mutde(F=F)  # 生成差分变异算子对象
            self.muOpter2 = ea.Mutde(F=F / 2)
            self.muOpter3 = ea.Mutde(F=F / 3)  # F delay
            self.recOper = ea.Xovexp(XOVR=CR, Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')
        self.experiment_pop = None

    def setup(self, prophetPop=None):
        population = self.population
        self.NIND = population.sizes
        NIND = population.sizes * 2
        self.initialization()
        population.initChrom(NIND)
        self.population = population
        return self.population

    def run_online(self):
        population = self.population if self.experiment_pop is None else self.population + self.experiment_pop
        NIND = self.NIND
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度

        # select
        population = population[ea.selecting('otos', population.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        # choose_index = ea.tour(population.FitnV, NIND, 2)
        # population=population[choose_index] # tour select

        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)
        self.population = population
        self.problem.evaluation(self.population)
        self.terminated(self.population)

        # mute and crossover
        r0 = np.arange(NIND)
        r_best = ea.selecting('ecs', population.FitnV, NIND)  # 执行'ecs'精英复制选择
        experimentPop = ea.Population(population.Encoding, population.Field, NIND)
        mutOper = self.mutOper
        if self.currentGen >= 10:
            mutOper = self.muOpter2
        if self.currentGen >= 20:
            mutOper = self.muOpter3
        experimentPop.Chrom = mutOper.do(population.Encoding, population.Chrom, population.Field,
                                         [r0, None, None, r_best, r0])  # 变异
        experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
        self.experiment_pop = experimentPop

        return experimentPop
