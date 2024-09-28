from GAN_model import GAN
from tournament import tournament
from spea2_env import *
from PM_mutation import pm_mutation
from spea2_env import environment_selection


class GMOEA(object):
    def __init__(self, decs=None, gp=None):
        self.decs = decs
        self.gp = gp
        self.w_max = 5
        self.k = self.gp.n
        self.lr = 0.0001
        self.batch_size = 8
        self.epoch = 200
        self.n_noise = self.gp.d

    def run(self):
        pro = self.gp.pro
        if self.decs is None:
            population = pro.fit('init', self.gp.n)
        else:
            population = pro.fit(in_value=self.decs)

        evaluated = np.shape(population[0])[0]
        score = [[evaluated, pro.IGD(population[1]["F"])]]
        net = GAN(self.gp.d, self.batch_size, self.lr, self.epoch, self.n_noise)

        while evaluated <= self.gp.eva:
            _, index = environment_selection(population, self.k)
            ref_dec = population[0][index, :]
            pool = ref_dec / np.tile(pro.upper, (self.k, 1))

            label = np.zeros((self.gp.n, 1))
            label[index, :] = 1
            pop_dec = population[0]
            input_dec = (pop_dec - np.tile(pro.lower, (np.shape(pop_dec)[0], 1))) / \
                        np.tile(pro.upper - pro.lower, (np.shape(pop_dec)[0], 1))
            net.train(input_dec, label, pool)

            for i in range(self.w_max):
                if 1 - (i/self.w_max)**2 > np.random.random(1):
                    off = net.generate(ref_dec / np.tile(pro.upper, (np.shape(ref_dec)[0], 1)), self.gp.n) * \
                          np.tile(pro.upper, (self.gp.n, 1))
                    off = pm_mutation(off, [self.gp.lower, self.gp.upper])
                else:
                    fitness = cal_fit(population[1]["F"])
                    mating = tournament(k_size=2, n_size=self.gp.n, fit=fitness.reshape((len(fitness), 1)))
                    off = self.gp.operator(population[0][mating, :], boundary=[pro.lower, pro.upper])

                offspring = pro.fit(in_value=off)
                evaluated += np.shape(offspring[0])[0]

                # 合并 population 确保维度一致
                population[0] = np.vstack([population[0], offspring[0]])
                population[1]["F"] = np.vstack([population[1]["F"], offspring[1]["F"]])
                population[1]["G"] = np.vstack([population[1]["G"], offspring[1]["G"]])

                population, _ = environment_selection(population, self.gp.n)
                score.append([evaluated, pro.IGD(population[1]["F"])])

        return population, score
