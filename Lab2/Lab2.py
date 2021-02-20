# Варіант - 311 (x1_min - 10, x1_max = 60, x2_min = -30, x2_max = 45)
import numpy as np


class Experiment:
    def __init__(self,
                 x1_min_max,
                 x2_min_max,
                 y_min_max,
                 m):
        self.R_critical_values = {
            5: 2,
            6: 2.16,
            7: 2.3,
            8: 2.43,
            9: 2.5
        }
        self.x1_min_max = x1_min_max
        self.x2_min_max = x2_min_max
        self.y_min_max = y_min_max

        self.plan_matrix = np.array(
            [np.random.randint(*self.x1_min_max, size=3),
             np.random.randint(*self.x2_min_max, size=3)]).T

        self.x0 = [np.mean(self.x1_min_max), np.mean(self.x2_min_max)]

        self.normalized_matrix = self.create_normalized_plan_matrix()

        self.m = m

        self.experiment()
        self.b = self.find_b()
        self.a = self.find_a()

        self.check_b = self.check_coeffs_B()
        self.check_a = self.check_coeffs_A()

    def create_normalized_plan_matrix(self):
        self.N = self.plan_matrix.shape[0]
        self.k = self.plan_matrix.shape[1]

        interval_of_change = [self.x1_min_max[1] - self.x0[0],
                              self.x2_min_max[1] - self.x0[1]]
        x_normalized = [
            [(self.plan_matrix[i, j] - self.x0[j]) / interval_of_change[j]
             for j in range(self.k)]
            for i in range(self.N)
        ]
        return np.array(x_normalized)

    def experiment(self):
        self.y_matrix = np.random.randint(*self.y_min_max, size=(3, self.m))
        self.y_mean = np.mean(self.y_matrix, axis=1)

        self.y_var = np.var(self.y_matrix, axis=1)
        self.sigma = np.sqrt((2 * (2 * self.m - 2)) / (self.m * (self.m - 4)))

        if not self.check_r():  # К-сть експерементів збільшуємо лише тоді, коли дисперсія неоднорідна
            print(f'\n Дисперсія неоднорідна, змінимо m={self.m} на m={self.m+1}\n') # при однорідній залишаємо все як є
            self.m += 1
            self.experiment()



    def check_r(self):
        for i in range(len(self.y_var)):
            for j in range(len(self.y_var)):
                if i > j:
                    if self.y_var[i] >= self.y_var[j]:
                        R = (abs((self.m - 2) * self.y_var[i] /
                             (self.m * self.y_var[j]) - 1) / self.sigma)
                    else:
                        R = (abs((self.m - 2) * self.y_var[j] /
                             (self.m * self.y_var[i]) - 1) / self.sigma)
                    if R > self.R_critical_values[self.m]:
                        print('Variance isn\'t stable!')
                        return False
        return True

    def check_coeffs_B(self):
        return np.array([
            (self.b[0] + np.sum(self.b[1:3] * self.normalized_matrix[i]))
            for i in range(self.N)])

    def check_coeffs_A(self):
        return np.array([
            (self.a[0] + np.sum(self.a[1:3] * self.plan_matrix[i]))
            for i in range(self.N)])

    def find_b(self):
        mx1 = np.mean(self.normalized_matrix[:, 0])
        mx2 = np.mean(self.normalized_matrix[:, 1])

        a1 = np.mean(self.normalized_matrix[:, 0] ** 2)
        a2 = np.mean(self.normalized_matrix[:, 0] * self.normalized_matrix[:, 1])
        a3 = np.mean(self.normalized_matrix[:, 1] ** 2)

        my = np.mean(self.y_mean)
        a11 = np.mean(self.normalized_matrix[:, 0] * self.y_mean)
        a22 = np.mean(self.normalized_matrix[:, 1] * self.y_mean)

        b = np.linalg.solve([[1, mx1, mx2],
                             [mx1, a1, a2],
                             [mx2, a2, a3]],
                            [my, a11, a22])
        return b

    def find_a(self):
        delta_of_x = [abs(self.x1_min_max[1] - self.x1_min_max[0]) / 2,
                   abs(self.x2_min_max[1] - self.x2_min_max[0]) / 2]
        a = [(self.b[0] - self.b[1] * self.x0[0] / delta_of_x[0] -
              self.b[2] * self.x0[1] / delta_of_x[1]),
             self.b[1] / delta_of_x[0],
             self.b[2] / delta_of_x[1]]
        return np.array(a)

    def check_results(self):
        print('\nМатриця Y:\n', self.y_matrix )
        print('\nМатриця планування:\n', self.plan_matrix)
        print('\nНормована матриця:\n', self.normalized_matrix)
        print('\nНормовані коефіцієнти:     ', self.b)
        print('Натуралізовані коефіцієнти:', self.a)
        print('\nY середнє:                             ', self.y_mean)
        print('Перевірка нормованих коефіцієнтів:     ', self.check_b)
        print('Перевірка натуралізованих коефіцієнтів:', self.check_a)


if __name__ == '__main__':
    m = 5
    variant = 11
    x1_min_max = [10, 60]
    x2_min_max = [-30, 45]
    y_min_max = [((20 - variant) * 10), ((30 - variant) * 10)]
    experiment = Experiment(x1_min_max, x2_min_max, y_min_max, m)
    experiment.check_results()
