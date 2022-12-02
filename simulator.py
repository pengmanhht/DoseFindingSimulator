import numpy as np
import statsmodels.api as sm
from scipy.integrate import odeint


class KineticsCalculator():
    __slots__ = ('time', 'conc')

    def __init__(self, time, conc):
        self.time = np.asarray(time)
        self.conc = np.asarray(conc)

    def __str__(self):
        return f'Time:\n {self.time}\nConcentration:\n {self.conc}'

    # Cmax
    def cmax(self):
        return max(self.conc)

    # tmax
    def tmax(self):
        return self.time[np.nanargmax(self.conc)]

    # AUC
    def conc_time_check(self, start=None, end=None):
        # drop nan conc (BQL)
        time = np.delete(self.time, np.where(np.isnan(self.conc)))
        conc = np.delete(self.conc, np.where(np.isnan(self.conc)))

        # check specific time range
        if start is None:
            start = self.time.min()
        if end is None:
            end = self.time.max()

        # define time range based on start and end
        conc = conc[np.where((time >= start) & (time <= end))]
        time = time[np.where((time >= start) & (time <= end))]
        return conc, time

    def auc_lin_trap(self, start=None, end=None):

        conc, time = self.conc_time_check(start=start, end=end)

        # calculation of area under curve
        auc = 0
        for i in range(len(time) - 1):
            auc += 0.5 * (conc[i + 1] + conc[i]) * (time[i + 1] - time[i])
        return auc

    def auc_log_trap(self, start=None, end=None):

        conc, time = self.conc_time_check(start=start, end=end)

        # calculation of area under curve
        auc = 0
        for i in range(len(time) - 1):
            auc += (conc[i] - conc[i + 1]) * (time[i + 1] - time[i]) / (np.log(conc[i]) - np.log(conc[i + 1]))
        return auc

    def auc_lin_log(self, start=None, end=None):

        conc, time = self.conc_time_check(start=start, end=end)

        # calculation of area under curve
        auc = 0
        for i in range(len(time) - 1):
            if conc[i] <= conc[i + 1]:
                auc += 0.5 * (conc[i] + conc[i + 1]) * (time[i + 1] - time[i])
            else:
                auc += (conc[i] - conc[i + 1]) * (time[i + 1] - time[i]) / (np.log(conc[i]) - np.log(conc[i + 1]))
        return auc

    # lambda_z: terminal elimination rate constant
    def lambda_z(self):
        time = np.delete(self.time, np.where(np.isnan(self.conc)))
        conc = np.delete(self.conc, np.where(np.isnan(self.conc)))
        log_conc = np.log(conc)
        log_conc_max = max(log_conc)
        res = []
        for i in range(3, 11):
            x = time[-i:]
            x = sm.add_constant(x)
            y = log_conc[-i:]
            if (y < log_conc_max).all():
                fit = sm.OLS(y, x).fit()
                res.append([i, fit.rsquared_adj, fit.params[1]])
            else:
                # continue
                res.append([i, 0, np.nan])
        arr = np.array(res)
        indx = np.argmax(arr[:, 1])
        return arr[indx, 2] * (-1)

    # TODO: visual check for lambda_z
    # def lambda_z_vcheck(self):
    #     pass

    # terminal half-life:
    def t_half(self):
        return np.log(2) / self.lambda_z()

    # AUCextra
    def auc_extra(self, last_obs=1):
        # drop nan conc (BQL)
        # time = np.delete(self.time, np.where(np.isnan(self.conc)))
        conc = np.delete(self.conc, np.where(np.isnan(self.conc)))
        return conc[-2 + last_obs] / self.lambda_z()

    # CL/F apparent clearance
    def apparent_clearance(self, dose):
        return dose / (self.auc_extra() + self.auc_lin_log())


class ModelSimulator:
    def __init__(self, model=None, amount=None, interval=None, vd=1, cpt=2, **kwargs):
        if isinstance(amount, np.ndarray):
            self.amount = amount
        else:
            self.amount = np.asarray(amount).reshape((1,))
        if isinstance(interval, np.ndarray):
            self.interval = interval
        else:
            self.interval = np.asarray(interval).reshape((1,))
        self.Vd = vd
        self.cpt = cpt
        self.conc = self.amount / self.Vd
        self.model = model
        self.params = kwargs

    def solve(self, p0=None, t0=None):
        cpt = self.cpt
        t = np.asarray([0])
        c = np.asarray(np.concatenate([np.array([self.amount[0] / self.Vd]), np.zeros(cpt - 1)])).reshape(1, cpt)
        params = tuple(val for val in self.params.values())
        t1 = 0
        for i in range(len(self.amount)):
            if i == 0:
                p0 = np.concatenate([np.array([self.amount[i] / self.Vd]), np.zeros(cpt - 1)])
                t0 = np.linspace(0, self.interval[i], 300)
            else:
                t1 = t1 + self.interval[i - 1]
                t2 = t1 + self.interval[i]
                p0 = c0[-1] + np.concatenate([np.array([self.amount[i] / self.Vd]), np.zeros(cpt - 1)])
                t0 = np.linspace(t1, t2, 300)

            c0 = odeint(self.model, p0, t0, args=params)
            t = np.concatenate([t, t0])
            c = np.concatenate([c, c0])
        return {
            "time_point": t,
            "conc": c
        }

    def __add__(self, other):
        if isinstance(other, ModelSimulator):
            amount = np.concatenate([self.amount, other.amount])
            interval = np.concatenate([self.interval, other.interval])
            new = ModelSimulator(self.model, amount, interval, self.Vd, **self.params)
            return new

    def __mul__(self, num):
        new = self
        if isinstance(num, int):
            new.amount = np.repeat(self.amount, num)
            new.interval = np.repeat(self.interval, num)
        return new


def pk_model(a0, t, ka, cl, vc, vp, q, f):
    a, ac, ap = a0
    dd_dt = - ka * a * f
    dc_dt = ka * a * f - cl / vc * ac - q / vc * ac + q / vp * ap
    dp_dt = q / vc * ac - q / vp * ap
    return [dd_dt, dc_dt, dp_dt]

