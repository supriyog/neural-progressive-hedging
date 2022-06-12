import numpy as np

class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, num_stocks, steps=730, trading_cost=0.0025, time_cost=0.0, init_weights='cash'):
        #self.asset_names = asset_names
        self.num_stocks = num_stocks
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.init_weights = init_weights

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        p0 = self.p0

        dw1 = (y1 * self.weights) / (np.dot(y1, self.weights) ) #+ eps)  # (eq7) weights evolve into

        mu1 = 1 - self.cost * (np.abs(dw1[1:] - w1[1:])).sum()  # (eq16) cost to change portfolio

        #iterative for 5 steps
        for i in range(5):
            mu1 = (1-self.cost*dw1[0]-(2*self.cost-self.cost*self.cost)*np.sum(np.maximum(0, dw1[1:]-mu1*w1[1:]))) /(1-self.cost*w1[0])

        #assert 1-mu1 < 1.0, 'Cost is larger than current holding'

        self.p1_prime= p0 * np.dot(y1, self.weights)

        p1 = mu1 * self.p1_prime  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log(p1 /p0)  # log rate of return
        #r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        #reward = r1 / self.steps * 1000.  # (22) average logarithmic accumulated return
        reward = r1
        # remember for next step
        self.p0 = p1
        self.weights = w1 #new weights

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "prev_portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0
        if self.init_weights == 'cash':
            self.weights = np.concatenate(([1.0],np.zeros(self.num_stocks))) #need previous weights
        elif self.init_weights == 'eq':
            self.weights = np.ones(self.num_stocks+1) / (self.num_stocks+1)
        else:
            raise
