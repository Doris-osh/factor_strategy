import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 读取北京因子题factor_ret .csv数据
class FactorData:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.get_data()
    def get_data(self):
        data = pd.read_csv(self.filename, header=0, index_col=0)
        data = pd.DataFrame(data)
        data.dropna(how='all', inplace=True)
        data.index = pd.to_datetime(data.index, format='%Y%m%d')
        return data

# 编写策略，指定风险厌恶系数，追求一定风险约束下的收益最大化
class Strategy:
    file_data = FactorData('北京因子题factor_ret .csv')
    factor_data = file_data.data
    length = len(factor_data)
    def __init__(self, risk_aversion, industry_con, style_con, window, half):
        self.risk_aversion = risk_aversion
        self.industry_con = industry_con
        self.style_con = style_con
        self.window = window
        self.half = half
        self.factor_ewma = self.get_ewma()
    def optimize_fun(self, w, moving_weight_alpha_return, F_cov_matrix):
        factors_return = np.dot(w, moving_weight_alpha_return)
        risk_punish = self.risk_aversion * np.dot(np.dot(w, F_cov_matrix), w)
        return risk_punish - factors_return
    '''def half_period_weight(self, N):
        a = sum(list(map(lambda x:pow(0.5, x/self.half), [i for i in range(1,N+1)])))
        return list(map(lambda x:pow(0.5, x/self.half)/a, [i for i in range(N,0,-1)]))
    def get_moving_weight_alpha_return(self, t):
        current_window = min(t, self.window)
        return np.dot(self.half_period_weight(current_window), self.factor_data.iloc[t-current_window:t])'''
    def get_ewma(self):
        factor_ewma = pd.DataFrame(columns=self.factor_data.columns, index=self.factor_data.index)
        for factor in self.factor_data.columns:
            factor_ewma[factor] = self.factor_data[factor].ewm(halflife=self.half).mean()
        return factor_ewma
    def get_F_cov_matrix(self, t):
        current_window = min(t, self.window)
        #moving_weight_alpha_return = self.get_moving_weight_alpha_return(current_window)
        #for i in range(len(self.factor_data.columns))
        F_cov_matrix = pd.DataFrame(columns=self.factor_data.columns, index=self.factor_data.columns)
        for column1 in self.factor_data.columns:
            for column2 in self.factor_data.columns:
                F_cov_matrix.loc[column1, column2] = self.factor_data[column1].ewm(halflife=self.half).cov(self.factor_data[column2]).iloc[-1]
        return F_cov_matrix
    def cal_optimize_weight(self, w0, t):
        #n = len(self.factor_data.columns)
        #w0 = [1/n] * n
        moving_weight_alpha_return = self.factor_ewma.iloc[t]
        F_cov_matrix = self.get_F_cov_matrix(t)
        args = (moving_weight_alpha_return, F_cov_matrix)
        cons_industry1 = list(map(lambda x:{'type':'ineq', 'fun':lambda w:w[x]+self.industry_con}, [i for i in range(11,40)]))
        cons_industry2 = list(map(lambda x:{'type':'ineq', 'fun':lambda w:-w[x]+self.industry_con}, [i for i in range(11,40)]))
        cons_style = [{'type':'ineq', 'fun':lambda w:sum(w[:10])+self.style_con},{'type':'ineq', 'fun':lambda w:-sum(w[:10])+self.style_con}]
        cons_sumw = [{'type':'eq', 'fun':lambda w:sum(abs(x) for x in w)-1}]
        cons = cons_industry1 + cons_industry2 + cons_style + cons_sumw
        result = minimize(self.optimize_fun, w0, args=args, method='SLSQP', constraints=cons)
        return [-result.fun, result.x]

class Position:
    def __init__(self, strategy, init_total):
        self.init_total = init_total
        self.strategy = strategy
        position_columns = list(Strategy.factor_data.columns) + ['portfolio_value', 'cash', 'daily_return', 'daily_turnover', 'daily_turnover_rate']
        self.position = pd.DataFrame(columns=position_columns, index=Strategy.factor_data.index)
        self.weight = pd.DataFrame(columns=(list(Strategy.factor_data.columns)+['optimize_fun']), index=Strategy.factor_data.index)
    def execute_strategy(self):
        # 初始化因子权重：等权
        self.weight.iloc[0, :40] = 1 / 40
        w0 = self.weight.iloc[0,:40].tolist()
        # 初始化因子仓位：等仓位
        self.position.iloc[0, :40] = self.init_total * 1 / 40
        # 初始化资产组合价值：初始资金
        self.position.iloc[0, 40] = self.init_total
        # 初始化现金：0
        self.position.iloc[0, 41] = 0
        # 初始化收益、换手：0
        self.position.iloc[0, 42:45] = 0
        # 执行策略
        for t in range(1, Strategy.length):
            # 开盘后表现
            # 根据最新一日因子收益率，更新因子持仓价值
            self.position.iloc[t, :40] = np.multiply(self.position.iloc[t - 1, :40].to_numpy(),Strategy.factor_data.iloc[t, :40].to_numpy() + 1)
            # 更新总资产组合价值
            self.position.iloc[t, 40] = self.position.iloc[t, :40].sum() + self.position.iloc[t - 1, 41]
            # 计算资产组合日收益率
            self.position.iloc[t, 42] = self.position.iloc[t, 40] / self.position.iloc[t - 1, 40] - 1
            # 收盘后调仓
            # 优化因子权重
            optimize_res = self.strategy.cal_optimize_weight(w0, t)
            # 记录每日优化后权重与优化函数值
            self.weight.iloc[t,:40] = optimize_res[1]
            self.weight.iloc[t,40] = optimize_res[0]
            w0 = optimize_res[1]
            print(str(self.position.index[t]) + ': ' + str(self.position.iloc[t, 40]))
            print(self.position.iloc[t, :40].tolist())
            # 根据优化后的因子权重，调整仓位
            self.position.iloc[t,:40] = pd.Series(w0) * self.position.iloc[t,40]
            # 调整现金余额
            trade_value = (self.position.iloc[t,:40] - self.position.iloc[t-1,:40]).sum()
            self.position.iloc[t,41] = self.position.iloc[t-1,41] - trade_value
            # 计算资产组合换手
            self.position.iloc[t,43] = (self.position.iloc[t,:40]-self.position.iloc[t-1,:40]).abs().sum()
            self.position.iloc[t,44] = self.position.iloc[t,43] / self.position.iloc[t-1,:40].abs().sum()
        return 0
    def backtest(self):
        self.execute_strategy()
        self.position.to_csv('position_log.csv')
        self.weight.to_csv('weight_log.csv')
        return 0

strategy = Strategy(0.2,0.05,0.2,30,10)
position = Position(strategy,1000000)
position.backtest()
