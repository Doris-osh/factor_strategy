import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

position_data = pd.read_csv('position_log.csv', header=0, index_col=0)
position_data = pd.DataFrame(position_data)

# 绘制累计收益率折线图
position_data['cum_ret'] = position_data['portfolio_value'] / position_data.iloc[0, 40]
plt.plot(position_data.index, position_data['cum_ret'], linestyle='-')
plt.title('Cumulative Returns Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.show()

# 求最大回撤
peak_ret = position_data.iloc[0,45]
max_drawdown = 0
for i in range(len(position_data)):
    peak_ret = max(peak_ret, position_data.iloc[i, 45])
    max_drawdown = max(peak_ret - position_data.iloc[i, 45], max_drawdown)
# 求年化收益率、年化波动率
annual_ret = pow(1 + position_data['daily_return'].mean(), 252) - 1
annual_volatility = position_data['daily_return'].std() * np.sqrt(252)
# 求年化换手率
total_turnover = position_data['daily_turnover_rate'].sum()
annul_turnover = total_turnover * 252 / len(position_data)
# 求年化夏普比率
shape_ratio = np.sqrt(252) * position_data['daily_return'].mean() / position_data['daily_return'].std()

print('最大回撤：' + str(max_drawdown), '\n年化收益率：' + str(annual_ret), '\n年化波动率：' + str(annual_volatility), '\n年化换手率：' + str(annul_turnover), '\n年化夏普比率：' + str(shape_ratio))

'''    def get_F_cov_matrix(self, t):
        current_window = min(t, self.window)
        #moving_weight_alpha_return = self.get_moving_weight_alpha_return(current_window)
        #for i in range(len(self.factor_data.columns))
        F_cov_matrix = pd.DataFrame(columns=self.factor_data.columns, index=self.factor_data.columns)
        for column1 in self.factor_data.columns:
            for column2 in self.factor_data.columns:
                F_cov_matrix.loc[column1, column2] = self.factor_data[column1].iloc[t-current_window+1:t+1].ewm(halflife=self.half).cov(self.factor_data[column2]).iloc[-1]
        return F_cov_matrix'''
