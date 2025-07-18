import matplotlib.pyplot as plt

from benchstreet import util

dataset = util.getDataFrame()

plt.figure(figsize=(15, 5))

plt.title('S&P 500 Price Over Time (2005 - 2025)')
plt.xlabel('Date')
plt.ylabel('Price')

plt.plot(util.getPrice(dataset), label='S&P 500 Price')
plt.grid(True)
plt.legend()
plt.show()