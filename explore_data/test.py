# for VSCode
import os
import sys

TOP_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(TOP_DIR))

import pandas as pd
import matplotlib.pyplot as plt
from config import CORE_DATA_DIR


def plot_data(df, column_index, figsize=(25, 20), title=None):
    df1 = df.iloc[:, column_index]
    df1.plot(linewidth=0.5, figsize=figsize, title=title)
    plt.show()


# google
file_data_google = CORE_DATA_DIR + '/input_data/google_trace/1_job/10_mins.csv'
df = pd.read_csv(file_data_google, header=None)
meanCPUUsage, canonical_memory_usage = 3, 4

# plot_data(df, meanCPUUsage, title='Google Trace: mean CPU usage')
# plot_data(df, canonical_memory_usage, title='Google Trace: canonical memory usage')

df2 = df.iloc[:, [3, 4]]
df2 = (df2 - df2.min()) / (df2.max() - df2.min())
df2.plot(linewidth=0.5, figsize=(25, 10))
plt.show()

# box plot
# df.boxplot(column=[3])
# plt.show()


# Azure
# file_data_azure = CORE_DATA_DIR + '/input_data/azure/5_mins.csv'
# df = pd.read_csv(file_data_azure, header=None)
# plot_data(df, 1)
# plt.show()
