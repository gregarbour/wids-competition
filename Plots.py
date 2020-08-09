import pandas as pd
import seaborn as sns
import os

os.chdir('/Users/gregarbour/Desktop/WiDS Competition/')
df = pd.read_csv('exploratory_df.csv')
lab_vars = pd.Series([var for var in df.columns if 'h1' in var or 'd1' in var])

#Grab a random set of 10 lab value variables to construct correlation plot
df_cor = df[lab_vars.sample(15)]

corr = df_cor.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#Correlation plot with random subset of 20 bloor pressure variables
bp_vars = pd.Series([var for var in df.columns if 'bp' in var])
df_cor2 = df[bp_vars.sample(15)]

corr2 = df_cor2.corr()
ax = sns.heatmap(
    corr2, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
