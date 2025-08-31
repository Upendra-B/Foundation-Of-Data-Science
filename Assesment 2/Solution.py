# Solution.py
# Project: Investigation A - Bat vs Rat Behaviour
# Author: Your Team
# Date: 2025

# 1. Import libraries
import pandas as pd
import numpy as np
import statistics
from scipy import stats
from statsmodels.stats import proportion
import matplotlib.pyplot as plt

# 2. Load datasets
df1 = pd.read_csv("../Datasets/dataset1.csv")
df2 = pd.read_csv("../Datasets/dataset2.csv")

# 3. Explore & clean
print(df1.head())
print(df2.head())

# ---- Descriptive stats ----
# Risk-taking vs avoidance
risk_counts = df1['risk'].value_counts()
print("Risk counts:\n", risk_counts)

# Avoidance rate
avoidance_rate = (df1['risk'] == 0).mean()
print("Avoidance rate:", avoidance_rate)

# ---- Inferential stats ----
# Proportion CI
count_risk = (df1['risk'] == 1).sum()
n = len(df1)
ci_low, ci_high = proportion.proportion_confint(count_risk, n, alpha=0.05, method='normal')
print("Risk-taking 95% CI:", ci_low, ci_high)

# T-test delay (risk vs avoid)
delay_risk = df1[df1['risk'] == 1]['bat_landing_to_food']
delay_avoid = df1[df1['risk'] == 0]['bat_landing_to_food']
t_stat, p_val = stats.ttest_ind(delay_risk, delay_avoid, equal_var=False)
print("T-test delay risk vs avoid:", t_stat, p_val)

# ---- Correlation test (Dataset 2) ----
corr, p_val = stats.pearsonr(df2['rat_minutes'], df2['bat_landing_number'])
print("Correlation rats vs bats:", corr, "p-value:", p_val)

# ---- Plots ----
df1['risk'].value_counts().plot(kind='bar', title="Bat Risk vs Avoidance")
plt.show()
