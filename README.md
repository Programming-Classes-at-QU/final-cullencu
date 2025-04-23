# Cullens FINAL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Seed for reproducibility
np.random.seed(42)

# Create a DataFrame with realistic rugby player data
players = [
    "Antoine Dupont", "Ardie Savea", "Cheslin Kolbe", "Eben Etzebeth", "Marcus Smith",
    "Beauden Barrett", "Romain Ntamack", "Maro Itoje", "Siya Kolisi", "Finn Russell",
    "Aaron Smith", "Richie Mo'unga", "James Ryan", "Caelan Doris", "Damian Penaud",
    "Freddie Steward", "Ellis Genge", "Tadhg Furlong", "Tom Curry", "Sam Cane",
    "Pieter-Steph Du Toit", "Will Jordan", "Josh van der Flier", "Courtney Lawes", "Duhan van der Merwe"
]

positions = [
    "Back", "Forward", "Back", "Forward", "Kicker",
    "Kicker", "Kicker", "Forward", "Forward", "Kicker",
    "Back", "Kicker", "Forward", "Forward", "Back",
    "Back", "Forward", "Forward", "Forward", "Forward",
    "Forward", "Back", "Forward", "Forward", "Back"
]

data = pd.DataFrame({
    "Player": players,
    "Position": positions,
    "Tries": np.random.poisson(6, size=25),
    "Tackles": np.random.normal(100, 20, size=25).astype(int),
    "Meters_Gained": np.random.normal(400, 100, size=25).astype(int),
    "Kicking_Accuracy": np.random.normal(75, 10, size=25).clip(50, 100)
})

# Display the first few rows of the data
print("Data Head:")
print(data.head())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Visualization 1: Average tackles per player
plt.figure(figsize=(12, 6))
sns.barplot(data=data.sort_values('Tackles', ascending=False), x='Player', y='Tackles')
plt.xticks(rotation=90)
plt.title('Tackles per Player')
plt.tight_layout()
plt.show()

# T-test: Meters gained between Forwards and Backs
forward_mg = data[data['Position'] == 'Forward']['Meters_Gained']
back_mg = data[data['Position'] == 'Back']['Meters_Gained']
t_stat, p_value = stats.ttest_ind(forward_mg, back_mg)
print("T-test result (Forwards vs. Backs - Meters Gained):")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# ANOVA: Tries by Position
anova_result = stats.f_oneway(
    data[data['Position'] == 'Forward']['Tries'],
    data[data['Position'] == 'Back']['Tries'],
    data[data['Position'] == 'Kicker']['Tries']
)
print("\nANOVA result (Tries by Position):")
print(f"F-statistic: {anova_result.statistic:.3f}, p-value: {anova_result.pvalue:.3f}")

# Correlation test: Kicking Accuracy and Meters Gained
corr, corr_p = stats.pearsonr(data['Kicking_Accuracy'], data['Meters_Gained'])
print("\nCorrelation between Kicking Accuracy and Meters Gained:")
print(f"Correlation coefficient: {corr:.3f}, p-value: {corr_p:.3f}")

# Visualization 2: Box plot of tries by position
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='Position', y='Tries')
plt.title('Distribution of Tries by Position')
plt.show()

# Visualization 3: Scatter plot of Kicking Accuracy vs Meters Gained
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Kicking_Accuracy', y='Meters_Gained', hue='Position')
plt.title('Kicking Accuracy vs Meters Gained')
plt.show()
