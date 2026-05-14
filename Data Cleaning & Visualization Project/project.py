# =====================================================
# DATA CLEANING & VISUALIZATION PROJECT
# TITANIC DATASET
# =====================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_excel("titanic_dataset.xlsx")

print("========== FIRST 5 ROWS ==========")
print(df.head())

# -----------------------------
# BASIC INFORMATION
# -----------------------------

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# =====================================================
# DATA CLEANING
# =====================================================

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

# Fill missing Age values using median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values using mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column because it has many missing values
df.drop(columns=['Cabin'], inplace=True)

print("\n========== AFTER HANDLING MISSING VALUES ==========")
print(df.isnull().sum())

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

duplicates = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Duplicates Removed Successfully!")

# -----------------------------
# HANDLE OUTLIERS
# -----------------------------

# Using IQR Method for Fare column

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_limit) &
        (df['Fare'] <= upper_limit)]

print("\nOutliers Removed Successfully!")

print("\nDataset Shape After Cleaning:")
print(df.shape)

# =====================================================
# DATA VISUALIZATION
# =====================================================

sns.set(style="whitegrid")

# -----------------------------
# 1. SURVIVAL COUNT
# -----------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='Survived', data=df)

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.show()

# -----------------------------
# 2. GENDER VS SURVIVAL
# -----------------------------

plt.figure(figsize=(7,5))

sns.countplot(x='Sex',
              hue='Survived',
              data=df)

plt.title("Gender vs Survival")

plt.show()

# -----------------------------
# 3. AGE DISTRIBUTION
# -----------------------------

plt.figure(figsize=(8,5))

sns.histplot(df['Age'],
             bins=30,
             kde=True)

plt.title("Age Distribution")

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# -----------------------------
# 4. PASSENGER CLASS DISTRIBUTION
# -----------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='Pclass', data=df)

plt.title("Passenger Class Distribution")

plt.xlabel("Passenger Class")
plt.ylabel("Count")

plt.show()

# -----------------------------
# 5. FARE DISTRIBUTION
# -----------------------------

plt.figure(figsize=(8,5))

sns.boxplot(x=df['Fare'])

plt.title("Fare Distribution")

plt.show()

# -----------------------------
# 6. CORRELATION HEATMAP
# -----------------------------

plt.figure(figsize=(10,7))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# =====================================================
# SAVE CLEANED DATASET
# =====================================================

df.to_excel("cleaned_titanic_dataset.xlsx",
            index=False)

print("\nCleaned Dataset Saved Successfully!")

# =====================================================
# KEY INSIGHTS
# =====================================================

print("\n========== KEY INSIGHTS ==========")

print("1. Female passengers had a higher survival rate.")
print("2. First-class passengers survived more than others.")
print("3. Most passengers were between 20-40 years old.")
print("4. Higher fare passengers had better survival chances.")
print("5. Data cleaning improved dataset quality.")

# =====================================================
# END OF PROJECT
# =====================================================