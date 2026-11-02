import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
os.makedirs("build", exist_ok=True)

file_path = "data/processed/events_dataset.parquet"

#print(f"Loading a sample from {file_path} for fast exploratory analysis...")
df = pd.read_parquet(file_path)
if len(df) > 2_000_000:
    df = df.sample(n=2_000_000, random_state=42)

print(f"Working with {len(df):,} sampled interactions.")

plt.figure(figsize=(10, 6))
top_cats = df['category'].value_counts().head(10)
sns.barplot(x=top_cats.values, y=top_cats.index, hue=top_cats.index, palette='crest', legend=False)
plt.title('Top 10 Product Categories by Interaction Volume', fontsize=14, fontweight='bold')
plt.xlabel('Interaction Count', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.savefig('build/top_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: build/top_categories.png")


df_sorted = df.sort_values(by=['user_id', 'timestamp'])
df_sorted['next_category'] = df_sorted.groupby('user_id')['category'].shift(-1)

top_6 = df['category'].value_counts().head(6).index.tolist()
filtered_transitions = df_sorted[
    df_sorted['category'].isin(top_6) & df_sorted['next_category'].isin(top_6)
]
print("Processing timestamps for temporal trends...")
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df['year_month'] = df['datetime'].dt.to_period('M').astype(str)

transition_matrix = pd.crosstab(
    filtered_transitions['category'], 
    filtered_transitions['next_category'], 
    normalize='index' # Shows probabilities
)

plt.figure(figsize=(9, 7))
sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.title('Category Transition Probabilities (Current -> Next)', fontsize=14, fontweight='bold')
plt.xlabel('Next Category', fontsize=12)
plt.ylabel('Current Category', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig('build/category_transitions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: build/category_transitions.png")

print("\nAll graphs successfully created")
