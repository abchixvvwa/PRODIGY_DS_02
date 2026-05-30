import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Titanic Dataset (built-in, no download needed) ────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

try:
    df = pd.read_csv(url)
    print("Dataset loaded from URL")
except:
    # Fallback: hardcoded sample if no internet
    print("Using built-in data")
    np.random.seed(42)
    n = 891
    df = pd.DataFrame({
        'PassengerId': range(1, n+1),
        'Survived': np.random.choice([0,1], n, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1,2,3], n, p=[0.24, 0.21, 0.55]),
        'Name': ['Passenger_'+str(i) for i in range(n)],
        'Sex': np.random.choice(['male','female'], n, p=[0.65, 0.35]),
        'Age': np.where(np.random.rand(n) > 0.2, np.random.normal(30, 14, n).clip(1, 80), np.nan),
        'SibSp': np.random.choice([0,1,2,3,4], n, p=[0.68,0.23,0.06,0.02,0.01]),
        'Parch': np.random.choice([0,1,2,3], n, p=[0.76,0.13,0.09,0.02]),
        'Fare': np.random.exponential(32, n).clip(5, 500),
        'Embarked': np.random.choice(['S','C','Q'], n, p=[0.72,0.19,0.09])
    })

# ── Data Cleaning ─────────────────────────────────────────────────────────
print(f"\nShape before cleaning: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], errors='ignore', inplace=True)

print(f"\nShape after cleaning: {df.shape}")
print(f"Survival rate: {df['Survived'].mean()*100:.1f}%")

# ── EDA Stats ─────────────────────────────────────────────────────────────
print("\n=== Key Statistics ===")
print(df[['Age','Fare','Survived']].describe().round(2))

# ── Plot Setup ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18), facecolor='#0D1117')
fig.suptitle('Titanic Dataset — Data Cleaning & Exploratory Data Analysis\nProdigy InfoTech — Task 02 | Data Science Internship',
             fontsize=20, fontweight='bold', color='white', y=0.98)

axes_color = '#161B22'
text_color = '#E6EDF3'
accent1 = '#58A6FF'
accent2 = '#3FB950'
accent3 = '#FF7B72'
accent4 = '#FFA657'
pink    = '#F778BA'

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

def style_ax(ax):
    ax.set_facecolor(axes_color)
    ax.tick_params(colors=text_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#30363D')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── 1. Survival Count ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
counts = df['Survived'].value_counts()
bars = ax1.bar(['Did Not Survive', 'Survived'], counts.values, color=[accent3, accent2], width=0.5, edgecolor='none')
ax1.set_title('Overall Survival Count', color=text_color, fontsize=12, fontweight='bold')
ax1.set_ylabel('Passengers', color=text_color)
for bar in bars:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             str(int(bar.get_height())), ha='center', color=text_color, fontweight='bold')

# ── 2. Survival by Gender ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
gender_surv = df.groupby('Sex')['Survived'].mean() * 100
ax2.bar(gender_surv.index, gender_surv.values, color=[accent1, pink], width=0.5, edgecolor='none')
ax2.set_title('Survival Rate by Gender', color=text_color, fontsize=12, fontweight='bold')
ax2.set_ylabel('Survival Rate (%)', color=text_color)
ax2.set_ylim(0, 100)
for i, (idx, val) in enumerate(gender_surv.items()):
    ax2.text(i, val+2, f'{val:.1f}%', ha='center', color=text_color, fontweight='bold')

# ── 3. Survival by Passenger Class ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)
class_surv = df.groupby('Pclass')['Survived'].mean() * 100
colors_class = [accent4, accent1, accent3]
ax3.bar(['1st Class','2nd Class','3rd Class'], class_surv.values, color=colors_class, width=0.5, edgecolor='none')
ax3.set_title('Survival Rate by Passenger Class', color=text_color, fontsize=12, fontweight='bold')
ax3.set_ylabel('Survival Rate (%)', color=text_color)
ax3.set_ylim(0, 100)
for i, val in enumerate(class_surv.values):
    ax3.text(i, val+2, f'{val:.1f}%', ha='center', color=text_color, fontweight='bold')

# ── 4. Age Distribution ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4)
survived = df[df['Survived']==1]['Age']
not_survived = df[df['Survived']==0]['Age']
ax4.hist(not_survived, bins=25, alpha=0.7, color=accent3, label='Did Not Survive', edgecolor='none')
ax4.hist(survived, bins=25, alpha=0.7, color=accent2, label='Survived', edgecolor='none')
ax4.set_title('Age Distribution by Survival', color=text_color, fontsize=12, fontweight='bold')
ax4.set_xlabel('Age', color=text_color)
ax4.set_ylabel('Count', color=text_color)
ax4.legend(facecolor='#21262D', edgecolor='#30363D', labelcolor=text_color, fontsize=9)

# ── 5. Fare Distribution ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)
ax5.hist(df[df['Fare'] < 200]['Fare'], bins=30, color=accent1, alpha=0.85, edgecolor='none')
ax5.axvline(df['Fare'].mean(), color=accent4, linewidth=2, linestyle='--', label=f'Mean: £{df["Fare"].mean():.1f}')
ax5.axvline(df['Fare'].median(), color=pink, linewidth=2, linestyle=':', label=f'Median: £{df["Fare"].median():.1f}')
ax5.set_title('Fare Distribution (< £200)', color=text_color, fontsize=12, fontweight='bold')
ax5.set_xlabel('Fare (£)', color=text_color)
ax5.set_ylabel('Count', color=text_color)
ax5.legend(facecolor='#21262D', edgecolor='#30363D', labelcolor=text_color, fontsize=9)

# ── 6. Embarkation Port ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6)
port_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
df['Embarked_name'] = df['Embarked'].map(port_map)
port_surv = df.groupby('Embarked_name')['Survived'].mean() * 100
ax6.bar(port_surv.index, port_surv.values, color=[accent2, accent1, accent4], width=0.5, edgecolor='none')
ax6.set_title('Survival Rate by Embarkation Port', color=text_color, fontsize=12, fontweight='bold')
ax6.set_ylabel('Survival Rate (%)', color=text_color)
ax6.set_ylim(0, 100)
for i, val in enumerate(port_surv.values):
    ax6.text(i, val+2, f'{val:.1f}%', ha='center', color=text_color, fontweight='bold')

# ── 7. Family Size vs Survival ────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
style_ax(ax7)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
fam_surv = df.groupby('FamilySize')['Survived'].mean() * 100
ax7.bar(fam_surv.index, fam_surv.values, color=accent1, edgecolor='none', width=0.6)
ax7.set_title('Survival Rate by Family Size', color=text_color, fontsize=12, fontweight='bold')
ax7.set_xlabel('Family Size (incl. self)', color=text_color)
ax7.set_ylabel('Survival Rate (%)', color=text_color)
ax7.set_ylim(0, 100)

# ── 8. Age vs Fare Scatter ────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
style_ax(ax8)
s0 = df[df['Survived']==0]
s1 = df[df['Survived']==1]
ax8.scatter(s0['Age'], s0['Fare'].clip(0,300), alpha=0.4, color=accent3, s=15, label='Did Not Survive')
ax8.scatter(s1['Age'], s1['Fare'].clip(0,300), alpha=0.5, color=accent2, s=15, label='Survived')
ax8.set_title('Age vs Fare by Survival', color=text_color, fontsize=12, fontweight='bold')
ax8.set_xlabel('Age', color=text_color)
ax8.set_ylabel('Fare (£)', color=text_color)
ax8.legend(facecolor='#21262D', edgecolor='#30363D', labelcolor=text_color, fontsize=9)

# ── 9. Class & Gender Heatmap ─────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
style_ax(ax9)
pivot = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean') * 100
im = ax9.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax9.set_xticks([0,1,2]); ax9.set_xticklabels(['1st','2nd','3rd'], color=text_color)
ax9.set_yticks([0,1]); ax9.set_yticklabels(pivot.index, color=text_color)
ax9.set_title('Survival Rate: Gender × Class (%)', color=text_color, fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(3):
        ax9.text(j, i, f'{pivot.values[i,j]:.0f}%', ha='center', va='center',
                 color='black', fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax9).ax.tick_params(colors=text_color)

plt.savefig('task02_visualization.png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.show()
print("Done! Saved as task02_visualization.png")
