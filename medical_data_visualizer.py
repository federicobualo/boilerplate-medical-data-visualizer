import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def draw_cat_plot():
    # 1. Import data
    df = pd.read_csv("medical_examination.csv")

    # 2. Add 'overweight' column
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
    df['overweight'] = (df['BMI'] > 25).astype(int)

    # 3. Normalize cholesterol and gluc (0 = good, 1 = bad)
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # 4 & 5. Melt dataframe
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and reformat
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Create catplot
    fig = sns.catplot(
        x='variable', y='total', hue='value',
        col='cardio', data=df_cat, kind='bar'
    ).fig

    # 8. Do not modify
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():

    # 1. Import data
    df = pd.read_csv("medical_examination.csv")

    # 2. Add overweight
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
    df['overweight'] = (df['BMI'] > 25).astype(int)

    # Normalize
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # 3. Data cleaning for heatmap
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 4. Correlation matrix
    corr = df_heat.corr()

    # 5. Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 6. Set up figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # 7. Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        center=0,
        vmax=0.3,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5}
    )

    # 8. Do not modify
    fig.savefig('heatmap.png')
    return fig
