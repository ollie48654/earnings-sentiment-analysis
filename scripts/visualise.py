import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

EXPORTS = Path(__file__).parent.parent / 'exports'
DOCS    = Path(__file__).parent.parent / 'docs'
DOCS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150
})

def chart_sentiment_comparison():
    vader = pd.read_csv(EXPORTS / 'vader_by_company.csv')
    bert  = pd.read_csv(EXPORTS / 'bert_by_company.csv')
    
    merged = pd.merge(vader[['ticker','avg_compound']],
                      bert[['ticker','bert_avg_compound']], on='ticker')
    merged = merged.sort_values('bert_avg_compound', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(merged))
    width = 0.35
    
    ax.barh(x - width/2, merged['avg_compound'], width,
            label='VADER', color='#185FA5', alpha=0.8)
    ax.barh(x + width/2, merged['bert_avg_compound'], width,
            label='FinBERT', color='#E07B39', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(merged['ticker'])
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Sentiment Score (negative ← 0 → positive)')
    ax.set_title('Earnings Call Sentiment by Company VADER vs FinBERT',
                 fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(DOCS / 'sentiment_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved sentiment_comparison.png")

def chart_sentiment_vs_returns():
    df = pd.read_csv(EXPORTS / 'sentiment_vs_returns.csv')
    if 'return_3d_pct' not in df.columns or df['return_3d_pct'].isna().all():
        print("No return data available — skipping scatter chart")
        return
    
    df = df.dropna(subset=['bert_compound','return_3d_pct'])
    
    colour_map = {'Positive':'#185FA5','Neutral':'#888888','Negative':'#D94F3D'}
    colours = df['bert_sentiment'].map(colour_map).fillna('#888888')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['bert_compound'], df['return_3d_pct'],
               c=colours, s=120, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    for _, row in df.iterrows():
        ax.annotate(row['ticker'],
                    (row['bert_compound'], row['return_3d_pct']),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_xlabel('FinBERT Sentiment Score')
    ax.set_ylabel('3-Day Stock Return After Earnings (%)')
    ax.set_title('Earnings Sentiment vs Subsequent Stock Return(FinBERT)',
                 fontsize=14, fontweight='bold')
    
    patches = [mpatches.Patch(color=v, label=k) for k,v in colour_map.items()]
    ax.legend(handles=patches)
    plt.tight_layout()
    plt.savefig(DOCS / 'sentiment_vs_returns.png', bbox_inches='tight')
    plt.close()
    print("Saved sentiment_vs_returns.png")

def chart_model_agreement():
    vader = pd.read_csv(EXPORTS / 'vader_by_company.csv')
    bert  = pd.read_csv(EXPORTS / 'bert_by_company.csv')
    merged = pd.merge(vader[['ticker','avg_compound','overall_sentiment']],
                      bert[['ticker','bert_avg_compound','bert_sentiment']],
                      on='ticker')
    
    merged['agree'] = merged['overall_sentiment'] == merged['bert_sentiment']
    agreement_pct = merged['agree'].mean() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    for ax, col, title in [
        (axes[0], 'overall_sentiment', 'VADER'),
        (axes[1], 'bert_sentiment', 'FinBERT')
    ]:
        counts = merged[col].value_counts()
        colours = {'Positive':'#185FA5','Neutral':'#888888','Negative':'#D94F3D'}
        bar_colours = [colours.get(l,'#888888') for l in counts.index]
        ax.bar(counts.index, counts.values, color=bar_colours, alpha=0.85)
        ax.set_title(f'{title} Sentiment Distribution', fontweight='bold')
        ax.set_ylabel('Number of Companies')
    
    fig.suptitle(f'Model Agreement: {agreement_pct:.0f}% of companies classified the same by both models',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DOCS / 'model_agreement.png', bbox_inches='tight')
    plt.close()
    print("Saved model_agreement.png")

if __name__ == '__main__':
    chart_sentiment_comparison()
    chart_sentiment_vs_returns()
    chart_model_agreement()
    print("All charts generated.")