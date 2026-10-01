import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROC_DIR    = Path(__file__).parent.parent / 'data' / 'processed'
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'

def load_and_merge():
    print("Loading scored transcripts...")
    bert_df = pd.read_csv(PROC_DIR / 'bert_scored.csv')
    
    print("Loading stock returns...")
    returns = pd.read_csv(PROC_DIR / 'stock_returns.csv')
    
    # Aggregate sentiment by ticker
    group_col = 'ticker' if 'ticker' in bert_df.columns else bert_df.columns[0]
    
    sent_agg = (bert_df.groupby(group_col)
                .agg(
                    vader_compound=('vader_compound', 'mean'),
                    bert_compound=('bert_compound', 'mean'),
                    bert_positive=('bert_positive', 'mean'),
                    bert_negative=('bert_negative', 'mean'),
                    num_passages=(group_col, 'count')
                )
                .reset_index()
                .rename(columns={group_col: 'ticker'}))
    
    merged = pd.merge(sent_agg, returns, on='ticker', how='inner')
    print(f"Merged dataset: {len(merged)} companies")
    return merged

def analyse_correlation(df):
    print("Correlation Analysis:")
    print("="*50)
    
    for sentiment_col, label in [
        ('vader_compound', 'VADER Compound'),
        ('bert_compound',  'FinBERT Compound')
    ]:
        if sentiment_col not in df.columns:
            continue
        
        clean = df[[sentiment_col, 'return_3d_pct']].dropna()
        if len(clean) < 3:
            print(f"{label}: insufficient data")
            continue
        
        r, p = stats.pearsonr(clean[sentiment_col], clean['return_3d_pct'])
        print(f"{label} vs 3-day stock return:")
        print(f"  Pearson r = {r:.4f}")
        print(f"  p-value   = {p:.4f}")
        print(f"  {'Statistically significant (p<0.05)' if p < 0.05 else 'Not significant — small sample'}")
    
    return df

def categorise_and_export(df):
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'Positive' if x >= 0.05
                  else ('Negative' if x <= -0.05 else 'Neutral')
    )
    df['bert_sentiment'] = df['bert_compound'].apply(
        lambda x: 'Positive' if x > 0.1
                  else ('Negative' if x < -0.1 else 'Neutral')
    )
    df['model_agree'] = df['vader_sentiment'] == df['bert_sentiment']
    
    df.to_csv(EXPORTS_DIR / 'sentiment_vs_returns.csv', index=False)
    print("Saved sentiment_vs_returns.csv")
    
    print("Full results table:")
    cols = ['ticker','vader_compound','bert_compound',
            'bert_sentiment','return_3d_pct']
    available = [c for c in cols if c in df.columns]
    print(df[available].sort_values('bert_compound', ascending=False).to_string())
    
    return df

if __name__ == '__main__':
    df = load_and_merge()
    df = analyse_correlation(df)
    df = categorise_and_export(df)
    print("Correlation analysis complete.")