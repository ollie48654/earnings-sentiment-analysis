import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROC_DIR    = Path(__file__).parent.parent / 'data' / 'processed'
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
EXPORTS_DIR.mkdir(exist_ok=True)

def score_text(text, analyser):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
    return analyser.polarity_scores(text)

def run_vader(df):
    analyser = SentimentIntensityAnalyzer()
    
    # Determine which column has the text
    text_col = None
    for col in ['text', 'transcript', 'body', 'content', 'answer']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        text_col = df.columns[-1]
    
    print(f"Scoring text from column: {text_col}")
    
    scores = df[text_col].apply(lambda t: score_text(t, analyser))
    scores_df = pd.DataFrame(list(scores))
    
    df['vader_negative']  = scores_df['neg'].round(4)
    df['vader_neutral']   = scores_df['neu'].round(4)
    df['vader_positive']  = scores_df['pos'].round(4)
    df['vader_compound']  = scores_df['compound'].round(4)
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'Positive' if x >= 0.05
                  else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    return df

def aggregate_by_company(df):
    group_col = 'ticker' if 'ticker' in df.columns else df.columns[0]
    
    agg = (df.groupby(group_col)
             .agg(
                 avg_compound=('vader_compound', 'mean'),
                 avg_positive=('vader_positive', 'mean'),
                 avg_negative=('vader_negative', 'mean'),
                 num_passages=('vader_compound', 'count'),
                 pct_positive=('vader_sentiment',
                               lambda x: (x == 'Positive').mean() * 100),
                 pct_negative=('vader_sentiment',
                               lambda x: (x == 'Negative').mean() * 100)
             )
             .reset_index())
    
    agg = agg.rename(columns={group_col: 'ticker'})
    agg['avg_compound']  = agg['avg_compound'].round(4)
    agg['avg_positive']  = agg['avg_positive'].round(4)
    agg['avg_negative']  = agg['avg_negative'].round(4)
    agg['pct_positive']  = agg['pct_positive'].round(1)
    agg['pct_negative']  = agg['pct_negative'].round(1)
    agg['overall_sentiment'] = agg['avg_compound'].apply(
        lambda x: 'Positive' if x >= 0.05
                  else ('Negative' if x <= -0.05 else 'Neutral')
    )
    return agg

if __name__ == '__main__':
    print("Loading transcripts...")
    df = pd.read_csv(PROC_DIR / 'raw_transcripts.csv').head(500)
    print(f"Loaded {len(df):,} rows")
    
    print("Running VADER sentiment scoring...")
    df = run_vader(df)
    df.to_csv(PROC_DIR / 'vader_scored.csv', index=False)
    print("Saved vader_scored.csv")
    
    print("Aggregating by company...")
    agg = aggregate_by_company(df)
    agg.to_csv(EXPORTS_DIR / 'vader_by_company.csv', index=False)
    print("Saved vader_by_company.csv")
    
    print("Top 5 most positive companies:")
    print(agg.nlargest(5, 'avg_compound')[['ticker','avg_compound','overall_sentiment']])
    print("Top 5 most negative companies:")
    print(agg.nsmallest(5, 'avg_compound')[['ticker','avg_compound','overall_sentiment']])