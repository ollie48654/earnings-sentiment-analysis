import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

PROC_DIR    = Path(__file__).parent.parent / 'data' / 'processed'
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'

MODEL_NAME = "ProsusAI/finbert"

def load_finbert():
    print(f"Loading FinBERT model from Hugging Face...")
    print("(First run downloads ~440MB — subsequent runs load from cache)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    return tokenizer, model

def score_text(text, tokenizer, model, max_length=512):
    if not isinstance(text, str) or len(text.strip()) < 10:
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34,
                'label': 'neutral', 'score': 0.0}
    
    # Truncate long text to fit BERT's 512 token limit
    text = text[:1000]
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=max_length, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = softmax(outputs.logits, dim=1).squeeze().numpy()
    
    # FinBERT output order: positive, negative, neutral
    labels = ['positive', 'negative', 'neutral']
    label  = labels[probs.argmax()]
    score  = float(probs[probs.argmax()])
    
    # Convert to a -1 to +1 compound score similar to VADER
    compound = float(probs[0] - probs[1])
    
    return {
        'bert_positive': round(float(probs[0]), 4),
        'bert_negative': round(float(probs[1]), 4),
        'bert_neutral':  round(float(probs[2]), 4),
        'bert_label':    label,
        'bert_score':    round(score, 4),
        'bert_compound': round(compound, 4)
    }

def run_finbert(df, tokenizer, model):
    results = []
    text_col = None
    for col in ['text', 'transcript', 'body', 'content', 'answer']:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[-1]
    
    print(f"Scoring {len(df)} passages with FinBERT...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = score_text(row[text_col], tokenizer, model)
        results.append(result)
    
    scores_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True),
                      scores_df.reset_index(drop=True)], axis=1)

def aggregate_by_company(df):
    group_col = 'ticker' if 'ticker' in df.columns else df.columns[0]
    
    agg = (df.groupby(group_col)
             .agg(
                 bert_avg_compound=('bert_compound', 'mean'),
                 bert_avg_positive=('bert_positive', 'mean'),
                 bert_avg_negative=('bert_negative', 'mean'),
                 num_passages=('bert_compound', 'count')
             )
             .reset_index())
    
    agg = agg.rename(columns={group_col: 'ticker'})
    for col in ['bert_avg_compound','bert_avg_positive','bert_avg_negative']:
        agg[col] = agg[col].round(4)
    
    agg['bert_sentiment'] = agg['bert_avg_compound'].apply(
        lambda x: 'Positive' if x > 0.1
                  else ('Negative' if x < -0.1 else 'Neutral')
    )
    return agg

if __name__ == '__main__':
    print("Loading VADER-scored transcripts...")
    df = pd.read_csv(PROC_DIR / 'vader_scored.csv')
    
    tokenizer, model = load_finbert()
    
    print("Running FinBERT scoring...")
    df = run_finbert(df, tokenizer, model)
    df.to_csv(PROC_DIR / 'bert_scored.csv', index=False)
    print("Saved bert_scored.csv")
    
    print("Aggregating by company...")
    agg = aggregate_by_company(df)
    agg.to_csv(EXPORTS_DIR / 'bert_by_company.csv', index=False)
    print("Saved bert_by_company.csv")
    
    print("FinBERT results:")
    print(agg[['ticker','bert_avg_compound','bert_sentiment']].to_string())