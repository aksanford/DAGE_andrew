import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Paths
DATA_PATH = r'c:\Users\grant\Projects_PSYCH_755\final_project\DAGE\data\clean\PRCA_FileC_clean.csv'
OUT_DIR = r'c:\Users\grant\Projects_PSYCH_755\final_project\DAGE\data\clean\figures'
os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style='whitegrid', palette='muted')

STOPWORDS = {
    'a','about','all','also','am','an','and','are','as','at','be','been','being','but','by','can','could','did','do','does','doing','don','dont','for','from','get','go','got','had','has','have','he','her','him','his','how','i','if','in','is','it','its','just','like','ll','m','me','more','much','my','no','not','of','on','one','or','our','out','over','re','s','she','so','some','t','than','that','the','their','them','then','there','they','this','to','too','up','us','use','used','ve','very','was','we','were','what','when','where','which','who','will','with','would','you','your'
}

# --- Numeric/extraction mappings from clean_data.qmd ---
LIKERT_FORWARD = {
    'Strongly disagree': 1,
    'Somewhat disagree': 2,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 4,
    'Strongly agree': 5,
}
LIKERT_REVERSE = {
    'Strongly disagree': 5,
    'Somewhat disagree': 4,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 2,
    'Strongly agree': 1,
}

FORWARD_CODE = ['Q1', 'Q3', 'Q5', 'Q13', 'Q15', 'Q18']
REVERSE_CODE = ['Q2', 'Q4', 'Q6', 'Q14', 'Q16', 'Q17']
PRCA_COLS = FORWARD_CODE + REVERSE_CODE

GROUP_DISCUSSION = ['Q1', 'Q2', 'Q3', 'Q4', 'Q6']
SOLO_CONVERSATION = ['Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18']

USAGE_MONTHLY = {
    'Never': 0,
    '0-1 days a month': 1,
    '2-4 days a month': 3,
    '4-8 days a month': 6,
    '8 or more days a month': 8,
}
USAGE_DAILY = {
    '1-2 rides in a typical day': 1.5,
    '3-4 rides in a typical day': 3.5,
    '5-6 rides in a typical day': 5.5,
    '7 or more rides in a typical day': 7,
}
TRANSPORT_COLS = ['Q26', 'Q27', 'Q28', 'Q29']


def tokenize(text):
    '''Lowercase, strip punctuation, and return word tokens.'''
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [t.strip("'") for t in text.split() if t.strip("'")]
    return tokens


def extract_classes(df):
    '''Apply the PRCA and transport class extraction from clean_data.qmd.'''
    df = df.copy()
    for col in FORWARD_CODE:
        df[col] = df[col].map(LIKERT_FORWARD)
    for col in REVERSE_CODE:
        df[col] = df[col].map(LIKERT_REVERSE)
    df['group_apprehension'] = df[GROUP_DISCUSSION].mean(axis=1, skipna=True)
    df['solo_apprehension'] = df[SOLO_CONVERSATION].mean(axis=1, skipna=True)
    fear_cond = [df['group_apprehension'] > df['solo_apprehension'],
                 df['solo_apprehension'] > df['group_apprehension']]
    fear_choices = ['Group Discussion', 'Solo Conversation']
    df['biggest_fear'] = np.select(fear_cond, fear_choices, default='Equal')
    df['q26_num'] = df['Q26'].map(USAGE_MONTHLY)
    df['q27_num'] = df['Q27'].map(USAGE_DAILY)
    df['q28_num'] = df['Q28'].map(USAGE_MONTHLY)
    df['q29_num'] = df['Q29'].map(USAGE_DAILY)
    df['public_transit_count'] = df['q26_num'] * df['q27_num']
    df['rideshare_count'] = df['q28_num'] * df['q29_num']
    trans_cond = [df['public_transit_count'] > df['rideshare_count'],
                  df['rideshare_count'] > df['public_transit_count']]
    trans_choices = ['Public Transit', 'Rideshare']
    df['highest_transport'] = np.select(trans_cond, trans_choices, default='Equal')
    return df


def build_plot_df(df, text_col, class_col):
    '''Return a small DataFrame with text features and class labels.'''
    df = df.reset_index(drop=True)
    plot_df = pd.DataFrame()
    plot_df[class_col] = df[class_col]
    plot_df['tokens'] = df[text_col].astype(str).apply(tokenize)
    plot_df['word_count'] = plot_df['tokens'].apply(len)
    plot_df['avg_word_length'] = plot_df['tokens'].apply(lambda toks: np.mean([len(t) for t in toks]) if toks else np.nan)
    plot_df['unique_per_response'] = plot_df['tokens'].apply(lambda toks: len(set(toks)))
    return plot_df


def plot_text_stats_by_class(plot_df, class_col, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['word_count', 'avg_word_length', 'unique_per_response']
    ylabels = ['Words per response', 'Average word length (chars)', 'Unique words per response']
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        sns.violinplot(x=class_col, y=metric, hue=class_col, data=plot_df, inner='box', palette='muted', ax=ax, cut=0, legend=False)
        ax.set_xlabel(class_col.replace('_', ' ').title())
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def plot_common_words_by_class(df, text_col, class_col, n, title, filename):
    classes = sorted(df[class_col].dropna().unique())
    n_classes = len(classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 6), sharey=False)
    if n_classes == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        tokens = df.loc[df[class_col] == cls, text_col].astype(str).apply(tokenize)
        all_words = [t for toks in tokens for t in toks]
        content_words = [w for w in all_words if w not in STOPWORDS]
        counter = Counter(content_words)
        top = counter.most_common(n)
        if top:
            words, counts = zip(*top)
            y_pos = range(len(words))
            ax.barh(y_pos, counts, color='steelblue')
            ax.invert_yaxis()
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.set_xlabel('Frequency (stop words removed)')
        ax.set_title(f'{cls} (n={len(tokens)})')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def plot_unique_word_counts(df, text_col, class_col, title, filename):
    classes = sorted(df[class_col].dropna().unique())
    total_unique = []
    avg_unique = []
    for cls in classes:
        tokens = df.loc[df[class_col] == cls, text_col].astype(str).apply(tokenize)
        all_words = [t for toks in tokens for t in toks]
        total_unique.append(len(set(all_words)))
        per_resp = [len(set(toks)) for toks in tokens]
        avg = np.nanmean(per_resp) if per_resp else 0
        avg_unique.append(avg)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    bars = ax.bar(x, total_unique, color='teal')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Total unique words')
    ax.set_title(title)
    for bar, avg in zip(bars, avg_unique):
        height = bar.get_height()
        label = f'avg={avg:.1f}' if not np.isnan(avg) else ''
        ax.text(bar.get_x() + bar.get_width() / 2., height, label, ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def print_class_summary(df, text_col, class_col, question_label):
    print()
    print(f'=== {question_label} BY {class_col.upper()} ===')
    classes = sorted(df[class_col].dropna().unique())
    class_word_sets = {}
    for cls in classes:
        toks = df.loc[df[class_col] == cls, text_col].astype(str).apply(tokenize)
        class_word_sets[cls] = set(t for toks_list in toks for t in toks_list if t not in STOPWORDS)
    for cls in classes:
        subset = df[df[class_col] == cls]
        tokens = subset[text_col].astype(str).apply(tokenize)
        all_words = [t for toks in tokens for t in toks]
        content = [w for w in all_words if w not in STOPWORDS]
        counter = Counter(content)
        top_str = ', '.join([f'{w}({c})' for w, c in counter.most_common(5)])
        others = set()
        for other in classes:
            if other != cls:
                others |= class_word_sets[other]
        exclusive = sorted(class_word_sets[cls] - others)
        exclusive_str = ', '.join(exclusive[:15]) if exclusive else 'None'
        print()
        print(f'{cls} (n={len(subset)}):')
        print(f'  Avg words per response: {tokens.apply(len).mean():.2f}')
        print(f'  Avg word length: {tokens.apply(lambda toks: np.mean([len(t) for t in toks]) if toks else np.nan).mean():.2f}')
        print(f'  Total unique words: {len(set(all_words))}')
        print(f'  Top content words: {top_str}')
        print(f'  Words solely in this class (n={len(exclusive)}): {exclusive_str}')


def plot_exclusive_word_counts(df, text_col, class_col, title, filename):
    '''Plot the number of content words that appear only in each class.'''
    classes = sorted(df[class_col].dropna().unique())
    class_word_sets = {}
    for cls in classes:
        toks = df.loc[df[class_col] == cls, text_col].astype(str).apply(tokenize)
        class_word_sets[cls] = set(t for toks_list in toks for t in toks_list if t not in STOPWORDS)
    counts = []
    for cls in classes:
        others = set()
        for other in classes:
            if other != cls:
                others |= class_word_sets[other]
        counts.append(len(class_word_sets[cls] - others))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(np.arange(len(classes)), counts, color='coral')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel('Exclusive content words')
    ax.set_title(title)
    for bar, c in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, str(c), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)
    df = extract_classes(df)

    # Match the R filtering in clean_data.qmd / eda_dependents.qmd
    ap_df = df[df['Q18.1'].notna() & df[PRCA_COLS].notna().all(axis=1)].copy()
    tr_df = df[df['Q19'].notna() & df[TRANSPORT_COLS].notna().all(axis=1)].copy()

    print(f'Apprehension analysis sample: n={len(ap_df)}')
    print(f'Transportation analysis sample: n={len(tr_df)}')

    plot_df_ap = build_plot_df(ap_df, 'Q18.1', 'biggest_fear')
    plot_text_stats_by_class(plot_df_ap, 'biggest_fear',
                             'Apprehension Text Stats by Biggest Fear',
                             'apprehension_text_stats_by_class.png')
    plot_common_words_by_class(ap_df, 'Q18.1', 'biggest_fear', 10,
                               'Apprehension Common Words by Biggest Fear',
                               'apprehension_common_words_by_class.png')
    plot_unique_word_counts(ap_df, 'Q18.1', 'biggest_fear',
                            'Total Unique Words by Apprehension Type',
                            'apprehension_unique_words_by_class.png')
    plot_exclusive_word_counts(ap_df, 'Q18.1', 'biggest_fear',
                               'Exclusive Words by Apprehension Type',
                               'apprehension_exclusive_words_by_class.png')
    print_class_summary(ap_df, 'Q18.1', 'biggest_fear', 'APPREHENSION TEXT')

    plot_df_tr = build_plot_df(tr_df, 'Q19', 'highest_transport')
    plot_text_stats_by_class(plot_df_tr, 'highest_transport',
                             'Transport Text Stats by Dominant Method',
                             'transport_text_stats_by_class.png')
    plot_common_words_by_class(tr_df, 'Q19', 'highest_transport', 10,
                               'Transport Common Words by Dominant Method',
                               'transport_common_words_by_class.png')
    plot_unique_word_counts(tr_df, 'Q19', 'highest_transport',
                            'Total Unique Words by Transport Method',
                            'transport_unique_words_by_class.png')
    plot_exclusive_word_counts(tr_df, 'Q19', 'highest_transport',
                               'Exclusive Words by Transport Method',
                               'transport_exclusive_words_by_class.png')
    print_class_summary(tr_df, 'Q19', 'highest_transport', 'TRANSPORT TEXT')

    print()
    print(f'Plots saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
