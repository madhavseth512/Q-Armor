import pandas as pd

df = pd.read_csv('data/CICIOT23/train/train.csv')

print("="*70)
print("1. CLASS DISTRIBUTION")
print("="*70)
print(df['label'].value_counts())
print("\nTotal rows:", len(df))
print("Unique labels:", df['label'].nunique())

print("\n" + "="*70)
print("2. PER-CLASS MEANS OF FEATURE COLUMNS")
print("="*70)
cols = ['Rate','syn_count','rst_count','fin_count','Std','AVG',
        'Header_Length','Covariance','IAT','urg_count','ack_count',
        'TCP','UDP','ICMP','ARP','Telnet','HTTP','HTTPS','DNS']

# Check which columns actually exist
available_cols = [c for c in cols if c in df.columns]
print(f"\nNote: {len(cols)} columns requested, {len(available_cols)} found in dataset")
if len(cols) != len(available_cols):
    print(f"Missing: {set(cols) - set(available_cols)}")
print(f"Available: {available_cols}\n")

df['parent'] = df['label'].str.split('-').str[0]
print(df.groupby('parent')[available_cols].mean().round(2).T)

print("\n" + "="*70)
print("3. OUTLIER / RANGE CHECK")
print("="*70)
range_cols = ['Rate','Header_Length','Covariance','IAT','AVG','Std']
available_range = [c for c in range_cols if c in df.columns]
print(df[available_range].describe(percentiles=[.5,.95,.99]).round(2))

print("\n" + "="*70)
print("4. CORRELATION AMONG FEATURE-SOURCE COLUMNS")
print("="*70)
corr_cols = ['Rate','syn_count','rst_count','fin_count','Std','AVG',
             'Header_Length','Covariance']
available_corr = [c for c in corr_cols if c in df.columns]
print(df[available_corr].corr().round(2))

print("\n" + "="*70)
print("5. PARENT-CLASS EXTRACTION")
print("="*70)
print(df['parent'].value_counts())
