# Plots in Google Trends Manuscript

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import us # for state names
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import os
import re

import warnings
warnings.filterwarnings("ignore")

## Reading data

# hospitalization data
hosp = pd.read_csv('data/01_raw/hospitalizations.csv')
max_date = '2024-04-27'
hosp = hosp[hosp['date'] <= max_date] # end date for 2023-2024 flu season
cutoff = '2022-10-01' # cutoff between train and test sets
hosp_train = hosp[hosp['date'] <= cutoff]

# clusters
clusters = pd.read_csv('data/03_preprocessed/cluster_gt.csv')
# clusters = clusters[clusters['date'] >= hosp['date'].min()]
clusters = clusters[clusters['date'] <= hosp['date'].max()]
clusters = clusters.reset_index(drop = True)

# denoised
smooth = pd.read_csv('data/03_preprocessed/smooth_gt.csv')
# smooth = smooth[smooth['date'] >= hosp['date'].min()]
smooth = smooth[smooth['date'] <= hosp['date'].max()]
smooth = smooth.reset_index(drop = True)

# detrended
detrend = pd.read_csv('data/03_preprocessed/detrend_gt.csv')
# detrend = detrend[detrend['date'] >= hosp['date'].min()]
detrend = detrend[detrend['date'] <= hosp['date'].max()]
detrend = detrend.reset_index(drop = True)

# all locations
geo = pd.read_csv('data/01_raw/geo.txt', header = None)[0]

fig_save = 'figures/'

## Figure 2: Example of search volumes over time for “Cough” in Alabama

clusters['date'] = pd.to_datetime(clusters['date'])
clusters.set_index('date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 100))
clusters_scaled = pd.DataFrame(scaler.fit_transform(clusters), columns=clusters.columns)
clusters_scaled.insert(0, 'date', clusters.index)
clusters_scaled.set_index('date', inplace=True)

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap

kw = 'US-AL_(TOPIC)01b_21'
fig = plt.figure(figsize=(7, 3))  
sns.lineplot(clusters_scaled[kw].astype(int), linewidth=1) 
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig2_example_volumes.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 3: Examples of issues in Google Trends search volumes for California and Alaska

state = 'US-CA'
indiv = pd.read_csv(f'data/01_raw/individual_merged_trends/{state}_individual_trends.csv')
indiv = indiv[indiv['date'] <= max_date] # end date for 2023-2024 flu season
indiv['date'] = pd.to_datetime(indiv['date'])
indiv.set_index('date', inplace=True)

# scaling data
scaler = MinMaxScaler(feature_range=(0, 100))
indiv_scaled = pd.DataFrame(scaler.fit_transform(indiv), columns=indiv.columns)
indiv_scaled.insert(0, 'date', indiv.index)
indiv_scaled.set_index('date', inplace=True)

### (a): Missing Values in "Influenza Treatment" in California

kw = 'influenza treatment'

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
# fig = plt.figure(figsize=(6.8, 3.5))  # width 6.8 in
fig = plt.figure(figsize=(4, 3))
# fig = plt.figure(figsize=(3.2, 2.2))
sns.lineplot(data=indiv_scaled[kw].astype(int), color='orange', linewidth=1) 
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# plt.legend(fontsize=9)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(4)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig3a_missing_values.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (b): Noise in "Acute Bronchitis" in California

kw = 'acute bronchitis'

sns.set(style='whitegrid') 
fig = plt.figure(figsize=(4, 3))  # width 6.8 in
# fig = plt.figure(figsize=(3.2, 2.2))
sns.lineplot(data=indiv_scaled[kw].astype(int), color='orange', linewidth=1) 
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(4)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig3b_noise.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (c): Trend in "Cough" in California

kw = '/m/01b_21'

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(4, 3))  # width 6.8 in
# fig = plt.figure(figsize=(3.2, 2.2))
sns.lineplot(data=indiv_scaled[kw].astype(int), color='orange', linewidth=1) 
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# plt.legend(fontsize=9)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(4)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig3c_trend.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (d): Sampling Variability across different downloads for "Common Cold" in Alaska

api_save = "data/01_raw/api_downloads/api_zeros/"
files = sorted([folder for folder in os.listdir(api_save) if folder.endswith(".csv")])
files

j = 20
sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap

fig = plt.figure(figsize=(5, 3))  # width 6.8 in
# fig = plt.figure(figsize=(3.2, 2.2))
for file in files[9:11]:
    df = pd.read_csv(f"{api_save}{file}")
    date = re.search(r'\d{4}-\d{2}-\d{2}', file).group()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df)  # Assuming all columns except 'date' are numeric
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    sns.lineplot(x=df_scaled.index[df_scaled.iloc[:, j].notna()], y=df_scaled.iloc[:, j][df_scaled.iloc[:, j].notna()].astype(int), linewidth=1, label = f"Download {date}")

plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=8, loc='upper left')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(4)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig3d_sampling_variability.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Table S1: Overall percentage of zeros across multiple downloads of Google Trends search volumes for the same 161 keywords across 52 locations.

dataframes = []

# Loop over each file
for i, file_name in enumerate(files):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f"{api_save}{file_name}")
    # Assign the DataFrame to a variable with a specific name
    globals()[f"df{i}"] = df
    # Append the DataFrame to the list
    dataframes.append(df)

zeros_ratios = []

# Loop over each DataFrame
for i in range(11):
    # Get the DataFrame corresponding to df0, df1, ..., df10
    df = globals()[f"df{i}"]
    # Calculate the number of zeros
    zeros_count = (df == 0).sum().sum()
    # Calculate the ratio
    zeros_ratio = zeros_count / (df.shape[0] * df.shape[1])
    # Append the ratio to the list
    zeros_ratios.append(zeros_ratio)

for file_name, zeros_ratio in zip(files, zeros_ratios):
    print(f"{file_name}: {np.round(zeros_ratio * 100)}% zeros")

table_s1_rows = []
for file_name, zeros_ratio in zip(files, zeros_ratios):
    match = re.search(r'\d{4}-\d{2}-\d{2}', file_name)
    table_s1_rows.append({
        'date': match.group() if match else file_name,
        'percentage': np.round(zeros_ratio * 100)
    })

table_s1_df = pd.DataFrame(table_s1_rows)
keep_dates = ['2024-02-11', '2024-02-25', '2024-03-10', '2024-03-24', '2024-04-07', '2024-04-21']
table_s1_df = table_s1_df[table_s1_df['date'].isin(keep_dates)].reset_index(drop=True)
table_s1_df.to_csv('tables/tableS1_percentage_zeros.txt', sep='\t', index=False)

## Figure 4: Example of changes in the percentage of zeros within different downloads of "Nasal Congestion" in Alaska from February to April 2024

selected_files = [files[i] for i in [0, 3, 9]]
selected_files

plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap

j = 6

# fig = plt.figure(figsize=(6.8, 3.5))  # width 6.8 in
# fig = plt.figure(figsize=(3.2, 2.2))
fig = plt.figure(figsize=(7, 3))
legend_labels = ["February 2024", "March 2024", "April 2024"]  # Custom labels
for idx, file in enumerate(selected_files):
    df = pd.read_csv(f"{api_save}{file}")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df)  # Assuming all columns except 'date' are numeric
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    num_zeros = int(np.round(df_scaled.iloc[:, j].isin([0]).mean()*100))
    sns.lineplot(x=df_scaled.index[df_scaled.iloc[:, j].notna()], y=df_scaled.iloc[:, j][df_scaled.iloc[:, j].notna()].astype(int), linewidth=1, label=f"{legend_labels[idx]}: {num_zeros}%")

    print(df_scaled.iloc[:, j].name)
    print('Percentage of zeros', round(df_scaled.iloc[:, j].isin([0]).mean(), 2)*100)

plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=8)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig4_updates.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 5: Example comparison of search volumes in Tennessee

state = 'US-TN'

indiv = pd.read_csv(f'data/01_raw/individual_merged_trends/{state}_individual_trends.csv')
indiv = indiv[indiv['date'] <= max_date] # end date for 2023-2024 flu season
indiv['date'] = pd.to_datetime(indiv['date'])
indiv.set_index('date', inplace=True)

# scaling data
scaler = MinMaxScaler(feature_range=(0, 100))
indiv_scaled = pd.DataFrame(scaler.fit_transform(indiv), columns=indiv.columns)
indiv_scaled.insert(0, 'date', indiv.index)
indiv_scaled.set_index('date', inplace=True)

# retrieving the clusters for Tennessee
with open(f'data/02_intermediate/hierarchical_clusters/{state}_hierarchical.txt', 'r') as file:
    all_terms = [line.strip() for line in file.readlines()]

idx = 6 # cluster 6 which contains 3 terms
terms = all_terms[idx % len(all_terms)]
print("Terms in cluster 6 for Tennessee:", terms)
terms_list = terms.split(' + ')
terms_list = [term.replace('/m/', '(TOPIC)') for term in terms_list]
first_two_terms = ' + '.join(terms_list[:2])
sub = clusters_scaled[state + '_' + first_two_terms]
print(sub)

### (a) Individual (large values truncated)

# Process terms in the cluster
cluster_terms = terms.split(' + ')
c1 = [item if not item.startswith(state + '_') else item for item in cluster_terms]
c1 = [item.strip() for item in c1]
num_terms = len(c1)

cols = 3
rows = 1
# fig, axes = plt.subplots(rows, cols, figsize=(6.8, 2.5), sharey=True)
fig, axes = plt.subplots(rows, cols, figsize=(9, 3), sharey=True)
sns.set(style='whitegrid') 

for i in range(len(c1)):
    ax = axes[i]
    truncated_data = np.minimum(indiv_scaled[c1[i]], 20)
    print(f"Keyword {i}: {indiv_scaled[c1[i]].name} with {np.round(indiv_scaled[c1[i]].isin([0]).mean(), 2)*100}% zeros")
    sns.lineplot(truncated_data[truncated_data.notna()].astype(int), color='orange', linewidth=1, ax=ax)
    ax.set_ylabel('Index', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # For axis ticks every 10 units
    start = int(ax.get_ylim()[0])
    end = int(ax.get_ylim()[1])
    ax.set_yticks(np.arange(start+1, end + 1, 5))
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, axis='y', color='lightgray', alpha=0.5)
    ax.grid(True, axis='x', color='lightgray', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{fig_save}fig5a_individual.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (b) Combined (large values truncated)

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
truncated_data = np.minimum(sub, 30)  # Truncate values above 30
print(f"Cluster with {np.round(sub.isin([0]).mean(), 2)*100}% zeros")
sns.lineplot(truncated_data.astype(int), linewidth=1) 
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig5b_combined.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (c) Summation (large values truncated)

sum_trends = indiv_scaled[c1].sum(axis=1)

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
truncated_data = np.minimum(sum_trends, 30)  # Truncate values above 30
print(f"Sum of keywords with {np.round(sum_trends.isin([0]).mean(), 2)*100}% zeros")
sns.lineplot(truncated_data.astype(int), linewidth=1, color='#FF7F0E') 
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig5c_summation.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 6: Distribution of percentage of zeros in individual and combined search volumes across all locations

### (a) Individual

indiv_all = []

for i in range(0, len(geo)):
    state = geo[i]
    if state not in ["US", "PR"]:
        trends_indiv = pd.read_csv(f'data/01_raw/individual_merged_trends/{state}_individual_trends.csv')
        trends_indiv['date'] = pd.to_datetime(trends_indiv['date'])
        trends_indiv = trends_indiv[trends_indiv['date'] <= max_date]
        trends_indiv.set_index('date', inplace=True)
        indiv_all.append(trends_indiv)

all_trends_indiv = pd.concat(indiv_all, axis = 1)
all_indiv_zeros = all_trends_indiv.isin([0]).mean()
all_indiv_zeros.describe()

print("Total number of keywords across all states:", all_trends_indiv.shape[1])

perc = all_indiv_zeros*100
bins = [0, 10] + list(range(20, 110, 10))
# labels = [f"[{bins[i]}, {bins[i+1]}]" for i in range(len(bins)-1)]
# binned_perc = pd.cut(perc, bins=bins, right=True, include_lowest=True, labels=labels)

labels = ['[0, 10]'] + [f'({bins[i]}, {bins[i+1]}]' for i in range(1, len(bins)-1)]
binned_perc = pd.cut(perc, bins=bins, right=True, include_lowest=True, labels=labels)

counts = binned_perc.value_counts().sort_index()
counts_df = counts.reset_index()
counts_df.columns = ['Percentage of zeros', 'Number of terms']
counts_df

total_terms = counts_df['Number of terms'].sum()
print(f"Total number of individual terms: {total_terms}")

# Count terms with percentage of zeros in [0, 30]
terms_0_30 = (perc >= 0) & (perc <= 30)
count_0_30 = terms_0_30.sum()
print(f"Number of individual terms with 0-30% zeros: {count_0_30}")

# Count terms with percentage of zeros between 30% and 99%
terms_30_99 = (perc >= 30) & (perc <= 99.2)
count_30_99 = terms_30_99.sum()
print(f"Number of individual terms with 30-99% zeros: {count_30_99}")

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(5,4))
# sns.lineplot(truncated_data.astype(int), linewidth=1) 
barplot = sns.barplot(
    x='Percentage of zeros', 
    y='Number of terms', 
    data=counts_df, 
    # palette=palette,
    # alpha=0.8,
    dodge=False, 
    legend=False  
)
# Add number of terms above each bar
for index, row in counts_df.iterrows():
    barplot.text(
        index, row['Number of terms'], 
        round(row['Number of terms'], 2), 
        color='black', ha="center", 
        va='bottom', fontsize=10
    )
plt.xlabel('Percentage of Zeros', fontsize=12) 
plt.ylabel('Number of Terms', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().tick_params(axis='x', rotation=45) # Rotate date labels 
plt.grid(False)  # Remove all grid lines
sns.despine() # Remove top and right borders for a cleaner look
plt.tight_layout()
plt.savefig(f'{fig_save}fig6a_indiv_zeros.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (b) Combined

clusters_sub1 = clusters.loc[:, ~clusters.columns.str.startswith(("PR", "US-NY-501", "US_"))]
clusters_sub = clusters_sub1.filter(like='+', axis=1)
all_clusters_zeros = clusters_sub.isin([0]).mean()
all_clusters_zeros.describe()

print("Number of clusters across all states:", clusters_sub.shape[1])

perc = all_clusters_zeros*100
# bins = list(range(0, 99, 10)) + [99, 100]
# bins = list(range(0, 90, 10)) + [90,99]
# bins = list(range(0, 110, 10))
# binned_perc = pd.cut(perc, bins=bins, right=False, include_lowest=True)
# binned_perc

bins = [0, 10] + list(range(20, 110, 10))
# labels = [f"[{bins[i]}, {bins[i+1]}]" for i in range(len(bins)-1)]
# binned_perc = pd.cut(perc, bins=bins, right=True, include_lowest=True, labels=labels)

labels = ['[0, 10]'] + [f'({bins[i]}, {bins[i+1]}]' for i in range(1, len(bins)-1)]
binned_perc = pd.cut(perc, bins=bins, right=True, include_lowest=True, labels=labels)

counts = binned_perc.value_counts().sort_index()
counts_df = counts.reset_index()
counts_df.columns = ['Percentage of zeros', 'Number of terms']
counts_df
# counts

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(5,4))  # width 6.8 in
# sns.lineplot(truncated_data.astype(int), linewidth=1) 
barplot = sns.barplot(
    x='Percentage of zeros', 
    y='Number of terms', 
    data=counts_df, 
    # palette=palette,
    # alpha=0.8,
    dodge=False, 
    legend=False  
)
# Add number of terms above each bar
for index, row in counts_df.iterrows():
    barplot.text(
        index, row['Number of terms'], 
        round(row['Number of terms'], 2), 
        color='black', ha="center", 
        va='bottom', fontsize=10
    )
plt.xlabel('Percentage of Zeros', fontsize=12) 
plt.ylabel('Number of Clusters', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().tick_params(axis='x', rotation=45) # Rotate date labels 
plt.grid(False)  # Remove all grid lines
sns.despine() # Remove top and right borders for a cleaner look
plt.tight_layout()
plt.savefig(f'{fig_save}fig6b_combined_zeros.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 7: Example comparison of noise levels before and after denoising across five downloads in Alaska

### (a) Raw

save_dir = 'data/01_raw/api_downloads/api_raw'
files_raw = sorted([
    os.path.join(save_dir, file) 
    for file in os.listdir(save_dir) 
    if os.path.isfile(os.path.join(save_dir, file))
])
files_raw = files_raw
files_raw

j = 0
sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
for file in files_raw[0:5]:
    df = pd.read_csv(file)
    print(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= max_date]
    df.set_index('date', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df)  # Assuming all columns except 'date' are numeric
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    state = 'US-AK'
    sub = df_scaled[df_scaled.columns[df_scaled.columns.str.contains(state+'_')]]

    sns.lineplot(x=df_scaled.index, y=sub.iloc[:, j].astype(int), linewidth=1)
    
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig7a_variability_raw.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (b) Denoised

save_dir = 'data/01_raw/api_downloads/api_smooth'
files_smooth = sorted([
    os.path.join(save_dir, file) 
    for file in os.listdir(save_dir) 
    if os.path.isfile(os.path.join(save_dir, file))
])
files_smooth

smoothed_data = pd.read_csv(files_smooth[14])
# remove the columns that contain "US_" or "PR_"
smoothed_data = smoothed_data[smoothed_data.columns[~smoothed_data.columns.str.contains("US_|PR_|US-NY-501")]]
smooth_prefix_count = smoothed_data.columns.str.contains("smooth").sum()
print(f"Number of variables requiring denoising: {smooth_prefix_count}")
print(f"Total number of clusters and individual keywords kept: {smoothed_data.shape[1]}")
print(f"Percentage of variables requiring denoising: {smooth_prefix_count/smoothed_data.shape[1]*100:.2f}%")

j = 0
sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
# sns.lineplot(clusters_scaled[kw].astype(int), linewidth=1) 
for file in files_smooth[0:5]:
    df = pd.read_csv(file)
    print(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= max_date]
    df.set_index('date', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df)  # Assuming all columns except 'date' are numeric
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    state = 'US-AK'
    sub = df_scaled[df_scaled.columns[df_scaled.columns.str.contains(state+'_')]]

    sns.lineplot(x=df_scaled.index, y=sub.iloc[:, j].astype(int), linewidth=1)
    
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig7b_variability_smooth.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 8: Descriptive statistics before and after denoising across 27 downloads in Alaska example

# concatenating all raw and smooth data for a single state
# calculating standard deviation and average across downloads

file = files_smooth[0]
df = pd.read_csv(file)
start_date = df['date'].min()

state = 'US-AK'
j = 0
subs = []
for file in files_raw:
    df = pd.read_csv(file)
    df = df[df['date'] >= start_date]
    df = df[df['date'] <= max_date]
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    sub = df[df.columns[df.columns.str.contains(state+'_')]]
    subs.append(sub.iloc[:,j])

combined_raw = pd.concat(subs, axis=1)
std_raw = combined_raw.std(axis=1)
avg_raw = combined_raw.dropna().mean(axis=1)

subs = []
for file in files_smooth:
    df = pd.read_csv(file)
    df = df[df['date'] >= start_date]
    df = df[df['date'] <= max_date]
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    sub = df[df.columns[df.columns.str.contains(state+'_')]]
    subs.append(sub.iloc[:,j])

combined_smooth = pd.concat(subs, axis=1)
std_smooth = combined_smooth.std(axis=1)
avg_smooth = combined_smooth.dropna().mean(axis=1)

# Rescale each column in combined_smooth based on combined_raw
combined_raw_scaled = combined_raw.copy()
combined_smooth_copy = combined_smooth.copy()

rename_dict = {smooth_col: raw_col for smooth_col, raw_col in zip(combined_smooth_copy.columns, combined_raw.columns)}
combined_smooth_copy.rename(columns=rename_dict, inplace=True)
combined_smooth_scaled = combined_smooth_copy.copy()

combined_raw_scaled = combined_raw_scaled[combined_raw_scaled.index <= combined_smooth_scaled.index.max()]

for col in combined_raw.columns:
    raw_min = combined_raw[col].min()
    raw_max = combined_raw[col].max()
    combined_raw_scaled[col] = (combined_raw[col] - raw_min) / (raw_max - raw_min) * 100

std_raw_scaled = combined_raw_scaled.std(axis=1)
std_raw_scaled[std_raw_scaled < 0.1] = 0
avg_raw_scaled = combined_raw_scaled.dropna().mean(axis=1)

for col in combined_raw.columns:
    raw_min = combined_raw[col].min()
    raw_max = combined_raw[col].max()
    combined_smooth_scaled[col] = (combined_smooth_copy[col] - raw_min) / (raw_max - raw_min) * 100

std_smooth_scaled = combined_smooth_scaled.std(axis=1)
std_smooth_scaled[std_smooth_scaled < 0.1] = 0
avg_smooth_scaled = combined_smooth_scaled.dropna().mean(axis=1)

snr_raw_scaled = avg_raw_scaled / std_raw_scaled
snr_smooth_scaled = avg_smooth_scaled / std_smooth_scaled
ratio_snr = np.log(snr_smooth_scaled / snr_raw_scaled)

### (a) Average

sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
sns.lineplot(avg_raw_scaled, label = "Raw", linewidth=1)
sns.lineplot(avg_smooth_scaled, label = "Denoised", linewidth=1)
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.legend(fontsize=8)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig8a_average_raw_denoised.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (b) Standard deviation

j = 0
sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
sns.lineplot(std_raw_scaled, label = "Raw", linewidth=1)
sns.lineplot(std_smooth_scaled, label = "Denoised", linewidth=1)
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Index', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.legend(fontsize=8) 
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig8b_std_raw_denoised.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 9: Log ratio of SNR across 27 downloads. Higher is better

### (a) Single keyword in Alaska example

j = 0
sns.set(style='whitegrid') 
plt.rcParams['image.cmap'] = 'viridis'  # RGB colormap
fig = plt.figure(figsize=(7,3))  # width 6.8 in
sns.lineplot(ratio_snr, linewidth=1)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Date', fontsize=12) 
plt.ylabel('Log Ratio of SNR', fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Show only years 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Format the date as year 
plt.gca().tick_params(axis='x') # Rotate date labels 
plt.grid(True, axis='y', color='lightgray', alpha=0.5) 
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{fig_save}fig9a_ratio_snr_single.pdf', format='pdf', bbox_inches='tight')
plt.show()

### (b) Summary across all keywords and locations

variab_dir = 'data/02_intermediate/variability_analysis/'

avg_raws_df = pd.read_csv(os.path.join(variab_dir, 'avg_raws_df.csv'))
std_raws_df = pd.read_csv(os.path.join(variab_dir, 'std_raws_df.csv'))
snr_raws_df = pd.read_csv(os.path.join(variab_dir, 'snr_raws_df.csv'))

avg_smooths_df = pd.read_csv(os.path.join(variab_dir, 'avg_smooths_df.csv'))
std_smooths_df = pd.read_csv(os.path.join(variab_dir, 'std_smooths_df.csv'))
snr_smooths_df = pd.read_csv(os.path.join(variab_dir, 'snr_smooths_df.csv'))

avg_raws_df['date'] = pd.to_datetime(avg_raws_df['date'])
avg_raws_df.set_index('date', inplace=True)
avg_smooths_df['date'] = pd.to_datetime(avg_smooths_df['date'])
avg_smooths_df.set_index('date', inplace=True)
std_raws_df['date'] = pd.to_datetime(std_raws_df['date'])
std_raws_df.set_index('date', inplace=True)
std_smooths_df['date'] = pd.to_datetime(std_smooths_df['date'])
std_smooths_df.set_index('date', inplace=True)
snr_raws_df['date'] = pd.to_datetime(snr_raws_df['date'])
snr_raws_df.set_index('date', inplace=True)
snr_smooths_df['date'] = pd.to_datetime(snr_smooths_df['date'])
snr_smooths_df.set_index('date', inplace=True)

ratio_snr = np.log(snr_smooths_df / snr_raws_df).mean()
ratio_snr.describe()

import seaborn as sns
sns.set(style="whitegrid")  
plt.figure(figsize=(4,3))  # Increase figure size for better readability
sns.boxplot(data=ratio_snr,
            # color='skyblue',  # Box color
            linewidth=1,  # Set the thickness of the lines
            fliersize=7 ) # Set the size of the outliers
            # medianprops={'color': 'red', 'linewidth': 2})  # Red median line

# Title and axis labels
plt.ylabel('Ratio of SNR', fontsize=12)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

# Customize ticks and labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.tight_layout()  # Make sure everything fits in the plot area
plt.savefig(f'{fig_save}fig9b_ratio_snr_boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 10: Ratios of RMSEs across all locations and horizons for different denoising methods, relative to smoothing splines (benchmark). Values below 1 indicate better performance than smoothing splines.

save_dir = 'results/denoising_results/'
files = sorted([
    os.path.join(save_dir, file) 
    for file in os.listdir(save_dir) 
    if os.path.isfile(os.path.join(save_dir, file))
])
files

rmses_files = [file for file in files if "rmses" in os.path.basename(file)]
rmses_files

def rename_rmse_columns(df, method="arimax_111", data="smooth"):
    # print(f'rmse_{method}_h0')
    return df.rename(columns={
        f'rmse_{method}_h0': f'rmse_{data}_h0',
        f'rmse_{method}_h1': f'rmse_{data}_h1',
        f'rmse_{method}_h2': f'rmse_{data}_h2',
        f'rmse_{method}_h3': f'rmse_{data}_h3'
    })

# Example usage for clusters, smooth, detrend
dfs = {}
data = 'smooth'
file = f'results/forecast_rmses/arimax_111_smooth_rmses.csv'
df = pd.read_csv(file)
dfs[data] = rename_rmse_columns(df=df, method="arimax_111", data=data)

data = ['smooth_ma', 'smooth_ssa', 'smooth_wt']
for data in data:
    # print(data)
    file = f'results/denoising_results/arimax_111_{data}_rmses.csv'
    df = pd.read_csv(file)
    dfs[data] = rename_rmse_columns(df=df, method="arimax_111", data=data)

# Method names for columns (matching keys in dfs)
print("ARIMA")
method_names = ['smooth', 'smooth_ma', 'smooth_ssa', 'smooth_wt']

# Extract RMSEs for each horizon and concatenate into tables
rmse_tables = {}
horizons = ['h0', 'h1', 'h2', 'h3']

for h in horizons:
    dfs_h = []
    for method in method_names:
        df = dfs[method]
        col_name = f'rmse_{method}_{h}'
        dfs_h.append(df[['geo', col_name]].rename(columns={col_name: method}))
    # Merge all methods on 'geo'
    merged = dfs_h[0]
    for d in dfs_h[1:]:
        merged = merged.merge(d, on='geo')
    rmse_tables[h] = merged

mses_tables = {}
for h in ['h0', 'h1', 'h2', 'h3']:
    rmse_df = rmse_tables[h].copy()
    mse_df = rmse_df.copy()
    for method in method_names:
        # mse_df[method] = rmse_df[method] ** 2
        mse_df[method] = rmse_df[method] # CHANGED MSE to RMSE
    mses_tables[h] = mse_df

# relative efficiency
mses_df = mses_tables.copy()
h = 'h0'
for h in ['h0', 'h1', 'h2', 'h3']:
    mses_df[h]['ratio_smooth_ma'] = mses_df[h]['smooth_ma'] / mses_df[h]['smooth']
    mses_df[h]['ratio_smooth_ssa'] = mses_df[h]['smooth_ssa'] / mses_df[h]['smooth']
    mses_df[h]['ratio_smooth_wt'] = mses_df[h]['smooth_wt'] / mses_df[h]['smooth']

print("Mean Relative Efficiency at the US state level (omitting overall national level):")
for h in ['h0', 'h1', 'h2', 'h3']:
    mask = (mses_df[h]['geo'] != 'US') & (mses_df[h]['geo'] != 'PR')
    numerator = mses_df[h].loc[mask, ['smooth_ma', 'smooth_ssa', 'smooth_wt']].mean()
    denominator = mses_df[h].loc[mask, 'smooth'].mean()
    result = round(numerator / denominator, 2)
    print(f"{h}:")
    print(result)

sns.set(style="whitegrid")  
# fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
# fig, axes = plt.subplots(1, 4, figsize=(6.8, 3.5), sharey=True)
fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharey=True)  # Double-column width

horizons = ['h0', 'h1', 'h2', 'h3']
labels = ['MA', 'SSA', 'WT']

for idx, h in enumerate(horizons):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    # ax = axes[idx]
    sns.boxplot(
        data=mses_df[h][['ratio_smooth_ma', 'ratio_smooth_ssa', 'ratio_smooth_wt']],
        linewidth=1,
        fliersize=7,
        ax=ax
    )
    ax.set_title(f'Horizon {h[1:]}', fontsize=12)
    ax.set_ylabel('Ratio of RMSEs', fontsize=12)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
    ax.set_ylim(0, 2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig(f'{fig_save}fig10_smooth_boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 11: Example of estimated deterministic trend in raw and detrended data in Alabama

smooth['date'] = pd.to_datetime(smooth['date'])
smooth.set_index('date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 100))
smooth_scaled = pd.DataFrame(scaler.fit_transform(smooth), columns=smooth.columns)
smooth_scaled.insert(0, 'date', smooth.index)
smooth_scaled.set_index('date', inplace=True)

train = smooth_scaled.copy()
train = train[train.index < cutoff]
test = smooth_scaled.copy()
test = test[test.index >= cutoff]

detrend['date'] = pd.to_datetime(detrend['date'])
detrend.set_index('date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 100))
detrend_scaled = pd.DataFrame(scaler.fit_transform(detrend), columns=detrend.columns)
detrend_scaled.insert(0, 'date', detrend.index)
detrend_scaled.set_index('date', inplace=True)

train_detrend = detrend_scaled.copy()
train_detrend = train_detrend[train_detrend.index < cutoff]
test_detrend = detrend_scaled.copy()
test_detrend = test_detrend[test_detrend.index >= cutoff]

from sklearn.metrics import r2_score

# raw data
data = smooth_scaled['smooth_US-AL_(TOPIC)05s5v6 + (TOPIC)06n3pj']
time = np.arange(len(smooth_scaled))
train_size = len(train)
train_time = time[:train_size]
train_data = data[:train_size]
test_time = time[train_size:]

# Fit the linear trend on the train set
mean_time_train = np.mean(train_time)
mean_data_train = np.mean(train_data)
# compute trend on raw data
numerator = np.sum((train_time - mean_time_train) * (train_data - mean_data_train))
denominator = np.sum((train_time - mean_time_train) ** 2)
slope_train = numerator / denominator
intercept_train = mean_data_train - slope_train * mean_time_train
# Raw trend estimate
trend_estimate = slope_train * time + intercept_train
r2 = r2_score(data, trend_estimate)
print("Raw trend R^2", r2)

# detrended_data = data / trend_estimate
# min_detrended = np.min(detrended_data)
# max_detrended = np.max(detrended_data)
# detrended_data = 100 * (detrended_data - min_detrended) / (max_detrended - min_detrended)

# detrended data
detrended_data = detrend_scaled['smooth_US-AL_(TOPIC)05s5v6 + (TOPIC)06n3pj']
time = np.arange(len(detrend_scaled))
train_size = len(train_detrend)
train_time = time[:train_size]
train_data = detrended_data[:train_size]
# Fit the linear trend on the train set
mean_time_train = np.mean(train_time)
mean_data_train = np.mean(train_data)
# compute trend on detrended data
numerator = np.sum((train_time - mean_time_train) * (train_data - mean_data_train))
denominator = np.sum((train_time - mean_time_train) ** 2)
slope_train = numerator / denominator
intercept_train = mean_data_train - slope_train * mean_time_train
# Detrended trend estimate
trend_estimate2 = slope_train * time + intercept_train
r2_detrended = r2_score(detrended_data, trend_estimate2)
print("Detrended trend R^2", r2_detrended)

data = smooth_scaled['smooth_US-AL_(TOPIC)05s5v6 + (TOPIC)06n3pj']
sns.set(style='whitegrid')
plt.figure(figsize = (7,3))
sns.lineplot(data, label = 'Raw', linewidth=1)
sns.lineplot(detrend_scaled['smooth_US-AL_(TOPIC)05s5v6 + (TOPIC)06n3pj'], label='Detrended', linewidth=1)
plt.plot(smooth_scaled.index, trend_estimate, label=f'Raw Trend (R² = {r2:.2f})', linestyle='--', linewidth=1)
plt.plot(detrend_scaled.index, trend_estimate2, label=f'Detrended Trend (R² = {r2_detrended:.2f})', linestyle='--', linewidth=1)
# plt.title('Estimated Linear Trend vs Data', fontsize=10)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Index', fontsize=12)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Show only years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the date as year
plt.gca().tick_params(axis='x')  # Rotate date labels
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.grid(True, axis='y', color='lightgray', alpha=0.5)
plt.grid(True, axis='x', color='lightgray', alpha=0.5)
plt.legend(loc='lower right', fontsize=8)
# Add tight layout for better spacing
plt.tight_layout()
plt.savefig(f'{fig_save}fig11_raw_detrended.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Figure 12: Summary of R2 for deterministic trends before and after detrending across all locations

# exclude columns that contain "PR_" or "US_" or "NY-501_"
detrend_copy = detrend.loc[:, ~detrend.columns.str.contains("PR_|US_|NY-501_")]
smooth_copy = smooth.loc[:, ~smooth.columns.str.contains("PR_|US_|NY-501_")]

from statsmodels.tsa.stattools import adfuller

# ADF test to identify columns that have a trend
alpha = 0.05
detrended_cols = []
for i in range(smooth_copy.shape[1]):
    kw = smooth_copy.iloc[:,i].dropna()
    adf1 = adfuller(kw, regression='c')[1]
    adf2 = adfuller(kw, regression='ct')[1]
    adf3 = adfuller(kw, regression='ctt')[1]

    if (adf1 > alpha) & (adf2 < alpha): # linear
        detrended_cols.append(smooth_copy.columns[i])
    elif (adf1 > alpha) & (adf2 > alpha) & (adf3 < alpha): # quadratic
        detrended_cols.append(smooth_copy.columns[i])

# ADF test post linear and quadratic detrending
final_adf_c = []
final_adf_ct = []
final_adf_ctt = []

for i in range(detrend_copy[detrended_cols].shape[1]):
    kw = detrend_copy[detrended_cols].iloc[:,i].dropna()
    adf1 = adfuller(kw, regression='c')[1]
    final_adf_c.append(adf1)
    adf2 = adfuller(kw, regression='ct')[1]
    final_adf_ct.append(adf2)
    adf3 = adfuller(kw, regression='ctt')[1]
    final_adf_ctt.append(adf3)

pd.DatetimeIndex([cutoff])[0]

detrend_copy.index = pd.DatetimeIndex(detrend_copy.index)
smooth_copy.index = pd.DatetimeIndex(smooth_copy.index)
smooth_copy = smooth_copy[smooth_copy.index >= detrend_copy.index.min()]
train = smooth_copy.copy()
train = train[train.index < pd.DatetimeIndex([cutoff])[0]]
test = smooth_copy.copy()
test = test[test.index >= pd.DatetimeIndex([cutoff])[0]]

# using ADF test post detrending
from sklearn.metrics import r2_score

time = np.arange(len(detrend_copy))

r2s = []
r2s_after = []
alpha = 0.05

for i in range(detrend_copy[detrended_cols].shape[1]):
    data = smooth_copy[detrended_cols].iloc[:, i]
    data_detrend = detrend_copy[detrended_cols].iloc[:, i]
   
    if (final_adf_c[i] > alpha) & (final_adf_ct[i] < alpha): # linear

        # before detrending
        train_size = len(train)
        train_time = time[:train_size]
        train_data = data[:train_size]
        # Fit the linear trend on the train set
        mean_time_train = np.mean(train_time)
        mean_data_train = np.mean(train_data)
        numerator = np.sum((train_time - mean_time_train) * (train_data - mean_data_train))
        denominator = np.sum((train_time - mean_time_train) ** 2)
        slope_train = numerator / denominator
        intercept_train = mean_data_train - slope_train * mean_time_train
        # Apply the trend to detrend the entire dataset
        trend_estimate = slope_train * time + intercept_train
        # R2
        r2 = r2_score(data, trend_estimate)
        r2s.append(r2)

        # after detrending
        train_size = len(train_detrend)
        train_time = time[:train_size]
        train_data = data_detrend[:train_size]
        # Fit the linear trend on the train set
        mean_time_train = np.mean(train_time)
        mean_data_train = np.mean(train_data)
        numerator = np.sum((train_time - mean_time_train) * (train_data - mean_data_train))
        denominator = np.sum((train_time - mean_time_train) ** 2)
        slope_train = numerator / denominator
        intercept_train = mean_data_train - slope_train * mean_time_train
        # Apply the trend to detrend the entire dataset
        trend_estimate = slope_train * time + intercept_train
        r2 = r2_score(data, trend_estimate)
        r2s_after.append(r2)
        
    elif (final_adf_c[i] > alpha) & (final_adf_ct[i] > alpha) & (final_adf_ctt[i] < alpha): # quadratic
        # before detrending
        train_size = len(train)
        train_time = time[:train_size]
        train_data = data[:train_size]
        # Fit the quadratic trend on the train set
        # Calculate the coefficients for a second-degree polynomial: ax^2 + bx + c
        coeffs = np.polyfit(train_time, train_data, deg=2)
        a, b, c = coeffs
        # Apply the quadratic trend to detrend the entire dataset
        trend_estimate = a * time**2 + b * time + c        
        r2 = r2_score(data, trend_estimate)
        r2s.append(r2)

        # after detrending
        train_size = len(train_detrend)
        train_time = time[:train_size]
        train_data = data_detrend[:train_size]
        # Fit the quadratic trend on the train set
        # Calculate the coefficients for a second-degree polynomial: ax^2 + bx + c
        coeffs = np.polyfit(train_time, train_data, deg=2)
        a, b, c = coeffs
        # Apply the quadratic trend to detrend the entire dataset
        trend_estimate = a * time**2 + b * time + c
        r2 = r2_score(data_detrend, trend_estimate)
        r2s_after.append(r2)

    elif (final_adf_c[i] < alpha): # post detrending

        # before detrending
        train_size = len(train)
        train_time = time[:train_size]
        train_data = data[:train_size]
        # Fit the quadratic trend on the train set
        # Calculate the coefficients for a second-degree polynomial: ax^2 + bx + c
        coeffs = np.polyfit(train_time, train_data, deg=2)
        a, b, c = coeffs
        # Apply the quadratic trend to detrend the entire dataset
        trend_estimate = a * time**2 + b * time + c        
        r2 = r2_score(data, trend_estimate)
        r2s.append(r2)

        # after detrending
        train_size = len(train_detrend)
        train_time = time[:train_size]
        train_data = data_detrend[:train_size]
        # Fit the quadratic trend on the train set
        # Calculate the coefficients for a second-degree polynomial: ax^2 + bx + c
        coeffs = np.polyfit(train_time, train_data, deg=2)
        a, b, c = coeffs
        # Apply the quadratic trend to detrend the entire dataset
        trend_estimate = a * time**2 + b * time + c
        r2 = r2_score(data_detrend, trend_estimate)
        r2s_after.append(r2)

print("Mean R^2 before detrending:", round(np.mean(r2s), 2))
print("Mean R^2 after detrending:", round(np.mean(r2s_after), 2))

sns.set(style="whitegrid")  
plt.figure(figsize=(4,3))  # Increase figure size for better readability

# Define flier properties to make outliers empty circles
flierprops = dict(marker='o', markersize=7, markerfacecolor='none', markeredgecolor='black')

sns.boxplot(data=[r2s, r2s_after], 
            linewidth=1,  # Set the thickness of the lines
            flierprops=flierprops)  # Apply custom outlier style

# Title and axis labels
plt.ylabel('R$^2$', fontsize=12)

# Customize ticks and labels
plt.xticks(ticks=[0, 1], labels=["Raw", "Detrended"], fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.tight_layout()  # Make sure everything fits in the plot area
plt.savefig(f'{fig_save}fig12_r2_boxplot.pdf', format='pdf', bbox_inches='tight')

plt.show()

## Figure S2: Ratios RMSEs across all locations and horizons for different datasets, relative to the model without exogenous variables (baseline). Values below 1 indicate better performance than the baseline.

# function to rename RMSE columns based on their method
def rename_rmse_columns(df, model, data):
    return df.rename(columns={
        f'rmse_{model}_h0': f'rmse_{data}_h0',
        f'rmse_{model}_h1': f'rmse_{data}_h1',
        f'rmse_{model}_h2': f'rmse_{data}_h2',
        f'rmse_{model}_h3': f'rmse_{data}_h3'
    })

def mses_re(save_dir, model, horizons, method_names):
    dfs = {}
    # concatenating files
    for data in method_names:
        # print(data)
        file = f'{save_dir}{model}_{data}_rmses.csv'
        df = pd.read_csv(file)
        dfs[data] = rename_rmse_columns(df=df, model=model, data=data)

    # Extract RMSEs for each horizon and concatenate into tables
    rmse_tables = {}
    for h in horizons:
        dfs_h = []
        for method in method_names:
            df = dfs[method]
            col_name = f'rmse_{method}_{h}'
            dfs_h.append(df[['geo', col_name]].rename(columns={col_name: method}))
        # Merge all methods on 'geo'
        merged = dfs_h[0]
        for d in dfs_h[1:]:
            merged = merged.merge(d, on='geo')
        rmse_tables[h] = merged

    # compute MSEs from RMSEs
    mses_tables = {}
    for h in horizons:
        rmse_df = rmse_tables[h].copy()
        mse_df = rmse_df.copy()
        for method in method_names:
            # mse_df[method] = rmse_df[method] ** 2
            mse_df[method] = rmse_df[method] # CHANGED MSE to RMSE
        mses_tables[h] = mse_df

    # remove PR from the mses_tables
    mses_tables = {h: df[df['geo'] != 'PR'].reset_index(drop=True) for h, df in mses_tables.items()}

    # relative efficiency
    mses_df = mses_tables.copy()
    for h in horizons:
        mses_df[h]['ratio_indiv'] = mses_df[h]['indiv'] / mses_df[h]['noexog']
        mses_df[h]['ratio_topics'] = mses_df[h]['topics'] / mses_df[h]['noexog']
        mses_df[h]['ratio_cluster'] = mses_df[h]['clusters'] / mses_df[h]['noexog']
        mses_df[h]['ratio_smooth'] = mses_df[h]['smooth'] / mses_df[h]['noexog']
        mses_df[h]['ratio_detrend'] = mses_df[h]['detrend'] / mses_df[h]['noexog']

    return mses_df

def plot_all_models_boxplot(mses_dfs, horizons, model_names, labels=None):
    """
    mses_dfs: list of mses_df dicts, one per model
    horizons: list of horizon keys (e.g., ['h0', 'h1', 'h2', 'h3'])
    model_names: list of model names (e.g., ['ARIMA', 'SARIMA', ...])
    labels: list of method labels for x-axis
    """
    n_models = len(mses_dfs)
    n_horizons = len(horizons)
    if labels is None:
        labels = ['Non-preprocessed', 'Topics', 'Clustering', 'Denoising', 'Detrending']

    sns.set(style="whitegrid")
    # fig, axes = plt.subplots(n_models, n_horizons, figsize=(6*n_horizons, 4*n_models), sharey=True)
    fig, axes = plt.subplots(n_models, n_horizons, figsize=(5*n_horizons, 3*n_models), sharey=True)
    print("Figure size:", (5*n_horizons, 3*n_models))

    for i, (mses_df, model_name) in enumerate(zip(mses_dfs, model_names)):
        for j, h in enumerate(horizons):
            ax = axes[i, j] if n_models > 1 else axes[j]
            sns.boxplot(
                data=mses_df[h][['ratio_indiv', 'ratio_topics', 'ratio_cluster', 'ratio_smooth', 'ratio_detrend']],
                linewidth=1,
                fliersize=7,
                ax=ax
            )
            # Extract the number from the horizon key (e.g., 'h0' -> 0)
            horizon_num = h[1:] if isinstance(h, str) and h.startswith('h') else str(h)
            if i == 0:
                ax.set_title(f'Horizon {horizon_num}', fontsize=12)
            if j == 0:
                ax.set_ylabel(model_name, fontsize=12)
            else:
                ax.set_ylabel('')
            ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
            ax.set_ylim(0, 2)
            ax.set_xticklabels(labels, fontsize=10)
            ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    # save image
    plt.savefig(f'{fig_save}figS2_ratio_rmses.pdf', format='pdf', bbox_inches='tight')
    plt.show()

horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

# ARIMA
save_dir = 'results/forecast_rmses/'
model = "arimax_111"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_arima = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

# SARIMA
model = "sarimax_010"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_sarima = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

# ARGO
model = "argo"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_argo = {h: df[~df['geo'].isin(['US', 'Puerto.Rico'])] for h, df in mses_df.items()}

# LightGBM
model = "lgbm"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_lgbm = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

# ADAboost
model = "adaboost"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_adaboost = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

horizons = ['h0', 'h1', 'h2', 'h3']
model_names = ['ARIMAX', 'SARIMAX', 'ARGO', 'LightGBM', 'AdaBoost']

plot_all_models_boxplot(
    [mses_df_arima, mses_df_sarima, mses_df_argo, mses_df_lgbm, mses_df_adaboost],
    horizons,
    model_names
)

## Table 1: Median and interquartile range (Q1, Q3) of relative efficiency across all locations, horizons, models, and sets of exogenous variables

def print_all_models_median_results(mses_dfs_list, model_names, horizons):
    """
    Create a table with median results for all models.
    
    Parameters:
    -----------
    mses_dfs_list : list
        List of mses_df dictionaries, one per model
    model_names : list
        List of model names corresponding to mses_dfs_list
    horizons : list
        List of horizons (e.g., ['h0', 'h1', 'h2', 'h3'])
    
    Returns:
    --------
    pd.DataFrame
        Table with median results for all models
    """
    # Define method mapping and order
    method_order = ['Detrending', 'Denoising', 'Clustering', 'Topics', 'Non-preprocessed']
    ratio_cols = ['ratio_detrend', 'ratio_smooth', 'ratio_cluster', 'ratio_topics', 'ratio_indiv']
    method_names = ['Detrending', 'Denoising', 'Clustering', 'Topics', 'Non-preprocessed']
    
    # Initialize results list to build DataFrame
    results = []
    
    # For each horizon and method
    for h in horizons:
        for ratio_col, method_name in zip(ratio_cols, method_names):
            # Create a row entry for this horizon/method combination
            row = {'Horizon': h, 'Method': method_name}
            
            # Add data for each model
            for mses_df, model_name in zip(mses_dfs_list, model_names):
                median_value = mses_df[h][ratio_col].median()
                q1_value = mses_df[h][ratio_col].quantile(0.25)
                q3_value = mses_df[h][ratio_col].quantile(0.75)
                
                # Format as "Median (Q1, Q3)"
                row[model_name] = f"{median_value:.2f} ({q1_value:.2f}, {q3_value:.2f})"
            
            # Add this row to results
            results.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Set method order
    results_df['Method'] = pd.Categorical(results_df['Method'], categories=method_order, ordered=True)
    
    # Sort by horizon and method
    results_df = results_df.sort_values(by=['Horizon', 'Method']).reset_index(drop=True)
    
    # Print the table
    print(results_df.to_string(index=False))
    
    return results_df

horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']
model_names = ['ARIMA', 'SARIMA', 'ARGO', 'LightGBM', 'AdaBoost']

# ARIMA
save_dir = 'results/forecast_rmses/'
model = "arimax_111"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_arima = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

# SARIMA
model = "sarimax_010"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_sarima = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

# ARGO
model = "argo"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_argo = {h: df[~df['geo'].isin(['US', 'Puerto.Rico'])] for h, df in mses_df.items()}

# LightGBM
model = "lgbm"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_lgbm = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

# ADAboost
model = "adaboost"
mses_df = mses_re(save_dir, model, horizons, method_names)
mses_df_adaboost = {h: df[~df['geo'].isin(['US', 'PR'])] for h, df in mses_df.items()}

horizons = ['h0', 'h1', 'h2', 'h3']
model_names = ['ARIMA', 'SARIMA', 'ARGO', 'LightGBM', 'AdaBoost']
mses_dfs_list = [mses_df_arima, mses_df_sarima, mses_df_argo, 
                 mses_df_lgbm, mses_df_adaboost]

results_table = print_all_models_median_results(mses_dfs_list, model_names, horizons)
table_txt_out = 'tables/table1_median_iqr.txt'
with open(table_txt_out, 'w', encoding='utf-8') as f:
    f.write(results_table.to_string(index=False))
print(f"Saved table to: {table_txt_out}")

## Table 3: Average relative efficiency between detrending/denoising and clustering across all locations, horizons, and models during peak (December–January) and off (February–November) periods over two influenza seasons

# ---------- Config ----------
models = ["arimax_111", "sarimax_010", "argo", "lgbm", "adaboost"]
model_labels = {
    "arimax_111": "ARIMAX",
    "sarimax_010": "SARIMAX",
    "argo": "ARGO",
    "lgbm": "LightGBM",
    "adaboost": "AdaBoost"
}

# Use the same result directories you used in the notebook
save_dirs = {
    "arimax_111": "results/arimax_results/",
    "sarimax_010": "results/sarimax_results/",  # adjust if needed
    "argo": "results/argo_results/",
    "lgbm": "results/lgbm_results/",
    "adaboost": "results/adaboost_results/"
}

horizons = ["h0", "h1", "h2", "h3"]
methods = ["clusters", "detrend", "smooth"]  # denominator is clusters

# Peak/off season boundaries from your notebook
end_peak_2023 = "2023-02-15"
start_peak_2023 = "2023-12-01"
end_peak_2024 = "2024-02-01"

start_off_2023 = "2023-02-16"
end_off_2023 = "2023-12-01"
start_off_2024 = "2024-02-02"


# ---------- Helpers ----------
def season_masks(idx):
    # idx is datetime index
    peak_mask = ((idx <= pd.to_datetime(end_peak_2023)) |
                 ((idx >= pd.to_datetime(start_peak_2023)) & (idx <= pd.to_datetime(end_peak_2024))))
    off_mask = (((idx >= pd.to_datetime(start_off_2023)) & (idx <= pd.to_datetime(end_off_2023))) |
                (idx >= pd.to_datetime(start_off_2024)))
    return peak_mask, off_mask


def rmse_over_time(save_dir, model, method_names, h, hosp):
    """
    Same logic as your notebook:
    returns dict(method -> DataFrame indexed by date with RMSE column)
    """
    errors_dict = {}
    for data in method_names:
        file = f"{save_dir}{model}_{data}_{h}.csv"
        df = pd.read_csv(file)

        # keep date first; other cols alphabetical
        df = df.reindex(sorted(df.columns), axis=1)
        df = df[["date"] + [c for c in df.columns if c != "date"]]

        hosp_sub = hosp[hosp["date"] >= df["date"].min()].reset_index(drop=True)

        # prediction error by location
        errs = []
        for i in range(1, df.shape[1]):
            errs.append(df.iloc[:, i] - hosp_sub.iloc[:, i])

        errs = pd.DataFrame(errs).T
        errs = pd.concat([df["date"], errs], axis=1)
        errs.columns = ["date"] + [f"{c}" for c in df.columns[1:]]

        errs["RMSE"] = np.sqrt((errs.iloc[:, 1:] ** 2).mean(axis=1))
        errs["date"] = pd.to_datetime(errs["date"], errors="coerce")
        errs = errs.set_index("date")

        errors_dict[data] = errs

    return errors_dict


def remove_outliers(series, quantile=0.99):
    threshold = series.quantile(quantile)
    return series[series <= threshold]


def compute_rel_eff_by_season(errors_dict_h, model):
    """
    Compute relative efficiency:
      detrend/clusters and smooth/clusters
    for peak and off
    """
    idx = errors_dict_h["clusters"].index
    peak_mask, off_mask = season_masks(idx)

    peak_clusters = errors_dict_h["clusters"].loc[peak_mask, "RMSE"]
    peak_detrend = errors_dict_h["detrend"].loc[peak_mask, "RMSE"]
    peak_smooth = errors_dict_h["smooth"].loc[peak_mask, "RMSE"]

    off_clusters = errors_dict_h["clusters"].loc[off_mask, "RMSE"]
    off_detrend = errors_dict_h["detrend"].loc[off_mask, "RMSE"]
    off_smooth = errors_dict_h["smooth"].loc[off_mask, "RMSE"]

    # Match METHOD 1 behavior: remove top 1% outliers for SARIMAX only
    if model == "sarimax_010":
        peak_clusters = remove_outliers(peak_clusters, quantile=0.99)
        peak_detrend = remove_outliers(peak_detrend, quantile=0.99)
        peak_smooth = remove_outliers(peak_smooth, quantile=0.99)

        off_clusters = remove_outliers(off_clusters, quantile=0.99)
        off_detrend = remove_outliers(off_detrend, quantile=0.99)
        off_smooth = remove_outliers(off_smooth, quantile=0.99)

    peak_det = peak_detrend.mean() / peak_clusters.mean()
    peak_smo = peak_smooth.mean() / peak_clusters.mean()
    off_det = off_detrend.mean() / off_clusters.mean()
    off_smo = off_smooth.mean() / off_clusters.mean()

    return {
        ("Peak", "detrend"): peak_det,
        ("Peak", "smooth"): peak_smo,
        ("Off", "detrend"): off_det,
        ("Off", "smooth"): off_smo,
    }


# ---------- Build long results ----------
# assumes hosp dataframe already exists in notebook and date column is string/datetime
hosp["date"] = pd.to_datetime(hosp["date"])

rows = []
for model in models:
    save_dir = save_dirs[model]
    for h in horizons:
        errors_dict_h = rmse_over_time(save_dir, model, methods, h, hosp)
        rel = compute_rel_eff_by_season(errors_dict_h, model)

        for (season, method), value in rel.items():
            rows.append({
                "Horizon": int(h.replace("h", "")),
                "Season": season,
                "Method": method,   # detrend/smooth
                "Model": model,
                "Value": value
            })

eff_df = pd.DataFrame(rows)

# ---------- Pivot for table ----------
pivot = (
    eff_df.assign(
        Method=lambda d: d["Method"].map({"detrend": "Detrending", "smooth": "Denoising"}),
        Model=lambda d: d["Model"].map(model_labels)
    )
    .pivot_table(
        index=["Horizon", "Season", "Method"],
        columns="Model",
        values="Value",
        aggfunc="first"
    )
    .reset_index()
)

# enforce order
season_order = pd.CategoricalDtype(["Peak", "Off"], ordered=True)
method_order = pd.CategoricalDtype(["Detrending", "Denoising"], ordered=True)

pivot["Season"] = pivot["Season"].astype(season_order)
pivot["Method"] = pivot["Method"].astype(method_order)
pivot = pivot.sort_values(["Horizon", "Season", "Method"]).reset_index(drop=True)


# ---------- Write LaTeX ----------
out_file = "tables/table3_average_re.tex"

def fmt(x):
    return f"{x:.2f}"

with open(out_file, "w") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\resizebox{0.8\\textwidth}{!}{%\n")
    f.write("\\begin{tabular}{l l l c c c c c}\n")
    f.write("\\toprule\n")
    f.write("{Horizon} & {Season} & {Method} & ARIMAX & SARIMAX & ARGO & LightGBM & AdaBoost \\\\\n")
    f.write("\\midrule\n")

    for h in [0, 1, 2, 3]:
        sub_h = pivot[pivot["Horizon"] == h].copy()

        # 4 rows per horizon: Peak(2 methods), Off(2 methods)
        rows_h = []
        for season in ["Peak", "Off"]:
            sub_s = sub_h[sub_h["Season"] == season]
            for method in ["Detrending", "Denoising"]:
                r = sub_s[sub_s["Method"] == method].iloc[0]
                rows_h.append((season, method, r))

        # row 1 (Horizon + Season multirow)
        season, method, r = rows_h[0]
        f.write(
            f"\\multirow{{4}}{{*}}{{{h}}} "
            f"& \\multirow{{2}}{{*}}{{{season}}} "
            f"& {method} & {fmt(r['ARIMAX'])} & {fmt(r['SARIMAX'])} & {fmt(r['ARGO'])} & {fmt(r['LightGBM'])} & {fmt(r['AdaBoost'])} \\\\\n"
        )
        # row 2 (Peak second method)
        season, method, r = rows_h[1]
        f.write(
            f"& & {method} & {fmt(r['ARIMAX'])} & {fmt(r['SARIMAX'])} & {fmt(r['ARGO'])} & {fmt(r['LightGBM'])} & {fmt(r['AdaBoost'])} \\\\\n"
        )

        f.write("\\cmidrule(lr){3-8}\n")

        # row 3 (Off first method)
        season, method, r = rows_h[2]
        f.write(
            f"& \\multirow{{2}}{{*}}{{{season}}} "
            f"& {method} & {fmt(r['ARIMAX'])} & {fmt(r['SARIMAX'])} & {fmt(r['ARGO'])} & {fmt(r['LightGBM'])} & {fmt(r['AdaBoost'])} \\\\\n"
        )
        # row 4 (Off second method)
        season, method, r = rows_h[3]
        f.write(
            f"& & {method} & {fmt(r['ARIMAX'])} & {fmt(r['SARIMAX'])} & {fmt(r['ARGO'])} & {fmt(r['LightGBM'])} & {fmt(r['AdaBoost'])} \\\\\n"
        )

        if h < 3:
            f.write("\\midrule\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("} % end resizebox\n")
    f.write("\\vspace{5pt}\n")
    f.write("\\caption{Average relative efficiency between detrending/denoising and clustering across all locations, horizons, and models during peak (December--January) and off (February--November) periods over two influenza seasons. Values below 1 indicate better performance than clustering.}\n")
    f.write("\\label{tab:peak_off_mses}\n")
    f.write("\\end{table}\n")

print(f"Wrote LaTeX table to: {out_file}")

# ---------- Write readable text table ----------
text_out_file = "tables/table3_average_re.txt"

# Keep the same column order as in the LaTeX table
col_order = ["Horizon", "Season", "Method", "ARIMAX", "SARIMAX", "ARGO", "LightGBM", "AdaBoost"]
text_table = pivot[col_order].copy()

with open(text_out_file, "w") as f:
    f.write("Average relative efficiency table\n\n")
    f.write(text_table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print(f"Wrote readable text table to: {text_out_file}")

## Figure S3: Weekly RMSEs for each model across locations from October 2022 to May 2024 for clustered, denoised, and detrended data. Lower values indicate better performance. Differences are mostly observable during peaks.

# functions to generate results
def print_files(save_dir):
    files = sorted([
        os.path.join(save_dir, file) 
        for file in os.listdir(save_dir) 
        if os.path.isfile(os.path.join(save_dir, file))
    ])

    pred_files = [file for file in files if "h" in os.path.basename(file)]
    print(pred_files)

def rmse_over_time(save_dir, model, method_names, h):
    errors_dict = {}  # Dictionary to store errors for each method
    for data in method_names:
        file = f'{save_dir}{model}_{data}_{h}.csv'
        df = pd.read_csv(file)  # Predictions
        # Order df columns by alphabetical order, keeping the date as the first column
        df = df.reindex(sorted(df.columns), axis=1)
        df = df[['date'] + [col for col in df.columns if col != 'date']]

        hosp_sub = hosp[hosp['date'] >= min(df['date'])]
        hosp_sub = hosp_sub.reset_index(drop=True)

        errors = []
        for i in range(1, df.shape[1]):
            errors.append(df.iloc[:, i] - hosp_sub.iloc[:, i])

        # Error between hosp and prediction at each time step
        errors = pd.DataFrame(errors).T
        errors = pd.concat([df['date'], errors], axis=1)
        # Rename the columns with the column names from df
        errors.columns = ['date'] + [f'{col}' for col in df.columns[1:]]

        # Compute the RMSE for each row
        errors['RMSE'] = np.sqrt((errors.iloc[:, 1:] ** 2).mean(axis=1))

        errors['date'] = pd.to_datetime(errors['date'], errors='coerce')  # Convert to datetime, handle invalid values
        errors.set_index('date', inplace=True)

        # Add the errors DataFrame to the dictionary
        errors_dict[data] = errors

    return errors_dict

def plot_rmse_over_time_panels(errors_dict, horizons, model, fig_number):
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(len(horizons), 1, figsize=(7, 10), sharex=True)  # Create subplots for each horizon

    for i, h in enumerate(horizons):
        ax = axes[i]
        sns.lineplot(errors_dict[h]['clusters']['RMSE'], ax=ax, linewidth=1, label='Clustering')
        sns.lineplot(errors_dict[h]['smooth']['RMSE'], ax=ax, linewidth=1, label='Denoising')
        sns.lineplot(errors_dict[h]['detrend']['RMSE'], ax=ax, linewidth=1, label='Detrending')
        ax.set_title(f'Horizon {h[1:]}', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show only months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format the date as year-month
        ax.tick_params(axis='x', labelsize=10)  # Reduce x-axis tick label size
        ax.tick_params(axis='y', labelsize=10)  # Reduce y-axis tick label size
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', color='lightgray', alpha=0.5)
        ax.grid(True, axis='x', color='lightgray', alpha=0.5)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(f'{fig_save}figS3{fig_number}_{model}.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# removing outliers to discern differences better
def plot_rmse_over_time_panels_no_out(errors_dict, horizons, model, fig_number, quantile=0.99):
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(len(horizons), 1, figsize=(7, 10), sharex=True)  # Create subplots for each horizon

    for i, h in enumerate(horizons):
        ax = axes[i]

        for method, label in zip(['clusters', 'smooth', 'detrend'],
                                 ['Clustering', 'Denoising', 'Detrending']):
            rmse = errors_dict[h][method]['RMSE']
            # Remove outliers above quantile threshold
            threshold = rmse.quantile(quantile)
            rmse_no_outlier = rmse[rmse <= threshold]
            sns.lineplot(rmse_no_outlier, ax=ax, linewidth=1, label=label)
        
        ax.set_title(f'Horizon {h[1:]}', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show only months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format the date as year-month
        ax.tick_params(axis='x', labelsize=10)  # Reduce x-axis tick label size
        ax.tick_params(axis='y', labelsize=10)  # Reduce y-axis tick label size
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', color='lightgray', alpha=0.5)
        ax.grid(True, axis='x', color='lightgray', alpha=0.5)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(f'{fig_save}figS3{fig_number}_{model}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

### (a) ARIMAX

save_dir = 'results/arimax_results/' # directory where results are stored
model = "arimax_111"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['clusters', 'smooth', 'detrend']
fig_number = "a"
print_files(save_dir)

errors_dict = {}
for h in horizons:
    errors_dict[h] = rmse_over_time(save_dir, model, method_names, h)

plot_rmse_over_time_panels(errors_dict, horizons, model, fig_number)

### (b) SARIMAX

save_dir = 'results/sarimax_results/'
model = "sarimax_010"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['clusters', 'smooth', 'detrend']
fig_number = "b"
print_files(save_dir)

errors_dict = {}
for h in horizons:
    errors_dict[h] = rmse_over_time(save_dir, model, method_names, h)

plot_rmse_over_time_panels_no_out(errors_dict, horizons, model, fig_number)

### (c) ARGO

save_dir = 'results/argo_results/' # directory where results are stored
model = "argo"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['clusters', 'smooth', 'detrend']
fig_number = 'c'
print_files(save_dir)

errors_dict = {}
for h in horizons:
    errors_dict[h] = rmse_over_time(save_dir, model, method_names, h)

plot_rmse_over_time_panels(errors_dict, horizons, model, fig_number)

### (d) LightGBM

save_dir = 'results/lgbm_results/' # directory where results are stored
model = "lgbm"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['clusters', 'smooth', 'detrend']
fig_number = 'd'
print_files(save_dir)

errors_dict = {}
for h in horizons:
    errors_dict[h] = rmse_over_time(save_dir, model, method_names, h)

plot_rmse_over_time_panels(errors_dict, horizons, model, fig_number)

### (e) AdaBoost

save_dir = 'results/adaboost_results/' # directory where results are stored
model = "adaboost"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['clusters', 'smooth', 'detrend']
fig_number = 'e'
print_files(save_dir)

errors_dict = {}
for h in horizons:
    errors_dict[h] = rmse_over_time(save_dir, model, method_names, h)

plot_rmse_over_time_panels(errors_dict, horizons, model, fig_number)
