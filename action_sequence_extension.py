import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a modified function to include a second plot for overall action rates
def compare_user_action_stats(dataframes, names, figsize=(14, 14)):
    user_action_stats = {}
    overall_action_stats = {}

    # Calculate per-user action counts and frequencies
    for df, name in zip(dataframes, names):
        df['action_count'] = df.groupby(['user_id', 'action'])['action'].transform('count')
        total_actions = df.groupby('user_id')['action'].transform('count')
        df['action_frequency'] = df['action_count'] / total_actions
        user_action_stats[name] = df.groupby(['user_id', 'action']).first().reset_index()
       
        # Calculate overall action counts and frequencies for each dataframe
        action_counts = df['action'].value_counts()
        overall_action_stats[name] = pd.DataFrame({
            'action': action_counts.index,
            'action_count': action_counts.values,
            'action_frequency': action_counts.values / action_counts.sum()
        })

    # Combine all user dataframes for analysis
    combined_stats = pd.concat(user_action_stats.values(), keys=user_action_stats.keys(), names=['DataFrame', 'Index']).reset_index()

    # Melt the combined stats for user-based plotting
    melted_user_stats = combined_stats.melt(id_vars=['DataFrame', 'user_id', 'action'],
                                            value_vars=['action_frequency'],
                                            var_name='Statistic', value_name='Value')

    # Combine all overall action stats for analysis
    combined_overall_stats = pd.concat(overall_action_stats.values(), keys=overall_action_stats.keys(), names=['DataFrame']).reset_index()

    # Melt the combined stats for overall action rate plotting
    melted_overall_stats = combined_overall_stats.melt(id_vars=['DataFrame', 'action'],
                                                       value_vars=['action_frequency'],
                                                       var_name='Statistic', value_name='Value')

    # Start plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, constrained_layout=True)

    # Plot for user-based statistics
    sns.boxplot(ax=axes[0], x='action', y='Value', hue='DataFrame', data=melted_user_stats,
                palette="Set2", showfliers=False)
    axes[0].set_title('Per-User Action Frequencies Comparison')
    axes[0].set_ylabel('Frequency Value')
    axes[0].set_xlabel('Actions')
    axes[0].legend(title='Dataframe')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot for overall action rates
    sns.barplot(ax=axes[1], x='action', y='Value', hue='DataFrame', data=melted_overall_stats,
                palette="Set1")
    axes[1].set_title('Overall Action Frequencies Comparison')
    axes[1].set_ylabel('Frequency Value')
    axes[1].set_xlabel('Actions')
    axes[1].legend(title='Dataframe')
    axes[1].tick_params(axis='x', rotation=45)

    plt.show()

# This function can be used as follows:
# compare_user_action_stats([df1, df2, df3], ['DataFrame1', 'DataFrame2', 'DataFrame3'])

from collections import Counter
from itertools import chain
import pandas as pd
from nltk.util import ngrams

# Define the modified function to use the n most common n-grams
def create_combined_actions_df(dataframe, n, use_n_most_common=False):
    # Create a sequence of actions with their timestamps for each user
    user_actions = dataframe.groupby('user_id').apply(lambda df: df.sort_values('timestamp')[['action', 'timestamp']].values.tolist())

    # Generate n-grams for each user's action sequence
    n_grams_per_user = user_actions.apply(lambda actions: list(ngrams(actions, n)))

    # Identify common n-grams or the n most common n-grams
    all_n_grams = list(chain.from_iterable(n_grams_per_user))
    n_gram_counts = Counter(map(lambda x: tuple(action for action, timestamp in x), all_n_grams))
   
    if use_n_most_common:
        common_n_grams = set(dict(n_gram_counts.most_common(n)).keys())
    else:
        common_n_grams = {n_gram for n_gram, count in n_gram_counts.items() if count > 1}

    # Create a new dataframe for combined actions
    combined_actions_data = []

    for user_id, actions in user_actions.items():
        new_sequence = []
        skip = 0
        for i in range(len(actions)):
            if skip:
                skip -= 1
                continue
            found_n_gram = False
            for j in range(n, 0, -1):
                potential_n_gram = tuple(actions[i:i+j])
                if tuple(action for action, timestamp in potential_n_gram) in common_n_grams:
                    combined_action = '_'.join(action for action, _ in potential_n_gram)
                    timestamps = [timestamp for _, timestamp in potential_n_gram]
                    overall_time = (timestamps[-1] - timestamps[0]).total_seconds()
                    combined_actions_data.append([user_id, combined_action, timestamps[0], timestamps, overall_time])
                    skip = j - 1
                    found_n_gram = True
                    break
            if not found_n_gram:
                action, timestamp = actions[i]
                combined_actions_data.append([user_id, action, timestamp, [timestamp], 0])

    # Convert to DataFrame
    combined_actions_df = pd.DataFrame(combined_actions_data, columns=['user_id', 'action', 'timestamp', 'action_timestamps', 'action_time'])
   
    return combined_actions_df

# Define the function to process and analyze dataframes
def process_and_analyze(dataframes, names, n=2):
    # Process dataframes with common n-grams
    processed_dataframes_common = [create_combined_actions_df(df, n) for df in dataframes]
    # Process dataframes with the n most common n-grams
    processed_dataframes_most_common = [create_combined_actions_df(df, n, use_n_most_common=True) for df in dataframes]
   
    # Compare action counts for common n-grams (This function needs to be provided by the user)
    # compare_action_counts(processed_dataframes_common, names, figsize=(16,8))
   
    # Compare user action stats for common n-grams
    compare_user_action_stats(processed_dataframes_common, names, figsize=(16,8))
   
    # Compare action counts for the n most common n-grams (This function needs to be provided by the user)
    # compare_action_counts(processed_dataframes_most_common, names, figsize=(16,8))
   
    # Compare user action stats for the n most common n-grams
    compare_user_action_stats(processed_dataframes_most_common, names, figsize=(16,8))

# Note: The compare_action_counts function is not provided and should be implemented by the user.

# The function call would look something like this:
# process_and_analyze([df1, df2, df3], ['DataFrame1', 'DataFrame2', 'DataFrame3'], n=2)
