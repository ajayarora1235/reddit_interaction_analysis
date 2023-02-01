import numpy as np
import networkx as nx
import pandas as pd
import scipy as sp
from scipy.sparse import csr_matrix
import networkx.algorithms.community as nx_comm
import collections
import operator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# one thing to compare activity in various subreddits between two years
class ActivityComparison:
  def __init__():
    #create adjacency matrix
    author_list = pd.read_csv('authors_big.csv')
    interactions = pd.read_csv('interactions_cleaned.csv')
    author_sub_counts = pd.read_csv('author_subreddit_counts_big.csv')
    comment_sub_counts = pd.read_csv('comment_subreddit_counts.csv')

  def overlap_calculation():
    political_reddits = ["Conservative", "Republican", "tucker_carlson", "trump", "conservatives", "HillaryForPrison", "ConservativesOnly"]
    political_subs = author_sub_counts[author_sub_counts["subreddit"].isin(political_reddits)]

    political_comms = comment_sub_counts[comment_sub_counts["subreddit"].isin(political_reddits)]
    political_comms = political_comms[political_comms["comment_cnt"] > 1]

    political_comms.sort_values(by=['subreddit'], inplace=True)
    political_comms['subreddits'] = political_comms.groupby(['author','year'])['subreddit'].transform(lambda x: ','.join(x))
    
    df_two = political_comms[["author", "year", "comment_cnt"]].groupby(["author", "year"]).sum().reset_index()
    df = df_two.groupby(["author"]).sum()
    df = df[df["year"]==4041]
    df = df.reset_index()

    political_comms_two = political_comms.merge(df["author"], on="author")
    political_comms_three = political_comms_two[["author", "year", "subreddits"]].drop_duplicates()
    reddit_combs = political_comms_three["subreddits"].value_counts().index.tolist()
    year_one_subs = political_comms_three[political_comms_three["year"]==2020]
    year_two_subs = political_comms_three[political_comms_three["year"]==2021]

    overlap_matrix = np.zeros((len(reddit_combs), len(reddit_combs)))
    for ind in range(len(year_one_subs)):
      overlap_matrix[reddit_combs.index(year_one_subs.iloc[[ind]]["subreddits"].tolist()[0])][reddit_combs.index(year_two_subs.iloc[[ind]]["subreddits"].tolist()[0])] += 1
    
    return overlap_matrix



  def activity_plot():
    overlap_df = []
      for x in range(len(reddit_combs)):
        for y in range(len(reddit_combs)):
          d = {
              '2020activity' : reddit_combs[x],  # some formula for obtaining values
              '2021activity' : reddit_combs[y],
              'commSize2020': np.sum(overlap_matrix[x], axis=0),
              'commSize2021': np.sum(overlap_matrix[:, y], axis=0),
              'overlap' : round(float(overlap_matrix[x][y]/np.sum(overlap_matrix[x], axis=0)), 2),
              'overlap_raw' : overlap_matrix[x][y],
          }
          #if np.sum(overlap_matrix[x], axis=0) > 3 and np.sum(overlap_matrix[:, y], axis=0) > 3:
          overlap_df.append(d)

      overlap_df = pd.DataFrame(overlap_df)

              
      alt.data_transformers.disable_max_rows()
      alt.Chart(
        overlap_df,
        title="2010 Daily High Temperature (F) in Seattle, WA"
      ).mark_rect().encode(
        x='2020activity:O',
        y='2021activity:O',
        color=alt.Color('overlap:Q', scale=alt.Scale(scheme="inferno")),
        tooltip=[
            alt.Tooltip('2020activity:N', title='Reddits 2020'),
            alt.Tooltip('2021activity:N', title='Reddits 2021'),
            alt.Tooltip('commSize2020:Q', title='Comm Size 2020'),
            alt.Tooltip('commSize2021:Q', title='Comm Size 2021'),
            alt.Tooltip('overlap:Q', title='overlap'),
            alt.Tooltip('overlap_raw:Q', title='overlap raw count')
        ]
      ).properties(width=550)

# compare overlap between communities in two different years
class CommunityComparison:
  '''
  authors_csv: a one column csv containing all author usernames whose posts are being studied
  intx_csv: a three column csv containing the author, link_id, and subreddit for each comment 
    with at least one interaction with another comment by another author in the dataset
  subreddit_posts_csv: a four column csv containing the author, subreddit, year, and the number of posts
    made by said author in the subreddit ('post_count') for that year
  '''
  def __init__(self, author_csv, intx_csv, subreddit_posts_csv):
    self.authors = pd.read_csv(authors_csv)
    self.intx_posts = pd.read_csv(intx_posts_csv) 
    self.intx_comments = pd.read_csv(intx_comments_csv)
    self.subreddit_counts = pd.read_csv(subreddit_posts_csv)
    self.author_dict = self.create_author_dict()

  def create_author_dict(self):
    a = list(self.authors['author'])
    author_dict = {}
    for i in range(len(a)):
      author_dict[a[i]] = i
    return author_dict

  def intx_preprocessing(self):        
    interactions_one['parent_index'] = interactions_one.apply(lambda row: self.author_dict[row['submissions_author']], axis=1)
    interactions_one['child_index'] = interactions_one.apply(lambda row: self.author_dict[row['comment_authors']], axis=1)
    interactions_two['parent_index'] = interactions_two.apply(lambda row: self.author_dict[row['parentAuthor']], axis=1)
    interactions_two['child_index'] = interactions_two.apply(lambda row: self.author_dict[row['replyAuthor']], axis=1)

    final_df = pd.concat([interactions_one[['parent_index', 'child_index', 'subreddit', 'day']], 
                          interactions_two[['parent_index', 'child_index', 'subreddit', 'day']]], axis=0).reset_index()[['parent_index', 'child_index', 'subreddit', 'day']]
    final_df.to_csv('interactions_cleaned_big.csv')

    year_one_actions = final_df[final_df['day'] <= '2020-12-31']
    year_two_actions = final_df[final_df['day'] >= '2021-01-01']

    #get all pairs and corresponding counts
    year_one_df = year_one_actions.groupby(['parent_index','child_index']).size().reset_index()
    year_one_df.rename(columns = {0: 'frequency'}, inplace = True)
    year_two_df = year_two_actions.groupby(['parent_index','child_index']).size().reset_index()
    year_two_df.rename(columns = {0: 'frequency'}, inplace = True)

  def louvain_detection(self, freq_df):
    row = np.array(freq_df['parent_index'])
    col = np.array(freq_df['child_index'])
    data = np.array(freq_df['frequency'])
    
    matr = csr_matrix((data, (row, col)), shape=(len(author_list), len(author_list)))
    G = nx.from_scipy_sparse_matrix(matr, create_using=nx.MultiGraph)
    z = nx_comm.louvain_communities(G, resolution=1.6)
    
    single_sets = []
    multisize_sets = 0
    multisize_size = 0
    final_z = []
    for i in z:
      if len(i) < 2:
        single_sets.append(list(i)[0])
      else:
        multisize_sets += 1
        multisize_size += len(i)
        final_z.append(i)
            
    print(multisize_sets, multisize_size)
    return final_z, single_sets

  def louvain_postprocessing(self):
    comm_year_one, single_sets_year_one = louvain_func(year_one_df)
    comm_year_two, single_sets_year_two = louvain_func(year_two_df)

    year_one_cluster_list = [-1 for x in range(len(a))]
    year_two_cluster_list = [-1 for x in range(len(a))]
    for comm_index in range(len(comm_year_one)):
      for ind in comm_year_one[comm_index]:
        year_one_cluster_list[ind] = comm_index
        
    for comm_index in range(len(comm_year_two)):
      for ind in comm_year_two[comm_index]:
        year_two_cluster_list[ind] = comm_index

  def community_subreddit_data(comm_segmentation, year, topX=True, exceptions=['AskReddit', 'memes', 'dankmemes', 'Showerthoughts']):
    top_subreddits_by_post = [0 for x in range(len(comm_segmentation))]
    top_subreddits_by_active_users = [0 for x in range(len(comm_segmentation))]
    
    all_subreddits = complete_reddit_counts[complete_reddit_counts["year"]==year].subreddit.unique().tolist()
    
    
    action_counts_comm = complete_reddit_counts[complete_reddit_counts["year"]==year].groupby([str(year)+"ind",'subreddit'])["post_cnt"].sum().reset_index()
    action_counts_comm.rename(columns = {0: 'frequency'}, inplace = True)
    
    action_users_comm = complete_reddit_counts[complete_reddit_counts["year"]==year].groupby([str(year)+"ind",'subreddit']).size().reset_index()
    action_users_comm.rename(columns = {0: 'frequency'}, inplace = True)
    
    for comm_index in range(len(comm_segmentation)):
      comm_cluster_counts = action_counts_comm[action_counts_comm[str(year)+"ind"] == comm_index]
      comm_cluster_counts.sort_values(['post_cnt'],ascending=False, inplace = True)        
      top_reddits = comm_cluster_counts[~comm_cluster_counts['subreddit'].isin(exceptions)].head(5)
      top_subreddits_by_post[comm_index] = dict(zip(top_reddits.subreddit, top_reddits.post_cnt))
      
      comm_cluster_users = action_users_comm[action_users_comm[str(year)+"ind"] == comm_index]
      comm_cluster_users.sort_values(['frequency'],ascending=False, inplace = True)        
      top_reddits_users = comm_cluster_users[~comm_cluster_users['subreddit'].isin(exceptions)].head(5)
      top_subreddits_by_active_users[comm_index] = dict(zip(top_reddits_users.subreddit, top_reddits_users.frequency))
      
      #print(comm_index)
        
    return top_subreddits_by_post, top_subreddits_by_active_users, all_subreddits

  def community_interaction_data(actions, comm_dict, year, num_clusters, topX=True, exceptions=[]):
    parent_clusters = pd.DataFrame(list(comm_dict.items()), columns = ['parent_index', 'parent_cluster'])
    child_clusters = pd.DataFrame(list(comm_dict.items()), columns = ['child_index', 'child_cluster'])
    
    #make df of parent_index and cluster
    df2 = actions.merge(parent_clusters, on='parent_index')
    action_cluster_df = df2.merge(child_clusters, on='child_index')
    
    action_counts_comm = action_cluster_df.groupby(['parent_cluster','child_cluster']).size().reset_index()
    action_counts_comm.rename(columns = {0: 'frequency'}, inplace = True)
    
    #create interaction matrix
    interaction_matrix = np.zeros((num_clusters, num_clusters))
    for ind, row in action_counts_comm.iterrows():
      interaction_matrix[row['parent_cluster']][row['child_cluster']] += row['frequency']
        
    #find top subreddits for each cluster-cluster interaction
    same_comm_intx = action_cluster_df[action_cluster_df['parent_cluster'] == action_cluster_df['child_cluster']]
    same_comm_subreddit_intx = same_comm_intx.groupby(['parent_cluster','child_cluster', 'subreddit']).size().reset_index()
    same_comm_subreddit_intx.rename(columns = {0: 'frequency'}, inplace = True)
    
    #for each cluster output dictionary of top 5 subreddits and counts
    top_subreddits_by_num_intx = [0 for x in range(num_clusters)]
    for cluster in range(num_clusters):
      cluster_intx = same_comm_subreddit_intx[same_comm_subreddit_intx['parent_cluster']==cluster]
      intx_freq_dict = dict(zip(cluster_intx.subreddit, cluster_intx.frequency))
      top_subreddits_by_num_intx[cluster] = dict(sorted(intx_freq_dict.items(), key=operator.itemgetter(1),reverse=True))
      if topX:
        top_subreddits_by_num_intx[cluster] = dict(sorted(intx_freq_dict.items(), key=operator.itemgetter(1),reverse=True)[:5])
      
    return interaction_matrix, top_subreddits_by_num_intx

  def community_rankings(self):
    post_ranking_year_one, user_rankings_year_one, all_subreddits_yr_one = community_subreddit_data(comm_year_one, 2020, topX=False)
    post_ranking_year_two, user_rankings_year_two, all_subreddits_yr_two = community_subreddit_data(comm_year_two, 2021, topX=False)
    intx_matrix, intx_subreddit_ranking = community_interaction_data(year_one_actions, 
                                                                    year_one_cluster_dict, 
                                                                    2020, 
                                                                    len(comm_year_one),
                                                                    topX=False)
    intx_matrix_two, intx_subreddit_ranking_two = community_interaction_data(year_two_actions, 
                                                                            year_two_cluster_dict, 
                                                                            2021, 
                                                                            len(comm_year_two),
                                                                            topX=False)

  def alt_community_rankings(self):
    _, user_rankings_year_one, all_subreddits = community_subreddit_data(comm_year_one, 2020)
    _, user_rankings_year_two, all_subreddits = community_subreddit_data(comm_year_two, 2021)
    intx_matrix, intx_subreddit_ranking = community_interaction_data(year_one_actions, 
                                                                    year_one_cluster_dict, 
                                                                    2020, 
                                                                    len(comm_year_one),
                                                                    topX=True)


  def get_unique_subreddits(self):
    cluster_unique_subreddits_year_one = [[] for x in range(len(comm_year_one))]
    for reddit in all_subreddits:
      i = 0
      for ranking in [user_rankings_year_one, intx_subreddit_ranking]:
        amt_list = []
        for amt_dict in ranking:
          if reddit in amt_dict:
            amt_list.append(amt_dict[reddit])
          else:
            amt_list.append(0)
        if sum(amt_list) >= 15:
          orig_sum = sum(amt_list)
          amt_list = [x/float(sum(amt_list)) for x in amt_list]
          yes = []
          num_nonzero = 0
          for ind in range(len(amt_list)):
            if amt_list[ind] >= 0.4:
              yes.append(ind)
            if amt_list[ind] > 0:
              num_nonzero += 1
          if num_nonzero <= 5:
            #print('ranking type ' + str(i))
            print(reddit)
            #print(orig_sum)
            #print(amt_list)
            #print(str(num_nonzero) + " comms with subreddit.")
            for ind in yes:
              cluster_unique_subreddits_year_one[ind].append(reddit)
          i += 1

  def yearwise_heatmap(self):
    #subreddit heat map
    overlap_matrix = np.zeros((len(comm_year_one), len(comm_year_two)))
    print(len(comm_year_one), len(comm_year_two))
    for comm_index in range(len(comm_year_one)):
        for yr_two_clusters in year_two_comm_dist[comm_index]:
            overlap_matrix[comm_index][yr_two_clusters] += 1

    overlap_df = []
    for x in range(len(comm_year_one)):
        for y in range(len(comm_year_two)):
            d = {
                '2020cluster' : x,  
                '2021cluster' : y,
                '2020size': len(comm_year_one[x]),
                '2021size': len(comm_year_two[y]),
                'overlap' : round(float(overlap_matrix[x][y]/len(comm_year_one[x])), 2),
                'overlap_raw' : overlap_matrix[x][y],
                '2020topSubreddits': str(user_rankings_year_one[x]),
                '2021topSubreddits': str(user_rankings_year_two[y]),
                '2020uniqueSubreddits': str(cluster_unique_subreddits_year_one[x]),
                '2021uniqueSubreddits': str(cluster_unique_subreddits_year_two[x])
            }
            if len(comm_year_two[y]) > 8 and len(comm_year_one[x]) > 8:
                overlap_df.append(d)

    overlap_df = pd.DataFrame(overlap_df)
    print(len(overlap_df))
            
    alt.data_transformers.disable_max_rows()
    alt.Chart(
        overlap_df,
        title="Louvain community detection: overlap from 2020 to 2021"
    ).mark_rect().encode(
        x='2020cluster:O',
        y='2021cluster:O',
        color=alt.Color('overlap:Q', scale=alt.Scale(scheme="inferno")),
        tooltip=[
            alt.Tooltip('2020size:Q', title='Size of 2020 cluster'),
            alt.Tooltip('2021size:Q', title='Size of 2021 cluster'),
            alt.Tooltip('overlap:Q', title='overlap'),
            alt.Tooltip('overlap_raw:Q', title='overlap raw count'),
            alt.Tooltip('2020topSubreddits:N', title='Top subreddits 2020'),
            alt.Tooltip('2021topSubreddits:N', title='Top subreddits 2021'),
            alt.Tooltip('2020uniqueSubreddits:N', title='Unique subreddits 2020'),
            alt.Tooltip('2021uniqueSubreddits:N', title='Unique subreddits 2021')
        ]
    ).properties(width=550)
    
    

  
    

