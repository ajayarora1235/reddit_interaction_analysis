import numpy as np
import networkx as nx
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.sparse import csr_matrix
import networkx.algorithms.community as nx_comm
import community as community_louvain
from utils import *

class CommunityDetection:
  '''
  authors_csv: a one column csv containing all author usernames whose posts are being studied
  intx_csv: a three column csv containing the author, link_id, and subreddit for each comment 
    with at least one interaction with another comment by another author in the dataset
  subreddit_posts_csv: a three column csv containing the author, subreddit, and the number of posts
    made by said author in the subreddit ('post_count')
  '''
  def __init__(self, authors_csv, intx_csv, subreddit_posts_csv):
    self.authors = pd.read_csv(authors_csv)
    self.intx = pd.read_csv(intx_csv) 
    self.subreddit_counts = pd.read_csv(subreddit_posts_csv)
    self.author_dict = utils.create_author_dict(self.authors)

  '''
  returns a dataframe returning interaction counts of author pairs
  '''
  def intx_preprocessing(self):
    intx_grouped = self.intx.groupby(['link_id','subreddit'])['author']
      .apply(lambda x: ','.join(x))
      .reset_index()
    numPosts = len(intx_grouped)

    rowVals = np.zeros(numPosts*3)
    col = np.zeros(numPosts*3)
    value = np.zeros(numPosts*3)

    intx_dict = defaultdict(lambda: 0)
    for index,row in intx_grouped.iterrows():
      if (index % 10000 == 0):
          print(index)
      for pair in list(combinations(row['author'].split(','), 2)):
          if author_dict[pair[1]] < author_dict[pair[0]]:
              rowVals[index] = author_dict[pair[1]]
              col[index] = author_dict[pair[0]]
          else:
              rowVals[index] = author_dict[pair[1]]
              col[index] = author_dict[pair[0]]
          value[index] = 1

    final_df = pd.DataFrame({'parent_index': rowVals, 'child_index': col, 'freq': value})
      .groupby(['parent_index','child_index']).size().reset_index()
    final_df.rename(columns = {0: 'frequency'}, inplace = True)
    final_df = final_df.iloc[1: , :]

    return final_df

  def subreddit_cnt_preprocessing(self):
    all_subreddits = list(set(self.subreddit_counts.subreddit.unique().tolist()))
    starting_ind = len(a)
    subreddit_dict = {}
    for i in range(len(all_subreddits)):
        subreddit_dict[all_subreddits[i]] = starting_ind + i

    comment_counts = self.subreddit_counts
    comment_counts['author_index'] = comment_counts.apply(lambda row: author_dict[row['author']], axis=1)
    comment_counts['subreddit_index'] = comment_counts.apply(lambda row: subreddit_dict[row['subreddit']], axis=1)

    activity_intx = comment_counts[['author_index', 'subreddit_index', 'post_count']]

    return activity_intx, comment_counts, all_subreddits

  def community_analysis(self):
    freq_df = self.intx_preprocessing()
    activity_intx, comment_counts, all_subreddits = self.subreddit_cnt_preprocessing()
    dim = len(a) + len(all_subreddits)
    comms, _, G = utils.louvain_detection(freq_df, dim, res=1.6, activity_intx)
    year_one_cluster_list, _, _, _ = utils.louvain_postprocessing(comms, comm_year_two=None, dim)

    conservative_counts = comment_counts[comment_counts['subreddit'] == 'Conservative']
    author_df = pd.DataFrame(data={'author': a + all_subreddits, "author_index": range(dim), "comm #": year_one_cluster_list})
    conservative_counts_two = conservative_counts.merge(author_df, on='author_index')

    conservative_counts_comm = conservative_counts_two.groupby(['comm #'])["post_count"].sum().reset_index()

    cons_counts = []

    for comm_index in range(len(comms)):
      new_G = G.subgraph(comms[comm_index])
      people = [x for x in comms[comm_index] if x < len(a)]
      num_people = len(people)
      print("\nsubreddits in comm " + str(comm_index) + " (" + str(num_people) + " people, "  + str(len(comms[comm_index]) - num_people) + " reddits)")
      print(str(round(conservative_counts_comm[conservative_counts_comm["comm #"] == comm_index]["post_count"].tolist()[0]/num_people, 2)) + " conservative comments per person in community")
      cons_counts.append(conservative_counts_comm[conservative_counts_comm["comm #"] == comm_index]["post_count"].tolist()[0])
            
      total_num_edges = new_G.number_of_edges()
      num_intx_edges = new_G.subgraph(people).number_of_edges()
      num_comment_edges = total_num_edges - num_intx_edges
      
      total_deg = 0
      pp_total_deg = 0 #only between people
      rp_total_deg = 0
      
      for ind in comms[comm_index]:
        total_deg += G.degree(ind, "weight")
        if (ind > len(a)):
          rp_total_deg += G.degree(ind, "weight")
          if (G.degree(ind)>800):
            print(all_subreddits[ind-len(a)] + " " + str(G.degree(ind, "weight")) + " " + str(new_G.degree(ind, "weight")/G.degree(ind, "weight")))
                  
      pp_total_deg = total_deg - rp_total_deg
      print(str(total_num_edges) + " in-edges, " + str(round(num_intx_edges/total_num_edges, 2)) + " proportion interactions")
      print(str(round(num_intx_edges/pp_total_deg, 3)) + " proportion interaction edges in-community, " + str(round(num_comment_edges/rp_total_deg, 2)) + " proportion comment edges in-community.")
      