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
    self.author_dict = self.create_author_dict()


  def create_author_dict():
    a = list(self.authors['author'])
    author_dict = {}
    for i in range(len(a)):
      author_dict[a[i]] = i
    return author_dict

  '''
  returns a dataframe returning interaction counts of author pairs
  '''
  def intx_preprocessing():
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

    final_df = pd.DataFrame({'parent': rowVals, 'child': col, 'freq': value})
      .groupby(['parent','child']).size().reset_index()
    final_df.rename(columns = {0: 'frequency'}, inplace = True)
    final_df = final_df.iloc[1: , :]

    return final_df

  def subreddit_cnt_preprocessing():
    all_subreddits = list(set(self.subreddit_counts.subreddit.unique().tolist()))
    starting_ind = len(a)
    subreddit_dict = {}
    for i in range(len(all_subreddits)):
        subreddit_dict[all_subreddits[i]] = starting_ind + i

    comment_counts = self.subreddit_counts
    comment_counts['author_index'] = comment_counts.apply(lambda row: author_dict[row['author']], axis=1)
    comment_counts['subreddit_index'] = comment_counts.apply(lambda row: subreddit_dict[row['subreddit']], axis=1)

    activity_intx = comment_counts[['author_index', 'subreddit_index', 'post_count']]

    return activity_intx

  def louvain_detection():
    rowLov = np.concatenate([rowVals,activity_intx.author_index.tolist()])
    colLov = np.concatenate([col, activity_intx.subreddit_index.tolist()])
    dataLov = np.concatenate([value, activity_intx.post_count.tolist()])

    print(rowLov.shape)
    print(colLov.shape)
    print(dataLov.shape)

    matr = csr_matrix((dataLov, (rowLov, colLov)), shape=(len(author_list)+len(all_subreddits), len(author_list)+len(all_subreddits)))
    G = nx.from_scipy_sparse_matrix(matr)
    G.remove_nodes_from(list(nx.isolates(G)))
    z = nx_comm.louvain_communities(G, resolution=1)

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
    #final_z, single_sets

  def clustering_postprocessing():
    partition = {}
    for comm_index in range(len(final_z)):
        for ind in final_z[comm_index]:
            partition[ind] = comm_index
            
    year_one_cluster_list = [-1 for x in range(len(a) + len(all_subreddits))]
    for comm_index in range(len(final_z)):
        for ind in final_z[comm_index]:
            year_one_cluster_list[ind] = comm_index

  def community_analysis():
    conservative_counts = comment_counts[comment_counts['subreddit'] == 'Conservative']
    author_df = pd.DataFrame(data={'author': a + all_subreddits, "author_index": range(len(a) + len(all_subreddits)), "comm #": year_one_cluster_list})
    conservative_counts_two = conservative_counts.merge(author_df, on='author_index')

    conservative_counts_comm = conservative_counts_two.groupby(['comm #'])["post_count"].sum().reset_index()

    cons_counts = []

    for comm_index in range(len(final_z)):
      new_G = G.subgraph(final_z[comm_index])
      people = [x for x in final_z[comm_index] if x < len(a)]
      num_people = len(people)
      print("\nsubreddits in comm " + str(comm_index) + " (" + str(num_people) + " people, "  + str(len(final_z[comm_index]) - num_people) + " reddits)")
      print(str(round(conservative_counts_comm[conservative_counts_comm["comm #"] == comm_index]["post_count"].tolist()[0]/num_people, 2)) + " conservative comments per person in community")
      cons_counts.append(conservative_counts_comm[conservative_counts_comm["comm #"] == comm_index]["post_count"].tolist()[0])
            
      total_num_edges = new_G.number_of_edges()
      num_intx_edges = new_G.subgraph(people).number_of_edges()
      num_comment_edges = total_num_edges - num_intx_edges
      
      total_deg = 0
      pp_total_deg = 0 #only between people
      rp_total_deg = 0
      
      for ind in final_z[comm_index]:
        total_deg += G.degree(ind, "weight")
        if (ind > len(a)):
          rp_total_deg += G.degree(ind, "weight")
          if (G.degree(ind)>800):
            print(all_subreddits[ind-len(a)] + " " + str(G.degree(ind, "weight")) + " " + str(new_G.degree(ind, "weight")/G.degree(ind, "weight")))
                  
      pp_total_deg = total_deg - rp_total_deg
      print(str(total_num_edges) + " in-edges, " + str(round(num_intx_edges/total_num_edges, 2)) + " proportion interactions")
      print(str(round(num_intx_edges/pp_total_deg, 3)) + " proportion interaction edges in-community, " + str(round(num_comment_edges/rp_total_deg, 2)) + " proportion comment edges in-community.")
      