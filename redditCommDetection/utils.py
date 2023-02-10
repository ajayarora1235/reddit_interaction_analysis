def create_author_dict(authors):
  a = list(authors['author'])
  author_dict = {}
  for i in range(len(a)):
    author_dict[a[i]] = i
  return author_dict

def louvain_detection(self, freq_df, dim, res=1, activity_intx=None):
  row = np.array(freq_df['parent_index'])
  col = np.array(freq_df['child_index'])
  data = np.array(freq_df['frequency'])

  if activity_intx is not None:
    row = np.concatenate([row, activity_intx.author_index.tolist()])
    col = np.concatenate([col, activity_intx.subreddit_index.tolist()])
    data = np.concatenate([data, activity_intx.post_count.tolist()])

  #dim = len(author_list)+len(all_subreddits)
  matr = csr_matrix((data, (row, col)), shape=(dim, dim))
  G = nx.from_scipy_sparse_matrix(matr)
  G.remove_nodes_from(list(nx.isolates(G)))
  z = nx_comm.louvain_communities(G, resolution=res) #1.6 w/o activity, 1 with

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
  return final_z, single_sets, G
  #final_z, single_sets

def louvain_postprocessing(self, comm_year_one, comm_year_two=None, dim):
  comm_year_one, _ = louvain_func(year_one_df)
  comm_year_two, _ = louvain_func(year_two_df)

  year_one_cluster_list = [-1 for x in range(dim)]
  year_two_cluster_list = [-1 for x in range(dim)]
  partition_year_one = {}
  partition_year_two = {}

  for comm_index in range(len(comm_year_one)):
    for ind in comm_year_one[comm_index]:
      year_one_cluster_list[ind] = comm_index
      partition_year_one[ind] = comm_index 
      
  if comm_year_two is not None:
    for comm_index in range(len(comm_year_two)):
      for ind in comm_year_two[comm_index]:
        year_two_cluster_list[ind] = comm_index
        partition_year_two[ind] = comm_index 

  return year_one_cluster_list, year_two_cluster_list, partition_year_one, partition_year_two



