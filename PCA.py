class PCA():
  def __init__(self):
    return
  ''' This Centers the Data'''
  def center_data(self,data):
    categories = []
    for i in data:
      categories.append(i)
    means = [0 for i in range(len(categories))]
    for i in range(len(categories)):
      for j in range(len(data)):
        #print(i,j)
        means[i]+=data[categories[i]][j]
      means[i] = means[i]/len(data)

    for i in range(len(categories)):
      for j in range(len(data)):
        data[categories[i]][j] = data[categories[i]][j]-means[i]
    return data
  '''Finds the Covariance'''
  def covariance(self,data):
    categories = []
    for i in data:
      categories.append(i)
    means = [0 for i in range(len(categories))]
    for i in range(len(categories)):
      for j in range(len(data)):
        means[i]+=data[categories[i]][j]
      means[i] = means[i]/len(data)
    cov = [[0 for i in range(len(categories))] for j in range(len(categories))]

    for i in range(len(categories)):
      for j in range(len(categories)):
        for k in range(len(data)):
          cov[i][j]+=((data[categories[i]][k]-means[i])*(data[categories[j]][k]-means[j]))
        cov[i][j] = cov[i][j]/(len(data)-1)
    return np.array(cov)

  '''Finds the principal Components'''
  def principal_components(self,data):
    #print('in')
    data = self.center_data(data)
    #print('out')
    cov = self.covariance(data)
    #print('out')
    '''QR decomposition for eigenvalues'''
    values = QR_eigenvalues(cov)
    vectors = []
    '''find the eigen vectors from eigen values'''
    for i in range(len(values)):
      vectors.append(power_iteration(cov,values[i]))
    vectors = np.array(vectors)
    #print(inv_power(cov,values[0]))
    idx = np.argsort(abs(values))[::-1]
    categories = [i for i in data]
    self.categories_sorted = []
    for i in idx:
      self.categories_sorted.append(categories[i])
    print(self.categories_sorted)
    #print('in')
    self.variables = []
    for i in range(len(values)):
      self.variables.append(values[i]/np.sum(values))
    print(np.sum(self.variables),"\n",self.variables)
    '''doing matrix multiplication to find the transformed vectors'''
    projected1 = data.dot(vectors.T[0])
    projected2 = data.dot(vectors.T[1])
    res = pd.DataFrame(projected1,columns = ['PC1'])
    res['PC2'] = projected2
    #print(len(res['PC1'].unique()),len(res['PC2'].unique()))
    return res
  '''plots the graph'''
  def plot_graph(self):
    feature_no = [i for i in range(1,len(self.variables)+1)]
    cummulative_sum = [0 for i in range(1,len(self.variables)+1)]
    for i in range(len(self.variables)):
      if(i==0):
        cummulative_sum[i]+=self.variables[i]
      else:
        cummulative_sum[i]+=self.variables[i]+cummulative_sum[i-1]
    plt.figure()
    sns.lineplot(feature_no,cummulative_sum)
    plt.xlabel('Feature No')
    plt.ylabel('Cumullative Sum of Variances')





