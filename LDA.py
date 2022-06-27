class lineardiscriminantanalysis:
  def __init__(self):
    self.categories_sorted = []
  def find_scatter_mats(self,X,y):
    Scatter_Within = np.zeros((X.shape[1],X.shape[1]))
    Scatter_Between = np.zeros((X.shape[1],X.shape[1]))
    categories = [i for i in X]
    print(categories)
    total_mean = [0 for i in range(len(categories))]
    '''Find the means'''
    for i in range(len(categories)):
        for j in range(len(X)):
          total_mean[i]+=X[categories[i]][j]
        total_mean[i] = total_mean[i]/len(X)
    for cls in y['classes'].unique():
      temp = X[y['classes']==cls]
      temp.reset_index(inplace = True,drop = True)
      means = [0 for i in range(len(categories))]
      '''again calculating the means'''
      for i in range(len(categories)):
        for j in range(len(temp)):
          means[i]+=temp[categories[i]][j]
        means[i] = means[i]/len(temp)
      '''Scatte whiting is basically variances'''
      for i in range(len(categories)):
        for j in range(len(categories)):
          for k in range(len(temp)):
            Scatter_Within[i,j]+=((temp[categories[i]][k]-means[i])*(temp[categories[j]][k]-means[j]))
      dif_means = np.array(means)-np.array(total_mean)
      dif_means = dif_means.reshape(len(categories),1)
      '''Finding the scatter whiting matrix'''
      Scatter_Between +=(len(temp)*(dif_means@dif_means.T))
    return Scatter_Within,Scatter_Between
          
  def discriminant_selector(self,explained_variables,percent):
    n = 0
    s = 0
    percent = percent/100
    for i in range(len(explained_variables)):
      if(s<percent):
        s+=explained_variables[i]
        n+=1
    return n


  def fit_Transform(self,X,y,percent):
    '''finding the scatter matrix'''
    SW ,SB = self.find_scatter_mats(X,y)
    A = np.linalg.inv(SW).dot(SB)
    '''Finding the eigen values and vectors'''
    values = QR_eigenvalues(A)
    vectors = []
    for i in range(len(values)):
      vectors.append(power_iteration(A,values[i]))
    vectors = np.array(vectors)
    vectors = vectors.T
    idx = np.argsort(abs(values))[::-1]
    categories = [i for i in X]
    temp_cat = []
    
    for i in idx:
      self.categories_sorted.append(categories[i])
    values = values[idx]
    vectors = vectors[idx]
    variables = []
    for i in range(len(values)):
      variables.append(values[i]/np.sum(values))
    print(np.sum(variables),"\n",variables)
    n_components = self.discriminant_selector(variables,percent)
    print(n_components)
    linear_discriminants = vectors[0:n_components]
    print(self.categories_sorted)
    return np.dot(X,linear_discriminants.T)
