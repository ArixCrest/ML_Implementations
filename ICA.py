class ICA():
  def __init__(self,n_components,n_iters,tolerance):
    self.n_components = n_components
    self.means = []
    self.n_iters = n_iters
    self.tolerance = tolerance
  def center(self,data):
    n,m = data.shape
    for i in range(m):
      mean = 0
      for j in range(n):
        mean+=data[j,i]
      mean = mean/n
      self.means.append(mean)
      for j in range(m):
        data[i,j] -=mean
    return data
  def D_g(self,data):
    return 1-data*data
  def g(self,data):
    return np.tanh(data)
  def covariance(self,data):
    n,m = data.shape
    print(n,m)
    cov = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
      for j in range(n):
        print(i,j)
        for k in range(m):
          cov[i][j]+=((data[i,k]-self.means[i])*(data[j,k]-self.means[j]))
        cov[i][j] = cov[i][j]/(n)
    return np.array(cov)
  def whiten(self,data):
    cov = self.covariance(data)
    eig,vec = np.linalg.eigh(cov)
    D = np.diag(eig)
    print(D.shape,vec.shape,data.shape)
    data_whiten = np.dot(vec,np.dot(np.sqrt(np.linalg.inv(D)),np.dot(vec,data)))
    return data_whiten
  def demix_mat(self,data,mat):
    #print("in\n")
    mat_new = (data*self.g(np.dot(mat.T,data))).mean(axis=1)
    #print("out2")
    mat_new-= self.D_g(np.dot(mat.T,data)).mean(axis = 0)*mat
    #print('out')
    mat_new /= np.sqrt((mat**2).sum())
    #print('out')
    return mat_new


  def ica(self,data):
    n,m = data.shape
    W = np.zeros((n,n),dtype = 'float64')
    print(n,m)
    for i in range(n):
      print('in')
      w = np.random.rand(n)
      for j in range(self.n_iters):
        w_new = self.demix_mat(data,w)
        if(i>=1):
          w_new = w_new-np.dot(np.dot(w_new,W[:i].T),W[:i])
        #print(np.abs(np.abs(w*w_new).sum()))
        dist = np.abs(np.abs(w*w_new).sum()-1)
        #print(dist)
        if dist<=self.tolerance:
          w = w_new
          break
        w = w_new
      W[i,:] = w
    final =  np.dot(W,data)
    return final
  def fit_transform(self,data):
    data = self.center(data)
    data = self.whiten(data.T)
    data = self.ica(data)
    return data.T



