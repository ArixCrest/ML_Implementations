class Kmeans():
  def __init__(self,clusters,n_iters):
    self.centroids = []
    self.clusters = clusters
    self.size = 784
    self.n_iters = n_iters
  def initialize_centroids(self,X,random = True,points = None):
    n = len(X)
    if(random):
      ids = np.random.choice(n,self.clusters,replace = False)
      self.centroids = X[ids,:]
      #print(self.centroids.shape)
    else:
      self.centroids = np.array(points)
      #print(self.centroids.shape)
  def euclidean_distance(self,X):
    n,m = X.shape
    print(X[0].shape)
    ans = 0
    temp = []
    for i in range(self.size):
      ans+=(X[0,i]-X[1,i])**2
    print(ans)
    print(np.sqrt(ans))
    for i in range(1,len(X)):
      a = X[:1,:]-self.centroids
      for j in range(self.size):
        print(a[0,i],(self.centroids[0,j]-X[1,j])**2,X[1,i],self.centroids[0,i])
      print(a.sum(axis =1))
      print(a.shape)
      #k = np.sqrt(((self.centroids-a[i])**2).sum(axis = 1))
      #print(k)
      return
    ans = np.sqrt(ans)
    return ans
  def euclidean_numpy(self,X):
    n,m = X.shape
    ans = []
    for i in range(len(X)):
      temp = []
      for j in range(self.clusters):
        temp.append(np.linalg.norm(X[i,:]-self.centroids[j,:]))
      ans.append(temp)
    return np.array(ans)
  def noob_euclidean(self,a):
    temp = []
    for i in range(len(a)):
      temp2 = []
      for j in range(self.clusters):
        c = 0
        for k in range(self.size):
          c+=(a[i,k]-self.centroids[j,k])**2
        temp2.append(np.sqrt(c))
      temp.append(temp2)
    return np.array(temp)
  def calc_loss(self,a):
    ans = 0
    for i in range(self.clusters):
      k = 0
      for j in range(self.size):
        k+=(a[i,j]-self.centroids[i,j])**2
      k = np.sqrt(k)
      ans+=k
    ans = ans/10
    return ans
        

  def train(self,X):
    dist = self.euclidean_numpy(X)
    groups = np.array([np.argmin(i) for i in dist])
    for i in range(self.n_iters):
      updates = []
      for j in range(self.clusters):
        temp = X[groups==j].mean(axis = 0)
        updates.append(temp)
      loss = self.calc_loss(np.array(updates))
      print(i,loss)
      if(loss<1):
        break
      self.centroids = np.vstack(updates)
      dist = self.euclidean_numpy(X)
      groups = np.array([np.argmin(i) for i in dist])
  def preds(self,X):
    dist = self.euclidean_numpy(X)
    groups = np.array([np.argmin(i) for i in dist])
    return groups





