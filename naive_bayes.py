class Naive_Bayes_Gaussian_Classifier():
  def __init__(self,categorical,numeric,target):
    self.categorical = categorical
    self.numeric = numeric
    self.target = target
    self.model = {}
    self.yes = 0
    self.no = 0
  def train(self,df):
    n = len(df)
    y = len(df.loc[df[self.target] == 1])
    no = n-y;
    p_y = y/n
    p_n = 1-p_y
    self.yes = p_y
    self.no = p_n
    for i in self.categorical:
      l = df[i].unique()
      a = {}
      for j in l:
        k_y = len(df[i].loc[df[i]==j])
        p_y_n = len(df[(df[self.target]==1)&(df[i]==j)])
        p_n_n = len(df[(df[self.target]==0)&(df[i]==j)])
        p_f = k_y/n
        temp = [0]*2
        temp[0] = ((p_n_n/no)*p_n)/p_f
        temp[1] = ((p_y_n/y)*p_y)/p_f
        a[j] = temp
      self.model[i] = a
    for i in self.numeric:
      a = {}
      for j in range(0,2):
        temp = df[df[self.target] ==j]
        mean = temp[i].mean()
        std = temp[i].std()
        a[j] = {'mean' : mean,'std' :std}
      self.model[i] = a
  def find_val(self,val,feature,outcome):
    mean = self.model[feature][outcome]['mean']
    std = self.model[feature][outcome]['std']
    exponent = -((val-mean)*(val-mean))/(std*std)
    first_term = -np.log(std)
    second_term = -np.log(2*np.pi)/2
    return first_term+second_term+exponent
    

  def predict(self,df):
    preds = []
    probs = []
    for i in range(len(df)):
      p_yes = np.log(self.yes)
      p_no = np.log(self.no)
      for j in self.categorical:
        val = df.iloc[i][j]
        val1 = self.model[j][val][0]
        val2 = self.model[j][val][1]
        p_no += np.log(val1)
        p_yes += np.log(val2)

      for j in self.numeric:
        val = df.iloc[i][j]
        p_no+=self.find_val(val,j,0)
        p_yes+=self.find_val(val,j,1)
      if(p_no>p_yes):
        prob = np.e**(p_no)
        probs.append(prob)
        preds.append(0)
      else:
        prob = np.e**(p_yes)
        probs.append(prob)
        preds.append(1)
    return preds,probs
