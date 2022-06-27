class CrossVal():
  def __init__(self,folds):
    self.folds = folds
  def slice_data(self,n,df):
    sz = int(len(df)/n)
    l = [df.iloc[sz*i:(i+1)*sz].copy() for i in range(n)]
    return l
  def R_Squared_Value(self,test,pred):
    # test = np.array(test)
    # pred = np.array(pred)
    m = np.mean(test)
    y = 0
    g = 0
    for i in range(len(test)):
      y+=(test[i]-m)**2
    for i in range(len(pred)):
      g+=(test[i]-pred[i])**2
    score = 1 - g/y
    return score

  def kfoldCross_validation(self,df,regressor,target):
    df = df.sample(frac = 1).reset_index(drop=True)
    k = self.folds
    slices = self.slice_data(k,df)
    scores = []
    for i in range(k):
      reg = regressor
      x_val = slices[i].copy()
      y_val = np.array(x_val.pop(target))
      train = []
      for j in range(1,k):
        train.append(slices[(i+j)%k].copy())
      x_train = pd.concat(train)
      y_train = x_train.pop(target)
      reg.fit(x_train,y_train)
      pred = reg.predict(x_val)
      score = self.R_Squared_Value(y_val,pred)
      #print(a,score)
      scores.append(score)
    return scores
