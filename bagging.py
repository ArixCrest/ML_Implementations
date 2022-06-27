class Bagging:
  def __init__(self,n_estimators):
    self.n_estimators = n_estimators
    self.models = []
  def create_datasets(self,df):
    l = []
    for _ in range(self.n_estimators):
      m = []
      for i in range(len(df)):
        k = random.randint(0,len(df)-1)
        m.append(df.iloc[[k]])
      data = pd.concat(m)
      data.reset_index()
      l.append(data)
    return l
  def create_models(self,df,target,max_depth):
    datasets = self.create_datasets(df)
    for i in range(self.n_estimators):
      #using best max depth of 5
      reg = DecisionTreeRegressor(max_depth= max_depth)
      x_train = datasets[i].copy()
      y_train = x_train.pop(target)
      reg.fit(x_train,y_train)
      self.models.append(reg)

  def predict(self,df,target):
    x_test = df.copy()
    y_test = x_test.pop(target)
    scores = []
    for i in range(self.n_estimators):
      pred = self.models[i].predict(x_test)
      score = self.R_Squared_Value(y_test,pred)
      scores.append(score)
    return scores
  def combined_pred(self,df,target):
    x_test = df.copy()
    y_test = x_test.pop(target)
    predictions = [0 for i in range(len(df))]
    for i in range(self.n_estimators):
      pred = self.models[i].predict(x_test)
      for j in range(len(pred)):
        predictions[j]+=pred[j]
    for i in range(len(predictions)):
      predictions[i] = predictions[i]/(self.n_estimators)
    score = self.R_Squared_Value(y_test,predictions)
    return score

  def R_Squared_Value(self,test,pred):
    test = np.array(test)
    pred = np.array(pred)
    m = np.mean(test)
    y = 0
    g = 0
    for i in range(len(test)):
      y+=(test[i]-m)**2
    for i in range(len(pred)):
      g+=(test[i]-pred[i])**2
    score = 1 - g/y
    return score




