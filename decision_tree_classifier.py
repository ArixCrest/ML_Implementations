#Decision Tree Implementation
class Node():
  def __init__(self,feature=None,Gain=None,split_val=None,left=None,right=None,value=None):
    self.feature = feature
    self.Gain = Gain
    self.split_val = split_val
    self.left = left
    self.right = right
    self.value = value
class DecisionTreeClassifier():
  def __init__(self,max_depth,categorical_cols,numeric_cols):
    self.root = None
    self.max_depth = max_depth
    self.numeric_cols = numeric_cols
    self.categorical_col = categorical_cols

  def build(self,X,Y,depth):
    max_Gain = -1e9+5
    #print(depth)
    if depth<=self.max_depth:
      best = {}
      ##print(depth)
      for i in self.categorical_col:
        temp = self.find_best_split(X,Y,i)
        ##print(temp)
        #print(i,temp['Gain'])
        if temp['Gain']>max_Gain:
          best = temp
          max_Gain = temp['Gain']
      #print('\n')
      for i in self.numeric_cols:
        temp = self.find_best_split(X,Y,i,feature_type = "numeric")
        #print(i,temp['Gain'])
        if temp['Gain']>max_Gain:
          best = temp
          max_Gain = temp['Gain']
      #print('\n')
      #print("best split" ,best['Feature'])
      if(best['Gain']>0):
        left_tree = self.build(best['leftX'],best['leftY'],depth+1)
        right_tree = self.build(best['rightX'],best['rightY'],depth+1)
        #print("done1")
        return Node(best['Feature'],best['Gain'],best['split_val'],left_tree,right_tree)
    a = list(Y['species'])
    #print(a)
    val = max(a,key=a.count)
    return Node(value = val)
  #finding the best split
  def find_best_split(self,X,Y,feature_name,feature_type = "categorical"):
    best = {}
    max_Gain = -1e9+5
    best['Gain'] = 0
    if feature_type == "categorical":
      ##print(type(X),feature_name)
      types = X[feature_name].unique()
      for values in types:
        left_X ,left_Y,right_X,right_Y = self.split(X,Y,feature_name,values)
        if(len(left_X)>0 and len(right_X)>0):
          Gain = self.information_gain_IE3(Y,left_Y,right_Y)
          if Gain>max_Gain:
            best['Feature'] = feature_name
            best['Gain'] = Gain
            best['split_val'] = values
            best['leftX'] = left_X
            best['rightX'] = right_X
            best['leftY'] = left_Y
            best['rightY'] = right_Y
            max_Gain = Gain
    else:
      types = list(X[feature_name].unique())
      types.sort()
      ##print(types)
      for i in range(len(types)-1):
        avg_val = (types[i]+types[i+1])/2
        ##print(feature_name,avg_val)
        left_X ,left_Y,right_X,right_Y = self.split(X,Y,feature_name,avg_val)
        if(len(left_X)>0 and len(right_X)>0):
          Gain = self.information_gain_IE3(Y,left_Y,right_Y)
          if Gain>max_Gain:
            best['Feature'] = feature_name
            best['Gain'] = Gain
            best['split_val'] = avg_val
            best['leftX'] = left_X
            best['rightX'] = right_X
            best['leftY'] = left_Y
            best['rightY'] = right_Y
            max_Gain = Gain
    return best
  #splitting the dataframe with respect to a value
  def split(self,X,Y,feature_name,value):
    dataframe = pd.concat([X,Y],axis=1)
    left = dataframe[dataframe[feature_name]<=value]
    right = dataframe[dataframe[feature_name]>value]
    left_X = left.iloc[:,:-1]
    left_Y = left.iloc[:,-1:]
    right_X = right.iloc[:,:-1]
    right_Y = right.iloc[:,-1:]
    return left_X,left_Y,right_X,right_Y
  #entropy calculation
  def find_entropy(self,y):
    entropy = 0
    ##print(y)
    n = len(y)
    y =  np.array(y)
    unique_y = np.unique(y)
    for i in unique_y:
      d = [j for j in y if j==i]
      d = len(d)/n
      entropy += -d*np.log2(d)
    return entropy
  
  def information_gain_IE3(self,parent,left,right):
    entropy_par = self.find_entropy(parent)
    entropy_left = self.find_entropy(left)
    entropy_right = self.find_entropy(right)
    left_wt = len(left)/len(parent)
    right_wt = len(right)/len(parent)
    return entropy_par-(left_wt*entropy_left+right_wt*entropy_right)
  #fitting the model
  def fit(self,train_X,train_Y):
    self.root = self.build(train_X,train_Y,0)

  #predictions
  def make_prediction(self,x,node):
    if node.value!=None:
      return node.value
    #print(type(x),node.feature)
    #print(type(x[node.feature]),x[node.feature])
    a = list(x[node.feature])
    #print(a)
    if a[0]<=node.split_val:
      return self.make_prediction(x,node.left)
    else:
      return self.make_prediction(x,node.right)
  #test_X and test_Y are pandas dataframe
  def predict(self,test_X,test_Y):
    predictions = []
    #print(len(test_X))
    for i in range(len(test_X)):
      a = test_X.iloc[i:i+1,:]
      #print(a)
      a = self.make_prediction(a,self.root)
      predictions.append(a)
    n = len(test_Y.unique())
    confusion_mat = np.zeros((n,n))
    for x,y in zip(predictions,test_Y):
      confusion_mat[x,y]+=1
    acc = (confusion_mat.diagonal().sum())/(len(test_Y))
    class_wise_acc = confusion_mat.diagonal()/(confusion_mat.sum(axis=1))
    prediction = {'Accuracy' : acc,'Predictions' : predictions,'Class-Wise Accuracy': class_wise_acc}
    return prediction
