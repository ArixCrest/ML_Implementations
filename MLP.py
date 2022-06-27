from math import nan
class multi_layer_perceptron():
  def __init__(self,input_size,output_size,activations,hidden_layers = [],epochs = 50,learning_rate = 0.01):
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layers = hidden_layers
    self.epochs = epochs
    self.lr_rate = learning_rate
    self.weights = []
    self.biases = []
    self.activations = activations
  def build(self,s,num):
    prev = self.input_size
    if(s=='random'):
      for i in range(len(self.hidden_layers)):
        self.weights.append(np.random.rand(prev,self.hidden_layers[i]))
        prev = self.hidden_layers[i]
        self.biases.append(np.random.rand(self.hidden_layers[i],1))
      self.weights.append(np.random.rand(prev,self.output_size))
      self.biases.append(np.random.rand(self.output_size,1))
    if(s=='zeros'):
      for i in range(len(self.hidden_layers)):
        self.weights.append(np.zeros((prev,self.hidden_layers[i])))
        self.biases.append(np.zeros((self.hidden_layers[i],1)))
        prev = self.hidden_layers[i]
      self.weights.append(np.zeros((prev,self.output_size)))
      self.biases.append(np.zeros((self.output_size,1)))
    if(s=='constant'):
      for i in range(len(self.hidden_layers)):
        self.weights.append(num*np.ones((prev,self.hidden_layers[i])))
        self.biases.append(num*np.ones((self.hidden_layers[i],1)))
        prev = self.hidden_layers[i]
      self.weights.append(num*np.ones((prev,self.output_size)))
      self.biases.append(num*np.ones((self.output_size,1)))
  def load(self,weights, biases):
    self.weights = weights
    self.biases = biases
  def activation(self,arr,s):
    temp = []
    if(s=='Sigmoid'):
      for i in arr:
        k = 1/(1+np.exp(-1*i))
        temp.append(k)
    if(s=='Relu'):
      for i in arr:
        if(i[0]>=0):
          temp.append(i)
        else:
          temp.append(0*i)
    if(s=='Tanh'):
      for i in arr:
        k = 2/(1+np.exp(-2*i))-1
        temp.append(k)
    return np.array(temp)
    
  def derivative(self,arr,s):
    temp = []
    if(s=='Sigmoid'):
      #print('in')
      for i in arr:
        temp.append(i*(1-i))
    if(s=='Relu'):
      for i in arr:
        if(i[0]>=0):
          temp.append([1])
        else:
          temp.append([0])
    if(s=='Tanh'):
      for i in arr:
        temp.append(1-i**2)
    return np.array(temp)


  def cost_function(self,actual,pred):
    error = 0
    for i in range(len(actual)):
      error+=(actual[i,0]-pred[i,0])**2
    error = error/7
    return error

      

  def forward_propagate(self,X,y= None,training = True):
    n,m = X.shape
    preds = []
    error = 0
    for i in range(n):
      temp = []
      k = X[i].reshape(m,1)
      for j in range(len(self.biases)):
        k = self.weights[j].T@k+self.biases[j]
        k = self.activation(k,self.activations[j])
        temp.append(k)
      pred = k.tolist()
      pred = pred.index(max(pred))
      preds.append(pred)
      if(i%2==0 and training == True):
        out = np.zeros((7,1))
        out[y[i]] = 1
        error+=self.cost_function(out,k)
        self.back_propagate(X[i].reshape(m,1),k,temp,out)
    error = error/n
    return error,preds 


  def back_propagate(self,inputs, out,outputs,actual):
    dws = []
    dbs = []
    n = len(self.biases)
    k = out-actual
    for i in range(n-1,-1,-1):
      if(i==0):
        k = k*(self.derivative(outputs[i],self.activations[i]))
        dws.append(inputs@k.T)
      else:
        k = k*(self.derivative(outputs[i],self.activations[i]))
        dws.append(outputs[i-1]@k.T)
        k = self.weights[i]@k
    k = out-actual
    for i in range(n-1,-1,-1):
      if(i==0):
        k = k*(self.derivative(outputs[i],self.activations[i]))
        dbs.append(k)
      else:
        k = k*(self.derivative(outputs[i],self.activations[i]))
        dbs.append(k)
        k = self.weights[i]@k
    
    #updating weights and biases
    for i in range(n):
      self.weights[n-1-i] -= self.lr_rate*dws[i]
      self.biases[n-1-i] -= self.lr_rate*dbs[i]




  def train(self,X,y,valx,valy):
    accuracies = []
    X = np.array(X)
    y = np.array(y)
    valy = np.array(valy)
    for epch in range(self.epochs):
      cnt = 0
      loss,preds = self.forward_propagate(X,y)
      print("Epoch: ",epch,end = " ")
      print("Loss :",loss,end = " ")
      preds = self.predict(valx)
      for i,j in zip(preds,valy):
        if(i==j):
          cnt+=1
      accuracies.append(cnt/len(preds))
      print("Accuracy = ",cnt/len(preds))
    return accuracies
  def predict(self,X):
    X = np.array(X)
    loss,pred = self.forward_propagate(X,training = False)
    return pred



