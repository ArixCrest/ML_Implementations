from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

splitter = KFold(n_splits = 5,shuffle=True,random_state = 15)
splits = splitter.split(df)
def find_tp_tf(ytrue,preds,thres):
  TP = [0 for i in range(5)]
  FP = [0 for i in range(5)]
  TN = [0 for i in range(5)]
  FN = [0 for i in range(5)]
  for i in range(5):
    for j in range(len(ytrue)):
      if(ytrue[j][i]==1):
        if preds[j][i]>=thres:
          TP[i]+=1
        else:
          FN[i]+=1
      else:
        if preds[j][i]>=thres:
          FP[i]+=1
        else:
          TN[i]+=1
  tpr = [0 for i in range(5)]
  fpr = [0 for i in range(5)]
  for i in range(5):
    if(TP[i]==0):
      tpr[i] = 0
    else:
      tpr[i] = TP[i]/(TP[i]+FN[i])
    if(FP[i]==0):
      fpr[i] = 0
    else:
      fpr[i] = FP[i]/(TN[i]+FP[i])
  return tpr,fpr;

def roc_curve(ytrue,preds):
  tpr = []
  fpr = []
  for i in range(5):
    tpr.append([])
    fpr.append([])
  thresholds = np.arange(0.0,1.01,.01)
  '''taking the thresholds and find the tpr and fpr rate'''
  for i in thresholds:
    tmp1, tmp2 = find_tp_tf(ytrue,preds,i)
    for i in range(len(tmp1)):
      tpr[i].append(tmp1[i])
      fpr[i].append(tmp2[i])
  auc = []
  legends = []
  '''finding the auc'''
  for i in range(5):
    plt.plot(fpr[i],tpr[i]);
    area = np.trapz(tpr[i], dx=0.01)
    auc.append(area)
  plt.plot(thresholds,thresholds,'--')
  plt.xlabel("False Positive fraction")
  plt.ylabel("True Positive fraction")
  plt.title("ROC Curve")
  plt.legend([str("1"+"AUC = "+str(auc[0])),str("2"+"AUC = "+str(auc[1])),str("3"+"AUC = "+str(auc[2])),
              str("4"+"AUC = "+str(auc[3])),str("5"+"AUC = "+str(auc[4]))])
  plt.show()
  plt.savefig('roc.png')


for train_idx,test_idx in splits:
  xtrain = df.iloc[train_idx,:-1]
  ytrain = df.iloc[train_idx,-1:]
  xtest = df.iloc[test_idx,:-1]
  ytest = df.iloc[test_idx,-1:]
  '''Binarizing the labels'''
  ytrain = label_binarize(ytrain,classes = [0,1,2,3,4])
  ytest = label_binarize(ytest,classes = [0,1,2,3,4])
  model = OneVsRestClassifier(LinearDiscriminantAnalysis())
  model.fit(xtrain,ytrain)
  preds = model.predict_proba(xtest)
  roc_curve(ytest,preds)




