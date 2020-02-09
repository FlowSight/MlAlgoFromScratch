 from mlalgofromscratch.utils.importer import *

 importer()

 # This would do linear and ridge regression conditionally
 class LinearRegressionCustom:

  def __init__(self,x,y, epochs=10000,learning_rate=0.0001,ridge=False,ridgeCoef=1.0):  
        self.x = x
        self.y=y
        self.n_train=self.x.shape[0]
        self.m=self.x.shape[1]
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.ridge=ridge
        self.ridgeCoef=ridgeCoef
        if ridge==False:
          self.ridgeCoef=0.0
        self.bias=0.0
        self.weights=np.full((self.m, 1), 1.0)
        self.best_weights=np.full((self.m, 1), 1.0)
        self.best_error=2.0**30
        assert(y.shape[1]==1)
        assert(x.shape[0]==y.shape[0])
        self.errors = np.zeros(self.epochs)
        self.LinearRegression()

  def LSEGradDes(self):
   # initialize stuff
   error=0.0
   gradients=np.full((self.m, 1), 0.0)
   bias_grad=0
   
   y_pred = np.add(self.bias,np.dot(self.x,self.weights))
   error=np.square(np.subtract(self.y, y_pred)).mean(axis=0) # SE error
   gradients = -2/self.n_train*(self.x.T.dot(np.subtract(self.y,y_pred))+self.ridgeCoef*self.best_weights)  # Derivative wrt weights
   bias_grad = -2/self.n_train*np.sum(np.subtract(self.y, y_pred),axis=0)  # Derivative wrt bias
   return gradients,bias_grad,error
  
  def LinearRegression(self):
    # start action in epochs
    for epoch in range(self.epochs):
      gradients,bias_grad,epoch_loss = [],0.0,0.0
      gradients,bias_grad,epoch_loss=self.LSEGradDes()
      
      # update weights
      best_error=2.0**30
      self.bias = self.bias - self.learning_rate * bias_grad#bias update
      self.weights = np.subtract(self.weights,self.learning_rate * gradients)
      self.errors[epoch]=epoch_loss
      if epoch_loss < np.amin(self.errors):
        self.best_weights=self.best_weights
        self.best_error=epoch_loss
      if epoch%1000==0 :
        print("epoch: "+str(epoch+1))
        print("weights: "+str(self.weights))
        print("bias: "+str(self.bias))
        print("loss: "+str(epoch_loss))

# this will return result and mse error as a tuple
  def predict(self,X_test,y_test):
    y_pred = np.add(self.bias,np.dot(self.x,self.weights))
    error=np.square(np.subtract(y_test, y_pred)).mean(axis=0)
    return y_pred,error[0]