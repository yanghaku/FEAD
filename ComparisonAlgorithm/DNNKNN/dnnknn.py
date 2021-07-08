import time
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import CSVLogger
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class DNN:
  model_dnn = None
  imput_dim_neurons = 41
  number_neurons_hidden_layer = 41
  activation_function_hidden_layer = "tanh"
  number_neurons_output_layer = 2
  activation_function_output_layer = "softmax"
  n_epochs = 2
  optimizer = 'adam'
  loss = 'sparse_categorical_crossentropy'
  test_time = 0
  training_time = 0
  early_stopping = None
  name = "Deep Neural Network (DNN)"

  def setImputDimNeurons(self, a):
    self.imput_dim_neurons = a
  
  def setActivationFunctionHiddenLayer(self, a):
    self.activation_function_hidden_layer = a

  def setNumNeuronsHiddenLayer(self, a):
    self.number_neurons_hidden_layer = a
  
  def setActivationFunctionOutputLayer(self, a):
    self.activation_function_output_layer = a  

  def setNumNeuronsOutLayer(self, a):
    self.number_neurons_output_layer = a

  def setNumEpochs(self, a):
    self.n_epochs = a

  def setOptimizer(self, a):
    self.optimizer = a

  def setLoss(self, a):
    self.loss = a

  def getTestTime(self):
    return self.test_time

  def getTrainingTime(self):
    return self.training_time

  def getName(self):
    return self.name

  def getModel(self):
    return self.model_dnn
    
  def getInfos(self):
    texto = '    Deep Neural Network (DNN): \n'
    texto += f'      Structure:\n'
    texto += f'        Input dimension: {self.imput_dim_neurons}\n'
    texto += f'        Number of hidden layer neurons: {self.number_neurons_hidden_layer}\n'
    texto += f'        Activation function hidden layers: {self.activation_function_hidden_layer}\n'
    texto += f'        Number of output layer neurons: {self.number_neurons_output_layer}\n'
    texto += f'        Activation function output layers: {self.activation_function_output_layer}\n'
    texto += f'      Training:\n'
    texto += f'        Epochs: {self.n_epochs}\n'
    texto += f'        Optimizer: {self.optimizer}\n'
    texto += f'        Loss: {self.loss}\n'    
    return texto  

  def generateModel(self):

    self.model_dnn = Sequential()
    self.model_dnn.add(Dense(self.number_neurons_hidden_layer, input_dim=self.imput_dim_neurons, activation=self.activation_function_hidden_layer))
    self.model_dnn.add(Dense(self.number_neurons_hidden_layer, activation=self.activation_function_hidden_layer))
    self.model_dnn.add(Dense(self.number_neurons_output_layer, activation=self.activation_function_output_layer))

    print(self.model_dnn.summary())
    self.model_dnn.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
    csv_logger = CSVLogger('training.log')
    #funcao para interromper treinamento quando o erro for suficientemente pequeno
    self.early_stopping = EarlyStopping(monitor='loss', patience=20)

    return 0

  def fit (self, data_set_samples, data_set_labels): 

    train_time_start = time.time()

    fit = self.model_dnn.fit(data_set_samples, data_set_labels, epochs=self.n_epochs, verbose=2, callbacks=[self.early_stopping])

    self.train_time = time.time() - train_time_start

    return 0


  def predict(self, data_set_samples):
    test_time_start = time.time()

    predictions_dnn = self.model_dnn.predict(data_set_samples)

    self.test_time = time.time() - test_time_start

    return predictions_dnn


  def exec(self, data_set_samples, data_set_labels, test_data_set_samples):
    final_predictions = []

    self.generateModel()
    self.fit(data_set_samples, data_set_labels)

    final_predictions = self.predict(test_data_set_samples)

    return final_predictions


class kNN:
  knn = 0
  k_neighbors = 1
  algorithm = 'kd_tree'
  weights = 'uniform'
  test_time = 0
  training_time = 0
  name = "K-Nearest Neighbor (KNN)"
  
  def setKNeighbors(self, a):
    self.k_neighbors = a

  def setAlgorithm(self, a):
    self.algorithm = a 

  def setWeights(self, a):
    self.weights = a 

  def getTestTime(self):
    return self.test_time

  def getTrainingTime(self):
    return self.training_time

  def getName(self):
    return self.name

  def getModel(self):
    return self.knn

  def getInfos(self):
    texto = '    K-Nearest Neighbor (kNN): \n'
    texto += f'      K Neighbors: {self.k_neighbors}\n'
    texto += f'      Algorithm: {self.algorithm}\n'
    texto += f'      Weights: {self.weights}\n'
    return texto  

  def generateModel(self):
    
    self.knn = neighbors.KNeighborsClassifier(self.k_neighbors, weights=self.weights, algorithm=self.algorithm)

    return 0  

  def fit (self, data_set_samples, data_set_labels):

    training_time_start = time.time()

    self.knn.fit(data_set_samples, data_set_labels)
    
    self.training_time = time.time() - training_time_start

    return 0

  def predict(self, test_data_set_samples):

    test_time_start = time.time()

    predicoes = self.knn.predict(test_data_set_samples)

    self.test_time = time.time() - test_time_start
  
    return predicoes

  def exec(self, data_set_samples, data_set_labels, test_data_set_samples):
    final_predictions = []

    self.generateModel()
    self.fit(data_set_samples, data_set_labels)

    final_predictions = self.predict(test_data_set_samples)

    return final_predictions


class DNNkNN:
  dnn = None
  knn = None
  ACCEPTABLE_ERROR_RATE_FP = 0
  ACCEPTABLE_ERROR_RATE_FN = 0
  normal_neuron_limit = 0
  attack_neuron_limit = 0
  dnn_count = 0
  knn_count = 0
  test_time = 0
  training_time = 0
  deleted_attributes = []

  name = "Deep Neural Network and K-Nearest Neighbor (DNN-KNN)"

  def __init__(self):
    self.dnn = DNN()
    self.knn = kNN()
  
  def setAcceptableErrorRateFP (self, a):
    self.ACCEPTABLE_ERROR_RATE_FP = a

  def setAcceptableErrorRateFN (self, a):
    self.ACCEPTABLE_ERROR_RATE_FN = a

  def getName(self):
    return self.name

  def getTestTime(self):
    return self.test_time

  def getTrainingTime(self):
    return self.training_time

  def getDNN(self):
    return self.dnn

  def getKNN(self):
    return self.knn

  def getDNNCount(self):
    return self.dnn_count  

  def getKNNCount(self):
    return self.knn_count 

  def setDeleteAttributes(self, delete_att):
    self.deleted_attributes = delete_att

  def getInfos(self):
    texto = '    DNN-kNN Method: \n'
    texto += self.dnn.getInfos()
    texto += self.knn.getInfos()
    return texto  

  def generateModel(self):
    self.dnn.generateModel()
    self.knn.generateModel()

  def fit(self, data_set_samples, data_set_labels):
    predictions_dnn_knn_training = []
    knn_instances = []

    train_time_start = time.time()

    #Dnn training
    self.dnn.fit(data_set_samples, data_set_labels)

    #Knn training only for the training phase of the approach (same number of DNN attributes)
    self.knn.fit(data_set_samples, data_set_labels)

    print('Starting definition of threshold values...')
    acc = 0

    #sets the limits very low, so no example is sent to KNN
    self.attack_neuron_limit = 0.5  
    self.normal_neuron_limit = 0.5 

    for i in range(1, 10):
      predictions_dnn_knn_training = []
      list_id_sendto_knn = []
      knn_instances = []
      self.knn_count = 0
      self.dnn_count = 0

      #print(f'New Limit Neuron Normal: {self.normal_neuron_limit}')
      #print(f'New Limit Neuron Attack: {self.attack_neuron_limit}')

      predictions_dnn = self.dnn.predict(data_set_samples)

      for j in range(0,len(data_set_samples)):
        if(predictions_dnn[j][0] > self.normal_neuron_limit):
          self.dnn_count = self.dnn_count + 1
          predictions_dnn_knn_training.append(0) 
        elif(predictions_dnn[j][1] > self.attack_neuron_limit):
          self.dnn_count = self.dnn_count + 1
          predictions_dnn_knn_training.append(1) 
        else:
          self.knn_count = self.knn_count + 1
          predictions_dnn_knn_training.append(-1)
          list_id_sendto_knn.append(j)
          knn_instances.append(data_set_samples[j])
      
      if (len(knn_instances) != 0):
        predictions_knn = self.knn.predict(knn_instances)

        for k in range(0, len(knn_instances)):
          predictions_dnn_knn_training[list_id_sendto_knn[k]] = predictions_knn[k]

      acc = accuracy_score(data_set_labels, predictions_dnn_knn_training)
      matriz = confusion_matrix(data_set_labels, predictions_dnn_knn_training)

      tn = matriz[0][0]
      fp = matriz[0][1]
      fn = matriz[1][0]
      tp = matriz[1][1]
      #print(matriz)
      print(f'Training DNN acc: {acc}')
      #print(f'DNN Counter: {self.dnn_count}')
      #print(f'KNN Counter: {self.knn_count}')
      #print(f'FP Rate: {fp*(100/(fp+tp))}')
      #print(f'FN Rate: {fn*(100/(fn+tn))}')

      output_neuron_normal = []
      output_neuron_attack = []
      for p in range(0,len(data_set_samples)):
        if (predictions_dnn[p][0] > 0.5):
          output_neuron_normal.append(predictions_dnn[p][0])
        if (predictions_dnn[p][1] > 0.5):
          output_neuron_attack.append(predictions_dnn[p][1])

      if(len(output_neuron_attack) != 0):
        if ((fp*(100/(fp+tp)) > self.ACCEPTABLE_ERROR_RATE_FP) and (fp != 0)):
          self.attack_neuron_limit = np.percentile(output_neuron_attack, i*10)  
      
      if(len(output_neuron_normal) != 0):
        if ((fn*(100/(fn+tn)) > self.ACCEPTABLE_ERROR_RATE_FN) and (fn != 0)):
          self.normal_neuron_limit = np.percentile(output_neuron_normal, i*10)  

      if((fn*(100/(fn+tn)) <= self.ACCEPTABLE_ERROR_RATE_FP) and (fn*(100/(fn+tn)) <= self.ACCEPTABLE_ERROR_RATE_FN)):
        break


    print(f'Final value Normal neuron limit: {self.normal_neuron_limit}')
    print(f'Final value Attack neuron limit: {self.attack_neuron_limit}')



    #Attribute reduction (selection made with InfoGain)
    data_set_samples = np.delete(data_set_samples, self.deleted_attributes, axis=1)

    #Final knn training with reduced attributes
    self.knn.fit(data_set_samples, data_set_labels)

    self.training_time = time.time() - train_time_start

    return 0


  def predict(self, test_data_set_samples):

    predictions_ann_knn = []
    list_id_sendto_knn = []
    knn_instances = []
    self.dnn_count = 0
    self.knn_count = 0

    test_time_start = time.time()

    predictions_dnn = self.dnn.predict(test_data_set_samples)

    #Attribute reduction (selection made with InfoGain)
    test_data_set_samples = np.delete(test_data_set_samples, self.deleted_attributes, axis=1)


    for i in range(0,len(test_data_set_samples)):
          
      if(predictions_dnn[i][0] > self.normal_neuron_limit):
        self.dnn_count = self.dnn_count + 1
        predictions_ann_knn.append(0) 
      elif(predictions_dnn[i][1] > self.attack_neuron_limit):
          self.dnn_count = self.dnn_count + 1
          predictions_ann_knn.append(1) 
      else:
          self.knn_count = self.knn_count + 1
          predictions_ann_knn.append(-1)
          list_id_sendto_knn.append(i)
          knn_instances.append(test_data_set_samples[i])
      
    if (len(knn_instances) != 0):
      predictions_knn = self.knn.predict(knn_instances)
      for i in range(0, len(knn_instances)):
        predictions_ann_knn[list_id_sendto_knn[i]] = predictions_knn[i]
  
    self.test_time = time.time() - test_time_start

    return predictions_ann_knn

  def exec(self, data_set_samples, data_set_labels, test_data_set_samples):
    final_predictions = []

    self.generateModel()
    self.fit(data_set_samples, data_set_labels)

    final_predictions = self.predict(test_data_set_samples)

    return final_predictions
