from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
from ResNet import resnet34_cbam
from ResNet import resnet50_cbam
import torch
import time


numclass=10 #num of classes learned initially, will be updated in incremental learning
feature_extractor=resnet34_cbam() #try other resnets
img_size=32
batch_size=128 #was 128,32 
task_size=10 #was 10 #num of classes learned each task
memory_size= 2000#was 2000
epochs=70 #was 100
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

start_time = time.time()
for i in range(10): #was 10,5
    # if i==0:
    #     start_time = time.time()
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
    # if i==9:
end_time = time.time()
print('Total training time: {:.2f} seconds'.format(end_time - start_time))
