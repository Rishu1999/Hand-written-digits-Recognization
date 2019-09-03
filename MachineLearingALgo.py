import warnings
warnings.filterwarnings(action='ignore')
#standard scientific python import
import  matplotlib.pyplot as plt
#import datasets,classifiers and performance ,atrics
from sklearn import datasets,svm
#the digits of  datasets
digits=datasets.load_digits()
print("digit:",digits.keys())
print('digit.target----:',digits.target)
images_and_labels=list(zip(digits.images,digits.target))
print('len(images_and_labels)',len(images_and_labels))
for index,[image,label] in enumerate(images_and_labels[ :5]):
    print("index:",index,"image:\n",image,'labels:',label)
    plt.subplot(2,5,index+1)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    #imshow plot the matrix into the graph

    plt.title('training:%i'%label)
    #plt.show()
    #to apply the classifier on this data,we need to faltten the  image,to
    #TUrn the data in a sample ,feature
n_samples=len(digits.images)
print('n_sampels:',n_samples)
imageData=digits.images.reshape((n_samples,-1))
print('after Reshaped:len(imageData[0]:',len(imageData[0]))
classifier=svm.SVC(gamma=0.001)
classifier.fit(imageData[  :n_samples//2],digits.target[:n_samples//2])
#now predict the value from the second half
expectedY=digits.target[n_samples//2:]
predictedY=classifier.predict(imageData[n_samples//2:])
images_and_predictions=list(zip(digits.images[n_samples//2:],predictedY))
for index,[image,prediction] in enumerate(images_and_predictions[:5]):
    #foirst 5 value for test data
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('prediction:%i'%prediction)
print("originl values:",digits.target[n_samples//2:(n_samples//2)+5])
plt.show()
#scipy is for image processing

from scipy.misc import imread,imresize,bytescale
img=imread('Three2.jpeg')
img=imresize(img,(8,8))
classifier=svm.SVC(gamma=0.001)
classifier.fit(imageData[:],digits.target[:])
img=img.astype(digits.images.dtype)#given image pixel same as training image pixel
img=bytescale(img,high=16.0,low=0)
print('img.shape:',img.shape)
print('\n',img)
x_testData=[]
for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0)#average of every pixel
    print("x_testData:\n",x_testData)
    print('len(x_testData):',len(x_testData))

x_testData=[x_testData]
print('len(x_testData):',len(x_testData))
print("machine output=",classifier.predict(x_testData))
plt.show()