
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import tensorflow as tf
CIFAR_DIR='cifar-10-batches-py'
get_ipython().magic('matplotlib inline')


# In[2]:


def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


# In[3]:


CIFAR_PATH=['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']


# In[4]:


CIFAR_DATA=[]
for i in CIFAR_PATH:
    file=os.path.join(CIFAR_DIR,i)
    CIFAR_DATA.append(unpickle(file))


# In[5]:


def one_hot_encode(val,size=10):
    temp=np.zeros(size,dtype='uint8')
    temp[val]=1
    return temp


# In[6]:


def reshape(img):
    return img.reshape(-1,3,32,32).transpose(0,2,3,1).astype('uint8')    


# In[7]:


meta_data=CIFAR_DATA[0]
test_data=CIFAR_DATA[-1]
all_batches=CIFAR_DATA[1:-1]


# In[8]:


class cifar_train:
    def __init__(self):
        self.images=np.vstack([reshape(i[b'data']) for i in all_batches])
        self.labels=np.hstack([i[b'labels'] for i in all_batches])
    def next_batch(self,n):
        high=self.images.shape[0]
        indices=np.random.randint(0,high,size=n)
        return self.images[indices],np.vstack([one_hot_encode(i) for i in self.labels[indices]])
class cifar_test:
    def __init__(self):
        self.images=reshape(test_data[b'data'])
        self.labels=np.vstack([one_hot_encode(i) for i in test_data[b'labels']])


# In[9]:


def init_weights(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1),dtype=tf.float32)


# In[10]:


def init_bias(shape):
    return tf.Variable(tf.constant(value=0.1,shape=shape))


# In[11]:


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


# In[12]:


def pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[13]:


def convolution_layer(x_input,shape):
    W=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(x_input,W)+b)


# In[14]:


def dense_layer(input_layer,size):
    input_size=int(input_layer.shape[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,W)+b


# In[15]:


x_image=tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])
y_true=tf.placeholder(dtype=tf.float32,shape=[None,10])
hold_prob=tf.placeholder(dtype=tf.float32)


# In[70]:


convo_1 = convolution_layer(x_image,[2,2,3,24])
pool_1 = pooling(convo_1)
convo_2 = convolution_layer(pool_1,[2,2,24,48])
pool_2 = pooling(convo_2)
flat_layer = tf.reshape(pool_2,shape=[-1,8*8*48])
dense_1 = tf.nn.relu(dense_layer(flat_layer,512))
drop_out = tf.nn.dropout(dense_1,keep_prob=hold_prob)
y_pred = dense_layer(drop_out,10)


# In[71]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))


# In[72]:


optimizer = tf.train.AdamOptimizer(learning_rate=0.004)
train = optimizer.minimize(cross_entropy)


# In[73]:


init = tf.global_variables_initializer()


# In[74]:


c_train=cifar_train()
c_test=cifar_test()


# In[ ]:


steps=5000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps+1):
        x_batch,y_batch = c_train.next_batch(100)
        feed = {x_image:x_batch,y_true:y_batch,hold_prob:0.5}
        sess.run(train,feed_dict=feed)
        #print(i)
        if i%500==0:
            print("ACCURACY...")
            acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1)),dtype=tf.float32))
            accuracy=sess.run(acc,feed_dict={x_image:c_test.images,y_true:c_test.labels,hold_prob:1.0})
            print(accuracy)

