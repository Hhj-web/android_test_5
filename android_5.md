
# 在本地 Windows 环境中完成图像分类模型的训练和导出

## 项目概述
本项目旨在通过本地 Windows 环境完成一个图像分类模型（花卉分类）的训练和导出，过程中涉及自定义虚拟环境以切换到兼容的 Python 3.9 版本，并安装一系列精确版本的依赖，最终克服依赖冲突问题，完成模型的训练、导出。

## 环境准备
### 安装 Python 3.9 及必要工具
在 Windows 上，你需要手动从 [Python 官方网站](https://www.python.org/downloads/release/python-390/) 下载并安装 Python 3.9。安装过程中，请确保勾选“Add Python 3.9 to PATH”选项。

### 创建虚拟环境
```bash
python -m venv C:\tflite_env
```

### 验证虚拟环境和 pip
```bash
C:\tflite_env\Scripts\python.exe -m pip --version
```

## 依赖安装
### 安装核心依赖
```bash
C:\tflite_env\Scripts\pip.exe install -q ^
  tensorflow==2.10.0 ^
  keras==2.10.0 ^
  numpy==1.23.5 ^
  protobuf==3.19.6 ^
  tensorflow-hub==0.12.0 ^
  tflite-support==0.4.2 ^
  tensorflow-datasets==4.8.3 ^
  sentencepiece==0.1.99 ^
  sounddevice==0.4.5 ^
  librosa==0.8.1 ^
  flatbuffers==23.5.26 ^
  matplotlib==3.5.3 ^
  opencv-python==4.8.0.76
```

### 安装 tflite-model-maker 本体
```bash
C:\tflite_env\Scripts\pip.exe install tflite-model-maker==0.4.2
```

### 补充缺失依赖
```bash
C:\tflite_env\Scripts\pip.exe install matplotlib_inline IPython
```

### 验证是否成功安装
```bash
C:\tflite_env\Scripts\python.exe -c "from tflite_model_maker import image_classifier; print('TFLite Model Maker 已成功导入')"
```

## 模型训练
### 编写训练脚本
```python
with open('C:/step_train.py', 'w') as f:
    f.write("""
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

image_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)
loss, acc = model.evaluate(test_data)
print(f'✅ 测试准确率: {acc:.4f}')
model.export(export_dir='.')
""")
```

### 执行训练脚本
```bash
C:\tflite_env\Scripts\python.exe C:/step_train.py
```

## 模型下载
在本地 Windows 环境中，模型会直接保存在当前工作目录下，你可以在资源管理器中找到 `model.tflite` 文件。

## 总结
通过本项目，我们掌握了在特定环境下解决依赖冲突的有效方法，深入理解了 Python 虚拟环境的隔离作用以及不同版本库之间相互配合的重要性。同时，也体会到了在本地 Windows 环境中操作的独特性，对 TFLite Model Maker 的工作机制有了更直观的感受，对机器学习模型从开发到部署的完整链路有了清晰的认知。


# TensorFlow训练石头剪刀布数据集

解压下载的数据集
```python
import os
import zipfile

local_zip = 'C:/Users/86188/Downloads/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/Users/86188/Downloads/')
zip_ref.close()

local_zip = 'C:/Users/86188/Downloads/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/Users/86188/Downloads/')
zip_ref.close()
```

检测数据集的解压结果，打印相关信息。
```python
rock_dir = os.path.join('C:/Users/86188/Downloads/rps/rock')
paper_dir = os.path.join('C:/Users/86188/Downloads/rps/paper')
scissors_dir = os.path.join('C:/Users/86188/Downloads/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```

各打印一张石头剪刀布训练集图片
```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 1

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-1:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-1:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-1:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```



    
![png](石头.png)

    



    
![png](布.png)

    



    

![png](剪刀.png)
    



调用 TensorFlow 的 keras 进行数据模型的训练和评估。
```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


TRAINING_DIR = "C:/Users/86188/Downloads/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "C:/Users/86188/Downloads/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")
```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    WARNING:tensorflow:From D:\anaconda3\Lib\site-packages\keras\src\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From D:\anaconda3\Lib\site-packages\keras\src\layers\pooling\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2  (None, 74, 74, 64)        0         
     D)                                                              
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPoolin  (None, 36, 36, 64)        0         
     g2D)                                                            
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPoolin  (None, 17, 17, 128)       0         
     g2D)                                                            
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPoolin  (None, 7, 7, 128)         0         
     g2D)                                                            
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3473475 (13.25 MB)
    Trainable params: 3473475 (13.25 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    WARNING:tensorflow:From D:\anaconda3\Lib\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    Epoch 1/25
    WARNING:tensorflow:From D:\anaconda3\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
    
    WARNING:tensorflow:From D:\anaconda3\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    20/20 [==============================] - 39s 2s/step - loss: 1.1771 - accuracy: 0.3401 - val_loss: 1.0969 - val_accuracy: 0.4812
    Epoch 2/25
    20/20 [==============================] - 38s 2s/step - loss: 1.0973 - accuracy: 0.3817 - val_loss: 1.0825 - val_accuracy: 0.3387
    Epoch 3/25
    20/20 [==============================] - 35s 2s/step - loss: 1.0723 - accuracy: 0.4310 - val_loss: 0.8486 - val_accuracy: 0.7930
    Epoch 4/25
    20/20 [==============================] - 32s 2s/step - loss: 1.0393 - accuracy: 0.4663 - val_loss: 0.9628 - val_accuracy: 0.5833
    Epoch 5/25
    20/20 [==============================] - 31s 2s/step - loss: 0.9070 - accuracy: 0.5714 - val_loss: 0.7064 - val_accuracy: 0.6640
    Epoch 6/25
    20/20 [==============================] - 31s 2s/step - loss: 0.7650 - accuracy: 0.6452 - val_loss: 0.4253 - val_accuracy: 0.9435
    Epoch 7/25
    20/20 [==============================] - 32s 2s/step - loss: 0.6397 - accuracy: 0.7254 - val_loss: 0.4302 - val_accuracy: 0.6855
    Epoch 8/25
    20/20 [==============================] - 34s 2s/step - loss: 0.5159 - accuracy: 0.7726 - val_loss: 0.2129 - val_accuracy: 0.9409
    Epoch 9/25
    20/20 [==============================] - 32s 2s/step - loss: 0.4909 - accuracy: 0.8127 - val_loss: 0.3706 - val_accuracy: 0.8306
    Epoch 10/25
    20/20 [==============================] - 31s 2s/step - loss: 0.4112 - accuracy: 0.8381 - val_loss: 0.1871 - val_accuracy: 0.9489
    Epoch 11/25
    20/20 [==============================] - 31s 2s/step - loss: 0.2892 - accuracy: 0.8893 - val_loss: 0.1667 - val_accuracy: 0.9435
    Epoch 12/25
    20/20 [==============================] - 35s 2s/step - loss: 0.2997 - accuracy: 0.8861 - val_loss: 0.2650 - val_accuracy: 0.8898
    Epoch 13/25
    20/20 [==============================] - 32s 2s/step - loss: 0.1893 - accuracy: 0.9306 - val_loss: 0.1020 - val_accuracy: 0.9597
    Epoch 14/25
    20/20 [==============================] - 31s 2s/step - loss: 0.3214 - accuracy: 0.8813 - val_loss: 0.0747 - val_accuracy: 0.9677
    Epoch 15/25
    20/20 [==============================] - 34s 2s/step - loss: 0.1632 - accuracy: 0.9397 - val_loss: 0.0703 - val_accuracy: 0.9758
    Epoch 16/25
    20/20 [==============================] - 33s 2s/step - loss: 0.1319 - accuracy: 0.9492 - val_loss: 0.0577 - val_accuracy: 0.9731
    Epoch 17/25
    20/20 [==============================] - 33s 2s/step - loss: 0.1981 - accuracy: 0.9183 - val_loss: 0.0480 - val_accuracy: 0.9866
    Epoch 18/25
    20/20 [==============================] - 35s 2s/step - loss: 0.2249 - accuracy: 0.9143 - val_loss: 0.0657 - val_accuracy: 0.9570
    Epoch 19/25
    20/20 [==============================] - 31s 2s/step - loss: 0.1097 - accuracy: 0.9659 - val_loss: 0.0660 - val_accuracy: 0.9651
    Epoch 20/25
    20/20 [==============================] - 32s 2s/step - loss: 0.1263 - accuracy: 0.9591 - val_loss: 0.1977 - val_accuracy: 0.9113
    Epoch 21/25
    20/20 [==============================] - 31s 2s/step - loss: 0.1425 - accuracy: 0.9452 - val_loss: 0.2150 - val_accuracy: 0.9059
    Epoch 22/25
    20/20 [==============================] - 32s 2s/step - loss: 0.0786 - accuracy: 0.9726 - val_loss: 0.0820 - val_accuracy: 0.9677
    Epoch 23/25
    20/20 [==============================] - 31s 2s/step - loss: 0.0953 - accuracy: 0.9667 - val_loss: 0.0647 - val_accuracy: 0.9866
    Epoch 24/25
    20/20 [==============================] - 31s 2s/step - loss: 0.0948 - accuracy: 0.9675 - val_loss: 0.0297 - val_accuracy: 0.9866
    Epoch 25/25
    20/20 [==============================] - 32s 2s/step - loss: 0.1363 - accuracy: 0.9500 - val_loss: 0.0080 - val_accuracy: 1.0000
    

完成模型训练之后，我们绘制训练和验证结果的相关信息。
```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

```


    
![png](折线图.png)
    



    <Figure size 640x480 with 0 Axes>



```python

```