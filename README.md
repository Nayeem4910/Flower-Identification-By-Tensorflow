# Flower-Identification-Using-Tensorflow

## Abstract
Beauty is incomplete without flower. Bangladesh is the land of flower. In our everyday life, On the way of walking, beside the rail line or in our garden we used to see a lot of flower. But in most case we have no knowledge about that flower. Even we don’t know its name. In that case we choose this idea to research and develop our project. That will introduce people about that unknown flower which they see but don't know about that. Our application can recognizes the flower in real time by using mobile camera. This project is an attempt at using the concepts of neural networks to create an image classifier by Tensorflow on Android platform.

![identification](https://user-images.githubusercontent.com/46413461/50774294-3fcc8e00-12bd-11e9-9b22-d55255c25ac3.png)

## Introduction to Machine Learning
In Artificial Intelligence Learning is a very important feature. Many scientists tried to give a proper definition for learning. Some scientists think that learning is an adaptive skill that can perform the same process better later on (Simon 1987). Others claim that learning is a process of collecting knowledge (Feigenbaum 1977). In general, machine learning has to be identified on how to improve the computer algorithm automatically through experience (Mitchell 1997).
Machine learning is one of the important field of Artificial Intelligence. At the beginning of development of AI, the system does not have a thorough learning ability so the whole system is not perfect. For instance when the computer faces problems, it can not be self-adjusting. Moreover, the computer cannot automatically collect and discover new knowledge. Therefore, computer only can conducted by already existing truths. It does not have the ability to discover a new logical theory, rules and so on. Moreover It always needs original knowledge to understand the information from environment. Then the computer can use this information to learn new knowledge step by step. In conclusion, learning process in the whole system is a process of expansion and perfection of the knowledge base.

![capture](https://user-images.githubusercontent.com/46413461/50776359-f5e6a680-12c2-11e9-94d8-cf48a85189ac.jpg)

## Introduction to Deep Learning
Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Similarly to how we learn from experience, the deep learning algorithm would perform a task repeatedly, each time tweaking it a little to improve the outcome. Deep learning allows machines to solve complex problems even when using a data set that is very diverse, unstructured and inter-connected. The more deep learning algorithms learn, the better they perform.

## Convolutional Neural Network
Convolutional neural networks are a class of machine learning networks which are commonly applied to image visualization problems such as classification. CNNs were inspired by the connections of the neurons and synapses in the brain. The design of these networks is made up of series of convolutional, pooling, and fully connected layers. The convolutional layer does what its name describes, it applies a number of convolutional filters to the input images in order to acquire the learning parameters for the network. Pooling layers are placed in between convolutional layers, and are used to reduce the number of parameters used for learning, and thus reduce the computation required. Finally, fully connected layers are full connections to the previous layer, rather than the small window the convolutional layers are connected to in the input. Convolutional neural networks are commonly used for image classification, however, there are limitations to this application. A human can identify the contents of certain images much more quickly than a computer, but CNNs have proven to have a 97.6% success rate when applied to facial recognition.

![a-general-framework-of-convolutional-neural-networks](https://user-images.githubusercontent.com/46413461/50772221-41935300-12b7-11e9-8c39-1c30fdda09e4.png)

## Deep Learning Using Tensorflow
TensorFlow is one of the best libraries to implement deep learning. TensorFlow is a software library for numerical computation of mathematical expressional, using data flow graphs. Nodes in the graph represent mathematical operations, while the edges represent the multidimensional data arrays (tensors) that flow between them. It was created by Google and tailored for Machine Learning. In fact, it is being widely used to develop solutions with Deep Learning. Tensorflow, in addition to providing developers a simple way to build neural network layers, can also be run on mobile platforms such as Android.

## Experiment By Tensorflow
Tensorflow classifier is designed to take a directory of images, a text file of the labels used in the network, and the trained model itself as inputs. The classifier tests the images with the specified model and displays the results comparing the correct label with the top four classes based on the confidence level of the predictions. The Tensorflow trainer will generates two text files: one containing the labels for the classifier, and the other lists which images were selected for training, testing, and validation which called pb file. The classifier uses these to read result for each image classification and show the output result.


![capture111](https://user-images.githubusercontent.com/46413461/50772035-98e4f380-12b6-11e9-8ebe-7a1a8f80ad15.jpg)
         
## Develop Particular Classification By Deep Learning
Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Convolutional neural networks are a class of deep learning networks which are commonly applied to image visualization problems such as classification. The design of these networks is made up of series of convolutional, pooling, and fully connected layers. The convolutional layer does what its name describes, it applies a number of convolutional filters to the input images in order to acquire the learning parameters for the network. Pooling layers are placed in between convolutional layers, are used to reduce the number of parameters used for learning, and thus reduce the computation required. Finally, fully connected layers are full connections to the previous layer, rather than the small window the convolutional layers are connected to in the input.

### Network Design

#### Convolutional Layer 1
##### Input:
The image data is reduced to a size of 128x128 pixels in order to not overwhelm the hardware the program was normally tested on. Batches of 32 images are fed into the convolutional layer and 16 filters of 8x8 pixels are applied to the images.

###### conv_layer1 =
###### lb . build_convolutional_layer ( input = image_placeholder ,
###### num_channels = NUM_CHANNELS ,
###### filter_size = FILTER_SIZE ,
###### num_filters = NUM_FILTERS)

#### Pooling layer 1

##### Input:
###### Each pooling layer uses a pool size of 2x2 and a stride size of 2.
###### pool_layer1 = tf . layers . max_pooling2d ( inputs = conv_layer1 ,
###### pool_size =[ 2 , 2 ], strides = 2)

#### Convolutional Layer 2

The second convolutional layer has the same parameters as the first.
###### conv_layer2 =
###### lb . build_convolutional_layer ( input = pool_layer1 ,
###### num_channels = NUM_FILTERS ,
###### filter_size = FILTER_SIZE ,
###### num_filters = NUM_FILTERS)
###### Pooling layer 2

#### Fully connected layer 1

Each fully connected layer performs an activation on each of its inputs. The first, however, performs a RELU activation function on the data.

###### connected_layer1 =
###### lb . create_connected_layer ( input = flat_layer,
###### num_inputs = flat_layer . get_shape ()[ 1 : 4 ]. num_elements (),
###### num_outputs = 32 ,
###### use_relu = True)


#### Fully connected layer 2

The second FC layer does not perform the RELU activation.

###### connected_layer2 = \
###### lb . create_connected_layer ( input = connected_layer1 ,
###### num_inputs = 32 ,
###### num_outputs = num_classes ,use_relu = False)



#### Training Step

The number of training steps can be specified as a command line parameter. Each training step is validated and tested, and the results of each step are printed to standard out.

###### is_last_step = (i + 1 == FLAGS.how_many_training_steps)
###### if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
   ###### train_accuracy, cross_entropy_value = sess.run(
        [evaluation_step, cross_entropy],
   ###### feed_dict={bottleneck_input: train_bottlenecks,
       ground_truth_input: train_ground_truth})
   ###### tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
       (datetime.now(), i, train_accuracy * 100))
   ###### tf.logging.info('%s: Step %d: Cross entropy = %f' %
        (datetime.now(), i, cross_entropy_value))
   ###### validation_bottlenecks, validation_ground_truth, _ = (
         get_random_cached_bottlenecks(
         sess, image_lists, FLAGS.validation_batch_size, 'validation',
         FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
         decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture))
   ###### validation_summary, validation_accuracy = sess.run(
        [merged, evaluation_step],
        feed_dict={bottleneck_input: validation_bottlenecks,
        ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len (validation_bottlenecks)))







