import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# helper.maybe_download_pretrained_vgg("model")

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    
    vgg_tag = 'vgg16'

    # with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
    
    #vgg_input_tensor_name,vgg_keep_prob_tensor_name,vgg_layer3_out_tensor_name,vgg_layer4_out_tensor_name,vgg_layer7_out_tensor_name
    # return None, None, None, None, None
    # g=tf.graph()
    # g.get_operation()

    # return vgg_input_tensor_name,vgg_keep_prob_tensor_name,vgg_layer3_out_tensor_name,vgg_layer4_out_tensor_name,vgg_layer7_out_tensor_name

# tests.test_load_vgg(load_vgg(tf.Session(),"./model/vgg"), tf)
tests.test_load_vgg(load_vgg, tf)



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    #delete next.. because num classes = num outputs
    # num_classes = 2
    
    #replace fully into 1x1 conv
    output = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1))
    #upscale 1x1
    output2 = tf.layers.conv2d_transpose(output, num_classes, 4, 2, 'SAME')

    #replace fully into 1x1 conv
    output3 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1))
    #make a skip... 
    #The final step is adding skip connections to the model. In order to do this well combine the output of two layers. The first output is the output
    # of the current layer. The second output is the output of a layer further back in the network, typically a pooling layer
    #. In the following example we combine the result of the previous layer with the result of the 4th pooling layer through elementwise addition tf.add.

    #together with the lectures I followed the forum suggestions @:
    #https://discussions.udacity.com/t/what-is-the-output-layer-of-the-pre-trained-vgg16-to-be-fed-to-layers-project/327033/24
    output4 = tf.add(output2, output3)

    #upsample
    output5 = tf.layers.conv2d_transpose(output4, num_classes, 4, 2, 'SAME')

    #1s1 and make a skip
    output6 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1))
    output7 = tf.add(output5, output6)

    #upsample
    final = tf.layers.conv2d_transpose(output7, num_classes, 16, 8, 'SAME')

    return final



tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #reshape also labels to be sure its the same size
    labels = tf.reshape(correct_label, (-1, num_classes))

    #used exactly the same optimization / NN prep as module 1, project 2 (traffic sign classifier)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function


  # "with tf.Session() as sess:\n",
  #   "    sess.run(tf.global_variables_initializer())\n",
  #   "    num_examples = len(X_train_processed)\n",
  #   "    \n",
  #   "    print(\"Training...\")\n",
  #   "    print()\n",
  #   "    for i in range(EPOCHS):\n",
  #   "        X_train_processed, y_train = shuffle(X_train_processed, y_train)\n",
  #   "        for offset in range(0, num_examples, BATCH_SIZE):\n",
  #   "            end = offset + BATCH_SIZE\n",
  #   "            batch_x, batch_y = X_train_processed[offset:end], y_train[offset:end]\n",
  #   "            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})\n",
  #   "            \n",
  #   "        validation_accuracy = evaluate(X_validation, y_validation)\n",
  #   "        print(\"EPOCH {} ...\".format(i+1))\n",
  #   "        print(\"Validation Accuracy = {:.3f}\".format(validation_accuracy))\n",
  #   "        \n",
  #   "        print()\n",
  #   "        \n",
  #   "    saver.save(sess, 'lenet')\n",
  #   "    #print(\"Model saved\")\n",
  #   "    \n",

    #dropout
    keep_prob_stat = 0.7

    learning_rate_stat = 0.001
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: keep_prob_stat,
                                          learning_rate:learning_rate_stat})
        print("Epoch %d of %d: Training loss: %.4f" %(epoch+1, epochs, loss))


    pass

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/


    epochs = 10
    batch_size = 4

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        # TODO: Train NN using the train_nn function

        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)

        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,correct_label, keep_prob, learning_rate)


        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


        # saver = tf.train.Saver()
        # saver.save(sess, 'checkpoints/model1.ckpt')
        # saver.export_meta_graph('checkpoints/model1.meta')
        # tf.train.write_graph(sess.graph_def, './checkpoints/', 'model1.pb', False)

        # Save image outputs
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)


        # OPTIONAL: Apply the trained model to a video



if __name__ == '__main__':
    run()
