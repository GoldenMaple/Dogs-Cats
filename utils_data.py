from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input
from preprocessing import preprocess_image
import tensorflow as tf

img_size = 299

class Transfer():
    def __init__(self, image_size=299, is_training=True, is_show=False):
        self.sess = tf.Session()
        print('hahahaha')
        with self.sess.as_default():
            self.pl = tf.placeholder(dtype=tf.uint8)
            self.result = preprocess_image(self.pl, image_size, image_size, is_training=is_training)
            # here result.dtype should be tf.float32
            if is_show and self.result.dtype != tf.uint8:
                self.result = tf.cast(self.result, dtype=tf.uint8)
                
    def T(self, image):
        feed_dict = {self.pl: image}
        result = self.sess.run(self.result, feed_dict=feed_dict)
        return result
        
trans_tr = Transfer(image_size=img_size, is_training=True)
trans_val = Transfer(image_size=img_size, is_training=False)

pre_process_training = lambda image: preprocess_input(trans_tr.T(image))
pre_process_validation = lambda image: preprocess_input(trans_val.T(image))
    
def GetGens(size):
    gen_train = ImageDataGenerator(preprocessing_function=pre_process_training)
    gen_val = ImageDataGenerator(preprocessing_function=pre_process_validation)
    return gen_train, gen_val
    
    
    
    