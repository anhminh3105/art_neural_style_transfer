import os
import tensorflow as tf
from art_generate_model import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():
    with tf.Session() as test:
        tf.set_random_seed(1)
        # Test compute_content_cost() function:
        '''
        tf.global_variables_initializer()
        a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
        print("J_content = " + str(J_content.eval(session=test)))
        '''
        # test gram_matrix() function:
        """
        A = tf.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
        print("GA = ", str(GA.eval()))
        """
        # test compute_layer_style_cost() function:
        '''
        a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        print("J_style_layer =", str(J_style_layer.eval()))
        '''
        # test total_cost() function:
        '''
        np.random.seed(3)
        J_content = np.random.randn()
        J_style = np.random.randn()
        J = total_cost(J_content, J_style)
        print("J = ", str(J))
        '''
if __name__ == '__main__':
    main()
    