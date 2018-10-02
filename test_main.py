from nst_utils import *
import tensorflow as tf
def main():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    graph = sess.run(load_vgg_model(CONFIG.VGG_MODEL))
    print(graph)

if __name__ == '__main__':
    main()
    