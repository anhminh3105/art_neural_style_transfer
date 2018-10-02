import os
import sys
import cv2
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from model_utils import *
import numpy as np
import tensorflow as tf

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost.

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C.
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G.

    Return:
    J_content -- scalar value computed from the content cost equation.
    """

    # Retrieve dimensions from a_G.
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G.
    a_C_unrolled = tf.reshape(a_C, [n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [n_H*n_W, n_C])

    # Compute the cost with tensorflow.
    J_content = 1/(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

def gram_matrix(A):
    """
    Compute the Gram Matrix (i.e. Style Matrix) from activation A.

    Arguments:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(a=A, b=tf.transpose(A))

    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tesnsor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of image S.
    a_G -- tesnsor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of image G.
    
    Returns:
    J_style_layer -- tensor representing a scalar value computed from the style cost equation.
    """
    # retrieve dimensions from a_G.
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the style cost.
    J_style_layer = (1/(4*(n_H*n_C*n_W)**2))*tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer

def compute_style_cost(model,sess, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers.

    Arguments:
    model -- the target tensorflow model.
    STYLE_LAYERS -- A python list containing:
                    - the names of the layers we would like to extrack the style from.
                    - the coefficient for each of them.
    
    Returns:
    J_style -- tensor representing a scalar value of style cost.
    """

    #Initialise the overall style cost.
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer.
        out = model[layer_name]
        # Set a_S to be the hidden layer activation from the layer we have selected by running the session on out
        a_S = sess.run(out)
        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in code, we'll assign the image G as the model input, so that
        # when we run the session, this will the activations drawn from the appropriate layer, with G as input.
        a_G = out
        # Compute the style cost for the current layer.
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff*J_style_layer
    
    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost.
    J_style -- style cost
    alpha -- hyperparameter weighting the importance of the content cost.
    beta -- hyperparameter weighting the importance of the style cost.
    
    Returns:
    J -- total cost.
    """

    J = alpha*J_content + beta*J_style

    return J


def main():
    # We merge the style costs from several different layers to get better results.
    STYLE_LAYERS = [
        ("conv1_1", 0.2),
        ("conv2_1", 0.2),
        ("conv3_1", 0.2),
        ("conv4_1", 0.2),
        ("conv5_1", 0.2)
    ]
    """
    To do:
        1. Create an Interactive Session.
        2. Load the content image.
        3. Load the style image.
        4. Randomly initialize the image to be generated.
        5. Load the VGG19 model.
        6. Build the TensorFlow graph:
            - Run the content image through the VGG19 model and compute the content cost.
            - Run the style image through the VGG19 model and compute the style cost.
            - Compute the total cost.
            - Define the optimizer and the learning rate
        8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
    """    
    # Load content_img.
    content_img = cv2.imread(CONFIG.CONTENT_IMAGE, 1)
    content_img = cv2.resize(content_img, dsize=(CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT))
    cv2.imshow("content_img", content_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    content_img = reshape_and_normalise_image(content_img)
    print("content_img_normalised.shape = ", str(content_img.shape))

    # Load style_img.
    style_img = cv2.imread(CONFIG.STYLE_IMAGE, 1)
    style_img = cv2.resize(style_img, dsize=(CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT))
    cv2.imshow("style_img", style_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    style_img = reshape_and_normalise_image(style_img)
    print("style_img_normalised.shape = ", str(style_img.shape))
    
    # Create target generated_image.
    generated_img = generate_noise_image(content_img)
    print("generated_img.shape = ", generated_img.shape)
    cv2.imshow("generated_img", generated_img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    # Load VGG19 pre-trained model.
    model = load_vgg_model(CONFIG.VGG_MODEL)

    # Start interactive session.
    sess = tf.InteractiveSession()

    # Assign the content image to be the input of the VGG model. 
    sess.run(model["input"].assign(content_img))

    # Select the output tensor of layer conv4_2.
    out = model["conv3_3"]

    # Set a_C to be the hidden layer activation from the layer we have selected.
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from the same layer. Here, a_G references model["conv4_2"]
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost.
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image 
    sess.run(model["input"].assign(style_img))

    # Compute the style cost.
    J_style = compute_style_cost(model, sess, STYLE_LAYERS)

    # Compute the total cost.
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    optimiser = tf.train.AdamOptimizer(2.0)

    train_step = optimiser.minimize(J)
    
    # Initialise global variables.
    sess.run(tf.global_variables_initializer())
    # Run the noisy input image (initial generated image) through the model. 
    sess.run(model["input"].assign(generated_img))

    for i in range(200):
        # run the session train_step to minimise the total cost.
        sess.run(train_step)
        # Compute the generated image by running the session on the current model["input"].
        generated_img = sess.run(model["input"])

        # Print every 20 iterations.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("iteration " + str(i) + ":")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # Save current generated image in the "/output" directory.
            save_image(CONFIG.OUTPUT_DIR + str(i) + ".png", generated_img)
    
    # Save the last generated image.
    save_image(CONFIG.OUTPUT_DIR + "generated_img.jpg", generated_img)
    generated_img = cv2.imread(CONFIG.OUTPUT_DIR + "generated_img.jpg", 1)
    cv2.imshow("generated_img", generated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    


if __name__ == '__main__':
    main()
    