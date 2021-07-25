# -*- coding: utf-8 -*-

import tensorflow as tf


def mitochondria_nn_model(img_width = 256, img_height = 256, img_channels = 3):
    '''
    UNET model for detecting of mitochondria
    Created for solving problem described in: https://www.epfl.ch/labs/cvlab/data/data-em/
    Source: https://www.youtube.com/watch?v=csFGTLT6_WQ
    
    Parameters
    ----------
    img_width : uint8, optional
        DESCRIPTION. The default is 256.
    img_height : uint8, optional
        DESCRIPTION. The default is 256.
    img_channels : uint8, optional
        DESCRIPTION. The default is 3.
    Returns
    -------
    NN model

    '''
    # Input layer
    # Get initial tensor
    input_tensor = tf.keras.Input(shape = (img_width,img_height,img_channels))
    print(type(input_tensor))
    # Convert input values to float
    input_tensor_float = tf.keras.layers.Lambda(lambda x: x / 255)(input_tensor)
    
    # Contraction path
    # Convolution layers and MaxPooling layer. Between two convolution layers insert one dropout layer
    c_1_1 = tf.keras.layers.Conv2D(filters = 16, 
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(input_tensor_float)
    d_1 = tf.keras.layers.Dropout(rate = 0.1)(c_1_1)
    c_1_2 = tf.keras.layers.Conv2D(filters = 16, 
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(d_1)
    p_1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(c_1_2)
    
    
    c_2_1 = tf.keras.layers.Conv2D(filters = 32,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(p_1)
    d_2 = tf.keras.layers.Dropout(rate = 0.1)(c_2_1)
    c_2_2 = tf.keras.layers.Conv2D(filters = 32, 
                                    kernel_size = (3,3), 
                                    padding = "same",
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(d_2)
    p_2 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(c_2_2)
    
    
    c_3_1 = tf.keras.layers.Conv2D(filters = 64, 
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(p_2)
    d_3 = tf.keras.layers.Dropout(rate = 0.2)(c_3_1)
    c_3_2 = tf.keras.layers.Conv2D(filters = 64, 
                                    kernel_size = (3,3), 
                                    padding = "same",
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(d_3)
    p_3 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(c_3_2)
    
    
    c_4_1 = tf.keras.layers.Conv2D(filters = 128,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(p_3)
    d_4 = tf.keras.layers.Dropout(rate = 0.2)(c_4_1)
    c_4_2 = tf.keras.layers.Conv2D(filters = 128, 
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(d_4)
    p_4 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(c_4_2)
    
    
    c_5_1 = tf.keras.layers.Conv2D(filters = 256,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(p_4)
    d_5 = tf.keras.layers.Dropout(rate = 0.3)(c_5_1)
    c_5_2 = tf.keras.layers.Conv2D(filters = 256, 
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu", 
                                    kernel_initializer = "he_normal")(d_5)
    
    # Expansive path
    
    c_6_1 = tf.keras.layers.Conv2DTranspose(filters = 128, 
                                            kernel_size = (2,2), 
                                            strides = (2,2), 
                                            padding = "same")(c_5_2)
    u_6 = tf.keras.layers.concatenate([c_6_1,c_4_2])
    c_6_2 = tf.keras.layers.Conv2D(filters = 128,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(u_6)
    d_6 = tf.keras.layers.Dropout(rate = 0.2)(c_6_2)
    c_6_3 = tf.keras.layers.Conv2D(filters = 128,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(d_6)
    
    
    c_7_1 = tf.keras.layers.Conv2DTranspose(filters = 64, 
                                            kernel_size = (2,2), 
                                            strides = (2,2), 
                                            padding = "same")(c_6_3)
    u_7 = tf.keras.layers.concatenate([c_7_1,c_3_2])
    c_7_2 = tf.keras.layers.Conv2D(filters = 64,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(u_7)
    d_7 = tf.keras.layers.Dropout(rate = 0.2)(c_7_2)
    c_7_3 = tf.keras.layers.Conv2D(filters = 64,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(d_7)
    
    
    c_8_1 = tf.keras.layers.Conv2DTranspose(filters = 32, 
                                            kernel_size = (2,2), 
                                            strides = (2,2), 
                                            padding = "same")(c_7_3)
    u_8 = tf.keras.layers.concatenate([c_8_1,c_2_2])
    c_8_2 = tf.keras.layers.Conv2D(filters = 32,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(u_8)
    d_8 = tf.keras.layers.Dropout(rate = 0.1)(c_8_2)
    c_8_3 = tf.keras.layers.Conv2D(filters = 32,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(d_8)
    
    c_9_1 = tf.keras.layers.Conv2DTranspose(filters = 16, 
                                            kernel_size = (2,2), 
                                            strides = (2,2), 
                                            padding = "same")(c_8_3)
    u_9 = tf.keras.layers.concatenate([c_9_1,c_1_2])
    c_9_2 = tf.keras.layers.Conv2D(filters = 16,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(u_9)
    d_9 = tf.keras.layers.Dropout(rate = 0.1)(c_9_2)
    c_9_3 = tf.keras.layers.Conv2D(filters = 16,
                                    kernel_size = (3,3), 
                                    padding = "same", 
                                    activation = "relu",
                                    kernel_initializer = "he_normal")(d_9)
    
    output_layer = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = "same", activation = "sigmoid")(c_9_3)
    
    model = tf.keras.Model(inputs=[input_tensor], outputs=[output_layer])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model