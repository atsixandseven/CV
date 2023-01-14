# 출처: https://junstar92.tistory.com/150
'''다룰 수 있는 이미지 파일 형식으로는 PPM, PNG, JPEG, GIF, TIFF, BMP 등이 있으며,
지원하지 않는 파일 형식은 라이브러리를 확장해서 새로운 파일 디코더를 만드는 것이 가능하다고 합니다.
PIL 이미지 작업으로는 아래의 기능들이 가능합니다.
픽셀 단위의 조작
마스킹 및 투명도 제어
흐림, 윤곽 보정 다듬어 윤곽 검출 등의 이미지 필터
선명하게, 밝기 보정, 명암 보정, 색 보정 등의 화상 조정
이미지에 텍스트 추가
기타 등등
'''

import os
import zipfile
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns

print('Tensorflow version '  + tf.__version__)

# downlad the dataset (zipped file)

file_location = '~./fcnn'

# https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view
local_zip = '~./dataset1.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('~./fcnn')
zip_ref.close()

# pixel labels in the video frames
# 총 12개의 class, annotation image(label map)의 픽셀값은 각 class에 해당되는 값으로 되어 있음

class_names = ['sky', 'building','column/pole',
              'road', 'side walk', 'vegetation',
              'traffic light', 'fence', 'vehicle',
              'pedestrian', 'bicyclist', 'void']

# training/Validation Dataset 만들기

train_image_path = file_location + '/dataset1/images_prepped_train/'
train_label_path = file_location + '/dataset1/annotations_prepped_train/'
test_image_path = file_location + '/dataset1/images_prepped_test/'
test_label_path = file_location + '/dataset1/annotations_prepped_test/'

BATCH_SIZE = 64

def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):
    '''
    Preprocesses the dataset by:
        * resizing the input image and label maps
        * normalizing the input image pixels
        * reshaping the label maps from (height, width, 1) to (height, width, 12)

    Args:
        t_filename(string) -- path to the raw input image
        a_filename(string) -- path to the raw annotation (label map) file
        height(int) -- height in pixels to resize to
        width(int) -- width in pixels to resize to

    returns:
        image(tensor) -- preprocessed image
        annotation(tensor) -- preprocessed annotation

    '''

    # Convert image and mask file to tensors
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)

    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotation = tf.stack(stack_list, axis=2)

    # Normalize pixels in the input image
    image = image / 127.5
    image -= 1

    return image, annotation

def get_dataset_slice_paths(image_dir, label_map_dir):
    '''

    :param image_dir (string): path to the input images directory
    :param label_map_dir (string): path to the label map directory

    :return:
        image_paths (list of strings): paths to each image file
        label_map_paths (list of strings): paths to each label map
    '''

    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths

def get_training_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the training set.

    :param image_paths (string): path to the input images directory
    :param label_map_paths (string): path to the label map directory
    :return:
        tf Dataset containing the preprocessed train set
    '''

    # tensorflow Dataset 사용방법 참조 페이지: https://hiseon.me/data-analytics/tensorflow/tensorflow-dataset/
    # tf.data.Dataset.from_tensor_slices 함수는 tf.data.Dataset 를 생성하는 함수로 입력된 텐서로부터 slices를 생성합니다.
    # 예를 들어 MNIST의 학습데이터 (60000, 28, 28)가 입력되면, 60000개의 slices로 만들고 각각의 slice는 28×28의 이미지 크기를 갖게 됩니다.

    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))

    #  Maps `map_func` across the elements of this dataset.
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)

    # Randomly shuffles the elements of this dataset.
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    # shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None, name=None)
    #     This dataset fills a buffer with `buffer_size` elements, then randomly
    #     samples elements from this buffer, replacing the selected elements with new
    #     elements. For perfect shuffling, a buffer size greater than or equal to the
    #     full size of the dataset is required.

    # Combines consecutive elements of this dataset into batches.
    training_dataset = training_dataset.batch(BATCH_SIZE)

    # Repeats this dataset so each original value is seen `count` times.
    training_dataset = training_dataset.repeat()

    # Creates a `Dataset` that prefetches elements from this dataset.
    training_dataset = training_dataset.prefetch(-1)
    # prefetch(self, buffer_size, name=None)
    #     Most dataset input pipelines should end with a call to `prefetch`. This
    #     allows later elements to be prepared while the current element is being
    #     processed. This often improves latency and throughput, at the cost of
    #     using additional memory to store prefetched elements.

    return training_dataset

def get_validataion_dataset(image_paths, label_map_paths):
    '''

    :param image_paths (string): path to the input images directory
    :param label_map_paths (string): path to the label map directory
    :return:
        tf Dataset containing the preprocessed train set

    '''

    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()

    return validation_dataset

# Dataset 만들기

# get the paths to the images
training_image_paths, training_label_map_paths = get_dataset_slice_paths(train_image_path, train_label_path)
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(test_image_path, test_label_path)

# generate the train and valid sets
training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
validation_dataset = get_validataion_dataset(validation_image_paths, validation_label_map_paths)

# 각 클래스의 segmentation 색상을 지정
# seaborn의 color_pallette 사용해, RGB값 불러오기

# color_palette(palette=None, n_colors=None, desat=None, as_cmap=False)
#     Return a list of colors or continuous colormap defining a palette.
# Calling this function with ``palette=None`` will return the current matplotlib color cycle.
colors = sns.color_palette(None, len(class_names))

# print class name - normalized RGB tuple pairs
# the tuple values will be multiplied by 255 in the helper functions later
# to convert to the (0, 0, 0) to (255, 255, 255) RGB values you might be familiar with
for class_name, color in zip(class_names, colors):
    print(f'{class_name} -- {color}')

# Visulization Utilities

def fuse_with_pil(images):
    '''
    Creates a blank image and pastes input images
    :param images (list of numpy arrays) : numpy array representations of the images to paste
    :return:
        PIL Image object containing the images
    '''

    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset, 0))
        x_offset += im.shape[1]

    return new_im

def give_color_to_annotation(annotation):
    '''
    Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    :param annotation (numpy array) : label map array
    :return:
        the annotation array with an additional color channel/axis
    '''

    seg_img = np.zeros((annotation.shape[0], annotation.shape[1], 3)).astype('float')

    for c in range(12):
        segc = (annotation == c)
        seg_img[:, :, 0] += segc*( color[c][0] * 255.0)
        seg_img[:, :, 1] += segc * (color[c][1] * 255.0)
        seg_img[:, :, 2] += segc * (color[c][2] * 255.0)

    return seg_img

def show_predictions(image, labelmaps, titles, iou_list, dice_score_list):
    '''

    Displays the images with the ground truth and predicted label maps

    :param image (numpy array) : the input image
    :param labelmaps (list of arrays): contains the predicted and ground truth label maps
    :param titles (list of strings) : display heading for the images to be displayed
    :param iou_list (list of floats) : the IOU values for each class
    :param dice_score_list (list of floats) : the Dice Score for each values
    :return:
    '''

    # Dice score: Sørensen–Dice coefficient 라고도 하며, F1 Score와 개념상 같음
    # Dice = 2 * TP / ((TP + FP) + (TP + FN))

    true_img = give_color_to_annotation(labelmaps[1])
    pred_img = give_color_to_annotation(labelmaps[0])

    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img, true_img])
    # unit8은 1. 양수만 표현이 가능하고 2. 2^8 개수 만큼 표현이 가능 = 0 ~ 255

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tip: tip[1], reverse=True) # sorts in place
    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score)
                           for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list)

    plt.figure(figsize=(15,4))
    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)

def show_annotation_and_image(image, annotation):
    '''
    Displays the image and its annotation side by side

    :param image (numpy array): the input image
    :param annotation (numpy array) : the label image
    '''

    new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(new_ann)

    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, seg_img]

    fused_img = fuse_with_pil(images)
    plt.imshow(fused_img)

def list_show_anotation(dataset):
    '''
    Displays images and its annotations side by side

    :param dataset (tf Dataset) : batch of images and annotations
    '''

    ds = dataset.unbatch()
    # unbatch(self, name=None)
    #     Splits elements of a dataset into multiple elements.
    #
    #     For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
    #     where `B` may vary for each input element, then for each element in the
    #     dataset, the unbatched dataset will contain `B` consecutive elements
    #     of shape `[a0, a1, ...]`.
    #     elements = [ [1, 2, 3], [1, 2], [1, 2, 3, 4] ]
    #     dataset = tf.data.Dataset.from_generator(lambda: elements, tf.int64)
    #     dataset = dataset.unbatch()
    #     list(dataset.as_numpy_iterator())
    #          [1, 2, 3, 1, 2, 1, 2, 3, 4]

    ds = ds.shuffle(buffeer_size = 100)

    plt.figure(figsize=(25, 15))
    plt.title("Image and Annotation")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    # we set the number of image-annotation pairs to 9
    # feel free to make this a function parameter if you want

    for idx, (image, annotation) in enumerate(ds.take(9)):
        # take(self, count, name=None)
        #     Creates a `Dataset` with at most `count` elements from this dataset.
        #
        #     >>> dataset = tf.data.Dataset.range(10)
        #     >>> dataset = dataset.take(3)
        #     >>> list(dataset.as_numpy_iterator())
        #     [0, 1, 2]
        plt.subplot(3, 3, idx+1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy())

    list_show_anotation(training_dataset)


# Encoder
# VGG-16 feature extractor
# [conv1_1 | conv1_2 | pooling1] + [conv2_1 | conv2_2 | pooling2] +
# [conv3_1 | conv3_2 | conv3_3 | pooling3] + [conv4_1 | conv4_2 | conv4_3 | pooling1] +
# [conv5_1 | conv5_2 | conv5_3 | pooling5]

def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    '''
    Defines a block in the VGG block

    :param x (tensor) : input image
    :param n_convs (int) : number of convolution layers to append
    :param filters (int) : number of filters(kernels) for the convolution layers
    :param kernel_size (int): size of the kernel
    :param activation (string or object) : activation func. to use in teh covolution
    :param pool_size (int): size of the pooling layer
    :param pool_stride (int) : stride of the pooling layer
    :param block_name (string): name of the block
    :return:
        tensor containing the max-pooled output of the convolution
    '''

    for i in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   padding='same',
                                   name=f'{block_name}_cov{i+1}')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=pool_stride,
                                     name=f'{block_name}_pool{i+1}')(x)

    return x

# 학습된 VGG-16의 weight 다운로드
# !wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

vgg_weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG-16 Network 구현
# 입력 shape는 (244, 244, 3), block 함수를 통해서 VGG-16 network 구현

def VGG_16(image_input):
    '''
    This function defiens the VGG encoder.

    :param image_input (tensor): batch of images
    :return:
        tuple of tensors -- output of all encoder blocks plus the final convolution layer
    '''

    # create 5 blocks with increasing filters at each stage

    x = block(image_input, n_convs=2, filters=64, kernel_size=(3,3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block1')
    p1 = x # (112, 112, 64)

    x = block(x, n_convs=2, filters=128, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block2')

    p2 = x # (56, 56, 128)

    x = block(x, n_convs=3, filters=256, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block3')

    p3 = x # (28, 28, 256)

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block4')

    p4 = x  # (14, 14, 512)

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block5')

    p5 = x  # (7, 7, 512)

    # create the vgg model
    vgg = tf.keras.Model(image_input, p5)

    # load the pretrained weights downloaded
    vgg.load_weights(vgg_weights_path)

    # number of filters for the output convolutional layers
    n = 4096

    # our input images are 224 * 224 pixels so they will be down-sampled to 7 * 7 after the pooling layers above.
    # we can extract more features by chaining two more convolution layers.
    c6 = tf.keras.layers.Conv2D(n, [7, 7], activation='relu', padding='same', name="conv6")(p5)
    # 마지막 layer는 1 * 1 convolution layer를 통해서 depth를 class 개수로 변경함
    c7 = tf.keras.layers.Conv2D(n, [1, 1], activation='relu', padding='same', name='conv7')(c6)

    # class Conv2D(keras.layers.convolutional.base_conv.Conv)
    #  |  Conv2D(filters, kernel_size, strides=(1, 1),
    #  padding='valid', data_format=None, dilation_rate=(1, 1),
    #  groups=1, activation=None, use_bias=True,
    #  kernel_initializer='glorot_uniform',
    #  bias_initializer='zeros', kernel_regularizer=None,
    #  bias_regularizer=None, activity_regularizer=None,
    #  kernel_constraint=None, bias_constraint=None, **kwargs)

    # return the outputs at each stage. you will only need two of these in this particular exercise
    # But we included it all in case you want to experiment with other types of decoders.

    return (p1, p2, p3, p4, c7) # skip connection을 위해서 각 layer의 pooling layer도 return함

# Decoder
# 비교를 위해 FCN-32, FCN-16, FCN-8 모두 구현
def decoder(convs, n_classes):
    '''

    Defines the FCN 32, 16, 8 decoder.

    :param convs (tuple of tensors): output of teh encoder network
    :param n_classes (int): number of classes
    :return:
        tensor with shape (height, width, n_classes) containing class probabilites(FCN-32, FCN-16, FCN-8)

    '''

    # unpack the output of the encoder
    f1, f2, f3, f4, f5 = convs
    '''
    f1 = (112, 112, 64)
    f2 = (56, 56, 128)
    f3 = (28, 28, 256)
    f4 = (14, 14, 512)
    f5 = (7, 7, 512)
    '''


    # FCN-32 output
    fcn32_o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(32, 32), strides=(32, 32), use_bias=False)(f5)
    fcn32_o = tf.keras.layers.Activation('softmax')(fcn32_o)

    # up-sample the output of the encoder then crop extra pixels that were introduced
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(f5) # (16, 16, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (14, 14, n)

    # load the pool4 prediction and do a 1*1 convolution to reshape it to the same shape of 'o' above.
    o2 = f4 # (14, 14, 512)
    o2 = tf.keras.layers.Conv2D(n_classes, (1,1), activation='relu', padding='same')(o2) # (14, 14, n)

    # add the result of the up-sampling and pool4 prediction
    o = tf.keras.layers.Add()([o, o2]) # (14, 14, n)

    # FCN-16 output
    fcn16_o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(16, 16), use_bias=False)(o)
    fcn16_o = tf.keras.layers.Activation('softmax')(fcn16_o)

    # up-sample the output of the encoder then crop extra pixels that were introduced
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)  # (30, 30, n)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)  # (28, 28, n)

    # load the pool3 prediction and do a 1*1 convolution to reshape it to the same shape of 'o' above.
    o2 = f3  # (28, 28, 512)
    o2 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='relu', padding='same')(o2)  # (28, 28, n)

    # add the result of the up-sampling and pool3 prediction
    o = tf.keras.layers.Add()([o, o2])  # (28, 28, n)

    # up-sample upto the size of the original image
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o) # (224, 224, n)

    # append a softmax to get the class probabilities
    fcn8_o = tf.keras.layers.Activation('softmax')(o)

    return fcn32_o, fcn16_o, fcn8_o

# Encoder + Decoder
def segmentation_model():
    '''
    defines the final segmentation model by chaining together the encoder and decoder.

    :return:
        keras Model that connects the encoder and decoder networks of the segmentation model
    '''

    inputs = tf.keras.layers.Input(shape=(224, 224, 3, ))
    # Input(shape=None, batch_size=None, name=None, dtype=None,
    #       sparse=None, tensor=None, ragged=None, type_spec=None, **kwargs)
    #     `Input()` is used to instantiate a Keras tensor.
    # For instance, if `a`, `b` and `c` are Keras tensors,
    #     it becomes possible to do:
    #     `model = Model(input=[a, b], output=c)`

    convs = VGG_16(inputs)
    fcn32, fcn16, fcn8 = decoder(convs, 12)

    model_fcn32 = tf.keras.Model(inputs, fcn32)
    model_fcn16 = tf.keras.Model(inputs, fcn16)
    model_fcn8 = tf.keras.Model(inputs, fcn8)

    return model_fcn32, model_fcn16, model_fcn8

model_fcn32, model_fcn16, model_fcn8  = segmentation_model()

# Compile the Model

sgd = tf.keras.optimizers.SGD(learning_rate=10^-4, momentum=0.9, nesterov=True)
model_fcn32.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['acc'])
model_fcn16.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['acc'])
model_fcn8.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['acc'])

# number of training images
train_count = len(training_image_paths)

# number of validation images
valid_count = len(validation_image_paths)

EPOCHS = 170

steps_per_epoch = train_count // BATCH_SIZE
validation_steps = valid_count // BATCH_SIZE

history_fcn32 = model_fcn32.fit(training_dataset,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_dataset,
                                validation_steps=validation_steps,
                                epochs=70)

history_fcn16 = model_fcn16.fit(training_dataset,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_dataset,
                                validation_steps=validation_steps,
                                epochs=100)

history_fcn8 = model_fcn8.fit(training_dataset,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_dataset,
                                validation_steps=validation_steps,
                                epochs=100)

# Evaluate the Model
def get_images_and_segments_test_arrays():
    '''
    Gets a subsample of the val set as your test set
    :return:
        Test set containing ground truth images and label maps
    '''

    y_ture_segments = []
    y_true_images = []
    test_count = 64

    ds = validation_dataset.unbatch()
    ds = ds.batch(101) # batch size = 101

    for image, annotation in ds.take(1):
        y_true_images = image
        y_true_segments = annotation

    y_true_segments = y_true_segments[:test_count, :, :, :]
    y_true_segments = np.argmax(y_true_segments, axis=3)

# load the ground truth images and segmentation masks
y_true_images, y_true_segments = get_images_and_segments_test_arrays()

# get the model prediction
results_fcn32 = model_fcn32.predict(validation_dataset, steps=validation_steps)
results_fcn16 = model_fcn16.predict(validation_dataset, steps=validation_steps)
results_fcn8 = model_fcn8.predict(validation_dataset, steps=validation_steps)

# for each pixel, get the slice number which has the highest probability

results_fcn32 = np.argmax(results_fcn32, axis=3)
results_fcn16 = np.argmax(results_fcn16, axis=3)
results_fcn8 = np.argmax(results_fcn8, axis=3)

# Evaluation
# IoU = area of overlap / area of union // intersection of union, 0-1 사이 값으 가짐
# DiceScore = 2 * (area of overlap / combined area) // 0-1 사이 값을 가짐
# 두 평가 지표 모두 1에 가까울 수록 segmentation model 성능이 우수

def compute_metrics(y_true, y_pred):
    '''
    Compute IoU and Dice Score

    :param y_true (tensor): ground truth label map
    :param y_pred (tensor): predicted label map
    '''

    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(12):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)

        dice_score = 2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score

# input a number from 0 to 63 to pick an image from the test set
integer_slider = 20

# compute metrics
iou_fcn32, dice_score_fcn32 = compute_metrics(y_true_segments[integer_slider], results_fcn32[integer_slider])
iou_fcn16, dice_score_fcn16 = compute_metrics(y_true_segments[integer_slider], results_fcn16[integer_slider])
iou_fcn8, dice_score_fcn8 = compute_metrics(y_true_segments[integer_slider], results_fcn8[integer_slider])

# visualize the output and metrics
show_predictions(y_true_images[integer_slider],
                 [results_fcn32[integer_slider], y_true_segments[integer_slider]],
                  ['Image', 'Prediction Mask', 'True Mask'],
                  iou_fcn32,
                  dice_score_fcn32)
show_predictions(y_true_images[integer_slider],
                 [results_fcn16[integer_slider], y_true_segments[integer_slider]],
                 ['Image', 'Prediction Mask', 'True Mask'],
                 iou_fcn16, dice_score_fcn16)
show_predictions(y_true_images[integer_slider],
                 [results_fcn8[integer_slider], y_true_segments[integer_slider]],
                 ['Image', 'Prediction Mask', 'True Mask'],
                 iou_fcn8, dice_score_fcn8)

cls_wise_iou_fcn32, cls_wise_dice_score_fcn32 = compute_metrics(y_true_segments, results_fcn32)
cls_wise_iou_fcn16, cls_wise_dice_score_fcn16 = compute_metrics(y_true_segments, results_fcn16)
cls_wise_iou_fcn8, cls_wise_dice_score_fcn8 = compute_metrics(y_true_segments, results_fcn8)
