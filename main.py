import tensorflow as tf
import numpy as np
import tqdm
import PIL.Image
import argparse

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--disable_gpu', type=bool, default=False)
parser.add_argument('--content_image', type=str, default=None)
parser.add_argument('--style_image', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='output/test.png')
args = parser.parse_args()

if args.disable_gpu:
    tf.config.set_visible_devices([], 'GPU')


def load_img(path_to_img, target_size=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = target_size / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis]

def extract_intermediate_layers(model, layer_names):
    model = model(include_top=False, weights='imagenet')
    outputs = [
        model.get_layer(name).output for name in layer_names
    ]
    return tf.keras.Model(inputs=model.input, outputs=outputs)

def compute_gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


class StyleContentExtractor(tf.keras.Model):

    def __init__(self, model, content_layers, style_layers, **kwargs):

        super(StyleContentExtractor, self).__init__(**kwargs)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_style_layers = len(self.style_layers)

        self.model = extract_intermediate_layers(
            model,
            self.style_layers + self.content_layers)

        self.model.trainable=False

    def call(self, inputs):

        inputs = inputs * 255.0
        inputs  = tf.keras.applications.resnet.preprocess_input(
            inputs)

        outputs = self.model(inputs)
        content_outputs = outputs[self.num_style_layers:]
        style_outputs = [
            compute_gram_matrix(style_output)
            for style_output in outputs[:self.num_style_layers]
        ]

        content_dict = {
            name:value for name, value in zip(self.content_layers, content_outputs)
        }
        style_dict = {
            name:value for name, value in zip(self.style_layers, style_outputs)
        }

        return {'content': content_dict, 'style': style_dict}


class NeuralStyleTransfer:

    def __init__(self,
                 model,
                 optimizer,
                 content_layers,
                 style_layers,
                 content_loss_weight=10_000,
                 style_loss_weight=0.01,
                 total_variation_loss_weight=30):

        self.optimizer = optimizer
        self.extractor = StyleContentExtractor(
            model,
            content_layers=content_layers, style_layers=style_layers
        )
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.total_variation_loss_weight = total_variation_loss_weight

    def _compute_loss(self, targets, outputs):
        style_outputs = outputs['style']
        style_targets = targets['style']
        content_outputs = outputs['content']
        content_targets = targets['content']

        content_loss = tf.add_n(
            [tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
            for name in content_outputs.keys()]
        )
        content_loss *= self.content_loss_weight / self.num_content_layers

        style_loss = tf.add_n(
            [tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
             for name in style_outputs.keys()]
        )
        style_loss *= self.style_loss_weight / self.num_style_layers

        return content_loss + style_loss

    def _backprop_loss(self, tape, loss, weights):
        gradients = tape.gradient(loss, weights)
        self.optimizer.apply_gradients([(gradients, weights)])
        weights.assign(tf.clip_by_value(weights, 0.0, 1.0))

    def _get_train_step(self):
        @tf.function
        def _train_step(image, targets):
            with tf.GradientTape() as tape:
                outputs = self.extractor(image)
                loss = self._compute_loss(targets, outputs)
                loss += (
                    self.total_variation_loss_weight
                    * tf.image.total_variation(image)
                )[0]
            self._backprop_loss(tape, loss, image)
            return loss
        return _train_step

    def fit_style(self, image, targets, iterations):

        self.image = image

        train_step = self._get_train_step()

        pbar = tqdm.tqdm(range(iterations))
        for i in pbar:
            loss = train_step(self.image, targets)
            pbar.set_description('Loss = {}'.format(loss))

    @property
    def get_styled_image(self):
        return self.image


if __name__ == '__main__':

    if args.content_image is None:
        content_path = tf.keras.utils.get_file(
            'YellowLabradorLooking_new.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/'
             + 'example_images/YellowLabradorLooking_new.jpg')
        content_image = load_img(content_path)
    else:
        content_image = load_img(args.content_image)

    if args.style_image is None:
        style_path = tf.keras.utils.get_file(
            'kandinsky5.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/'
            + 'example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
        style_image = load_img(style_path)
    else:
        style_image = load_img(args.style_image)


    model = NeuralStyleTransfer(
        model=tf.keras.applications.VGG19,
        optimizer=tf.optimizers.Adam(learning_rate=1e-1),
        content_layers=[
            'block5_conv2'
        ],
        style_layers=[
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ],
        content_loss_weight=10_000,
        style_loss_weight=0.01,
        total_variation_loss_weight=30)

    targets = {
        'content': model.extractor(content_image)['content'],
        'style': model.extractor(style_image)['style']
    }

    image = tf.Variable(content_image)

    model.fit_style(image, targets, iterations=3000)
    image = np.squeeze(model.get_styled_image.numpy()*255.0, 0)
    image = image.astype(np.uint8)
    image = PIL.Image.fromarray(image)
    image.save(args.output_dir)
