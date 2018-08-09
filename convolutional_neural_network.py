import numpy as np

class CNN:
    def _init_conv_layers_filters(self, conv_layers_filters_shapes):
        self.conv_layers_filters = []

        for _, conv_layer_filters_shapes in enumerate(conv_layers_filters_shapes):
            conv_layer_filters = np.random.rand(*(conv_layer_filters_shapes))
            self.conv_layers_filters.append(conv_layer_filters)

    def _init_dense_layers_weights(self, dense_layers_neurons_count):
        self.dense_layers_weights = []

        dense_layers_count = len(dense_layers_neurons_count)

        for end_layer_index in range(1, dense_layers_count):
            start_layer_index = end_layer_index - 1

            start_neuron_count = dense_layers_neurons_count[start_layer_index]
            end_neuron_count = dense_layers_neurons_count[end_layer_index]

            end_layer_weights = np.random.rand(*(end_neuron_count, start_neuron_count))

            self.dense_layers_weights.append(end_layer_weights)

    def _init_conv_values(self, conv_layers_filters_shapes):
        self.conv_layers_values = []

        for _, conv_layer_filters_shapes in enumerate(conv_layers_filters_shapes):
            filter_width, filter_height, output_channel_count =


    def __init__(self, conv_layers_filters_shapes, dense_layers_neurons_count):
        self._init_conv_layers_filters(conv_layers_filters_shapes)
        self._init_dense_layers_weights(dense_layers_neurons_count)

    def conv_channel_forward(self, input_feature_map, conv_layer_filter):
        input_feature_map_height, input_feature_map_width = input_feature_map.shape
        filter_height, filter_width, input_channel_count = conv_layer_filter.shape

        output_feature_map_height = input_feature_map_height - filter_height + 1
        output_feature_map_width = input_feature_map_width - filter_width + 1

        output_feature_map = np.zeros((output_feature_map_height, output_feature_map_width))

        for output_feature_map_y_index in range(output_feature_map_height):
            for output_feature_map_x_index in range(output_feature_map_width):
                input_feature_map_region = input_feature_map[
                                           output_feature_map_y_index:output_feature_map_y_index + filter_height,
                                           output_feature_map_x_index:output_feature_map_x_index + filter_height,
                                           :]

                output_feature_map[output_feature_map_y_index, output_feature_map_x_index] = np.sum(
                    input_feature_map_region * conv_layer_filter)

        cache = (input_feature_map, conv_layer_filter)

        return output_feature_map, cache

    def conv_channels_forward(self, input_feature_maps, conv_layer_filters):
        output_feature_maps = np.empty(input_feature_maps.shape)

        for output_feature_map_index in range(len(output_feature_maps)):
            filter = conv_layer_filters[output_feature_map_index]

            output_feature_map = conv_channel_forward(input_feature_maps, filter)

            output_feature_maps[output_feature_map_index] = output_feature_map
        return output_feature_maps

    def dense_forward(self, input_layer_neurons, weights):
        end_layer_neurons_count = weights.shape[0]
        end_layer_neurons = np.zeros((end_layer_neurons_count,))

        for output_layer_neuron_index in range(end_layer_neurons_count):
            end_neuron_value = 0

            for input_layer_neuron_index in range(len(input_layer_neurons)):
                weight = weights[0, output_layer_neuron_index]
                start_neuron_value = input_layer_neurons[input_layer_neuron_index]

                end_neuron_value += weight * start_neuron_value

            end_layer_neurons[output_layer_neuron_index] = end_neuron_value

        return end_layer_neurons

    def dense_backward(self, layers_weights, layers_neurons, costs):
        layers_weight_gradients = []
        dense_layer_count = layers_neurons.shape[0]

        output_layer_index = dense_layer_count - 1
        layers_weights_gradient = []

        for end_layer_index in range(output_layer_index, 0, -1):
            start_layer_index = end_layer_index - 1

            start_layer_neurons = layers_neurons[start_layer_index]
            end_layer_neurons = layers_neurons[end_layer_index]

            if end_layer_index == output_layer_index:
                end_layer_neurons_gradient = dense_backward_output_layer(end_layer_neurons, costs)
            else:
                end_layer_weights = layers_weights[end_layer_index]
                end_layer_neurons_gradient = dense_backward_non_output_layer(end_layer_weights,
                                                                             end_layer_neurons_gradient,
                                                                             start_layer_neurons)

            layer_weights_gradient = np.shape((end_layer_neurons, start_layer_neurons))

            start_layer_neuron_count = len(start_layer_neurons)
            end_layer_neuron_count = len(layers_neurons[end_layer_index])

            layer_weight_gradients = np.zeros((start_layer_neuron_count, end_layer_neuron_count))

            for end_neuron_index in range(end_layer_neuron_count):
                for start_neuron_index in range(start_layer_neuron_count):
                    start_neuron_value = start_layer_neurons[start_layer_index][start_neuron_index]
                    weight = end_layer_neurons_gradient[end_neuron_index][start_neuron_index]

                    weight_gradient = start_neuron_value * weight
                    layer_weight_gradients[end_neuron_index, start_neuron_index] = weight_gradient

            layers_weight_gradients.append(layer_weight_gradients)

        return layers_weight_gradients

    def dense_backward_output_layer(self, observed_output_layer_neurons, costs):
        return observed_output_layer_neurons * (1 - observed_output_layer_neurons) * costs

    def dense_backward_non_output_layer(self, layer_weights, end_layer_neurons_gradient, start_layer_neurons):
        start_layer_neurons_gradients = np.empty(start_layer_neurons)

        for start_layer_neuron_index in range(len(start_layer_neurons)):
            start_neuron_gradient = 0

            for end_neuron_gradient_index in range(len(end_layer_neurons_gradient)):
                weight = layer_weights[end_neuron_gradient_index, start_layer_neuron_index]
                start_neuron = start_layer_neurons[start_layer_neuron_index]
                end_neuron_gradient = end_layer_neurons_gradient[end_neuron_gradient_index]

                start_neuron_gradient += weight * end_neuron_gradient * start_neuron * (1 - start_neuron)

            start_layer_neurons_gradients[start_layer_neuron_index] = start_neuron_gradient

        return start_layer_neurons_gradients

    def conv_backward(self, output_feature_map_gradient, cache):
        input_feature_map, filter = cache

        input_feature_map_height, input_feature_map_width = input_feature_map.shape
        filter_height, filter_width = filter.shape

        # We can retrieve shape of output feature map from its gradient since those shapes are equivalent
        output_feature_map_height, output_feature_map_width = output_feature_map_gradient.shape

        input_feature_map_gradient = np.zeros(input_feature_map.shape)
        filter_gradient = np.zeros(filter.shape)

        for output_feature_map_y_index in range(output_feature_map_height):
            for output_feature_map_x_index in range(input_feature_map_width):
                input_feature_map_region_gradient = input_feature_map_gradient[
                                                    output_feature_map_y_index:output_feature_map_y_index + filter_height,
                                                    output_feature_map_x_index:output_feature_map_x_index + filter_width,
                                                    :]

                output_feature_map_region_gradient = output_feature_map_gradient[output_feature_map_x_index,
                                                     output_feature_map_y_index,
                                                     :]

                input_feature_map_region_gradient += filter * output_feature_map_region_gradient

                input_feature_map_region = input_feature_map[
                                           output_feature_map_y_index:output_feature_map_y_index + filter_height,
                                           output_feature_map_x_index:output_feature_map_x_index + filter_width]

                filter_gradient += input_feature_map_region * output_feature_map_region_gradient

        return input_feature_map_gradient, filter_gradient

    def forward(self, image_pixel_maps, conv_layers_filters, dense_weights):
        conv_layer_count = conv_layers_filters.shape[0]

        current_feature_maps = image_pixel_maps

        for conv_layer_index in range(conv_layer_count):
            conv_layer_filters = conv_layers_filters[0]

            current_feature_maps = conv_channels_forward(current_feature_maps, conv_layer_filters)

        flattened_feature_maps = np.ndarray.flatten(current_feature_maps)

        dense_layer_count = dense_weights.shape[0]

        current_layer_neurons = flattened_feature_maps

        for dense_layer_index in range(dense_layer_count):
            dense_layer_weights = dense_weights[dense_layer_index]

            neuron_weighted_sums = np.matmul(dense_layer_weights, current_layer_neurons)
            activated_neurons = np.tanh(neuron_weighted_sums)

            current_layer_neurons = activated_neurons

        return current_layer_neurons

    def backward(self, conv_layer_filters, dense_layers_weights, costs):
        self.dense_backward(dense_layers_weights)

    @staticmethod
    def new_one_hot_encoder(image_classes):
        unique_classes = np.unique(image_classes)

        def encode(image_class):
            class_index = unique_classes[image_class]

            class_one_hot_vector = np.zeros(len(unique_classes))
            class_one_hot_vector[class_index] = 1

            return class_one_hot_vector

        def decode(class_one_hot_vector):
            class_index = np.where(class_one_hot_vector == 1)

            return class_index

        return encode, decode

    def train(self, images_pixel_maps, image_classes, conv_layers_filters, dense_layers_weights):
        encode, decode = CNN.new_one_hot_encoder(image_classes)

        for _, image_pixel_maps in images_pixel_maps:
            self.forward(image_pixel_maps, conv_layers_filters, dense_layers_weights)

