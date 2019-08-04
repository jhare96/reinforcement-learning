class ActorCritic_CNN_Keras():
    def __init__(self, input_size, action_size, conv_sizes, kernel_sizes, strides, dense_sizes, padding, lr, activation=tf.keras.activations.relu, *args, **kwargs):
        self.model = CNN(input_size, action_size, conv_sizes, kernel_sizes, strides, dense_sizes, padding, activation=tf.keras.activations.relu)
        self.optimiser = tf.keras.optimizers.RMSprop(lr=lr)
        self.action_size = action_size

    @tf.function
    def loss(self, x, y, actions):
        policy_distrib, value = self.model(x)
        actions_onehot = tf.keras.backend.one_hot(actions, self.action_size)
        Advantage = y - value
        value_loss = tf.keras.backend.square(Advantage)
        log_policy = tf.keras.backend.log(policy_distrib)
        log_policy_actions = tf.keras.backend.sum(log_policy * actions_onehot, axis=1)
        policy_loss =  -log_policy_actions * tf.stop_gradient(Advantage)
        entropy =  tf.keras.backend.sum(policy_distrib * -log_policy, axis=1)
        loss =  tf.keras.backend.sum( policy_loss + 0.5 * value_loss - 0.01 * entropy)
        return loss
    
    @tf.function
    def grads(self,x,y,actions):
        #actions = tf.convert_to_tensor(actions,dtype=tf.int32)
        #y = tf.convert_to_tensor(y,dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss_value = self.loss(x, y, actions)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
    
    @tf.function
    def backprop(self,x,y,actions):
        loss_value, grads = self.grads(x,y,actions)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
    
    @tf.function
    def forward(self,x):
        return self.model(x)


class CNN(tf.keras.Model):
    def __init__(self, input_size, action_size, conv_sizes, kernel_sizes, strides, dense_sizes, padding, activation=tf.keras.activations.relu, *args, **kwargs):
        super(CNN, self).__init__()
        self.conv_layers = []
        for output_size, kernel_size, stride in zip(conv_sizes, kernel_sizes, strides):
            conv = tf.keras.layers.Conv2D(output_size,
                                        kernel_size,
                                        stride,
                                        padding,
                                        activation=activation)
            self.conv_layers.append(conv)
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense_layers = []
        for dense in dense_sizes:
            self.dense_layers.append(tf.keras.layers.Dense(dense,activation=activation))

        self.policy_distrib = tf.keras.layers.Dense(action_size,activation=tf.keras.activations.softmax)
        self.value =  tf.keras.layers.Dense(1)

    def call(self, states):
        x = states/255
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.flatten(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        #print('dense shape', x)
        policy_distrib = self.policy_distrib(x)
        value = self.value(x)
        return policy_distrib, value

    
    
def loss(self, x, y, actions):
    policy_distrib, value = self.model(x)
    actions_onehot = tf.keras.backend.one_hot(actions, self.action_size)
    Advantage = y - value
    value_loss = tf.keras.backend.square(Advantage)
    log_policy = tf.keras.backend.log(policy_distrib)
    log_policy_actions = tf.keras.backend.sum(log_policy * actions_onehot, axis=1)
    policy_loss =  -log_policy_actions * tf.stop_gradient(Advantage)
    entropy =  tf.keras.backend.sum(policy_distrib * -log_policy, axis=1)
    loss =  tf.keras.backend.sum( policy_loss + 0.5 * value_loss - 0.01 * entropy)
    return loss   