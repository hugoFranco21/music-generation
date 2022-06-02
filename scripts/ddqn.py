class DDQNAgent:
    def __init__(self, expirience_replay, state_size, actions_size, optimizer):
        
        # Initialize atributes
        self._state_size = state_size
        self._action_size = actions_size
        self._optimizer = optimizer
        
        self.expirience_replay = expirience_replay
        
        # Initialize discount and exploration rate
        self.epsilon = MAX_EPSILON
        
        # Build networks
        self.primary_network = self._build_network()
        self.primary_network.compile(loss='mse', optimizer=self._optimizer)

        self.target_network = self._build_network()   

    def _build_network(self):
        network = Sequential()
        network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(self._action_size))
        
        return network
    
    def align_epsilon(self, step):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * step)
    
    def align_target_network(self):
        for t, e in zip(self.target_network.trainable_variables, 
                    self.primary_network.trainable_variables): t.assign(t * (1 - TAU) + e * TAU)
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self._action_size - 1)
        else:
            q_values = self.primary_network(state.reshape(1, -1))
            return np.argmax(q_values)
    
    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.store(state, action, reward, next_state, terminated)
    
    def train(self, batch_size):
        if self.expirience_replay.buffer_size < BATCH_SIZE * 3:
            return 0
        
        batch = self.expirience_replay.get_batch(batch_size)
        states, actions, rewards, next_states = expirience_replay.get_arrays_from_batch(batch)
        
        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_values_state = self.primary_network(states).numpy()
        q_values_next_state = self.primary_network(next_states).numpy()
        
        # Copy the q_values_state into the target
        target = q_values_state
        updates = np.zeros(rewards.shape)
                
        valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(BATCH_SIZE)

        action = np.argmax(q_values_next_state, axis=1)
        q_next_state_target = self.target_network(next_states)
        updates[valid_indexes] = rewards[valid_indexes] + GAMMA * 
                q_next_state_target.numpy()[batch_indexes[valid_indexes], action[valid_indexes]]
        
        target[batch_indexes, actions] = updates
        loss = self.primary_network.train_on_batch(states, target)

        # update target network parameters slowly from primary network
        self.align_target_network()
        
        return loss