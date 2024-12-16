# What this does ?

This loads the oliviette faces dataset and uses a neural network to classify faces. 


# Hyperparameters Used

- Varying Hidden Layer Size
- Activation Functions
- Learning Rate
- Maximum Iterations

# Results

![image](https://github.com/user-attachments/assets/e336af44-a59d-49a4-86bb-28674ab643fd)

As we can see, the choice of hyperparameters significantly impacts the model's performance. From the results, we can derive the best configuration for this data set is, 

hidden_layers= (100,), activation=tanh, learning_rate=constant/adaptive with an accuracy of 0.975
