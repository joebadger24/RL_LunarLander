Project Information:

To run the analysis on the different agent configurations run **'main.py'**.
NOTE: There is a variable in **'main.py'** called **load_data**, set this equal to **True** to show the learning curves from the last time the agent was trained.
NOTE: Set **load_data = False** if you want to run the models from scratch.

The same process can be followed to see the results of the hyper-parameter tuning in **'learning_rate.py'** and **'epsilon_decay.py'** files.

To view the output of the game at different stages in the learning, run **'test.py'** with the specified model.
NOTE: The network including its parameters is saved every 100 episodes in the **'models'** folder.
