import os
from tensorflow.python import pywrap_tensorflow

current_path = os.getcwd()
model_dir = os.path.join(current_path, 'csdn_rnn_model\checkpoints')
checkpoint_path = os.path.join(model_dir,'checkpoint_100.ckpt') # 保存的ckpt文件名，不一定是这个
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # 打印变量的值，对我们查找问题没啥影响，打印出来反而影响找问题