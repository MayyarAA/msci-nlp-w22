How to run inference.py script:

pass the following cmd args: <.txt file> <.type of model>
    
e.g: python3 inference.py ./infer_test.txt  sigmoid

list of valid models to pass in cmd args: sigmoid, relu, tanh

| model type| Accuracy(Test set) |
|------------|--------------------|
| sigmoid    | 0.50188901         |
| relu       | 0.51122908         |
| tanh       | 0.50041498         |
