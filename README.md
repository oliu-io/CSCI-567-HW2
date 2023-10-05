# CSCI-567 Assignment 2

## Work on the assignment
Working on the assignment in an Anaconda environment is highly encouraged.
In this assignment, please use Python `3.9`.
You will need to make sure that your conda setup is of the correct version of python.

Please refer to [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) for instructions on installing Anaconda.

## The objectives of this assignment
We have prepared to problems for this assignment, CNN and BERT
### CNN
* Implement the forward and backward passes as well as the neural network training procedure
* Implement the widely-used optimizers and training tricks including dropout, weight decay, and L1&L2 regularizations
* Implement convolutional neural networks from scratch
* Use convolutional neural networks for image classification
### BERT
* Implement the forward passes of the BERT architecture
* Verify your implementation against tested Hugging Face reference checkpoints.
* We strongly recommend you read through the [BERT](https://arxiv.org/abs/1810.04805) paper and Jay Alammar's [tutorial](http://jalammar.github.io/illustrated-bert/) before getting started.


Please see below for instructions on how to create and activate a conda environment on your terminal.
```shell
cd CSCI-567-HW2

conda init                      # If your conda isn't active already
conda create -n hw2 python=3.9 # If you didn't install it

# replace your_virtual_env with the virtual env name you want
conda activate hw2

# install dependencies
pip3 install -r requirements.txt
```


## Problems
In each of the notebook files, we indicate `TODO`, `Your Code`, or `NotImplementedError` for you to fill in with your implementation.
Majority of implementations will also be required under `lib` with specified tags.
* CNN: The IPython Notebook `CNN/Problem.ipynb` will walk you through implementing the basics of neural networks, and then then implementing a convolutional neural network (CNN) from scratch.
* BERT: You should implement all the code segments in `BYOB/bert_impl.py`. To verify your implementation, run
```shell
cd BYOB
python bert_impl.py
```

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are more than welcome to assist through Piazza.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS.

## FAQ

- **Cannot get 30% accuracy for TinyNet in CNN-Problem 1**\
You can try to vary the batch size, epochs and learning rate decay. Please don't modify any code outside the TODO block.

- **What is a good starting learning rate?**\
There is a good article: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

- **When parsing the text data file for CNN-Problem 2 I get a `charmap codec can't decode` error.**\
This might be a platform dependent issue. In past years adding `encoding="utf8"` to the file `open` command helped in these cases.

- **What is the `meta` variable used for in CNN-Problem 2?**\
This variable is used to pass all values to the backward pass that are necessary to compute the gradients. You can use it as a dictionary to pass any desired values over to the `backward` function.

- **I experimented with the hyperparameters and tried many different combinations, which ones should I report?**\
The usual rule of thumb is to report results with the best hyperparameters you found. \

- **Am I allowed to change code outside the TODO blocks?**\
Unless specified otherwise (eg for hyperparameters) please do not change any code outside TODO blocks.

- **My %reload_ext autoreload command does not work, how to fix it?**\
This has been observed in the past and, whenever it was a problem, could be fixed by downgrading IPython to version 7.5.0: `pip3 install ipython==7.5`.

- **General debugging tips**
1. Make sure your implementations matches the specified model layers perfectly.
2. Put print statements at various places inside your implementation code to make sure every module is working as it should. 
