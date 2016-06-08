# NegNN (Neural Network for Automatic Negation Detection)

The code implements a feed-forward NN (```/feed_forward```) and a BiLSTM(```/bilstm```) to perform automatic negation scope detection. 

## Data
Training, test and development data can be found in the ```/data``` folder.
For training, development and initial testing, we used the data released for the [*SEM2012 shared task](http://www.clips.ua.ac.be/sem2012-st-neg/); please refer to the shared task and related papers for information regarding the annotation style.
Additional test data extracted from Simple Wikipedia is available in ```/data/test/simple_wiki```. The data was manually annotated following the guidelines released during the *SEM2012 shared task. Please refer to the .pdf file for ulterior information.
The python code in ```/reader``` used to read in the data is part of the code made available by Packard et al. (2014) ([Simple Negation Scope Resolution through Deep Parsing: A Semantic Solution to a Semantic Problem](https://aclweb.org/anthology/P/P14/P14-1007.pdf)).

## Dependencies
- Tensorflow (tested on v. 0.7.1)
- scikit-learn (tested on v. 0.17.1) - for score report purposes only, feel free to use any other library instead -
- numpy (tested on v. 1.11.0)

## Train
To train the model, first go to the parent directory of the repository and run ```python NegNN/(bilstm|feed_forward)/train.py```.
```train.py``` accepts the following flags
```
```

