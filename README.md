# NegNN (Neural Network for Automatic Negation Detection)

The code implements a feed-forward NN (```/feed_forward```) and a BiLSTM(```/bilstm_class``` or ```/bilstm```) to perform automatic negation scope detection. 

## Data
Training, test and development data can be found in the ```/data``` folder.
For training, development and initial testing, we used the data released for the [*SEM2012 shared task](http://www.clips.ua.ac.be/sem2012-st-neg/); please refer to the shared task and related papers for information regarding the annotation style.
Additional test data extracted from Simple Wikipedia is available in ```/data/test/simple_wiki```. The data was manually annotated following the guidelines released during the *SEM2012 shared task. Please refer to the .pdf file for ulterior information.
The python code in ```/reader``` used to read in the data is part of the code made available by Packard et al. (2014) (["Simple Negation Scope Resolution through Deep Parsing: A Semantic Solution to a Semantic Problem"](https://aclweb.org/anthology/P/P14/P14-1007.pdf)).

## Dependencies
- Tensorflow (tested on v. 0.7.1)
- scikit-learn (tested on v. 0.17.1) - for score report purposes only, feel free to use any other library instead -
- numpy (tested on v. 1.11.0)

## Train
To train the model, first go to the parent directory of the repository and run ```python NegNN/(bilstm_class|bilstm|feed_forward)/train.py```. ```/bilstm_class``` is a more elegant implementation that wraps the BiLSTM code inside a separate class so to avoid any repetition. There seems to be however problems with this implementation when run on MacOsX El capitan 10.11.5; if so, please run the less elegant implementation in the ```/bilstm``` folder
```train.py``` accepts the following flags
```
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("max_sent_length", 100, "Maximum sentence length for padding (default:100)")
tf.flags.DEFINE_integer("num_hidden", 200, "Number of hidden units per layer (default:200)")
tf.flags.DEFINE_integer("num_classes", 2, "Number of y classes (default:2)")
# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate(default: 1e-4)")
tf.flags.DEFINE_boolean("scope_detection", True, "True if the task is scope detection or joined scope/event detection")
tf.flags.DEFINE_boolean("event_detection", False, "True is the task is event detection or joined scope/event detection")
tf.flags.DEFINE_integer("POS_emb",0,"0: no POS embeddings; 1: normal POS; 2: universal POS")
tf.flags.DEFINE_boolean("emb_update",False,"True if input embeddings should be updated (default: False)")
tf.flags.DEFINE_boolean("normalize_emb",False,"True to apply L2 regularization on input embeddings (default: False)")
# Data Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_boolean("pre_training", False, "True to use pretrained embeddings")
tf.flags.DEFINE_string("training_lang",'en', "Language of the tranining data (default: en)")
```

