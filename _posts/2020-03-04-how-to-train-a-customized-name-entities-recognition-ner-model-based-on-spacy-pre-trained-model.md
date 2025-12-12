---
layout: post
title: How to train a customized Name Entities Recognition (NER) model based on spaCy
  pre-trained model
description: 
image: 
category: [Learning Journey]
tags: ner spacy nlp
date: 2020-03-04 00:00 +0000
---
## How to train a customized Name Entities Recognition (NER) model based on spaCy pre-trained model

There are a bunch of online resources to teach you how to train your own NER model by spaCy, so I will attach some links for reference and skip this part.


[spaCy: Training the named entity recognizer](https://spacy.io/usage/training#ner)
[Convert SpaCy training data to Dataturks NER JSON output](https://medium.com/@dataturks/convert-spacy-training-data-to-dataturks-ner-json-output-949259865c47)
[Custom Named Entity Recognition Using spaCy](https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718)



Back to our main point of this article, and let's check the following code. I'm trying to train a customized NER model and retain the other pipeline in spaCy, i.e. `tagger` and `parser`. You can also add your own pipeline into the model as well.

Write a function as below, and load the model if you've pointed it out. I used default pre-trained model `en_core_web_sm` here.
```python
def train_spacy(self, training_data, testing_data, dropout, display_freq=1, 
                output_dir=None, new_model_name="en_model"):
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    # create blank Language class
    
    if self.model:
        nlp = spacy.load(self.model)
    else:
        nlp = spacy.blank('en')
```

The following is just like the tutorial from spaCy official website, create the pipeline or grab it, then add labels from the training data you've annotated already.
```python
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in training_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
```

And now, let's training. The initial loss as `100000`, and I set up the early stop to avoid overfitting.
```python
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        print(">>>>>>>>>>  Training the model  <<<<<<<<<<\n")

        losses_best = 100000
        early_stop = 0
```

Use stochastic gradient descent as optimizer with mini-batch configuration, get more insights [here](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/).
```python
        for itn in range(self.n_iter):
            print(f"Starting iteration {itn + 1}")
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                text, annotations = zip(*batch)
                nlp.update(
                    text,  # batch of texts
                    annotations,  # batch of annotations
                    drop=dropout,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)

            if itn % display_freq == 0:
                print(f"Iteration {itn + 1} Loss: {losses}")

            if losses["ner"] < losses_best:
                early_stop = 0
                losses_best = int(losses["ner"])
            else:
                early_stop += 1

            print(f"Training will stop if the value reached {self.not_improve}, "
                  f"and it's {early_stop} now.\n")

            if early_stop >= self.not_improve:
                break

        print(">>>>>>>>>>  Finished training  <<<<<<<<<<")
```

After training, dump the model as a binary data to your on-premises disk. Put the last 10 lines inside the `with nlp.disable_pipes(*other_pipes):` will save the only `ner` pipeline. Otherwise, the entire model will be saved, that's exactly what I need.
```python
        if output_dir:
            path = output_dir + f"en_model_ner_{round(losses_best, 2)}"
        else:
            path = os.getcwd() + f"/lib/inactive_model/en_model_ner_{round(losses_best, 2)}"
            os.mkdir(path)
        if testing_data:
            self.validate_spacy(model=nlp, data=testing_data)

    with nlp.use_params(optimizer.averages):
        nlp.meta["name"] = new_model_name
        bytes_data = nlp.to_bytes()
        lang = nlp.meta["lang"]
        pipeline = nlp.meta["pipeline"]

    model_data = dict(bytes_data=bytes_data, lang=lang, pipeline=pipeline)

    with open(path + '/model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
```

You can load the model by the following function to load your model by your specified path on your disk.
```python
def load_model(model_path):
    with open(model_path + '/model.pkl', 'rb') as f:
        model = pickle.load(f)
    nlp = spacy.blank(model['lang'])
    for pipe_name in model['pipeline']:
        pipe = nlp.create_pipe(pipe_name)
        nlp.add_pipe(pipe)
    nlp.from_bytes(model['bytes_data'])
    return nlp
```
