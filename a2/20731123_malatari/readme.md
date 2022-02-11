How to run inference.py script:

pass the following cmd args: <.txt file> <.type of model>
    
e.g: python3 inference.py ./infer_test.txt  mnb_uni_bi

list of valid models to pass in cmd args: mnb_uni, mnb_bi, mnb_uni_bi, mnb_uni_ns, mnb_bi_ns, mnb_uni_bi_ns

| Stopwords removed | text feature        | Accuracy(Test set) |
|---------------|---------------------|--------------------|
| yes           | unigram             | 0.8054875          |
| yes           | bigram              | 0.794375           |
| yes           | unigram+bigram      | 0.8262375          |
| no            | unigram             | 0.807475           |
| no              | bigram              | 0.82545            |
| no              | unigram+bigram      | 0.8313875          |