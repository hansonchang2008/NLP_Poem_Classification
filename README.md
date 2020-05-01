# Poem Classification study NLP course University of Ottawa

In this project we implemeted and compared the accuracy of 8 machine learning model on mentiod NLP task.

### Report 
<a href="https://www.overleaf.com/read/bcqcthkxtycb"> report file </a>

### Data
Please find the data in ```data/``` directory.
The data has been crawed through poem website and it has 3 main categories.

### Results and Conclution

In this study we used NLP as a tool to classify poems also used statistical measurement to find out which Machine learning algorithm is more robust and it can be generalized better. As you can see in the report the Multy layer perceptron (MLP) with 82% accuracy had the best performace however there is no significant gap betwen performace of MLP and next best model (random forest) with 81% accuracy .

In this task using current pre-trainder word embedding such as Glove, Fasttext, etc, were no beneficial and we think it's because  of two following reasons.
1. First, since the meaning of some words chang gradually through the time
2. Second, regular writings are very different compare to poems in many terms such as order of words in a sentence.

