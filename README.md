## Package requirements

**Python 3.7**
- numpy>=1.17.4
- pandas>=0.25.3
- scikit-learn==0.21.3
- tensorflow==1.15.0

## Included files

- run_bert_fv.sh: Supplied script for generating feature vectors that has been customized to a specific directory structure and file naming scheme
- prep.sh: Bash script that separates a two-column comma-separated file into two separate files by column and trims any leading or trailing quotation marks, optionally removing the first line
- model.py: Python code that performs the analysis using the *tensorflow* and *sklearn* modules

## Required directory tree

```
.
  |-bert
     |-models
         |-uncased_L-12_H-768_A-12
             |-bert_input_data
                |-prep.sh
                |-lang_id_eval.csv
                |-lang_id_test.csv
                |-lang_id_train.csv
                |-eval_data.txt
                |-eval_label.txt
                |-test_data.txt
                |-test_label.txt
                |-train_data.txt
                |-train_label.txt
             |-bert_output_data
                |-eval.jsonlines
                |-test.jsonlines
                |-train.jsonlines
  |-model.py
  |-run_bert_fv.sh
```

\newpage

## Instructions

1. Create the 'bert_input_data' and 'bert_output_data' directories as shown in the required tree.

2. Move the BERT files, provided data files, and attached files to their assigned places in the tree.

3. Run 'prep.sh' three times (once for train, test, and eval) with the 'output_file_prefix' option set to 'train', 'test', and 'eval' respectively. The output files will be named [dataset]_label.txt and [dataset]_data.txt and saved to the locations shown in the tree.

4. Run 'run_bert_fv.sh' to extract the feature vectors, which will be named [dataset].jsonlines and saved to the locations shown in the tree.

5. Run the Python analysis by calling 'python model.py' from the base of the tree..
