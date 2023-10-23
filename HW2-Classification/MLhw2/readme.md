# **ML hw2 - Classification**
# **how to run codes**
> ## **Dependency**
>> run `pip install -r requriements.txt` 
> ## **Overall Structure**
> ### **Codes** 
>* `augment_dataset.py`
>>> code for data preprocess, including saving data into h5 files, data augmentation, train_valid split.
>* `train.py`
>>> code for training classification model, generate the best model `model.h5`. 
>* `test.py`
>>> code for testing, generate the final classification result `result.csv` for submission. Also plot filters of first CNN layer `filter_pic/...` and confusion matrix `confusion_matrix.png` of validation with this file.
>* **`run.sh`**
>>> directly run this shell script to complete data preprocessing, training and testing. And further get the final result.
> ### **Files** 
>* `result.csv`
>>> result for submission.
>* `model.h5`
>>> best model saved in `train.py` with lowest validation loss.
>* `confusion_matrix.png`
>>> confusion matrix of validation result.
>* `filter_pic/...`
>>> filters of first CNN layer in `model.h5`.

> ## **Path Setting in each .py file**
>* **`augment_dataset.py`**
>>> `train_data_path`：
>>>> path to the folder where train data in.
>>>> 
>>> `test_data_path`：
>>>> path to the folder where test data in.
>>>> 
>>> `output_path`：
>>>> path to the folder where you want to save processed dataset in. (create this folder first)
>* **`train.py`**
>>> `model_output_path`：
>>>> path to where you want to save best model at.
>* **`test.py`**
>>> `model_path`：
>>>> path to where best model saved at. (must be the same path which is set as `model_output_path` in the `train.py`)
>>>> 
>>> `result_path`：
>>>> path to where you want to save prediction result at.
>>>> 
>>> `filter_path`：
>>>> path to the folder where you want to save CNN filter pictures in. (create this folder first)
>>>> 
>>> `matrix_path`：
>>>> path to where you want to save confusion matrix at.
>>>>  
# **報告**
>> 寫於`report.pdf`