# path parameters
DATA_PATH="../../data"
CORR_FIG_PATH="../../corr_fig"
PROCESSED_DATA_PATH="../../processed_data"

# run.sh part (DO NOT modify this part)
echo "start data processing"
python data_preprocess.py \
$DATA_PATH \
$CORR_FIG_PATH \
$PROCESSED_DATA_PATH
echo "done data preprocessing"