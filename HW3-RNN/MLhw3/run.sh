# train path setting
TRAIN_PATH='./data/new_train/'
LABEL_PATH='./data/train-toneless.csv'
MODEL_PATH='model.h5'
WER_MODEL_PATH='wer_model.h5'

python train.py \
$TRAIN_PATH \
$LABEL_PATH \
$MODEL_PATH \
$WER_MODEL_PATH
echo "done training"


# test path setting
TEST_PATH='./data/new_test/'
OUTPUT_PATH='./result.csv'
OUTPUT_WER_PATH='./wer_result.csv'

python test.py \
$TEST_PATH \
$OUTPUT_PATH \
$OUTPUT_WER_PATH
echo "done testing"