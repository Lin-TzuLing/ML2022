# data preprocess
DICT_PATH='./voc_dict/'
CH_PATH='./data/train-ZH.csv'
TL_PATH='./data/train-TL.csv'
TEST_PATH='./data/test-ZH-nospace.csv'
RESULT_PATH='./data/train.pkl.gz'
TEST_RESULT_PATH='./data/test.pkl.gz'

python preprocess.py \
$DICT_PATH \
$CH_PATH \
$TL_PATH \
$TEST_PATH \
$RESULT_PATH \
$TEST_RESULT_PATH
echo "Done data preprocessing"


# train/test path setting
TRAIN_PATH=$RESULT_PATH
TEST_PATH=$TEST_RESULT_PATH
MODEL_DIR='./model/'
OUTPUT_PATH='./result.csv'

python main.py \
$DICT_PATH \
$TRAIN_PATH \
$TEST_PATH \
$MODEL_DIR \
$OUTPUT_PATH
echo "Done training and testing"