# parameters
PROCESSED_DATA_PATH="./processed_data/"
LOG_NAME="test72"
LOG_PATH="./log/"
RESULT_PATH="./result.csv"


# run.sh part (DO NOT modify this part)
echo "start training and testing"
python main.py \
$PROCESSED_DATA_PATH \
$LOG_NAME \
$LOG_PATH \
$RESULT_PATH
echo "done training and testing, result store at $RESULT_PATH"