# path setting
TRAIN_INPUT="./data/train"
TEST_INPUT="./data/test"
TRAIN_OUTPUT="./data/new_train"
TEST_OUTPUT="./data/new_test"
# create folder
mkdir -p $TRAIN_OUTPUT
mkdir -p $TEST_OUTPUT

# loop for audio transform
for file in $TRAIN_INPUT/*.wav; do
    sox $file -r 16000 -e signed-integer -b 16  "$TRAIN_OUTPUT/${file##*/}"
done
echo 'done train audio transform'
for file in $TEST_INPUT/*.wav; do
    sox $file -r 16000 -e signed-integer -b 16  "$TEST_OUTPUT/${file##*/}"
done
echo 'done test audio transform'