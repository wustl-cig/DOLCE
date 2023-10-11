pip install gdown==4.7.1
mkdir downloads

FILE_ID="1L7KSDwnjch0L78NYrGdM1fDSHRuoiGdi"
OUTPUT_FILE_2="downloads/dataset.zip"
gdown --id $FILE_ID -O $OUTPUT_FILE_2

FILE_ID="1BYBZhd4IKR1cGM0qRNN125ey4Vfcd38U"
OUTPUT_FILE_2="downloads/model512_all.pt"
gdown --id $FILE_ID -O $OUTPUT_FILE_2

FILE_ID="1MUY-NtlupJlCrSqOXQfti1aHQLvrr9qN"
OUTPUT_FILE_2="downloads/model512_ckc.pt"
gdown --id $FILE_ID -O $OUTPUT_FILE_2

FILE_ID="1DrcPYSp9RHYBMHKs1zrvYCY3rK36o7Xv"
OUTPUT_FILE_2="downloads/model512_coe.pt"
gdown --id $FILE_ID -O $OUTPUT_FILE_2
