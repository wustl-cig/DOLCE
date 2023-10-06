#!/bin/bash
source ~/.bashrc
conda activate dolce
cd [directory/dolce]

rm -r guided_diffusion.egg-info
pip install -e .

pwd
ls -all
python -c "import sys; print(':'.join(x for x in sys.path if x))"

nvidia-smi

############################################################
# securityCT (COE) #
############################################################
INDEX=187 # Dataset
NUMANGLES=90 # {60, 90, 120}
Test_FOLDER=securityCT_S_$INDEX
############################################################.
echo "Testing on Folder: securityCT_S_$INDEX Using $NUMANGLES Angles"

SUBJETR_ID="--sub_id S_$INDEX"
ANGLES="--num_angs $NUMANGLES"
DATA_DIR="--data_dir ./dataset/dataset_coe"
PARAM_DIR="--param_fn ./config/param_parallel512_la$NUMANGLES.cfg"
############################################################
MODEL_FLAGS="--attention_resolutions 32,16,8 --image_size 512 --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --dropout 0.0"
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
CONDTIONS="--weighted_condition False --use_condtion rls --deterministic False"
MODEL_PATH="--model_path ./model_zoo/model512_all.pt"
SAMPLE_FLAGS="--batch_size 1 --num_samples 30 --sampler ddim --timestep_respacing ddim40 --prox_solver cgrad"
###########################################################
#For example, apply 3 gpus in parallel.
export CUDA_VISIBLE_DEVICES=1,2,3
mpiexec -n 3 python limited_ct_sample.py $CONDTIONS $PARAM_DIR \
                                         $DATA_DIR $MODEL_FLAGS \
                                         $DIFFUSION_FLAGS $SAMPLE_FLAGS \
                                         $MODEL_PATH $RS_MODEL_PATH \
                                         $SUBJETR_ID $ANGLES