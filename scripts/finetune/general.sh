
containerID=$1
task=$2
mode=$3
agent=$4
prompt=$5
bot=$6
type=$7
exp_name=$8
end_batch=$9
sample_time=${10}
tags=${11}
init_step=${12}
save_path=${13}
model=${14}
extra_label=${15}
save_interval=${16}

echo "task name = ${2},"
echo "mode= ${3},"
echo "agent= ${4}"
echo "prompt = ${5},"
echo "bot = ${6},"
echo "type = ${7},"
echo "exp_name = ${8},"
echo "end_batch = ${9},"
echo "sample_time = ${10},"
echo "wandb tag = ${11},"
echo "init step = ${12},"
echo "save path = ${13},"
echo "model = ${14},"
echo "extra_label = ${extra_label},"
echo "save_interval = ${16}"


CUDA_VISIBLE_DEVICES=0 python main.py \
                        --task $task \
                        --mode $mode \
                        --agent $agent \
                        --prompt $prompt \
                        --bot $bot \
                        --type $type \
                        --exp_name $exp_name \
                        --end_batch $end_batch \
                        --sample_time $sample_time \
                        --tags $tags \
                        --init_step $init_step \
                        --save_path $save_path \
                        --model $model \
                        --extra_label $extra_label \
                        --save_interval $save_interval