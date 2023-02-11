source ~/.bashrc
pushd /work/twsqfkt532/blackboxbot
conda activate pohan

task=$1
mode=$2
agent=$3
prompt=$4
bot=$5
type=$6
exp_nam=$7
end_batch=$8
sample_time=$9
tags=${10}
init_step=${11}
save_path=${12}
model=${13}
extra_label=${14}
save_interval=${15}
curid=${16:-" "}

if [ -z ${extra_label} ]; 
then
    extra_label="train_word_list.txt"
fi

echo "container ID = ${curid}"
echo "task name = ${1},"
echo "mode= ${2},"
echo "agent= ${3}"
echo "prompt = ${4},"
echo "bot = ${5},"
echo "type = ${6},"
echo "exp_name = ${7},"
echo "end_batch = ${8},"
echo "sample_time = ${9},"
echo "wandb tag = ${10},"
echo "init step = ${11},"
echo "save path = ${12},"
echo "model = ${13},"
echo "extra_label = ${extra_label},"
echo "save_interval = ${15}"

sleep 15
echo "exec script = scripts/${mode}/general.sh"

if [[ ${task:1:1} != '<' && ${type} == "emotion" ]] 
then
    task="<$task>"
fi
# current not support mode = pretrain
bash ./scripts/${mode}/general.sh $curid $task $2 $3 $4 $5 $6 $7 $8 ${9} ${10} ${11} ${12} ${13} ${extra_label} ${15}
popd

sleep 15

if [[ ${curid} != "NULL" ]]
then
source ~/.bashrc; twccrm "$curid"
fi
