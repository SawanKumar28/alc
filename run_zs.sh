gpuid=$1
datainp=$2
splitinp=$3
extraargs=$4

split=${splitinp}  #dev
echo "Evaluating on "${splitinp}
modelnames="gpt2-xl"
datanames=${datainp}
for modelname in ${modelnames}
do
    for dataname in ${datanames}
    do
        echo $dataname
        CUDA_VISIBLE_DEVICES=${gpuid} python3 run_model.py --num_processes 1 --data_name ${dataname}  --model ${modelname}  --data_split ${split} --compute_ece  ${extraargs} --cache_dir /scratche/home/sawan/cache
    done
done
