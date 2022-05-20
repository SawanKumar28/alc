gpuid=$1
datainp=$2
splitinp=$3
fslk=$4
extraargs=$5

split=${splitinp}  #dev

if [ "$split" = "dev" ]; then
    fssource="train"
else
    fssource="dev"
fi

echo "Evaluating on "${splitinp}
modelnames="gpt2-xl"
datanames=${datainp}
for modelname in ${modelnames}
do
    for dataname in ${datanames}
    do
        echo $dataname
        CUDA_VISIBLE_DEVICES=${gpuid} python3 run_model.py --num_processes 1 --data_name ${dataname}  --model ${modelname}  --data_split ${split} --compute_ece  ${extraargs} --cache_dir /scratche/home/sawan/cache --fsl_n_samples ${fslk} --fsl_sampling_split ${fssource}
    done
done
