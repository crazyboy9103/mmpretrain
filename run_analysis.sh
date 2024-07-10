data=$1

if [ -z "$data" ]; then
    echo "need data name"
    exit 1
fi

shape=$2
if [ -z "$shape" ]; then
    shape=224
fi

warmup=$3
if [ -z "$warmup" ]; then
    warmup=10
fi

iter=$4
if [ -z "$iter" ]; then
    iter=100
fi

models=("swin-tiny" "convnext-tiny" "deit-small" "efficientnet-b4" "efficientnetv2-b3" "mobilenet-v3-large" "regnetx-4.0gf" "resnet50")

for model in "${models[@]}"; do
    python tools/analysis_tools/get_flops.py ./work_dirs/"$data"/"$model"/"$model"_b8_"$data".py --shape $shape --warmup $warmup --iter $iter 2>&1 | tee ./work_dirs/"$data"/"$model"_perf.log
done