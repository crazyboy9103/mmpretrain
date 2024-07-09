data=$1

if [ -z "$data" ]; then
    echo "need data name"
    exit 1
fi

folders=("swin_transformer" "convnext" "deit" "efficientnet" "efficientnet_v2" "mobilenet_v3" "regnet" "resnet")
models=("swin-tiny" "convnext-tiny" "deit-small" "efficientnet-b4" "efficientnetv2-b3" "mobilenet-v3-large" "regnetx-4.0gf" "resnet50")

for i in "${!folders[@]}"; do
    model="${models[$i]}"
    folder="${folders[$i]}"
    log_dir="logs/$data/$folder"
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi
    python tools/train.py configs/neurocle/"$folder"/"$model"_b8_"$data".py --work-dir "./work_dirs/$data/$model" 2>&1 | tee "$log_dir/$model.log"
done