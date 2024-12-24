data_dir=$1
echo "[$0 arg] data_dir = ${data_dir}"

scene_ids=("ai_001_001" "ai_001_002" "ai_001_003" "ai_001_004" "ai_001_005" "ai_001_006" "ai_001_007" "ai_001_008")

for scene_id in ${scene_ids[*]}; do
    wget https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/${scene_id}.zip
    unzip ${scene_id}.zip
    rm ${scene_id}.zip
    mkdir -p ${data_dir}
    mv ${scene_id} ${data_dir}
done