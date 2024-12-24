data_dir=$1
all_output_dir=$2
detector=$3
test_scene=$4
all_load_dir=$5
echo "[$0 arg] data_dir = ${data_dir}"
echo "[$0 arg] all_output_dir = ${all_output_dir}"
echo "[$0 arg] detector = ${detector}"
echo "[$0 arg] test_scene = ${test_scene}"
echo "[$0 arg] all_load_dir = ${all_load_dir}"

# Note:
#  - If all_load_dir is not empty, we use the precomputed data (i.e. results of line detection & matching, VPs and SfM points)

if [ "$test_scene" = "all" ]; then
    scene_ids=("ai_001_001" "ai_001_002" "ai_001_003" "ai_001_004" "ai_001_005" "ai_001_006" "ai_001_007" "ai_001_008")
else
    scene_ids=($test_scene)
fi

config_file=cfgs_clmap/triangulation/hypersim.yaml
default_config_file=cfgs_clmap/triangulation/default.yaml

for scene_id in ${scene_ids[*]}; do
    output_dir="${all_output_dir}/${scene_id}"
    mkdir -p ${output_dir}
    log_path="${output_dir}/log.txt"

    if [ "$all_load_dir" = "" ]; then
        # do not use precomputed data

        python runners_clmap/hypersim/triangulation.py --data_dir ${data_dir} \
            --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --scene_id ${scene_id} \
            --visualize False \
            --line2d.detector.method ${detector} \
            --line2d.visualize True \
            --line2d.save_l3dpp True \
            --line2d.save_l3dpp_matches True \
            --output_dir ${output_dir} | tee ${log_path}

    else
        # use precomputed data
        load_dir=${all_load_dir}/${scene_id}

        python runners_clmap/hypersim/triangulation.py --data_dir ${data_dir} \
            --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --scene_id ${scene_id} \
            --visualize False \
            --load_dir ${load_dir} \
            --load_meta True \
            --load_det True \
            --load_match True \
            --load_vpdet True \
            --line2d.detector.method ${detector} \
            --output_dir ${output_dir} | tee ${log_path}

    fi
done
