all_input_dir=$1
all_output_dir=$2
detector=$3
test_scene=$4
all_colmap_folder=$5
all_load_dir=$6
echo "[$0 arg] all_input_dir = ${all_input_dir}"
echo "[$0 arg] all_output_dir = ${all_output_dir}"
echo "[$0 arg] detector = ${detector}"
echo "[$0 arg] test_scene = ${test_scene}"
echo "[$0 arg] all_colmap_folder = ${all_colmap_folder}"
echo "[$0 arg] all_load_dir = ${all_load_dir}"

# Note:
#  - If all_load_dir is not empty, we use the precomputed data (i.e. results of VPs and SfM points)

if [ "$test_scene" = "all" ]; then
    scene_ids=("ai_001_001" "ai_001_002" "ai_001_003" "ai_001_004" "ai_001_005" "ai_001_006" "ai_001_007" "ai_001_008")
else
    scene_ids=($test_scene)
fi

config_file="cfgs_clmap/plp_association/default.yaml"
default_config_file="cfgs_clmap/triangulation/default.yaml"

for scene_id in ${scene_ids[*]}; do
    input_folder="${all_input_dir}/${scene_id}/finaltracks"
    output_dir="${all_output_dir}/${scene_id}"
    mkdir -p ${output_dir}
    log_path="${output_dir}/log.txt"

    if [ "$all_load_dir" = "" ]; then
        # do not use precomputed data

        # re-triangulate point tracks using COLMAP and re-detect VPs
        python runners_clmap/plp_association.py --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --input_folder ${input_folder} \
            --visualize False \
            --line2d.detector.method ${detector} \
            --output_dir ${output_dir} | tee ${log_path}

    else
        # use precomputed data
        load_dir=${all_load_dir}/${scene_id}

        # use the known point tracks from COLMAP
        colmap_model_path="${all_colmap_folder}/${scene_id}/colmap_outputs/sparse"

        python runners_clmap/plp_association.py --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --input_folder ${input_folder} \
            --colmap_model_path ${colmap_model_path} \
            --visualize False \
            --load_dir ${load_dir} \
            --load_vpdet True \
            --line2d.detector.method ${detector} \
            --output_dir ${output_dir} | tee ${log_path}
    fi
done
