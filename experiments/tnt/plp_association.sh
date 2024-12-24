all_input_dir=$1
all_colmap_folder=$2
all_output_dir=$3
detector=$4
test_scene=$5
use_known_retriangulation_colmap_model=$6
all_load_dir=$7
echo "[$0 arg] all_input_dir = ${all_input_dir}"
echo "[$0 arg] all_colmap_folder = ${all_colmap_folder}"
echo "[$0 arg] all_output_dir = ${all_output_dir}"
echo "[$0 arg] detector = ${detector}"
echo "[$0 arg] test_scene = ${test_scene}"
echo "[$0 arg] use_known_retriangulation_colmap_model = ${use_known_retriangulation_colmap_model}" # True or False
echo "[$0 arg] all_load_dir = ${all_load_dir}"

# Note:
#  - If all_load_dir is not empty, we use the precomputed data (i.e. results of VPs)

if [ "$test_scene" = "all" ]; then
    scene_ids=("Barn" "Caterpillar" "Church" "Courthouse" "Meetingroom" "Truck")
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

    # use the known point tracks from COLMAP
    if [ "$use_known_retriangulation_colmap_model" = "True" ]; then
        # note: cfg["triangulation"]["pointsfm_config"]["retriangulation"] must be True
        colmap_model_path="${all_input_dir}/${scene_id}/colmap_retriangulation_outputs/sparse"
    elif [ "$use_known_retriangulation_colmap_model" = "False" ]; then
        colmap_model_path="${all_colmap_folder}/${scene_id}/dense/aligned"
    else
        echo "Error: the parameter use_known_retriangulation_colmap_model must be True or False."
        exit 1
    fi

    if [ "$all_load_dir" = "" ]; then
        # do not use precomputed data

        python runners_clmap/plp_association.py --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --input_folder ${input_folder} \
            --colmap_model_path ${colmap_model_path} \
            --visualize False \
            --output_dir ${output_dir} | tee ${log_path}

    else
        # use precomputed data
        load_dir=${all_load_dir}/${scene_id}

        python runners_clmap/plp_association.py --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --input_folder ${input_folder} \
            --colmap_model_path ${colmap_model_path} \
            --visualize False \
            --line2d.detector.method ${detector} \
            --load_dir ${load_dir} \
            --load_vpdet True \
            --output_dir ${output_dir} | tee ${log_path}
    fi
done
