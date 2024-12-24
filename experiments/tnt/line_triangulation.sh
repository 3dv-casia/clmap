all_colmap_dir=$1
all_output_dir=$2
detector=$3
test_scene=$4
all_load_dir=$5
echo "[$0 arg] all_colmap_dir = ${all_colmap_dir}"
echo "[$0 arg] all_output_dir = ${all_output_dir}"
echo "[$0 arg] detector = ${detector}"
echo "[$0 arg] test_scene = ${test_scene}"
echo "[$0 arg] all_load_dir = ${all_load_dir}"

# Note:
#  - If all_load_dir is not empty, we use the precomputed data (i.e. results of line detection & matching and VPs)

if [ "$test_scene" = "all" ]; then
    scene_ids=("Barn" "Caterpillar" "Church" "Courthouse" "Meetingroom" "Truck")
else
    scene_ids=($test_scene)
fi

config_file="cfgs_clmap/triangulation/tnt.yaml"
default_config_file="cfgs_clmap/triangulation/default.yaml"

for scene_id in ${scene_ids[*]}; do
    colmap_path=${all_colmap_dir}/${scene_id}/dense
    model_path=aligned
    image_path=images
    output_dir=${all_output_dir}/${scene_id}
    mkdir -p ${output_dir}
    log_path=${output_dir}/log.txt

    if [ "$all_load_dir" = "" ]; then
        # do not use precomputed data

        python runners_clmap/colmap_triangulation.py --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --colmap_path ${colmap_path} \
            --model_path ${model_path} \
            --image_path ${image_path} \
            --visualize False \
            --line2d.detector.method ${detector} \
            --line2d.visualize True \
            --line2d.save_l3dpp True \
            --line2d.save_l3dpp_matches True \
            --output_dir ${output_dir} | tee ${log_path}

    else
        # use precomputed data
        load_dir=${all_load_dir}/${scene_id}

        # note: `--load_meta` is useless in colmap_triangulation.py.

        python runners_clmap/colmap_triangulation.py --config_file ${config_file} \
            --default_config_file ${default_config_file} \
            --colmap_path ${colmap_path} \
            --model_path ${model_path} \
            --image_path ${image_path} \
            --visualize False \
            --load_dir ${load_dir} \
            --load_meta True \
            --load_det True \
            --load_vpdet True \
            --load_match True \
            --line2d.detector.method ${detector} \
            --output_dir ${output_dir} | tee ${log_path}
    fi
done
