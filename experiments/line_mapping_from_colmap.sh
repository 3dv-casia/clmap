colmap_path=$1
model_path=$2
image_path=$3
output_dir=$4
echo "[$0 arg] colmap_path = ${colmap_path}"
echo "[$0 arg] model_path = ${model_path}"
echo "[$0 arg] image_path = ${image_path}"
echo "[$0 arg] output_dir = ${output_dir}"

# directory tree:
# --${colmap_path}
#     --${model_path}
#         cameras.txt/bin
#         images.txt/bin
#         points3D.txt/bin
#     --${image_path}

tri_config_file="cfgs_clmap/triangulation/default.yaml"
tri_default_config_file="cfgs_clmap/triangulation/default.yaml"
plp_config_file="cfgs_clmap/plp_association/default.yaml"
plp_default_config_file="cfgs_clmap/triangulation/default.yaml"

tri_output_dir="${output_dir}/tri"
plp_output_dir="${output_dir}/plp"
log_path="${output_dir}/log.txt"
mkdir -p "${output_dir}"

python runners_clmap/line_mapping_from_colmap.py --tri_config_file ${tri_config_file} \
    --tri_default_config_file ${tri_default_config_file} \
    --plp_config_file ${plp_config_file} \
    --plp_default_config_file ${plp_default_config_file} \
    --colmap_path ${colmap_path} \
    --model_path ${model_path} \
    --image_path ${image_path} \
    --use_optim True \
    --visualize False \
    --tri_output_dir ${tri_output_dir} \
    --plp_output_dir ${plp_output_dir} | tee ${log_path}
