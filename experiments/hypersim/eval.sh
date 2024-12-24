data_dir=$1
all_input_dir=$2
nv=$3
test_scene=$4
echo "[$0 arg] data_dir = ${data_dir}"
echo "[$0 arg] all_input_dir = ${all_input_dir}"
echo "[$0 arg] nv = ${nv}"
echo "[$0 arg] test_scene = ${test_scene}"

config_file="cfgs_clmap/eval/hypersim.yaml"
default_config_file="cfgs_clmap/eval/default.yaml"
log_path="${all_input_dir}/eval_log_nv_${nv}.txt"

if [ "${test_scene}" = "all" ]; then
    # format: [['scene id', 'line map path'], ...]
    input_path_list="[
        ['ai_001_001', '${all_input_dir}/ai_001_001/finaltracks'],
        ['ai_001_002', '${all_input_dir}/ai_001_002/finaltracks'],
        ['ai_001_003', '${all_input_dir}/ai_001_003/finaltracks'],
        ['ai_001_004', '${all_input_dir}/ai_001_004/finaltracks'],
        ['ai_001_005', '${all_input_dir}/ai_001_005/finaltracks'],
        ['ai_001_006', '${all_input_dir}/ai_001_006/finaltracks'],
        ['ai_001_007', '${all_input_dir}/ai_001_007/finaltracks'],
        ['ai_001_008', '${all_input_dir}/ai_001_008/finaltracks']]"
else
    # evaluate one 3D line map
    input_path_list="[
            ['${test_scene}', '${all_input_dir}/${test_scene}/finaltracks']]"
fi

python scripts_clmap/eval_multiple_hypersim.py --data_dir ${data_dir} \
    --config_file ${config_file} \
    --default_config_file ${default_config_file} \
    --input_path_list "${input_path_list}" \
    -nv ${nv} | tee ${log_path}
