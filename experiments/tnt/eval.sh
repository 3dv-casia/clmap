meta_train_dir=$1
all_input_dir=$2
nv=$3
test_scene=$4
echo "[$0 arg] meta_train_dir = ${meta_train_dir}"
echo "[$0 arg] all_input_dir = ${all_input_dir}"
echo "[$0 arg] nv = ${nv}"
echo "[$0 arg] test_scene = ${test_scene}"

config_file="cfgs_clmap/eval/tnt.yaml"
default_config_file="cfgs_clmap/eval/default.yaml"
log_path="${all_input_dir}/eval_log_nv_${nv}.txt"

if [ "${test_scene}" = "all" ]; then
    # format: [['GT point cloud (.ply)', 'transform txt', 'line map path'], ...]
    input_path_list="[
        ['${meta_train_dir}/Barn/Barn.ply', '', '${all_input_dir}/Barn/finaltracks'],
        ['${meta_train_dir}/Caterpillar/Caterpillar.ply', '', '${all_input_dir}/Caterpillar/finaltracks'],
        ['${meta_train_dir}/Church/Church.ply', '', '${all_input_dir}/Church/finaltracks'],
        ['${meta_train_dir}/Courthouse/Courthouse.ply', '', '${all_input_dir}/Courthouse/finaltracks'],
        ['${meta_train_dir}/Meetingroom/Meetingroom.ply', '', '${all_input_dir}/Meetingroom/finaltracks'],
        ['${meta_train_dir}/Truck/Truck.ply', '', '${all_input_dir}/Truck/finaltracks']]"
else
    # evaluate one 3D line map
    input_path_list="[
        ['${meta_train_dir}/${test_scene}/${test_scene}.ply', '', '${all_input_dir}/${test_scene}/finaltracks']]"
fi

python scripts_clmap/eval_multiple_tnt.py --config_file ${config_file} \
    --default_config_file ${default_config_file} \
    --input_path_list "${input_path_list}" \
    --use_ranges \
    -nv ${nv} | tee ${log_path}
