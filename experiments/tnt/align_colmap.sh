meta_train_dir=$1
colmap_output_path=$2
echo "[$0 arg] meta_train_dir = ${meta_train_dir}"
echo "[$0 arg] colmap_output_path = ${colmap_output_path}"

scene_ids=("Barn" "Caterpillar" "Church" "Courthouse" "Meetingroom" "Truck")

for scene_id in ${scene_ids[*]}; do
    python scripts_clmap/tnt_align.py --scene_id ${scene_id} \
        --input_meta_path ${meta_train_dir} \
        --colmap_output_path ${colmap_output_path}
done
