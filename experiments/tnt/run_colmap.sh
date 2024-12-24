all_image_path=$1
all_output_path=$2
echo "[$0 arg] all_image_path = ${all_image_path}"
echo "[$0 arg] all_output_path = ${all_output_path}"

scene_ids=("Barn" "Caterpillar" "Church" "Courthouse" "Meetingroom" "Truck")

for scene_id in ${scene_ids[*]}; do
    image_path=${all_image_path}/${scene_id}
    output_path=${all_output_path}/${scene_id}
    mkdir -p ${output_path}
    log_path=${output_path}/log.txt
    python scripts_clmap/tnt_colmap_runner.py --image_path ${image_path} --output_path ${output_path} | tee ${log_path}
done
