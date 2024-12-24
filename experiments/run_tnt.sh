tnt_output_dir=$1
tnt_meta_train_dir=$2
tnt_colmap_dir=$3
use_known_retriangulation_colmap_model=$4
tnt_load_dir=$5
echo "[$0 arg] tnt_output_dir = ${tnt_output_dir}"
echo "[$0 arg] tnt_meta_train_dir = ${tnt_meta_train_dir}"
echo "[$0 arg] tnt_colmap_dir = ${tnt_colmap_dir}"
echo "[$0 arg] use_known_retriangulation_colmap_model = ${use_known_retriangulation_colmap_model}" # True or False
echo "[$0 arg] tnt_load_dir = ${tnt_load_dir}"

# detectors=("deeplsd")
detectors=("lsd" "deeplsd")
nvs=(4)

# test_scene="Barn"
test_scene="all"

for detector in ${detectors[*]}; do
    tnt_tri_output_dir=${tnt_output_dir}/${detector}/tri
    bash experiments/tnt/line_triangulation.sh ${tnt_colmap_dir} ${tnt_tri_output_dir} ${detector} ${test_scene} ${tnt_load_dir}

    tnt_plp_output_dir=${tnt_output_dir}/${detector}/plp
    if [ "$tnt_load_dir" = "" ]; then
        bash experiments/tnt/plp_association.sh ${tnt_tri_output_dir} ${tnt_colmap_dir} ${tnt_plp_output_dir} ${detector} ${test_scene} ${use_known_retriangulation_colmap_model} ${tnt_tri_output_dir}
    else
        bash experiments/tnt/plp_association.sh ${tnt_tri_output_dir} ${tnt_colmap_dir} ${tnt_plp_output_dir} ${detector} ${test_scene} ${use_known_retriangulation_colmap_model} ${tnt_load_dir}
    fi

    for nv in ${nvs[*]}; do
        bash experiments/tnt/eval.sh ${tnt_meta_train_dir} ${tnt_tri_output_dir} ${nv} ${test_scene}
        bash experiments/tnt/eval.sh ${tnt_meta_train_dir} ${tnt_plp_output_dir} ${nv} ${test_scene}
    done
done
