hypersim_data_dir=$1
hypersim_output_dir=$2
hypersim_load_dir=$3
echo "[$0 arg] hypersim_data_dir = ${hypersim_data_dir}"
echo "[$0 arg] hypersim_output_dir = ${hypersim_output_dir}"
echo "[$0 arg] hypersim_load_dir = ${hypersim_load_dir}"

# detectors=("deeplsd")
detectors=("lsd" "deeplsd")
nvs=(4)

# test_scene="ai_001_001"
test_scene="all"


for detector in ${detectors[*]}; do
    hypersim_tri_output_dir=${hypersim_output_dir}/${detector}/tri
    bash experiments/hypersim/line_triangulation.sh ${hypersim_data_dir} ${hypersim_tri_output_dir} ${detector} ${test_scene} ${hypersim_load_dir}

    hypersim_plp_output_dir=${hypersim_output_dir}/${detector}/plp
    if [ "$hypersim_load_dir" = "" ]; then
        bash experiments/hypersim/plp_association.sh ${hypersim_tri_output_dir} ${hypersim_plp_output_dir} ${detector} ${test_scene} ${hypersim_tri_output_dir} ${hypersim_tri_output_dir}
    else
        bash experiments/hypersim/plp_association.sh ${hypersim_tri_output_dir} ${hypersim_plp_output_dir} ${detector} ${test_scene} ${hypersim_load_dir} ${hypersim_load_dir}
    fi

    for nv in ${nvs[*]}; do
        bash experiments/hypersim/eval.sh ${hypersim_data_dir} ${hypersim_tri_output_dir} ${nv} ${test_scene}
        bash experiments/hypersim/eval.sh ${hypersim_data_dir} ${hypersim_plp_output_dir} ${nv} ${test_scene}
    done
done
