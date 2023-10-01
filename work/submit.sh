submit() {
    local amp_e_order=1
    local amp_p_order=$1
    local max_h_order=$2

    echo "amp_e_order=$amp_e_order"
    echo "amp_p_order=$amp_p_order"
    echo "max_h_order=$max_h_order"

    local epccdir="./epcc-e${amp_e_order}-p${amp_p_order}-h${max_h_order}/"

    for ie in $(seq 0 $amp_e_order); do
        for ip in $(seq 0 $amp_p_order); do
            for ih in $(seq 0 $max_h_order); do
                local workdir="${epccdir}/e${ie}-p${ip}-h${ih}/"

                if [ -d "$workdir" ]; then
                    echo "workdir=$workdir exists, will be removed"
                    rm -rf "$workdir"
                fi

                mkdir -p "$workdir" || exit 1

                cd "$workdir" || exit 1

                cp ../../../src/main.py .
                cp ../../../src/run.sh  .

                echo "python main.py --bra_e_order=$ie --bra_p_order=$ip --com_h_order=$ih --amp_p_order=$amp_p_order" >> run.sh

                # Uncomment and modify the following line to submit the job
                sbatch --partition=serial --time=120:00:00 --job-name=$workdir run.sh

                cd - || exit 1
            done
        done
    done
}

submit 2 3
submit 4 3
submit 6 3
submit 8 3
