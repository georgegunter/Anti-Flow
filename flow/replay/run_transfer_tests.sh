# Example usage: 
# run_rl_transfer_tests "/path/to/checkpoint" "200" "my_test"
# run_controller_transfer_tests "follower_stopper" "fs_0" "--num_cpus 2"

######## Transfer testing script. ########
TEST_PARAMS="--no_warnings --no_warmup"
##########################################
######### RL Related Commands ############
run_rl() {
    python transfer_tests.py -r $1 -c $2 --no_warnings --gen_emission --exp_title $3 $4
}

run_rl_outflow_sweep() {
    python transfer_tests.py -r $1 -c $2 --no_warnings --gen_emission --outflow_sweep --exp_title $3 $4 
}

run_rl_idm_sweep() {
    python transfer_tests.py -r $1 -c $2 --no_warnings --gen_emission --default_controller idm --idm_sweep --exp_title $3 $4
}

run_rl_inflow_sweep() {
    python transfer_tests.py -r $1 -c $2 --no_warnings --gen_emission --inflow_sweep --exp_title $3 $4
}

run_rl_lanefreq_sweep() {
    python transfer_tests.py -r $1 -c $2 --no_warnings --gen_emission --default_controller idm --lane_freq_sweep --exp_title $3 $4
}

run_rl_transfer_tests() {
    echo "\n" $1 $2 $3 $4"\n"
    run_rl $1 $2 $3 "$4"
    run_rl_outflow_sweep $1 $2 $3 "$4"
    run_rl_idm_sweep $1 $2 $3 "$4"
    run_rl_inflow_sweep $1 $2 $3 "$4"
    run_rl_lanefreq_sweep $1 $2 $3 "$4"
}
##########################################

########## Regular controller ############
run_controller() {
     python transfer_tests.py --controller $1 --no_warnings --gen_emission --exp_title $2 $3
}

run_controller_outflow_sweep() {
    python transfer_tests.py --controller $1 --no_warnings --gen_emission --outflow_sweep --exp_title $2 $3
}

run_controller_idm_sweep() {
    python transfer_tests.py --controller $1 --no_warnings --gen_emission --default_controller idm --idm_sweep --exp_title $2 $3
}

run_controller_inflow_sweep() {
    python transfer_tests.py --controller $1 --no_warnings --gen_emission --inflow_sweep --exp_title $2 $3
}

run_controller_lanefreq_sweep() {
    python transfer_tests.py --controller $1 --no_warnings --gen_emission --default_controller idm --lane_freq_sweep --exp_title $2 $3
}

run_controller_transfer_tests() {
    # 1: controller, 2: exp title, 3: additional
    echo "\n" $1 $2 $3"\n"
    run_controller $1 $2 "$3"
    run_controller_outflow_sweep $1 $2 "$3"
    run_controller_idm_sweep $1 $2 "$3"
    run_controller_inflow_sweep $1 $2 "$3"
    run_controller_lanefreq_sweep $1 $2 "$3"
 }

##########################################
test_rl_runs() {
    run_rl_transfer_tests $1 $2 $3 "$4 --horizon 10 --no_warmup"
}

test_controller_runs() {
    run_controller_transfer_tests $1 $2 "$3 --horizon 10 --no_warmup"
}

### Write tests below ###
# run_rl_transfer_tests "/path/to/checkpoint" "200" "my_test"
