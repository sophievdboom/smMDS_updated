from burst_search import recreate_smds_burst_search

ptu_file = r"Test_Data/2026-04-01_test_data/2026-04-01_Final_Test_Atto488_T2_All_Modes/conc-10pM_laser-485nm_640nm_Mode-T2_ch-2__StepScan_stepnumber-20_time-5s/3/216.34um_steps_PhotonData_T2.ptu"

df_photons, df_bursts = recreate_smds_burst_search(
    ptu_file,
    set_lee_filter=4,
    threshold_iT_signal_ms=0.05,
    threshold_iT_noise_ms=0.05,
    min_phs_burst=5,
    min_phs_noise=100,
    verbose=True,
    use_placeholder_filter=True,
)
