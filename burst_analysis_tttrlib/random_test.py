import time
import tttrlib

ptufilename = "Test_Data/2026-04-01_test_data/2026-04-01_Final_Test_Atto488_T2_All_Modes/conc-10pM_laser-485nm_640nm_Mode-T2_ch-2__StepScan_stepnumber-20_time-5s/3/216.34um_steps_PhotonData_T2.ptu"
start = time.time()
tttr = tttrlib.TTTR(ptufilename)
print("Load time:", time.time() - start)