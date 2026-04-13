import tttrlib
import json
import pandas as pd
import os

# %%

# Load the two PTU files
file1 = r"Example/HSA_20pM_PBS_0p01tween_100ulph_400steps_2sec_27/HSA_20pM_PBS_0p01tween_100ulph_400steps_2sec_27_23.33um_steps.ptu"
file2 = r"Test_Data/2026-04-01_test_data/2026-04-01_Final_Test_Atto488_T2_All_Modes/conc-10pM_laser-485nm_640nm_Mode-T2_ch-2__StepScan_stepnumber-20_time-5s/3/1081.68um_steps_PhotonData_T2.ptu"

# %%
# read metadata:
def read_metadata(file_name, csv_name):
    tttr = tttrlib.TTTR(file_name)

    header_raw = tttr.header.json
    print(header_raw[:500])

    header = json.loads(header_raw)
    print(type(header))

    tags = header["tags"]

    df = pd.DataFrame(tags)[["name", "idx", "type", "value"]]
    df = df.sort_values("name") # alphabetically ordered on name

    # Save CSV
    output_path = os.path.join(os.path.dirname(__file__), csv_name)
    df.to_csv(output_path, index=False)

    print("Saved metadata to:", output_path)

read_metadata(file1, "ptu_metadata_Example_smMDS_step_scan.csv")