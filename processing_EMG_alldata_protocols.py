import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

#constants
fs = 3500 #sample frequency
th = 2 #trigger threshold
windows_time = 1 #seconds time window, previously the stimulation, to quantify the RMS noise

samples_windows_time = int(windows_time*fs)  #windows size

matplotlib.use('TkAgg')
flag_save_plot = True
flag_show_plot = True #True to see the graphs of each file (slow the process)

# This is the path where you want to search
path = 'D:\\Repository\\RMS-noise-TMS-MEPs\\example_data_protocols\\02'

# this is the extension you want to detect
extension = '.csv'

# initialise data of lists.
data_out = {'Subject': [],
            'Muscle': [],
            'Hemisphere': [],
            'Protocol': [],
            'RMS': []}

#create data_out dir if not exist
out_directory = path+"\\Data_out\\"
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            print(file_name_path)  # This is the full path of the filter file

            data = pd.read_csv(file_name_path)

            # Split filename
            path_out, file = os.path.split(file_name_path)
            filename = os.path.splitext(file)[0]
            print(filename)

            x = data.iloc[:, 0]
            y1_protocol_1 = data.iloc[:, 1]
            y2_protocol_3 = data.iloc[:, 2]
            y3_protocol_4 = data.iloc[:, 3]
            trigger = data.iloc[:, 4]

            #windows creation
            mask = np.diff(1 * (trigger > th) != 0)
            a = x[:-1][mask]
            time_ramp = a[::2]
            time_ramp_list = time_ramp.index.tolist()
            index_windows = time_ramp.index.tolist()[1]
            x_ramp = []
            y1_protocol_1_ramp = []
            y2_protocol_3_ramp = []
            y3_protocol_4_ramp = []

            for i in range(len(time_ramp_list)):
                index = time_ramp.index.tolist()[i]
                #if the first stimulation is faster than the windows_time, gets the first point (0) until the stimulation (index)
                if index - samples_windows_time < 0:
                    back_index = 0
                else:
                    back_index = index - samples_windows_time

                y1_protocol_1_ramp.append(y1_protocol_1.array[back_index:index])
                y2_protocol_3_ramp.append(y2_protocol_3.array[back_index:index])
                y3_protocol_4_ramp.append(y3_protocol_4.array[back_index:index])
                x_ramp.append(x.array[back_index:index])

            # RMS estimation
            rms_y1_protocol_1_ramp = np.sqrt(sum(np.hstack(y1_protocol_1_ramp) * np.hstack(y1_protocol_1_ramp)) / len(np.hstack(y1_protocol_1_ramp)))
            rms_y2_protocol_3_ramp = np.sqrt(sum(np.hstack(y2_protocol_3_ramp) * np.hstack(y2_protocol_3_ramp)) / len(np.hstack(y2_protocol_3_ramp)))
            rms_y3_protocol_4_ramp = np.sqrt(sum(np.hstack(y3_protocol_4_ramp) * np.hstack(y3_protocol_4_ramp)) / len(np.hstack(y3_protocol_4_ramp)))
            rms_y1_protocol_1 = np.sqrt(sum(y1_protocol_1 * y1_protocol_1) / len(y1_protocol_1))
            rms_y2_protocol_3 = np.sqrt(sum(y2_protocol_3 * y2_protocol_3) / len(y2_protocol_3))
            rms_y3_protocol_4 = np.sqrt(sum(y3_protocol_4 * y3_protocol_4) / len(y3_protocol_4))

            subject, muscle, hemisphere = filename.split('_')
            data_out['Subject'] += [subject, subject, subject]
            data_out['Muscle'] += [muscle, muscle, muscle]
            data_out['Hemisphere'] += [hemisphere, hemisphere, hemisphere]
            data_out['Protocol'] += ['Protocol_1', 'Protocol_3', 'Protocol_4']
            data_out['RMS'] += [round(rms_y1_protocol_1_ramp, 4), round(rms_y2_protocol_3_ramp, 4), round(rms_y3_protocol_4_ramp, 4)]

            if flag_show_plot or flag_save_plot:
                fig, axs = plt.subplots(3)
                fig.suptitle(filename)

                markersize = 5
                transparency = 0.15
                axs[0].plot(x, y1_protocol_1, label='Protocol_1', color='orange', alpha=transparency)
                axs[0].plot(time_ramp, trigger[:-1][mask][::2], 'go', markersize=markersize)
                axs[1].plot(x, y2_protocol_3, label='Protocol_3', color='orange', alpha=transparency)
                axs[1].plot(time_ramp, trigger[:-1][mask][::2], 'go', markersize=markersize)
                axs[2].plot(x, y3_protocol_4, label='Protocol_4', color='orange', alpha=transparency)
                axs[2].plot(time_ramp, trigger[:-1][mask][::2], 'go', markersize=markersize)

                for i in range(len(time_ramp_list)):
                    axs[0].plot(np.hstack(x_ramp[i]), np.hstack(y1_protocol_1_ramp[i]), color='orange', alpha=0.5)
                    axs[1].plot(np.hstack(x_ramp[i]), np.hstack(y2_protocol_3_ramp[i]), color='orange', alpha=0.5)
                    axs[2].plot(np.hstack(x_ramp[i]), np.hstack(y3_protocol_4_ramp[i]), color='orange', alpha=0.5)

                axs[0].axhline(y=rms_y1_protocol_1_ramp, color='orange', linestyle='-', label=f"rms_y1_protocol_1_ramp = {round(rms_y1_protocol_1,2)}")
                axs[1].axhline(y=rms_y2_protocol_3_ramp, color='orange', linestyle='-', label=f"rms_y2_protocol_3_ramp = {round(rms_y2_protocol_3,2)}")
                axs[2].axhline(y=rms_y3_protocol_4_ramp, color='orange', linestyle='-', label=f"rms_y3_protocol_4_ramp = {round(rms_y3_protocol_4,2)}")

                # axs[0].axhline(y=rms_y1_FCR, color='bisque', linestyle='-', label=f"rms_y2_raw = {round(rms_y1_FCR,2)}")
                # axs[1].axhline(y=rms_y2_ECR, color='bisque', linestyle='-', label=f"rms_y2_raw = {round(rms_y2_ECR,2)}")
                # axs[2].axhline(y=rms_y3_FPB, color='bisque', linestyle='-', label=f"rms_y2_raw = {round(rms_y3_FPB,2)}")

                axs[0].legend(loc='upper right')
                axs[1].legend(loc='upper right')
                axs[2].legend(loc='upper right')

                axs[0].set_ylabel("Voltage [μV]")
                axs[1].set_ylabel("Voltage [μV]")
                axs[2].set_ylabel("Voltage [μV]")
                axs[2].set_xlabel("Time [s]")

                fig.set_size_inches(16, 9)

                if flag_save_plot:
                    plt.savefig(path+"\\Data_out\\"+filename+".png", dpi=300)
                if flag_show_plot:
                    plt.show()

                plt.close(fig)


# Create DataFrame to export
print("saving...")
print(data_out)
df = pd.DataFrame(data_out)

out_path = out_directory+"RMS_alldata"+".xlsx"
df.to_excel(out_path, index=False)
print("saved")
