

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime



class DasPlotter:
    def __init__(self, save_folder=None, mode='show', orientation='grid'):
        self.save_folder = save_folder if save_folder else os.getcwd()
        self.mode = mode.lower()
        self.orientation = orientation.lower()
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def plot(self, datamap, datasets, common_time=None, title=None):
        num_plots = len([key for key in datamap if key != "time"])
        
        if self.orientation == 'vertical':
            num_rows, num_cols = num_plots, 1
        elif self.orientation == 'grid':
            num_rows, num_cols = self.determine_plot_layout(num_plots)
        else:
            raise ValueError("Invalid orientation. Use 'vertical' or 'grid'.")
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 2.5*num_rows))
        
        if title:
            fig.suptitle(title, fontsize=12)
        
        axs = np.array(axs).flatten()
        
        plot_index = 0
        for key, value in datamap.items():
            if key == "time" or plot_index >= len(axs):
                continue
            
            ax = axs[plot_index]
            for i, data in enumerate(datasets):
                if common_time is not None:
                    time = common_time
                elif "time" in datamap:
                    time = data[:, datamap["time"]]
                else:
                    time = np.arange(len(data))
                
                if isinstance(value, list):
                    for j, idx in enumerate(value):
                        label = f"{key}[{j}]" if len(datasets) == 1 else f"{key}[{j}] {i+1}"
                        ax.plot(time, data[:, idx], label=label)
                else:
                    label = key if len(datasets) == 1 else f"{key} {i+1}"
                    ax.plot(time, data[:, value], label=label)
            
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True)
            plot_index += 1
        
        for i in range(plot_index, len(axs)):
            fig.delaxes(axs[i])
        
        plt.tight_layout()

        if self.mode == 'show':
            plt.show()
        elif self.mode == 'save':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_folder}/{'plot' if title is None else title}_{timestamp}.png"
            print(f"Saving: {filename}")
            plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
        else:
            raise ValueError("Invalid mode. Use 'show' or 'save'.")
        plt.close()

    def determine_plot_layout(self, num_plots):
        if num_plots <= 3:
            return 1, num_plots
        elif num_plots == 4:
            return 2, 2
        elif num_plots <= 6:
            return 2, 3
        elif num_plots <= 9:
            return 3, 3
        else:
            rows = min(5, (num_plots + 2) // 3)
            cols = min(3, (num_plots + rows - 1) // rows)
            return rows, cols


# class DasPlotter:
#     def __init__(self, save_folder=None, mode='show', orientation='grid'):
#         self.save_folder = save_folder if save_folder else os.getcwd()
#         self.mode = mode.lower()
#         self.orientation = orientation.lower()
#         # Ensure the save folder exists
#         if not os.path.exists(self.save_folder):
#             os.makedirs(self.save_folder)


#     def plot(self, datamap, datasets, common_time=None, title = None):
#         """
#         Plot multiple datasets on the same graph.
        
#         :param datasets: List of numpy arrays containing the data for each dataset
#         :param datamap: Dictionary mapping variable names to column indices
#         :param common_time: Optional common time vector for all datasets
#         """
#         # Count the number of subplots needed (one for each key, excluding 'time')
#         num_plots = len(datamap) - 1  # Subtract 1 to exclude 'time'
        
#         if self.orientation == 'vertical':
#             num_rows, num_cols = num_plots, 1
#         elif self.orientation == 'grid':  # 'grid' orientation
#             num_rows, num_cols = self.determine_plot_layout(num_plots)
#         else:
#             raise ValueError("Invalid orientation. Use 'vertical' or 'grid'.")
        
#         # Create the figure and subplots
#         # fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 2*num_rows))
#         fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 2.5*num_rows))    
        
#         if title:
#             fig.suptitle(title, fontsize=12)
#         if num_plots == 1:
#             axs = np.array([axs])
#         axs = axs.flatten()
        
#         # Plot each variable group
#         plot_index = 0
#         for key, value in datamap.items():
#             if key == "time":
#                 continue
            
#             ax = axs[plot_index]
#             for i, data in enumerate(datasets):
#                 if common_time is not None:
#                     time = common_time
#                 else:
#                     time = data[:, datamap["time"]]
                
#                 # if isinstance(value, list):
#                 #     for j, idx in enumerate(value):
#                 #         ax.plot(time, data[:, idx], label=f"{key} {i+1}")
#                 # else:
#                 #     ax.plot(time, data[:, value], label=f"{key} {i+1}")
                    
#                 if isinstance(value, list):
#                     for j, idx in enumerate(value):
#                         label = f"{key}[{j}]" if len(datasets) == 1 else f"{key}[{j}] {i+1}"
#                         ax.plot(time, data[:, idx], label=label)
#                 else:
#                     label = key if len(datasets) == 1 else f"{key} {i+1}"
#                     ax.plot(time, data[:, value], label=label)
            
#             # ax.set_title(key)
#             # ax.set_xlabel("Time")
#             # ax.set_ylabel("Value")
#             # # Create a smaller legend in the top right corner
#             # ax.legend(fontsize='x-small', loc='upper right')
#             # ax.grid(True)
#             # plot_index += 1
                    
#             ax.set_title(key, fontsize=10)
#             ax.set_xlabel("Time", fontsize=10)
#             ax.set_ylabel("Value", fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=8)
#             ax.legend(fontsize=8, loc='upper right')
#             ax.grid(True)
#             plot_index += 1
        
#         # Remove any unused subplots
#         for i in range(plot_index, len(axs)):
#             fig.delaxes(axs[i])
        
#         plt.tight_layout()
#         # plt.show()

        


#         if self.mode == 'show':
#             plt.show()
#         elif self.mode == 'save':
#             filename = None
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             if title is None:
#                 filename = f"{self.save_folder}/plot_{timestamp}.png"
#             else:
#                 filename = f"{self.save_folder}/{title}_{timestamp}.png"
#             print(f"Filename: {filename}")
#             # Save the figure as a JPEG file with the generated filename
#             print(f"Saving: {filename}")
#             plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
#         else:
#             raise ValueError("Invalid mode. Use 'show' or 'save'.")
#         plt.close()


#     def determine_plot_layout(self, num_plots):
#         if num_plots <= 3:
#             return 1, num_plots
#         elif num_plots == 4:
#             return 2, 2
#         elif num_plots == 5:
#             return 1, num_plots
#         elif num_plots == 6:
#             return 3, 2
#         elif num_plots == 7:
#             return 3,3
#         elif num_plots == 8: 
#             return 4, 2
#         elif num_plots == 9:
#             return 3, 3
#         else:
#             rows = min(5, (num_plots + 2) // 3)
#             cols = min(3, (num_plots + rows - 1) // rows)
#             return rows, cols