import ctypes
from ctypes import wintypes
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import numbers
import math

# Load necessary Windows API functions
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
kernel32.GetProcAddress.restype = ctypes.c_void_p

IMAGE_DIRECTORY_ENTRY_EXPORT = 0
IMAGE_DOS_SIGNATURE = 0x5A4D
IMAGE_NT_SIGNATURE = 0x00004550

class IMAGE_DOS_HEADER(ctypes.Structure):
    _fields_ = [
        ("e_magic", wintypes.WORD),
        ("e_cblp", wintypes.WORD),
        ("e_cp", wintypes.WORD),
        ("e_crlc", wintypes.WORD),
        ("e_cparhdr", wintypes.WORD),
        ("e_minalloc", wintypes.WORD),
        ("e_maxalloc", wintypes.WORD),
        ("e_ss", wintypes.WORD),
        ("e_sp", wintypes.WORD),
        ("e_csum", wintypes.WORD),
        ("e_ip", wintypes.WORD),
        ("e_cs", wintypes.WORD),
        ("e_lfarlc", wintypes.WORD),
        ("e_ovno", wintypes.WORD),
        ("e_res", wintypes.WORD * 4),
        ("e_oemid", wintypes.WORD),
        ("e_oeminfo", wintypes.WORD),
        ("e_res2", wintypes.WORD * 10),
        ("e_lfanew", wintypes.LONG),
    ]

class IMAGE_DATA_DIRECTORY(ctypes.Structure):
    _fields_ = [
        ("VirtualAddress", wintypes.DWORD),
        ("Size", wintypes.DWORD),
    ]

class IMAGE_OPTIONAL_HEADER64(ctypes.Structure):
    _fields_ = [
        ("Magic", wintypes.WORD),
        ("MajorLinkerVersion", ctypes.c_ubyte),
        ("MinorLinkerVersion", ctypes.c_ubyte),
        ("SizeOfCode", wintypes.DWORD),
        ("SizeOfInitializedData", wintypes.DWORD),
        ("SizeOfUninitializedData", wintypes.DWORD),
        ("AddressOfEntryPoint", wintypes.DWORD),
        ("BaseOfCode", wintypes.DWORD),
        ("ImageBase", ctypes.c_ulonglong),
        ("SectionAlignment", wintypes.DWORD),
        ("FileAlignment", wintypes.DWORD),
        ("MajorOperatingSystemVersion", wintypes.WORD),
        ("MinorOperatingSystemVersion", wintypes.WORD),
        ("MajorImageVersion", wintypes.WORD),
        ("MinorImageVersion", wintypes.WORD),
        ("MajorSubsystemVersion", wintypes.WORD),
        ("MinorSubsystemVersion", wintypes.WORD),
        ("Win32VersionValue", wintypes.DWORD),
        ("SizeOfImage", wintypes.DWORD),
        ("SizeOfHeaders", wintypes.DWORD),
        ("CheckSum", wintypes.DWORD),
        ("Subsystem", wintypes.WORD),
        ("DllCharacteristics", wintypes.WORD),
        ("SizeOfStackReserve", ctypes.c_ulonglong),
        ("SizeOfStackCommit", ctypes.c_ulonglong),
        ("SizeOfHeapReserve", ctypes.c_ulonglong),
        ("SizeOfHeapCommit", ctypes.c_ulonglong),
        ("LoaderFlags", wintypes.DWORD),
        ("NumberOfRvaAndSizes", wintypes.DWORD),
        ("DataDirectory", IMAGE_DATA_DIRECTORY * 16),
    ]

class IMAGE_FILE_HEADER(ctypes.Structure):
    _fields_ = [
        ("Machine", wintypes.WORD),
        ("NumberOfSections", wintypes.WORD),
        ("TimeDateStamp", wintypes.DWORD),
        ("PointerToSymbolTable", wintypes.DWORD),
        ("NumberOfSymbols", wintypes.DWORD),
        ("SizeOfOptionalHeader", wintypes.WORD),
        ("Characteristics", wintypes.WORD),
    ]

class IMAGE_NT_HEADERS64(ctypes.Structure):
    _fields_ = [
        ("Signature", wintypes.DWORD),
        ("FileHeader", IMAGE_FILE_HEADER),
        ("OptionalHeader", IMAGE_OPTIONAL_HEADER64),
    ]

class IMAGE_EXPORT_DIRECTORY(ctypes.Structure):
    _fields_ = [
        ("Characteristics", wintypes.DWORD),
        ("TimeDateStamp", wintypes.DWORD),
        ("MajorVersion", wintypes.WORD),
        ("MinorVersion", wintypes.WORD),
        ("Name", wintypes.DWORD),
        ("Base", wintypes.DWORD),
        ("NumberOfFunctions", wintypes.DWORD),
        ("NumberOfNames", wintypes.DWORD),
        ("AddressOfFunctions", wintypes.DWORD),
        ("AddressOfNames", wintypes.DWORD),
        ("AddressOfNameOrdinals", wintypes.DWORD),
    ]

class InvalidInputFieldError(Exception):
    """Exception raised for invalid input fields."""
    pass

class SimDLLError(Exception):
    """Custom exception class for SimDLL errors."""
    pass


class SimDLL:
    def __init__(self, dll_path, inputStruct = None, outputStruct = None, dt = 1/24e3):
        self.dll_path = dll_path
        self.dll = ctypes.WinDLL(dll_path)
        self.dt = dt
        self.exported_functions = self.get_exported_functions()
        self.dll_prefix = self.get_dll_prefix() # dll name
        self.functions = self._setup_functions()
        self.input_var, self.output_var = self.get_io_vars()
        # print("input vars test: ", self.input_var)
        # print("output vars test: ",self.output_var)

        if inputStruct and self.input_var:
            self.input = inputStruct.in_dll(self.dll, self.input_var)
            self.input_fields = {field[0]: field[1] for field in inputStruct._fields_}
            self.input_setters = self._create_setters(self.input_fields)
        else:
            print(f"Warning: Input structure or variable not found.")
            self.input = None
            self.input_fields = {}
            self.input_setters = {}
        
        if outputStruct and self.output_var:
            self.output = outputStruct.in_dll(self.dll, self.output_var)
            self.output_fields = {field[0]: field[1] for field in outputStruct._fields_}
        else:
            print(f"Warning: Output structure or variable not found.")
            self.output = None
            self.output_fields = {}

        # Initialize the dll
        self.initialize()

    # It returns DLL input, output, function
    def get_exported_functions(self):
        base_addr = self.dll._handle

        # Read DOS header
        dos_header = IMAGE_DOS_HEADER.from_address(base_addr)
        if dos_header.e_magic != IMAGE_DOS_SIGNATURE:
            raise ValueError("Invalid DOS signature")

        # Read NT headers
        nt_headers = IMAGE_NT_HEADERS64.from_address(base_addr + dos_header.e_lfanew)
        if nt_headers.Signature != IMAGE_NT_SIGNATURE:
            raise ValueError("Invalid NT signature")

        # Get export directory
        export_dir_rva = nt_headers.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress
        export_dir = IMAGE_EXPORT_DIRECTORY.from_address(base_addr + export_dir_rva)

        # Get function names
        names_rva = export_dir.AddressOfNames
        function_names = []
        for i in range(export_dir.NumberOfNames):
            name_rva = struct.unpack('<I', ctypes.string_at(base_addr + names_rva + i * 4, 4))[0]
            name = ctypes.c_char_p(base_addr + name_rva).value.decode('ascii')
            function_names.append(name)

        return function_names

     # This return the project name as string
    def get_dll_prefix(self):
        """
        This return the project name as string
        This project name is utilized to find out other variables
        Find the common prefix for initialize, step, and terminate functions
        """
       
        func_names = ['initialize', 'step', 'terminate']
        prefixes = [name.rsplit('_', 1)[0] for name in self.exported_functions 
                    if name.split('_')[-1] in func_names]
        # print("prefixes: ",prefixes)
        prefix = max(set(prefixes), key=prefixes.count) if prefixes else ""
        # print("Common prefix: ", prefix)
        return prefix


    
    def get_io_vars(self):
        """
        This returns the input and output variable names
        """
        input_var = next((var for var in self.exported_functions if var.endswith('_U')), None)
        output_var = next((var for var in self.exported_functions if var.endswith('_Y')), None)
        # print("input var: ",input_var)
        # print("output_var: ",output_var)
        return input_var, output_var


    def _setup_functions(self):
        """
        This function returns the name of three important functions. step, initialize, terminate
        """
        function_names = ['initialize', 'step', 'terminate']
        functions = {}  # Initialize as an empty dictionary
        for name in function_names:
            func_name = f"{self.dll_prefix}_{name}"
            # print(f"Looking for function: {func_name}")
            if func_name in self.exported_functions:
                # print(f"Found function: {func_name}")
                try:
                    func = getattr(self.dll, func_name)
                    func.argtypes = []
                    func.restype = None
                    functions[name] = func
                    # print(f"Successfully set up function: {name}")
                except Exception as e:
                    print(f"Error setting up function {func_name}: {str(e)}")
            else:
                print(f"Function {func_name} not found in exported functions")
        return functions
    

    def _create_setters(self, fields):
        setters = {}
        for field, field_type in fields.items():
            if isinstance(field_type, type(ctypes.Array)):
                setters[field] = lambda value, ft=field_type: ft(*value)
            else:
                setters[field] = lambda value, ft=field_type: ft(value)
        return setters


    def _set_input_values_v2(self, inputs):
        """
        This functiont  does two things 
        1. verfy the user input is following the inputStruct that was given during instance creation 
        2. converts the user input into the ctypes for dll. It uses the types that was given during instance creation
        """
        if not inputs or not self.input:
            return
        
        for field, value in inputs.items():
            try:
                setter = self.input_setters[field]
                setattr(self.input, field, setter(value))
            except KeyError:
                print(f"Warning: Field '{field}' not found in input structure")
                raise ValueError(f"Warning: Field '{field}' not found in input structure")
            except Exception as e:
                print(f"Error setting field '{field}': {e}")
                raise ValueError(f"Error setting field '{field}': {e}")
            
    def step(self, inputs=None):
        self._set_input_values_v2(inputs)
    
        if 'step' in self.functions:
            self.functions['step']()
        else:
            print("Warning: Step function not found in DLL.")
        
        if self.output:
            return {field: self._convert_output(getattr(self.output, field)) for field in self.output_fields}
        else:
            return None
        
    def reset(self):
        """Reset the state of the model to an initial state"""
        self.terminate()
        self.initialize()

    def _convert_output(self, value):
        if isinstance(value, ctypes.Array):
            return list(value)
        return value

    def initialize(self):
        # print("Available functions:", self.functions)
        if 'initialize' in self.functions:
            try:
                self.functions['initialize']()
            except Exception as e:
                raise SimDLLError(f"Error during initialization: {e}")
        else:
            raise SimDLLError("Initialize function not found in DLL.")
            
    def terminate(self):
        if 'terminate' in self.functions:
            try:
                self.functions['terminate']()
            except Exception as e:
                raise SimDLLError(f"Error during termination: {e}")
        else:
            raise SimDLLError("Terminate function not found in DLL.")
            

    ## This function is faster
    # def generate_step_input(self, step_info, current_time):
    #     """
    #     Convert user inputs based on their type:
    #     1. If the input is a scalar (any numeric type), it returns it directly.
    #     2. If the input is a dictionary of step command, it converts the input according to the current time.

    #     :param step_info: The input value or step command dictionary
    #     :param current_time: The current simulation time
    #     :return: The appropriate value for the current time
    #     :raises ValueError: If the input format is invalid
    #     """
    #     # if isinstance(step_info, (int, numbers.Number)):
    #     if isinstance(step_info, (int, float)):
    #         return step_info
    #     elif isinstance(step_info, dict) and 'initial' in step_info and 'final' in step_info and 'step_time' in step_info:
    #         return step_info['final'] if current_time >= step_info['step_time'] else step_info['initial']
    #     else:
    #         raise ValueError(f"Invalid step input format or data type: {step_info}")
        
    # This function is slower than the previous  one
    def generate_step_input(self, step_info, current_time):
        """
        Convert user inputs based on their type:
        1. If the input is a scalar (any numeric type), it returns it directly.
        2. If the input is a dictionary of step command, it converts the input according to the current time.

        :param step_info: The input value or step command dictionary
        :param current_time: The current simulation time
        :return: The appropriate value for the current time
        :raises ValueError: If the input format is invalid
        """
        if isinstance(step_info, numbers.Number):
            return float(step_info)  # Convert all numeric types to float
        elif isinstance(step_info, np.ndarray) and step_info.size == 1:
            return float(step_info.item())  # Convert single-element numpy arrays to float
        elif isinstance(step_info, dict) and all(key in step_info for key in ['initial', 'final', 'step_time']):
            return float(step_info['final']) if current_time >= step_info['step_time'] else float(step_info['initial'])
        else:
            raise ValueError(f"Invalid step input format or data type: {type(step_info)}, value: {step_info}")


    # def generate_inputs_for_time(self, inputs, current_time):
    #     """
    #     Generate input values for a given time.
        
    #     :param inputs: Dictionary of input values or step definitions
    #     :param current_time: Current simulation time
    #     :return: Dictionary of input values for the given time
    #     """
    #     # for key, values in inputs.items():
    #     #     current_inputs[key] = [self.generate_step_input(v, current_time) for v in values]
    #     #     input_arrays[key][step] = current_inputs[key]
    #     return {key: [self.generate_step_input(v, current_time) for v in values] for key, values in inputs.items()}


    def generate_inputs_for_time(self, inputs, current_time):
        """
        Generate input values for a given time.
        
        :param inputs: Dictionary of input values or step definitions
        :param current_time: Current simulation time
        :return: Dictionary of input values for the given time
        """
        result = {}
        for key, values in inputs.items():
            if isinstance(values, (list, tuple, np.ndarray)):
                result[key] = [self.generate_step_input(v, current_time) for v in values]
            else:
                result[key] = self.generate_step_input(values, current_time)
        return result
    

    def run_simulation(self, inputs, duration):
        num_steps = int(duration / self.dt)
        time_array = np.linspace(0, duration, num_steps, endpoint=False)
        
        # Initialize input arrays
        input_arrays = {key: np.zeros((num_steps, len(value))) for key, value in inputs.items()}
        
        # Initialize output arrays after the first step
        # initial_inputs = {key: [self.generate_step_input(v, 0) for v in value] for key, value in inputs.items()}
        initial_inputs = self.generate_inputs_for_time(inputs, 0)
        outputs = self.step(initial_inputs)
        output_arrays = {key: np.zeros((num_steps, len(value) if isinstance(value, (list, tuple)) else 1)) 
                         for key, value in outputs.items()}
        
        start_time = time.time()
        
        for step in range(num_steps):
            current_time = step * self.dt
            
            # Update inputs based on step inputs
            current_inputs = self.generate_inputs_for_time(inputs, current_time)

            # Storing the current 
            for key, values in current_inputs.items():
                    input_arrays[key][step] = values

            outputs = self.step(current_inputs)

            for key, value in outputs.items():
                output_arrays[key][step] = value if isinstance(value, (list, tuple)) else [value]
        
        self.terminate()
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        result = {'time': time_array, **input_arrays, **output_arrays}
        return result
    




   


################################# End of simDLL class###################
def determine_plot_layout(num_plots):
    if num_plots <= 3:
        return 1, num_plots
    elif num_plots == 4:
        return 2, 2
    elif num_plots == 5:
        return 1, num_plots
    elif num_plots == 6:
        return 3, 2
    elif num_plots == 7:
        return 3,3
    elif num_plots == 8: 
        return 4, 2
    elif num_plots == 9:
        return 3, 3
    else:
        rows = min(5, (num_plots + 2) // 3)
        cols = min(3, (num_plots + rows - 1) // rows)
        return rows, cols

def plot_data_from_map(data, datamap):
    """
    # Example usage:
    # Assuming 'data' is your numpy array of temporal data
    # plot_inverter_data(data, index)
    """
    # Count the number of subplots needed (one for each key, excluding 'time')
    num_plots = len(datamap) - 1  # Subtract 1 to exclude 'time'
    num_rows, num_cols = determine_plot_layout(num_plots)
    # print((num_rows,num_cols))
    
    # Create the figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 2*num_rows))
    if num_plots == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    # Get the time data
    time = data[:, datamap["time"]]
    
    # Plot each variable group
    plot_index = 0
    for key, value in datamap.items():
        if key == "time":
            continue
        
        ax = axs[plot_index]
        if isinstance(value, list):
            for i, idx in enumerate(value):
                ax.plot(time, data[:, idx], label=f"{key}[{i}]")
        else:
            ax.plot(time, data[:, value], label=key)
        
        ax.set_title(key)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        plot_index += 1
    
    # Remove any unused subplots
    for i in range(plot_index, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.show()



def plot_combined_from_map(datasets, datamap, common_time=None):
    """
    Plot multiple datasets on the same graph.
    
    :param datasets: List of numpy arrays containing the data for each dataset
    :param datamap: Dictionary mapping variable names to column indices
    :param common_time: Optional common time vector for all datasets
    """
    # Count the number of subplots needed (one for each key, excluding 'time')
    num_plots = len(datamap) - 1  # Subtract 1 to exclude 'time'
    num_rows, num_cols = determine_plot_layout(num_plots)
    
    # Create the figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 2*num_rows))
    if num_plots == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    # Plot each variable group
    plot_index = 0
    for key, value in datamap.items():
        if key == "time":
            continue
        
        ax = axs[plot_index]
        for i, data in enumerate(datasets):
            if common_time is not None:
                time = common_time
            else:
                time = data[:, datamap["time"]]
            
            if isinstance(value, list):
                for j, idx in enumerate(value):
                    ax.plot(time, data[:, idx], label=f"{key} {i+1}")
            else:
                ax.plot(time, data[:, value], label=f"{key} {i+1}")
        
        ax.set_title(key)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        # Create a smaller legend in the top right corner
        ax.legend(fontsize='x-small', loc='upper right')
        ax.grid(True)
        plot_index += 1
    
    # Remove any unused subplots
    for i in range(plot_index, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.show()

# Example usage:
# datasets = [data_debug_gfm_N, data_debug_gfm_W]
# common_time = np.linspace(0, 10, len(data_debug_gfm_N))  # Adjust as needed
# plot_data_from_map(datasets, datamap, common_time)