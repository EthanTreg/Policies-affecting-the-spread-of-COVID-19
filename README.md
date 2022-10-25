# Policies Affecting the Spread of COVID-19

## Requirements
* Install dependencies:  
`pip install -r requirements.txt`
* CuPy dependencies:  
NVIDIA GPU with CUDA toolkit > v10.2 [^1]
[^1]: If NVIDIA GPU not avaliable, remove `mempool` in `simulation` under `COVID_Simulation_V11.py`
and change all `cp.` references to `np.` and remove `cp.asnumpy` in `simulation`

## Report
PDF of the report can be found under `Simulating_Government_Policies_on_the_Spread_of_COVID`

## File Descriptions
### COVID_Simulation_V11:
Perfoms random walk to identify the number of people who would get infected from COVID-19.  
`main` function can be run to simulate one day.  
Configurable parameters are:
* day (int) - Current day number, used for file name saving and loading
* total_people (int) - Total number of people to simulate
* num_contagious (int) - Number of people infected with COVID-19 and can spread it
* initial_infections (int) - Number of people infected with COVID-19 who can't spread it
* alpha (ndarray) - Transmissibility and protection constant
* buildings (int) - Number of rooms to simulate
* density_avg (float) - Average density of each room

Options for the function are:
* load (bool) - Whether to laod previous results
* save (bool) - Whether to save results
* animate (bool) - Whether to produce random walk animation

### COVID Analysis V8
Simulates the spread of COVID-19 over several days.  
`main` function can be run to begin the simulation.  
Options for the function are:
* days (int) - number of days to simulate
* file_number (int) - Run number for file saving, if < 0, auto calculate new number from existing file numbers
* load (bool) - If previous results for the given file number should be loaded
* testing (bool) - If COVID-19 testing should be used
* vaccinating (bool) - If vaccinations should be used
* distancing (bool) - If social distancing should be used
* lockdown (bool) - If lockdowns should be used

### COVID Graph Plot
Plots graphs from data generated from `COVID Analsysis V8` to show the numbe of new infections, number of recoveries,
the effective reproductive value, and any parameter related to the policies implemented in the run.  
`main` function can be run to generate the graphs.  
Options for the function are:
* days (int) - Number of days to plot
* files (list) - File numbers to load

### Graph Plot
Imports and plots CSV COVID-19 data collected from the UK government website.  
Fits linear, logarithmic and sigmoid functions to the data to get the fit parameters.  
Fit parameters are found from `popt` and covariance matrix is given by `pcov`.  
`main` function can be run to import data and fit a model to it.  
Options for the function are:
* File location under `Retrieve data files` to import data
* Fit function type under `Fit function to data`
