"""
Simulation of the spread of COVID-19 over and the affect that polices have on reducing the spread

Methods
-------
main()
    Main function to control the simulation and configure which polices are active
"""


import os
import pickle
import numpy as np
from time import time
from typing import Tuple
from matplotlib import pyplot as plt

from COVID_Simulation_V11 import main as sim


def log_normal(x: np.ndarray, A: float, mu: float, sig: float):
    """
    Log-normal equation with a given amplitude

    y = A * exp(-(-ln(x) - mu)^2 / (2 * sig^2)) / x

    Parameters
    ----------
    x : ndarray
        Input values
    A : float
        Amplitude
    mu : float
        Log-normal mu
    sig : float
        Log-normal sigma

    Returns
    -------
    y : ndarray
        Output values
    """
    x += 1e-6
    return A * np.exp(-(np.log(x) - mu) ** 2 / (2 * sig ** 2)) / x


def sigmoid(x: np.ndarray, A=1.9e5, m=2.1e-2, c=9.4):
    """
    Sigmoid function with a defined amplitude

    A / (1 + exp(-m * x + c))

    Parameters
    ----------
    x : ndarray
        Input values
    A : float, default=1.9e5
        Amplitude
    m : float, default=2.1e-2
        Gradient
    c : float, default=9.4
        Offset

    Returns
    -------
    y : ndarray
        Output values
    """
    return A / (1 + np.exp(-m * x + c))


def vaccine_plot(days: int, vaccine_2_param: list):
    """
    Plots vaccinations over time

    Parameters
    ----------
    days : int
        Total number of days
    vaccine_2_param : list
        Parameters for the second dose sigmoid equation
    """
    minor = 16
    x = np.arange(days)
    y1 = sigmoid(x)
    y2 = sigmoid(x, *vaccine_2_param)

    plt.figure(figsize=(16, 8), constrained_layout=True)
    plt.plot(x, y1, label='First dose')
    plt.plot(x, y2, label='Second dose')

    plt.xlabel('Days', fontsize=minor)
    plt.legend(loc='upper left', fontsize=minor)
    plt.gca().tick_params(axis='both', labelsize=minor)
    plt.show()


def infection_plot(incubation: list, asym: list, presym: list, sym: list):
    """
    Plots the incubation, asymptomatic, pre-symptomatic and symptomatic period distributions

    Parameters
    ----------
    incubation : list
        Parameters for incubation log-normal distribution
    asym : list
        Parameters for asymptomatic log-normal distribution
    presym : list
        Parameters for pre-symptomatic log-normal distribution
    sym : list
        Parameters for symptomatic log-normal distribution
    """
    major = 20
    minor = 16

    # Incubation subplot
    _, ax = plt.subplots(nrows = 4, figsize=(16, 12), constrained_layout=True)
    x = np.linspace(0, 20, 1000)
    y1 = log_normal(x, 1 / (incubation[1] * np.sqrt(2 * np.pi)), incubation[0], incubation[1])
    ax[0].plot(x, y1)
    ax[0].set_title('Incubation period', fontsize=major)
    ax[0].tick_params(axis='both', labelsize=minor)

    # Asymptomatic suplot
    y2 = log_normal(x, 1 / (asym[1] * np.sqrt(2 * np.pi)), asym[0], asym[1])
    ax[1].plot(x, y2)
    ax[1].set_title('Asymptomatic period', fontsize=major)
    ax[1].tick_params(axis='both', labelsize=minor)

    # Pre-symptomatic subplot
    x = np.linspace(0, 10, 1000)
    y3 = log_normal(x, 1 / (presym[1] * np.sqrt(2 * np.pi)), presym[0], presym[1])
    ax[2].plot(x, y3)
    ax[2].set_title('Pre-symptomatic period', fontsize=major)
    ax[2].tick_params(axis='both', labelsize=minor)

    # Symptomatic subplot
    x = np.linspace(0, 40, 1000)
    y4 = log_normal(x, 1 / (sym[1] * np.sqrt(2 * np.pi)), sym[0], sym[1])
    ax[3].plot(x, y4)
    ax[3].set_title('Symptomatic period', fontsize=major)
    ax[3].tick_params(axis='both', labelsize=minor)

    plt.xlabel('Days', fontsize=major)
    plt.show()


def plot(new_infections: np.ndarray, recovered: np.ndarray, rt: np.ndarray, tests: np.ndarray, isolating: np.ndarray, vaccinated: np.ndarray):
    """
    Plots the evolution of the virus over time
    Plot varies depending on model type with additional plots for testing and vaccinations

    Parameters
    ----------
    new_infections : ndarray
        Number of infections per day
    recovered : ndarray
        Number of people recovered per day
    rt : ndarray
        Effective reporductive value per day
    tests : ndarray
        Number of tests taken per day and number of positive tests
    isolating : ndarray
        Number of people isolating per day
    vaccinated : ndarray
        Total nunmber of people vaccinated for one dose and both doses
    """
    nrows = 3
    major = 20
    minor = 16

    if np.count_nonzero(tests[:, 1]) != 0:
        nrows += 2

    if np.count_nonzero(vaccinated[:, 1]) != 0:
        nrows += 1

    _, ax = plt.subplots(nrows=nrows, sharex=True, constrained_layout=True, figsize=(16, 3 * nrows))

    # Infections
    ax[0].set_title('New Infections', fontsize=major)
    ax[0].plot(new_infections)
    ax[0].tick_params(axis='y', labelsize=minor)

    # Recoveries
    ax[1].set_title('Recovered', fontsize=major)
    ax[1].plot(recovered)
    ax[1].tick_params(axis='y', labelsize=minor)

    # Effective reproductive number
    ax[2].set_title(r'$R_t$', fontsize=major)
    ax[2].plot(rt[:, 0], rt[:, 1])
    ax[2].text(0.75, 0.75, f'$R_t$ final value: {round(rt[-1, 1], 2)}', transform=ax[2].transAxes, fontsize=minor)
    ax[2].tick_params(axis='y', labelsize=minor)

    # Testing & isolating
    if np.count_nonzero(tests[:, 1]) != 0:
        ax[3].set_title('Testing', fontsize=major)
        ax[3].plot(tests[:, 0], tests[:, 1], label='Tests taken')
        ax[3].plot(tests[:, 0], tests[:, 2], label='Positive tests')
        ax[3].legend(loc='upper left', fontsize=minor)
        ax[3].tick_params(axis='y', labelsize=minor)

        ax[4].set_title('Isolating', fontsize=major)
        ax[4].plot(isolating[:, 0], isolating[:, 1])
        ax[4].tick_params(axis='y', labelsize=minor)

    # Vaccinated
    if np.count_nonzero(vaccinated[:, 1]) != 0:
        ax[-1].set_title('Vaccinated', fontsize=major)
        ax[-1].plot(vaccinated[:, 0], vaccinated[:, 1] + vaccinated[:, 2], label='First dose')
        ax[-1].plot(vaccinated[:, 0], vaccinated[:, 2], label='Second dose')
        ax[-1].tick_params(axis='y', labelsize=minor)
        ax[-1].legend()

    plt.xlabel('Days', fontsize=minor)
    plt.xticks(fontsize=minor)
    plt.show()


def initialisation(total_people: int, buildings: int, days: int, infection_params: list, testing=False, vaccinating=False, distancing=False, lockdown=False):
    """
    Initialises the simulation by creating the initial uninfected and infected arrays as well as the parameters for the different policies

    Parameters
    ----------
    total_people : int
        Total number of people in the simulation
    buildings : int
        Total number of buildings in the simulation
    days : int
        Total number of days to run the simulation
    infection_params : list
        List of probability for asymptomatic cases and asymptomatic, pre-symptomatic and symptomatic log-normal parameters
    testing : bool, default=False
        If testing is enacted
    vaccinating : bool, default=False
        If vaccinations are in affect
    distancing : bool, default=False
        If social distancing is enacted
    lockdown : bool, default=False
        If lockdowns are in affect

    Returns
    -------
    uninfected : ndarray
        Immunity and infection status for all people
    infected : ndarray
        Detailed infection status for infected people
    test_rate : ndarray
        Testing rate for each day, 0 for no testing
    new_vaccines : ndarray
        Number of first doses per day, 0 for no vaccinations
    new_vaccines_2 : ndarray
        Number of second doses per day, 0 for no vaccinations
    density_avg : ndarray
        Average density of buildings per day, 0.86 for no social distancing
    buildings : ndarray
        Total number of buildings per day, equal to buildings parameter for no lockdowns
    """
    initial_infections = 10

    asym_probability = infection_params[0]
    asym = infection_params[1]
    presym = infection_params[2]
    sym = infection_params[3]

    test_start = 120
    test_ramp = 365
    test_probability = 1.5e-2

    infection_type = np.random.choice([0, 1], size=initial_infections, p=[asym_probability, 1 - asym_probability])
    presymptomatic_period = np.rint(np.where(infection_type == 0, 0, np.random.lognormal(*presym, size=initial_infections)))

    # Initialise infected array
    infected = np.array((
        np.zeros(initial_infections),
        np.rint(np.where(infection_type == 0, np.random.lognormal(*asym, size=initial_infections), np.random.lognormal(*sym, size=initial_infections))) + presymptomatic_period,
        np.zeros(initial_infections),
        presymptomatic_period,
        np.zeros(initial_infections),
        np.zeros(initial_infections),
        np.arange(initial_infections, dtype=int),
    ))

    # Initialise uninfected array
    uninfected = np.ones((2, total_people))
    uninfected = np.vstack((
        uninfected,
        np.append(np.ones(initial_infections), np.zeros(total_people - initial_infections)),
        np.zeros(total_people),
    ))

    # Testing
    if testing:
        test_rate = np.append(np.zeros(test_start), np.append(np.linspace(0, test_probability, test_ramp), np.ones(days) * test_probability))
    else:
        test_rate = np.zeros(days)

    # Vaccinations
    if vaccinating:
        new_vaccines = np.diff(sigmoid(np.arange(days + 1))).astype(int)
        new_vaccines_2 = np.diff(sigmoid(np.arange(days + 1), 1.8e5, 2.9e-2, 15)).astype(int)
    else:
        new_vaccines = np.zeros(days, dtype=int)
        new_vaccines_2 = np.zeros(days, dtype=int)

    # Social distancing
    if distancing:
        density_avg = np.append(np.ones(60) * 0.86, np.ones(days) * 0.25)
    else:
        density_avg = np.ones(days) * 0.86

    # Lockdowns
    if lockdown:
        buildings = (buildings * np.hstack((
            np.ones(90),
            np.ones(60) * 0.76,
            np.ones(150),
            np.ones(30) * 0.89,
            np.ones(30),
            np.ones(60) * 0.88,
            np.ones(days),
        ))).astype(int)
    else:
        buildings = np.ones(days, dtype=int) * buildings

    return uninfected, infected, test_rate, new_vaccines, new_vaccines_2, density_avg, buildings


def load_data(results_location: str, days: int, j: int) -> Tuple[int, dict]:
    """
    Opens the result file for previously loaded data

    Parameters
    ----------
    results_location : str
        Location of save files
    days : int
        Total number of simulation days
    j : int
        Results file number

    Returns
    -------
    i : int
        Initial day number
    results : dict
        Dictionary of saved results
    """
    i = 0

    # Find most recent file
    while os.path.exists(results_location.format(j) + f'/Day {i}'):
        i += 1

    # Limit loaded file to the maximum number of days
    i = min(days, i)

    # Open saved file
    with open(results_location.format(j) + f'/Day {i - 1}', 'rb') as result_file:
        results = pickle.load(result_file)

    return i, results


def main():
    # Variables
    days = 800
    file_number = 40
    load = True
    testing = False
    vaccinating = True
    distancing = False
    lockdown = False

    # Constants
    ti = time()
    j = 0
    total_people = int(250e3)
    buildings = 1000
    results_location = 'Project 1/Results/Sim Results {}'

    # Variant evolution
    variant = np.hstack((
        np.ones(270), # Ancestral
        np.linspace(1, 2, 60), # Transition
        np.ones(480 - 270 - 60) * 2, # Alpha
        np.linspace(2, 3, 60), # Transition
        np.ones(days) * 3 # Delta
    ))

    # Testing
    pcr_probability = 0.4
    pcr = [3.9, 2, 0.9]
    lf = [3.1, 1.7, 0.4]

    # Infection type
    asym_probability = 0.25
    incubation = [1.63, 0.5]
    asym = [1.75, 0.4]
    presym = [0.8, 0.3]
    sym = [2.85, 0.3]

    # Vaccinations & natural immunity
    natural_alpha = np.append(np.linspace(0.16, 0.31, 365), np.ones(days) * 0.31)
    vaccine_alpha = [0.18, 0.05]

    # Arrays
    new_infections = np.array((), dtype=int)
    recovered = np.array((), dtype=int)
    rt = np.empty((0, 2))
    tests = np.empty((0, 3))
    isolating = np.empty((0, 2))
    vaccinated = np.empty((0, 3))

    # Find new file number if no file is to be loaded
    if file_number < 0:
        while os.path.exists(results_location.format(j)):
            j += 1

        j -= 1
    else:
        j = file_number

    # Create folder if it doesn't exist'
    if not(os.path.exists(results_location.format(j))):
        os.mkdir(results_location.format(j))

    # Initialise variables
    returns = initialisation(total_people, buildings, days, [asym_probability, asym, presym, sym], testing=testing, vaccinating=vaccinating, distancing=distancing, lockdown=lockdown)
    uninfected, infected, test_rate, new_vaccines, new_vaccines_2, density_avg, buildings = returns

    # Load previous results if to be loaded
    if load and os.path.exists(results_location.format(j) + '/Day 0'):
        initial_day, results = load_data(results_location, days, j)

        uninfected = results['uninfected']
        infected = results['infected']
        new_infections = results['new infections']
        recovered = results['recovered']
        rt = results['rt']
        total_people = results['total people']
        tests = results['tests']
        isolating = results['isolating']
        vaccinated = results['vaccinated']

    # Infected:
    # # Infection date [0]
    # # Infection duration [1]
    # # Incubation period [2]
    # # Pre-symptomatic period [3]
    # # Isolating [4]
    # # Infected [5]
    # # Uninfected index [6]

    # Uninfected:
    # # Natural immunity [0]
    # # Vaccine [1]
    # # Infected [2]
    # # Days since last infection [3]

    # Run simulation over time
    for i in range(initial_day, days):
        # Calculate natural immunity & recovered people
        natural_indices = np.argwhere(uninfected[0] != 1)[:, 0]
        uninfected[3, natural_indices] += 1
        uninfected[0, natural_indices] = natural_alpha[uninfected[3, natural_indices].astype(int)]

        recovered_indices = np.argwhere(infected[1] == i)[:, 0]
        recovered = np.append(recovered, recovered_indices.size)

        # Calculate Rt from number of people infected by recovered people
        if recovered_indices.size != 0:
            rt = np.append(rt, [[i, np.sum(infected[5, recovered_indices]) / recovered_indices.size]], axis=0)

            uninfected_indices = infected[6, recovered_indices].astype(int)
            uninfected[0, uninfected_indices] = natural_alpha[0]
            uninfected[2, uninfected_indices] = uninfected[3, uninfected_indices] = 0

        # Vaccinate people
        vaccine_indices = np.random.choice(np.argwhere(uninfected[1] == 1)[:, 0], size=new_vaccines[i], replace=False)
        uninfected[1, vaccine_indices] = vaccine_alpha[0]

        # Double vaccinate people
        vaccine_indices = np.random.choice(np.argwhere(uninfected[1] == vaccine_alpha[0])[:, 0], size=new_vaccines_2[i], replace=False)
        uninfected[1, vaccine_indices] = vaccine_alpha[1]
        vaccinated = np.append(vaccinated, [[i, np.count_nonzero(uninfected[1] == vaccine_alpha[0]), np.count_nonzero(uninfected[1] == vaccine_alpha[1])]], axis=0)

        # Remove recovered people from infected array
        infected = np.delete(infected, recovered_indices, axis=1)

        # If there are no more infected people, end program
        if infected.size == 0:
            break

        # Perform testing
        not_isolating = np.argwhere(infected[4] == 0)[:, 0]
        sym_not_isolating = np.argwhere((infected[4] == 0) & (infected[3] <= i) & (infected[3] != infected[2]))[:, 0]
        test_indices = np.unique(np.append(np.random.choice(not_isolating, size=int(not_isolating.size * test_rate[i]), replace=False), np.random.choice(sym_not_isolating, size=int(sym_not_isolating.size * test_rate[i]), replace=False)))
        test_type = np.random.choice([0, 1], size=test_indices.size, p=[pcr_probability, 1 - pcr_probability])
        probabilities = log_normal(i - infected[0, test_indices], np.where(test_type == 0, pcr[0], lf[0]), np.where(test_type == 0, pcr[1], lf[1]), np.where(test_type == 0, pcr[2], lf[2]))
        rand_numbers = np.random.rand(*probabilities.shape)
        positive_results = np.argwhere(rand_numbers < probabilities)
        infected[4, test_indices[positive_results]] = 1
        isolating = np.append(isolating, [[i, np.count_nonzero(infected[4])]], axis=0)

        tests = np.append(tests, [[i, test_indices.size, positive_results.size]], axis=0)

        print(
            f"\nDay {i}\t{round(time() - ti, 2)} s"
            f"\nCurrent infected: {infected.shape[1]}\tIsolating: {np.count_nonzero(infected[4] == 1)}\tRecovered: {np.sum(recovered)}"
            f"\nNatural immunity: {np.count_nonzero(uninfected[0] != 1)}\tVaccine 1st dose: {np.count_nonzero(uninfected[1] == vaccine_alpha[0])}\tVaccine 2nd dose: {np.count_nonzero(uninfected[1] == vaccine_alpha[1])}"
        )

        # Find contagious people
        contagious_indices = np.argwhere((infected[4] == 0) & (infected[3] <= i))[:, 0]

        # Run sim
        sim_return = sim(i, total_people - np.count_nonzero(infected[4] == 1), contagious_indices.size, np.count_nonzero((infected[4] == 0) & (infected[3] > i)), uninfected[:2, np.argwhere(uninfected[2] == 0)[:, 0]] * variant[i], buildings[i], density_avg[i])

        # Mark infected people as infected in uninfected array
        new_infection_indices = sim_return[0]
        uninfected_indices = np.argwhere(uninfected[2] == 0)[:, 0]
        uninfected[2, uninfected_indices[new_infection_indices]] = 1

        # Add infected people to infected array
        new_infections = np.append(new_infections, new_infection_indices.size)
        infected[5, contagious_indices[:sim_return[1]]] += new_infections[-1] / sim_return[1]

        # Define infection type for new infections
        infection_type = np.random.choice([0, 1], size=new_infections[-1], p=[asym_probability, 1 - asym_probability])
        incubation_period = np.rint(np.random.lognormal(*incubation, size=new_infections[-1])) + i
        presymptomatic_period = np.rint(np.where(infection_type == 0, 0, np.random.lognormal(*presym, size=new_infections[-1]))) + incubation_period
        contagious_period = np.rint(np.where(infection_type == 0, np.random.lognormal(*asym, size=new_infections[-1]), np.random.lognormal(*sym, size=new_infections[-1]))) + presymptomatic_period

        # Update infected array
        new_infected = np.array((
            np.ones(new_infections[-1]) * i,
            contagious_period,
            incubation_period,
            presymptomatic_period,
            np.zeros(new_infections[-1]),
            np.zeros(new_infections[-1]),
            uninfected_indices[new_infection_indices],
        ))

        infected = np.hstack((infected, new_infected))

        # Calculate test success rate
        if test_indices.size != 0:
            test_success = str(round(positive_results.size * 100 / test_indices.size, 2)) + ' %'
        else:
            test_success = 'N/A'

        results = {
            'uninfected': uninfected,
            'infected': infected,
            'new infections': new_infections,
            'recovered': recovered,
            'rt': rt,
            'total people': total_people,
            'tests': tests,
            'isolating': isolating,
            'vaccinated': vaccinated,
        }

        if rt.size != 0:
            print_rt = round(rt[-1, 1], 2)
        else:
            print_rt = 'N/A'

        # Save results
        with open(results_location.format(j) + f'/Day {i}', 'wb') as result_file:
            pickle.dump(results, result_file)

        print(f'New infections: {new_infections[-1]}\tTest success: {test_success}\tRt: {print_rt}')

    total_time = time() - ti

    if total_time > 60:
        print_time = str(int(total_time // 60)) + ':' + str(int(total_time % 60))
    else:
        print_time = f'{round(total_time, 2)} s'
    print(f'Total sim time: {print_time}')

    plot(new_infections, recovered, rt, tests, results['isolating'], results['vaccinated'])


# np.random.seed(2)
main()
