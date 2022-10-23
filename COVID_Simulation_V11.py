"""
Random walk simulation for COVID-19

Classes
-------
COVIDWalk
    Class that holds the instance and calculates the random walk

Methods
-------
main(day, total_people, num_contagious, initial_infections, alpha, buildings, density_avg)
    Main function to create the environment and run the simulation for random walk, returning the indices of new infections and number of active contagious people
"""


import os
import pickle
import cupy as cp
import numpy as np
import multiprocessing as mp
from PIL import Image
from typing import Tuple
from time import time, sleep
from scipy.stats import norm
from matplotlib import pyplot as plt


class COVIDWalk:
    """
    Simulates random walk and calculates new infections between people

    Attributes
    ----------
    travel_xy : ndarray
        x & y position of each person for each step in time
    infection_step : ndarray
        Infection event containing uninfected person index and the step when they were infected
    time : float

    Methods
    -------
    simulations()
        Runs random walk and calculates infection events
    plot_initialise()
        Initialises a figure and starting conditions for an animation
    plot(i)
        Plots all positions of people for one frame, i
    """
    def __init__(self, contagious: np.ndarray, alpha: np.ndarray, size: float, max_speed: float, steps: int, duration: float):
        """
        Defines initial variables for simulation instance

        Parameters
        ----------
        contagious : ndarray
            Infection status of each person
        aplha : ndarray
            Alpha of each person
        size : float
            Building size
        max_speed : float
            Maximum step size
        steps : int
            Maximum number of steps
        duration : float
            Contact duration for infection probability
        """
        self.tail_length = 2
        self.contagious = contagious
        self.alpha = alpha
        self.size = size
        self.steps = steps
        self.duration = duration
        self.people = contagious.size
        self.lines = np.array(())
        self.annotations = np.array(())
        self.colours = np.where(contagious == 1, 'r', 'g')
        self.colours = np.where(contagious == 2, 'y', self.colours)
        self.travel_xy = np.expand_dims(np.random.uniform(size=(self.people, 2)), axis=2) * size

        #Calculate step properties for each simulated step for each person
        step_length = np.random.uniform(high=max_speed, size=(self.people, steps))
        step_direction = np.random.uniform(0, 2 * np.pi, size=(self.people, steps))

        self.steps_xy = np.stack((step_length * np.cos(step_direction), step_length * np.sin(step_direction)), axis=1)


    def simulation(self):
        """
        Simulates random walk and calculates infection events
        """
        # Initial variables
        mempool = cp.get_default_memory_pool()
        contagious_indices = np.argwhere(self.contagious == 1)

        # Calculate position of each person at each step
        self.travel_xy = np.dstack((self.travel_xy, self.travel_xy + np.cumsum(self.steps_xy, axis=2))) % self.size

        # Calculate the distance between each uninfected person to each infected person
        uninfected = np.delete(self.travel_xy, np.argwhere(self.contagious != 0), axis=0)
        uninfected_index = np.squeeze(np.argwhere(self.contagious == 0), axis=1)
        distance = np.sqrt((self.travel_xy[contagious_indices, 0, :] - uninfected[:, 0]) ** 2 + (self.travel_xy[contagious_indices, 1, :] - uninfected[:, 1]) ** 2)

        # Calculate if an uninfected person gets infected at each step
        cu_distance = cp.array(distance)
        cu_alpha = cp.multiply(cp.expand_dims(cp.asarray(self.alpha), axis=1), cp.ones(cu_distance.shape))
        cu_infect_probability = covid_probability(cu_distance, alpha=cu_alpha, T=self.duration)
        cu_rand_numbers = cp.random.rand(*cu_infect_probability.shape)
        cu_infection = cp.where(cu_rand_numbers < cu_infect_probability, 1, 0)

        infection_event = cp.asnumpy(cp.argwhere(cu_infection == 1))
        mempool.free_all_blocks()

        # Format infection event to uninfected index and step index
        self.contagious[uninfected_index[np.unique(infection_event[:, 1])]] = 2
        self.infection_step = np.array([(uninfected_index[i], np.min(infection_event[infection_event[:, 1] == i, 2])) for i in np.unique(infection_event[:, 1])])


    def plot_initialise(self):
        """
        Initialise plot by plotting starting positions of each person and create array for the line of each person
        """
        self.scat = plt.scatter(self.travel_xy[:, 0, 0], self.travel_xy[:, 1, 0], c=self.colours)

        for i in range(self.people):
            self.lines = np.append(self.lines, plt.plot([], [], c=self.colours[i]))
            self.annotations = np.append(self.annotations, plt.annotate(i, (self.travel_xy[i, 0], self.travel_xy[i, 1])))

        plt.xlim(0, self.size)
        plt.ylim(0, self.size)


    def plot(self, i: int):
        """
        Plot frame with lines showing the previous step of each person and the current location via a scatter point

        Parameters
        ----------
        i : int
            Frame number
        """
        for j in range(self.people):
            if np.argwhere(self.infection_step[:, 0] == j).size != 0:
                if i >= self.infection_step[np.argwhere(self.infection_step[:, 0] == j), 1]:
                    self.colours[j] = 'y'

            self.lines[j].set_color(self.colours[j])
            self.lines[j].set_data(self.travel_xy[j, 0, np.clip(i - self.tail_length, a_min=0, a_max=i):i + 1], self.travel_xy[j, 1, np.clip(i - self.tail_length, a_min=0, a_max=i):i + 1])
            self.annotations[j].remove()
            self.annotations[j] = plt.annotate(j, (self.travel_xy[j, 0, i], self.travel_xy[j, 1, i]))

        self.scat.remove()
        self.scat = plt.scatter(self.travel_xy[:, 0, i], self.travel_xy[:, 1, i], c=self.colours)
        plt.title(f'Frame {i}')
        plt.savefig(f'Project 1/Plots & Animations/Frames/Frame {i}')


def clipped_gaussian(mean: float, std: float, lower: float, upper: float, size: int):
    """
    Randomly generates values from a Gaussian distribution between lower and upper limits

    Parameters
    ----------
    mean : float
        Mean of Gaussian distribution
    std : float
        Standard deviation of Gaussian distribution
    lower : float
        Lower limit of Gaussian distribution
    upper : float
        Upper limit of Gaussian distribution
    size : int
        Number of random numbers to generate

    Returns
    -------
    ndarray
        Random values from Gaussian distribution
    """
    y_values = np.random.uniform(norm.cdf(lower, loc=mean, scale=std), norm.cdf(upper, loc=mean, scale=std), size=size)
    return norm.ppf(y_values, loc=mean, scale=std)


def walking_velocity(r: np.ndarray, u_max=1.4, r_limit_1: list=[0.89, 3], r_limit_2: list=[1.33, 2], a=0.24, b=0.025, g=0.27) -> np.ndarray:
    """
    Calculates the maximum velocity for a crowded room

    Parameters
    ----------
    r : ndarray
        Densities
    u_max : float, default = 1.4
        Maximum walking velocity
    r_limit_1 : list, default = [0.89, 3]
        Critical and maximum values for front-back densities
    r_limit_2 : list, default = [1.33, 2]
        Critical and maximum values for lateral densities
    a : float, default = 0.24
        Alpha weight
    b : float, default = 0.025
        Beta weight
    g : float, default = 0.27
        Gamma weight

    Returns
    -------
    u : ndarray
    Maximum walking velocities per building
    """
    r = np.where(r < 0.01, 0.01, r)

    r_star_1 = (r_limit_1[0] + r_limit_1[1]) / 2
    r_star_2 = (r_limit_2[0] + r_limit_2[1]) / 2
    A = np.log(r_limit_1[1] * r_star_2 / r) / np.log(r_limit_1[1] / r_limit_1[0])
    B = (r_star_1 * r_limit_2[1] - r) / (r_star_1 * (r_limit_2[1] - r_limit_2[0]))
    return np.clip(u_max * (a * A + b * B + g), a_min=0, a_max=u_max)


def gaussian(x: np.ndarray, A=1., mean=0., std=1.):
    """
    Gaussian distribution with given amplitude

    Parameters
    ----------
    x : ndarray
        Input values
    A : float, default = 1
        Amplitude
    mean : float, default = 0
        Mean of Gaussian distribution
    std : float, default = 1
        Standard deviation of Gaussian distribution

    Returns
    -------
    y : ndarray
        Output values
    """
    return A * np.exp(-0.5 * ((x - mean) / std) ** 2)


def covid_probability(x: np.ndarray, alpha=1., T=60., B=0.54 / 3600, l=7e6 * 1e6, vB=0.1 * 0.1 * 0.15, N0=900, a=0.8e12, b=0.1e12, c=0.45e12) -> np.ndarray:
    """
    Calculates the probability of infection from a given distance

    Parameters
    ----------
    x : ndarray
        Distance values
    alpha : float, default = 1
        Product of transmissibility and protection
    T : float, default = 60
        Contact duration in seconds
    B : float, default = 0.54 / 3600
        Breathing rate in m^3 s^-1
    l : float, default = 7e12
        Virial count per m^3
    vB : float, default = 0.1 * 0.1 * 0.15 m^3
        Breathing volume in m^3
    N0 : int, default = 900
        Average number of virions for an infection
    a : float, default = 0.8e12
        Polynomial approximation x^2 coefficient
    b : float, default = 0.1e12
        Polynomial approximation x^1 coefficient
    c : float, default = 0.45e12
        Polynomial approximation x^0 coefficient

    Returns
    -------
    P : ndarray
        Probability of COVID-19 infection
    """
    vd = (a * x ** 2 + b * x + c) ** -1
    N = B * l * vd * T / vB
    return 1 - np.exp(-alpha * N / N0)


def progress_bar(i: int, total: int):
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar = '█' * filled + '-' * (length - filled)
    print('Progress: |{}| {:.1f}%\t'.format(bar, percent), end='\r')

    if i == total:
        print()


def generate_data(sim_params: dict, building_index: int, results: dict):
    """
    Runs random walk simulation and collects results

    Parameters
    ----------
    sim_params : dict
        Parameters to pass into the simulation
    building_index : int
        Current building number
    results : dict
        Current results file

    Returns
    -------
    sim : callable
        Simulation instance
    results : dict
        Results file
    """
    # Initialise and run simulation
    sim = COVIDWalk(sim_params['contagious'], sim_params['alpha'], sim_params['size'], sim_params['max speed'], sim_params['steps'], sim_params['duration'])
    sim.simulation()

    result = {
        'sim parameters': sim_params,
        'travel': sim.travel_xy,
        'infections': sim.infection_step
    }

    # Save current building result to results file
    results[f'Building {building_index}'] = result

    return sim, results


def load_data(building_index: int, results: dict) -> Tuple[COVIDWalk, dict]:
    """
    Loads simulation instance for a given building

    Parameters
    ----------
    building_index : int
        Current building number
    results : dict
        Current results file

    Returns
    -------
    sim : callable
        Simulation instance
    sim_params : dict
        Simulation parameters from simulation instance
    """
    if f'Building {building_index}' in results:
        result = results[f'Building {building_index}']
        sim_params = result['sim parameters']
        sim = COVIDWalk(sim_params['contagious'], sim_params['alpha'], sim_params['size'], sim_params['max speed'], sim_params['steps'], sim_params['duration'])

        sim.travel_xy = result['travel']
        sim.infection_step = result['infections']

        return sim, sim_params


def generate_frames(sim: COVIDWalk, sim_params: dict):
    """
    Genereates each frame in the animation

    Parameters
    ----------
    sim : callable
        Simulation instance
    sim_params : dict
        Parameters for the simulation instance
    """
    print('\nGenerating frames...')
    pool = mp.Pool(mp.cpu_count())
    tm = time()

    # Create multiprocessing tasks to plot each frame
    plt.figure(figsize=(16,9))
    sim.plot_initialise()
    job = pool.map_async(sim.plot, [i for i in range(sim_params['steps'])])
    previous_number_left = job._number_left

    # Check progress using progress bar
    while job._number_left > 0:
        if job._number_left != previous_number_left:
            progress_bar(sim_params['steps'] - job._number_left * job._chunksize, sim_params['steps'])

        previous_number_left = job._number_left
        sleep(0.5)

    pool.close()
    pool.join()

    print('\n{:.2f} s'.format(time() - tm))


def create_animation(sim_params: dict, index: int):
    """
    Stiches all animation frames into a GIF

    Parameters
    ----------
    sim_params : dict
        Simulation instance parameters
    index : int
        Animation file number
    """
    print('\nCreating animation...')
    time_ms = 1e4
    tm = time()

    frames = (Image.open(f'Project 1/Plots & Animations/Frames/Frame {i}.png') for i in range(sim_params['steps']))
    frame_1 = next(frames)
    frame_1.save(f'Project 1/Plots & Animations/COVID Animation {index + 1}.gif', format='gif', append_images=frames, save_all=True, loop=True, duration=max((time_ms / sim_params['steps']), 100))
    print('{:.2f} s'.format(time() - tm))


def building_calculations(i: int, buildings: int, load: bool, animation: bool, contagious: np.ndarray, cumsum_people: np.ndarray, alpha: np.ndarray, area: np.ndarray, speed: np.ndarray, results: dict={}):
    """
    Processes a given building

    Parameters
    ----------
    i : int
        Building index
    buildings : int
        Total number of buildings
    load : bool
        If loading building simulation instance
    animation : bool
        If running animtion for the building
    contagious : ndarray
        Infection type for each person
    cumsum_people : ndarray
        Total number of people up to the current building
    alpha : ndarray
        Product of transmissibility and protection
    area : ndarray
        Area for each building
    speed : ndarray
        Maximum speed for each building
    results : dict, default = {}
        Current results file
    """
    progress_bar(i, buildings)
    alpha_indices = [np.count_nonzero(contagious[:cumsum_people[i]] == 0), np.count_nonzero(contagious[:cumsum_people[i + 1]] == 0)]

    sim_params = {
        'contagious': contagious[cumsum_people[i]:cumsum_people[i + 1]],
        'alpha': alpha[alpha_indices[0]:alpha_indices[1]],
        'size': np.sqrt(area[i]),
        'max speed': speed[i],
        'steps': 50,
        'duration': 0.5,
    }

    if load:
        returns = load_data(i, results)

        if returns is not None:
            sim, sim_params = returns
        else:
            return

    # Run simulation if there are contagious and uninfected people
    elif np.argwhere(sim_params['contagious'] != 0).size != 0 and np.argwhere(sim_params['contagious'] == 0).size != 0:
        sim, results = generate_data(sim_params, i, results)

    if animation & (np.argwhere(sim_params['contagious'] != 0).size != 0 and np.argwhere(sim_params['contagious'] == 0).size != 0):
        generate_frames(sim, sim_params)
        create_animation(sim_params, i)


def plot_buildings(density: np.ndarray, area: np.ndarray, people: np.ndarray, speed: np.ndarray, bins=10):
    """
    Plot building property distributions

    Parameters
    ----------
    density : ndarray
        Distribution of building densities
    area : ndarray
        Distribution of building areas
    people : ndarray
        Distribution of number of people per building
    speed : ndarray
        Distribution of maximum speeds per building
    bins : int, default = 10
        Number of bins per histogram plot
    """
    major_size = 20
    minor_size = 16

    # Density
    plt.figure(figsize=(16, 9))
    ax = plt.subplot(221)
    ax.set_title(r'a) Density $(People/m^2)$', fontsize=major_size)
    plt.hist(density, bins=bins)
    plt.xticks(fontsize=minor_size)
    plt.yticks(fontsize=minor_size)

    # Building sizes
    ax = plt.subplot(222)
    ax.set_title('b) Length (m)', fontsize=major_size)
    plt.hist(np.sqrt(area), bins=bins)
    plt.xticks(fontsize=minor_size)
    plt.yticks(fontsize=minor_size)

    # People per building
    ax = plt.subplot(223)
    ax.set_title('c) People per building', fontsize=major_size)
    ax.text(0.9, 0.9, f'Total number: {np.sum(people)}', fontsize=minor_size, ha='right', va='center', transform=ax.transAxes)
    plt.hist(people, bins=bins)
    plt.xticks(fontsize=minor_size)
    plt.yticks(fontsize=minor_size)

    # Max speed per building
    ax = plt.subplot(224)
    ax.set_title(r'd) Max speed per building $(ms^{-1})$', fontsize=major_size)
    plt.hist(speed, bins=bins)
    plt.xticks(fontsize=minor_size)
    plt.yticks(fontsize=minor_size)

    plt.tight_layout()
    plt.show()


def plot_maximum_speed():
    """
    Plots the relationship between maximum walking velocity and building density
    """
    x = np.linspace(0.02, 2, 1000)
    y = walking_velocity(x, u_max=1.3)

    plt.figure(figsize=(16, 8), constrained_layout=True)
    plt.plot(x, y)
    plt.xlabel(r'Density $(people/m^2)$', fontsize=20)
    plt.ylabel(r'Walking speed $(ms^{-1})$', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


def plot_covid_probability():
    """
    Plots infection probability against distance from infected individual
    """
    x = np.linspace(0, 5, 1000)
    y = covid_probability(x, T=15 * 60)

    plt.plot(x, y)
    plt.xlabel('Distance (m)', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


def main(day: int, total_people: int, num_contagious: int, initial_infections: int, alpha: np.ndarray, buildings: int, density_avg: float):
    """
    Main function to run random walk for each building

    Parameters
    ----------
    day : int
        Current day
    total_people : int
        Total number of people
    num_contagious : int
        Number of contagious people
    initial_infections : int
        Number of infected, but not contagious, people
    alpha : ndarray
        Transmissibility and protection
    buildings : int
        Number of buildings
    density_avg : float
        Average density of the buildings

    Returns
    -------
    new_infections : ndarray
        Indices of new infections
    active_contagious : int
        Number of active contagious people
    """
    # Variables
    load = animation = False
    save = False

    # Constants
    cp.random.seed(0)
    j = 1
    density_std = 0.5
    area_avg = 164
    area_std = 164
    ti = time()
    new_infections = np.array((), dtype=int)
    alpha = np.min(alpha, axis=0)
    results = {}
    results_location = 'Project 1/Results/Run Result {} Day {}'

    # Calculate building properties
    density = clipped_gaussian(density_avg, density_std, 0.02, 2, buildings)
    area = clipped_gaussian(area_avg, area_std, 20, 2000, buildings)
    people = (area * density).astype(int)

    # Recaultes building properties if there aren't enough people
    if np.sum(people) > total_people:
        print('\nNot enough people!')
        people = (people.astype(float) * total_people / np.sum(people)).astype(int)
        density = people / area

    # Calcualtes maximum walking velocities and people per building
    speed = walking_velocity(density, u_max=1.3)
    cumsum_people = np.insert(np.cumsum(people), 0, 0)

    # Create array of infection types
    if num_contagious >= total_people:
        contagious = np.ones(total_people)
    else:
        contagious_indices = np.random.choice(total_people, size=num_contagious, replace=False)
        contagious = np.zeros(total_people)
        contagious[contagious_indices] = 1
        infected_indices = np.random.choice(np.argwhere(contagious == 0)[:, 0], size=initial_infections, replace=False)
        contagious[infected_indices] = 2

    # Number of people with different infection types
    active_uninfected = np.count_nonzero(contagious[:cumsum_people[-1]] == 0)
    active_contagious = np.count_nonzero(contagious[:cumsum_people[-1]] == 1)
    active_infected = np.count_nonzero(contagious[:cumsum_people[-1]] == 2)
    print(f'Active people: {np.sum(people)}\tUninfected: {active_uninfected}\tContagious: {active_contagious}\tInfected: {active_infected}')

    # Find most recent file
    while os.path.exists(results_location.format(j, 0)):
        j += 1

    if load:
        with open(results_location.format(j - 1, day), 'rb') as result_file:
            results = pickle.load(result_file)

    # Run simulation for each building
    for i in range(buildings):
        building_calculations(i, buildings, load, animation, contagious, cumsum_people, alpha, area, speed, results)

        # Record new infections
        if f'Building {i}' in results.keys():
            if results[f'Building {i}']['infections'].size != 0:
                uninfected_index = results[f'Building {i}']['infections'][:, 0] + cumsum_people[i] - np.count_nonzero(contagious[:cumsum_people[i]] != 0) + new_infections.size
                new_infections = np.append(new_infections, uninfected_index)

    # Save results
    if not(load) and save:
        with open(results_location.format(j, day), 'wb') as result_file:
            pickle.dump(results, result_file)

    print('Total time: {:.2f} s'.format(time() - ti))

    return new_infections, active_contagious


if __name__ == '__main__':
    np.random.seed(0)
    main(0, int(250e3), 50000, 50000, np.ones((2, 150000)), 1000, 0.86)
