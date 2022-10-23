import numpy as np
import pickle
from matplotlib import pyplot as plt
from time import time


def progress_bar(i, total):
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar = '█' * filled + '-' * (length - filled)
    print('Progress: |{}| {:.1f}%\t{} / {}'.format(bar, percent, i, total), end='\r')

    if i == total:
        print()


def main():
    result_number = 18
    days = 800
    isolating = np.empty((0, 2))
    vaccinated = np.empty((0, 2))
    rt_avg_new = np.empty((0, 2))
    ti = time()

    # with open(f'Project 1/Results/Sim Results {result_number}/Day {days}', 'rb') as result_file:
    #     results = pickle.load(result_file)

    # rt = results['rt']

    for i in range(0, days):
        progress_bar(i, days)
        with open(f'Project 1/Results/Sim Results {result_number}/Day {i}', 'rb') as result_file:
            results = pickle.load(result_file)

        uninfected = results['uninfected']
        infected = results['infected']
        new_infections = results['new infections']
        recovered = results['recovered']
        rt = results['rt']
        total_people = results['total people']
        buildings = results['buildings']
        tests = results['tests']
        isolating = results['isolating']
        vaccinated = results['vaccinated']

        # isolating = np.append(isolating, [[i, np.count_nonzero(infected[4])]], axis=0)
        # vaccinated = np.append(vaccinated, [[i, np.count_nonzero(uninfected[1] == 0.18)]], axis=0)
        # results['vaccinated'] = np.empty((0, 3))
        # results['isolating'] = np.count_nonzero(infected[4])

        with open(f'Project 1/Results/Sim Results {result_number}/Day {i}', 'wb') as result_file:
            pickle.dump(results, result_file)

    plt.figure(figsize=(16, 9), constrained_layout=True)
    # print(rt[0, 0])
    # rt_indices = [np.argwhere(rt[:, 0] == 30)[0, 0], np.argwhere(rt[:, 0] == 80)[0, 0]]
    # print(np.average(rt[rt_indices[0]:rt_indices[1], 1]), np.std(rt[rt_indices[0]:rt_indices[1], 1]))
    # plt.plot(rt[rt_indices[0]:rt_indices[1], 0], rt[rt_indices[0]:rt_indices[1], 1])
    # plt.plot(isolating[:, 0], isolating[:, 1])

    print(f'Time: {time() - ti} s')
    plt.show()


main()
