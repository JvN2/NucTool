import numpy as np
import math, re, random
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(x, w):
    average = np.convolve(x, np.ones(w), 'valid') / w
    average = np.concatenate((np.ones(w // 2) * average[0], average, np.ones(w - w // 2 - 1) * average[-1]), axis=None)
    return average


def conjugate_strand(watson, reverse=True):
    conjugate = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    crick = ''.join([conjugate[base] for base in watson.upper()])
    if not reverse:
        return crick
    else:
        return crick[::-1]


def get_probability_nucleosome_folding(w, p=10.1, b=0.2):
    df = pd.DataFrame()
    df['i'] = np.arange(w) - w // 2
    df['AA'] = 0.25 + b * np.sin(2 * math.pi * df['i'] / p)
    df['GC'] = 0.25 - b * np.sin(2 * math.pi * df['i'] / p)
    df['TA'] = df['TT'] = 0.25 + b * np.sin(2 * math.pi * df['i'] / p)

    df['CA'] = df['CC'] = df['CG'] = df['CT'] = df['i'] * 0 + 0.25

    df['AC'] = df['AG'] = df['AT'] = (1 - df['AA']) / 3
    df['GA'] = df['GG'] = df['GT'] = (1 - df['GC']) / 3
    df['TC'] = df['TG'] = (1 - (df['TA'] + df['TT'])) / 2

    E_bind = 3.5
    df['none'] = [np.exp(-E_bind) if i % 10 == 5 else 1 for i in df['i']]

    df.set_index('i', inplace=True)
    df.sort_index(axis=1, inplace=True)
    return df


def calc_folding_energy(seq, window):
    seq = seq.upper()
    df = get_probability_nucleosome_folding(window)
    reversed_seq = conjugate_strand(seq, reverse=False)
    p_forward = p_reverse = np.zeros(len(seq)) + 1e-99

    for di in range(len(seq) - 2):
        p_forward[di] = np.prod(
            [4 * df.at[i, seq[i + di:i + di + 2]] if 0 < i + di < len(seq) - 3 else df.at[i, 'none'] for i in df.index])
        p_reverse[di] = np.prod(
            [4 * df.at[i, reversed_seq[i + di:i + di + 2]] if 0 < i + di < len(seq) - 3 else df.at[i, 'none'] for i in
             df.index])

    E = -(p_reverse * np.log(p_reverse) + p_forward * np.log(p_forward)) / (p_reverse + p_forward)
    E[-2:] = E[-3]
    return E


def vanderlick(Energy, mu):
    E_out = Energy + mu
    footprint = 147

    forward = np.zeros(len(Energy))
    for i in range(len(Energy)):
        tmp = sum(forward[max(i - footprint, 0):i])
        forward[i] = np.exp(-E_out[i] - tmp)

    backward = np.zeros(len(Energy))
    r_forward = forward[::-1]
    for i in range(len(Energy)):
        backward[i] = 1 - sum(r_forward[max(i - footprint, 0):i] * backward[max(i - footprint, 0):i])

    P = forward * backward[::-1]
    return P


def create_dna(dnalength, dyads601=None):
    seq601 = 'ACAGGATGTATATATCTGACACGTGCCTGGAGACTAGGGAGTAATCCCCTTGGCGGTTAAAACGCGGGGGACAGCGCGTACGTGCGTTTAAGCGGTGCTAGAGCTGTCTACGACCAATTGAGCGGCCTCGGCACCGGGATTCTCCAG'
    seq = [random.choice('ACGT') for x in range(dnalength)]
    for dyad in dyads601:
        seq = insert_seq(seq, seq601, dyad)
    return ''.join(seq)


def calc_nucleosome_positions(seq, w, mu, flank_size=1000):
    flanked_seq = 'c' * flank_size + seq + 'c' * flank_size
    folding_energy = calc_folding_energy(flanked_seq, w)
    relaxed_folding_energy = moving_average(folding_energy, 10)
    dyad_probability = vanderlick(relaxed_folding_energy, mu)
    nucleosome_occupancy = np.convolve(dyad_probability, np.ones(146), mode='same')
    return nucleosome_occupancy[flank_size:flank_size + len(seq)]


def read_fasta(filename, name=None):
    with open(filename, 'r') as f:
        content = f.readlines()
        seq = ''
        for line in range(len(content)):
            if content[line][0] != '>':
                seq = seq + content[line]
    seq = ''.join([s for s in seq if s.upper() in ['A', 'C', 'G', 'T']])
    return seq


def plot_occupancy(seq, occupancy, filename=''):
    xlabels = np.asarray([s + f'\n:\n{i + 1}' if (i + 1) % 10 == 0 else s for i, s in enumerate(seq)])
    x = np.arange(len(seq))
    # plt.xticks(x, xlabels)

    # plt.scatter(np.arange(len(selection))[selection], occupancy[selection], color='r', s=6)


def plot_dinuc_probs(df):
    nucleotides = list('ACTG')
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
    for i, n1 in enumerate(nucleotides):
        for n2 in nucleotides:
            if n1 + n2 in ['AA', 'TT', 'TA', 'GC']:
                color = 'red'
            else:
                color = 'grey'
            # ax[i].plot(df.index, df[n1 + n2]*df[conjugate_strand(n1+n2, False)], label=n1 + n2)#, color=color)
            ax[i].plot(df.index, df[n1 + n2], label=n1 + n2, color=color)
            ax[i].legend()
            ax[i].set_ylim(0, 0.5)
            # Put a legend to the right of the current axis
            ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def create_new_seq(df, power=1):
    nucleotides = list('ACGT')
    seq = [np.random.choice(nucleotides)]
    for i, nucleotide_probability in df.iterrows():
        p = np.asarray(nucleotide_probability[[seq[-1] + b for b in nucleotides]]) ** power
        seq.append(np.random.choice(nucleotides, p=p / np.sum(p)))
    return ''.join(seq)


def insert_seq(seq, insert, position, show=False):
    keep = np.asarray(list(seq)) == np.asarray(list(seq.upper()))
    seq = np.asarray(list(seq))
    new_seq = seq.copy()
    change = np.arange(len(insert)) - len(insert) // 2 + position
    new_seq[change] = list(insert)
    new_seq[keep] = seq[keep]

    if show:
        print(f'\n\n{"".join(seq[change])}\n->\n{"".join(new_seq[change])}')

    return ''.join(new_seq)


if __name__ == '__main__':
    filename = r'data/PHO5.fasta'
    w = 100
    mu = -w * 8.0 / 74
    seq = read_fasta(filename)

    # seq = create_dna(len(seq), 300 + np.arange(10) * 197)
    # mu = -10

    new_seq = seq
    for i in [713, 904]:
    # for i in []:
        insert = create_new_seq(get_probability_nucleosome_folding(w, 10.1), -20)
        # pA = 0.6
        # p = [pA] + [(1 - pA) / 3] * 3
        # insert = ''.join(np.random.choice(list('ACGT'), w, p=p))
        pos = 2425

        insert = seq[pos - w//2: pos+w].upper()
        new_seq = insert_seq(new_seq, insert, i+17, show=True)

    fig = plt.figure(figsize=(13, 2))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.8, left=0.05, right=0.98)
    plt.ylim((0, 1))
    plt.xlim((0, len(seq) - 1))
    plt.ylabel(r'Occupancy')

    title = filename.split(r'/')[-1]
    plt.text(0, 1.0, title, horizontalalignment='left', verticalalignment='bottom',
             transform=plt.gca().transAxes)

    selection = [b.isupper() for b in seq]
    plt.fill(np.arange(len(seq)), selection, alpha=0.3)
    for s, line in zip([seq, new_seq], ['-', '--']):
        nucleosome_occupancy = calc_nucleosome_positions(s, w, mu)
        plt.plot(nucleosome_occupancy, color='grey', linestyle=line)
    mutations = [1 if n1 != n2 else 0 for n1, n2 in zip(seq, new_seq)]
    plt.fill(np.arange(len(seq)), mutations, color='red', alpha=0.3)
    plt.show()
