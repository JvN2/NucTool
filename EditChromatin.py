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
    dna601 = 'ACAGGATGTATATATCTGACACGTGCCTGGAGACTAGGGAGTAATCCCCTTGGCGGTTAAAACGCGGGGGACAGCGCGTACGTGCGTTTAAGCGGTGCTAGAGCTGTCTACGACCAATTGAGCGGCCTCGGCACCGGGATTCTCCAG'
    len601 = len(dna601)
    dna = [random.choice('ACGT') for x in range(dnalength)]
    dna = ['A'] * dnalength
    if dyads601 is not None:
        for d in dyads601:
            dna[d - len601 // 2: d - len601 // 2 + len601] = list(dna601)
    return ''.join(dna)


def calc_nucleosome_positions(seq, w, mu):
    folding_energy = calc_folding_energy(seq, w)
    relaxed_folding_energy = moving_average(folding_energy, 10)
    dyad_probability = vanderlick(relaxed_folding_energy, mu)
    nucleosome_occupancy = np.convolve(dyad_probability, np.ones(146), mode='same')
    return nucleosome_occupancy


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

    fig = plt.figure(figsize=(20, 3))
    plt.tight_layout()
    plt.ylim((-0.1, 1.1))
    plt.xlim((0, len(seq) - 1))
    plt.ylabel(r'occupancy')
    fig.subplots_adjust(bottom=0.2, top=0.8, left=0.05, right=0.98)
    title = filename.split(r'/')[-1]
    plt.text(0, 1.0, title, horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.plot(occupancy, color = 'grey')
    selection = [s.isupper() for s in seq]

    plt.plot(occupancy/selection, color= 'r')

    plt.show()


if __name__ == '__main__':
    filename = r'data/PHO5.fasta'
    seq = read_fasta(filename)
    flank_size = 1000
    flanked_seq = 'c' * flank_size + seq.upper() + 'c' * flank_size
    selection = [s.isupper() for s in flanked_seq]

    w = 74
    mu = -w*8.5/74
    # dna = create_dna(5000, 1000 + 197 * np.arange(16))
    # dna = create_dna(600, [300])
    nucleosome_occupancy = calc_nucleosome_positions(flanked_seq, w, mu)
    plot_occupancy(seq, nucleosome_occupancy[selection], filename)
