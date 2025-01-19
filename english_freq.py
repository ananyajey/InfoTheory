import pandas as pd
import numpy as np
from collections import Counter


def get_empirical_entropy(freqs):
    ps = freqs/sum(freqs)
    ps = ps[np.nonzero(ps)]

    logs = np.log2(1/ps)

    entropy = sum(ps*logs)
    return entropy



# Question 4.1: Compute the empirical entropy of the English alphabet given the frequency table in english_freq.dat.
df = pd.read_csv('english_freq.dat', header=None, index_col=False, delimiter=' ')
freqs = df[1].to_numpy()
english_freq_entropy = get_empirical_entropy(freqs)

print(f"Question 3.1: {english_freq_entropy: 0.3f}")



# Question 4.2: Compute the empirical entropy of the English alphabet by computing a frequency table using shakespeare.txt.
with open("shakespeare.txt", "r") as f:
    contents = f.read().upper()

counts = Counter(contents)

freqs_sp = []
for letter in counts:
    if letter.isalpha():
        freqs_sp.append(counts[letter])

shakespeare_entropy = get_empirical_entropy(np.array(freqs_sp))

print(f"Question 3.2: {shakespeare_entropy: 0.3f}")



# Question 5: Letting XY be a random two-letter substring drawn from Shakespeare’s English, compute the quantities H(X), H(Y ), H(Y | X), and I(X : Y ).
with open("shakespeare.txt", "r") as file:
    for _ in range(244):
        next(file)

    contents = file.read().upper()

special_chars = [x for x in set(contents) if not x.isalpha()]

for ch in special_chars:
    contents = contents.replace(ch, "$")

while "$$" in contents:
    contents = contents.replace("$$", "$")

letters = list(set(contents))
letter_index_dict = {letter: index for index, letter in enumerate(letters)}

transition_table = np.zeros((len(letters), len(letters)))

for i in range(len(contents) - 1):
    transition_table[letter_index_dict[(contents[i])]][letter_index_dict[(contents[i+1])]] += 1


H_X = get_empirical_entropy(np.sum(transition_table, axis=1))
H_Y = get_empirical_entropy(np.sum(transition_table, axis=0))

transition_probs = (transition_table.T/np.sum(transition_table, axis=1)).T
h_x_axis = np.array([get_empirical_entropy(x) for x in transition_probs])
x_probs = np.sum(transition_table, axis=1)/np.sum(np.sum(transition_table))
H_Y_given_X = np.sum(h_x_axis * x_probs)

I_X_Y = H_Y - H_Y_given_X

print(f"Question 4: \nH(X) = {H_X: 0.3f}\nH(Y) = {H_Y: 0.3f}\nH(Y|X) = {H_Y_given_X: 0.3f}\nI(X:Y) = {I_X_Y: 0.3f}")
