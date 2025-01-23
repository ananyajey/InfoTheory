import heapq
import numpy as np
from collections import Counter

'''
What is the mean length of a Huffman code for the English alphabet, constructed according to empirical let-
ter probabilities derived from the file shakespeare.txt? Use the letter frequencies you computed in Problem 4.2 of

Homework 1. (Do the same transformations to the file that Problem 5 of Homework 1 asked for.)
Suppose that you were to take the above-constructed Huffman code, but you used it to encode the file holmes.txt,
which contains a complete collection of the Sherlock Holmes stories. What would the mean length of the above code be,
now that the underlying empirical letter probabilities have changed? (For computing these new probabilities, do the same
transformations to the file as you did for shakespeare.txt, except do not delete any lines of the file.)

Note that I am not asking you to encode the actual contents of either file. Just compute the two mean lengths.

Finally, compare the two mean lengths to the empirical entropies of (i) a random letter in Shakespeare’s English and (ii) a
random letter in English as represented by the Sherlock Holmes stories. Comment on your findings.
'''
class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, nxt):
        return self.freq < nxt.freq

def print_codes(node, val='', code_lengths={}):
    newVal = val + str(node.huff)

    if node.left:
        print_codes(node.left, newVal, code_lengths)
    if node.right:
        print_codes(node.right, newVal, code_lengths)
    if not node.left and not node.right:
        print(f"{node.symbol} -> {newVal}")
        code_lengths[node.symbol] = len(newVal)
    
    return code_lengths


def huffman_coding(symbols, frequencies):
    nodes = []
    for x in range(len(symbols)):
        heapq.heappush(nodes, Node(frequencies[x], symbols[x]))

    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        left.huff = 0
        right.huff = 1

        newNode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        heapq.heappush(nodes, newNode)

    # Print the Huffman Codes and calculate their lengths
    code_lengths = print_codes(nodes[0])
    return code_lengths




def get_empirical_entropy(freqs):
    ps = freqs/sum(freqs)
    ps = ps[np.nonzero(ps)]

    logs = np.log2(1/ps)

    entropy = sum(ps*logs)
    return entropy

# Compute average codeword length and empirical entropy for shakespeare.txt
with open("hw2/shakespeare.txt", "r") as file:
    for _ in range(244):
        next(file)

    contents = file.read().upper()

counts_shakespeare = Counter(contents)

letters_shakespeare = []
frequencies_shakespeare = []
for letter in counts_shakespeare:
    if letter.isalpha():
        letters_shakespeare.append(letter)
        frequencies_shakespeare.append(counts_shakespeare[letter])


code_lengths = huffman_coding(letters_shakespeare, frequencies_shakespeare)
#print("Code lengths: ", code_lengths)


total_len_shakespeare = sum(freq * code_lengths[sym] for sym, freq in zip(letters_shakespeare, frequencies_shakespeare))
total_symbols_shakespeare = sum(frequencies_shakespeare)
avg_len_shakespeare = total_len_shakespeare / total_symbols_shakespeare

print(f"\nAverage Huffman Code Length for 'shakespeare.txt': {avg_len_shakespeare:.4f} bits")
print(f"\nEmpirical Entropy for 'shakespeare.txt': {get_empirical_entropy(np.array(frequencies_shakespeare)):.4f} bits")

# Compute average codeword length and empirical entropy for holmes.txt
with open("hw2/holmes.txt", "r") as file:
    contents = file.read().upper()

counts_holmes = Counter(contents)


letters_holmes = []
frequencies_holmes = []

total_length_holmes = 0
total_symbols_holmes = 0
for letter in counts_holmes:
    if letter.isalpha():
        letters_holmes.append(letter)
        frequencies_holmes.append(counts_holmes[letter])
        freq = counts_holmes[letter]

        #print(freq)
        total_length_holmes += freq * code_lengths.get(letter, 0)
        total_symbols_holmes += freq

average_length_holmes = total_length_holmes / total_symbols_holmes

print(f"\nAverage Huffman Code Length for 'holmes.txt' using codes from 'shakespeare.txt': {average_length_holmes:.4f} bits")
print(f"\nEmpirical Entropy for 'shakespeare.txt': {get_empirical_entropy(np.array(frequencies_holmes)):.4f} bits")





