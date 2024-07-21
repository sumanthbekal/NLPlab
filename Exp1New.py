from collections import defaultdict
corpus = [
    "There is a big garden",
    "Children play in a garden",
    "They play inside beautiful garden"
]

bigram_counts = defaultdict(int)
unigram_counts = defaultdict(int)

for sentence in corpus:
    words = sentence.lower().split()
    for i in range(len(words) - 1):
        bigram_counts[(words[i], words[i + 1])] += 1
        unigram_counts[words[i]] += 1
    unigram_counts[words[-1]] += 1

test_sentence = "They play in a big garden".lower().split()
prob = 1.0
for i in range(len(test_sentence) - 1):
    bigram = (test_sentence[i], test_sentence[i + 1])
    prob *= bigram_counts[bigram] / unigram_counts[test_sentence[i]]
print(f"Probability: {prob}")