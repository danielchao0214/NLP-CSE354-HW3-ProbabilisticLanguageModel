import sys
import gzip
import json #for reading json encoded files into opjects
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch
from collections import Counter
from pprint import pprint

sys.stdout = open('a3_chao_112412719_OUTPUT.txt', 'w')

# # ## # #
# PART 2 #
# # ## # #

# PART 2.1
def loadData(filename):
    file = open(filename, "r", encoding='utf-8')
    data = []

    word_counts = {}
    
    for line in file:
        line_id, line_sense, line_context = re.split(r'\t', line)
        line_context = line_context.lower()
        line_context = line_context.split(' ')

        for i in range(0, len(line_context)):
            # remove excess info
            context_word = re.sub(r'/[^/]*/[^/<]*', '', line_context[i])
            # remove head tag
            if("<head>" in context_word):
                context_word = context_word[6:-7]
            # put word
            line_context[i] = context_word
            # word counts
            if(context_word in word_counts):
                word_counts[context_word] = word_counts[context_word] + 1
            else:
                word_counts[context_word] = 1

        line_context = ["<s>"] + line_context + ["<\s>"]    
        word_counts["<s>"] = word_counts.get("<s>",0)+1
        word_counts["<\s>"] = word_counts.get("<\s>",0)+1
        data.append(line_context)
    
    word_counts_popular = sorted(word_counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    return data, word_counts_popular

# PART 2.2
def extractUnigram(data, frequent_words):
    unigramCounts = {}
    for sentence in data:
        for i in range(0, len(sentence)):
            word = sentence[i]
            if(word in frequent_words):
                unigramCounts[word] = unigramCounts[word]+1 if word in unigramCounts else 1
            else:
                unigramCounts["<OOV>"] = unigramCounts["<OOV>"]+1 if "<OOV>" in unigramCounts else 1
    return unigramCounts

def extractBigram(data, frequent_words):
    bigramCounts = {}
    for sentence in data:
        for i in range(0, len(sentence)-1):
            word0 = sentence[i] if sentence[i] in frequent_words else "<OOV>"
            word1 = sentence[i+1] if sentence[i+1] in frequent_words else "<OOV>"
            if(word0 in bigramCounts):
                if(word1 in bigramCounts[word0]):
                    bigramCounts[word0][word1] = bigramCounts[word0][word1]+1
                else:
                    bigramCounts[word0][word1] = 1
            else:
                bigramCounts[word0] = {word1:1}
    return bigramCounts

def extractTrigram(data, frequent_words):
    trigramCounts = {}
    for sentence in data:
        for i in range(0, len(sentence)-2):
            word0 = sentence[i] if sentence[i] in frequent_words else "<OOV>"
            word1 = sentence[i+1] if sentence[i+1] in frequent_words else "<OOV>"
            word2 = sentence[i+2] if sentence[i+2] in frequent_words else "<OOV>"
            if((word0,word1) in trigramCounts):
                if(word2 in trigramCounts[(word0,word1)]):
                    trigramCounts[(word0,word1)][word2] = trigramCounts[(word0,word1)][word2]+1
                else:
                    trigramCounts[(word0,word1)][word2] = 1
            else:
                trigramCounts[(word0,word1)] = {word2:1}
    return trigramCounts

# PART 2.3
def bigramProbability(wordMinus1):
    bigramWordFreq = bigramCounts.get(wordMinus1,0)
    probDict = {}
    for word in bigramWordFreq:
        prob = (bigramWordFreq[word]+1)/(5000 + sum(bigramWordFreq.values()))
        probDict[word] = prob
    return probDict

def trigramProbability(wordMinus1, wordMinus2):
    trigramWordFreq = trigramCounts.get((wordMinus2, wordMinus1), 0)
    probDict = {}
    for word in trigramWordFreq:
        prob = (trigramWordFreq[word]+1)/(bigramCounts[wordMinus2][wordMinus1] + 5000)
        probDict[word] = prob
    return probDict

def languageModelProbability(wordMinus1, wordMinus2 = None):
    probDict = bigramProbability(wordMinus1)
    if(wordMinus2 != None):
        for word in probDict:
            probDict[word] = probDict[word]/2
        triDict = trigramProbability(wordMinus1, wordMinus2)
        for word in triDict:
            probDict[word] = probDict[word] + triDict[word]/2
    return probDict

def languageModelProbabilityWord(word, wordMinus1, wordMinus2 = None):
    probDict = languageModelProbability(wordMinus1,wordMinus2)
    if word in probDict:
        return probDict[word]
    else:
        return "Not valid Wi"

# PART 2.4
def generate(words):
    flag = True
    if words == ["<s>"]:
        prob = bigramProbability("<s>")
        sum_prob = sum(prob.values())
        for word in prob:
            prob[word] = prob[word] / sum_prob
        new_word = np.random.choice(list(prob.keys()), p=list(prob.values()))
        words.append(new_word)
        if(new_word == "<\s>"):
            flag = False
    while flag:
        prob = languageModelProbability(words[-1], words[-2])
        sum_prob = sum(prob.values())
        for word in prob:
            prob[word] = prob[word] / sum_prob
        new_word = np.random.choice(list(prob.keys()), p=list(prob.values()))
        words.append(new_word)
        if(new_word == "<\s>" or len(words) == 32):
            flag = False
    return words


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("USAGE: python3 a3_lastname_id.py onesec_train.tsv")
        sys.exit(1)
    filename_train = sys.argv[1]

    # PART 2.1
    data, word_counts_dict = loadData(filename_train)
    frequent_words = [key for (key, value) in word_counts_dict]
    frequent_words = frequent_words[0:5000]

    # PART 2.2
    global unigramCounts 
    unigramCounts = extractUnigram(data, frequent_words)
    global bigramCounts
    bigramCounts = extractBigram(data, frequent_words)
    global trigramCounts
    trigramCounts = extractTrigram(data, frequent_words)

    print("CHECKPOINT 2.2 - counts")
    print("  1 grams:")
    print("    ('language')", unigramCounts.get("language", 0))
    print("    ('the')", unigramCounts.get("the", 0))
    print("    ('formal')", unigramCounts.get("formal", 0))
    print("  2 grams:")
    print("    ('the', 'language')", bigramCounts.get("the", 0).get("language", 0))
    print("    ('<OOV>', 'language')", bigramCounts.get("<OOV>", 0).get("language", 0))
    print("    ('to', 'process')", bigramCounts.get("to", 0).get("process", 0))
    print("  3 grams:")
    print("    ('specific', 'formal', 'languages')", trigramCounts.get(("specific", "formal"), 0).get("languages", 0))
    print("    ('to', 'process', '<OOV>')", trigramCounts.get(("to", "process"), 0).get("<OOV>", 0))
    print("    ('specific', 'formal', 'event')", trigramCounts.get(("specific", "formal"), 0).get("event", 0))

    print("")
    print("")
    print("")
    # PART 2.3
    print("CHECKPOINT 2.3 - Probs with addone")
    print("  2 grams:")
    print("    ('the', 'language')", languageModelProbabilityWord("language", "the"))
    print("    ('<OOV>', 'language')", languageModelProbabilityWord("language", "<OOV>"))
    print("    ('to', 'process')", languageModelProbabilityWord("process", "to"))
    print("  3 grams:")
    print("    ('specific', 'formal', 'languages')", languageModelProbabilityWord("languages", "formal", "specific"))
    print("    ('to', 'process', '<OOV>')", languageModelProbabilityWord("<OOV>", "process", "to"))
    print("    ('specific', 'formal', 'event')", languageModelProbabilityWord("event", "formal", "specific"))

    print("")
    print("")
    print("")
    # PART 2.4
    print("FINAL CHECKPOINT - Generated Language")
    print("")
    print("PROMPT: <s>")
    print(*generate(["<s>"]))
    print(*generate(["<s>"]))
    print(*generate(["<s>"]))
    print("")
    print("PROMPT: <s> language is")
    print(*generate(["<s>", "language", "is"]))
    print(*generate(["<s>", "language", "is"]))
    print(*generate(["<s>", "language", "is"]))
    print("")
    print("PROMPT: <s> machines")
    print(*generate(["<s>", "machines"]))
    print(*generate(["<s>", "machines"]))
    print(*generate(["<s>", "machines"]))
    print("")
    print("PROMPT: <s> they want to process")
    print(*generate(["<s>", "they", "want", "to", "process"]))
    print(*generate(["<s>", "they", "want", "to", "process"]))
    print(*generate(["<s>", "they", "want", "to", "process"]))