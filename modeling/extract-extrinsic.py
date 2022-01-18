import random, csv
inputfile = 'predictglaucomasurgerycombinedprocessedwolaserfilteredbalanced.csv'
outputfile = 'extrinsicdatabalanced.csv'
stopwords = [ # stop words should match what was used during embedding creation
    'a', 'all', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from', 'had', 'has', 'have', 
    'in', 'is', 'it', 'may', 'of', 'on', 'or', 'our', 'than', 'that', 'the', 'there', 'these', 'this', 'to', 
    'was', 'we', 'were', 'which', 'who', 'with'
]

 # group entries by positive and negative so that they can be uniformly allocated to different sets
print('loading data')
with open(inputfile, 'r') as f:
    r = csv.reader(f)
    positiveentries = []
    negativeentries = []
    i = 0
    positives = 0
    for row in r:
        i += 1
        if i == 1:
            continue
        else:
            if i % 1000 == 0:
                print('Row:', i)
            tempEntry = row.copy()
            tempEntry[2] = ' '.join([word.lower() for word in tempEntry[2].split() if word.lower() not in stopwords])
            if int(tempEntry[-1]) == 1:
                positives += 1
                positiveentries.append(tempEntry)
            else:
                negativeentries.append(tempEntry)

print('total positives:', positives, 'and total rows:', i)

# shuffle positive and negative sets
print('shuffling data')
random.shuffle(positiveentries)
random.shuffle(negativeentries)

# allocate same proportion of positives and negatives to each set
small_set_pop = 300
target_positives = int(small_set_pop*positives/i)
target_negatives = small_set_pop - target_positives

holdout_set = positiveentries[0:target_positives] + negativeentries[0:target_negatives]
validation_set = positiveentries[target_positives:target_positives*2] + negativeentries[target_negatives:target_negatives*2]
training_set = positiveentries[target_positives*2:] + negativeentries[target_negatives*2:]

# shuffle again and concatenate
random.shuffle(holdout_set)
random.shuffle(validation_set)
random.shuffle(training_set)

entries = holdout_set + validation_set + training_set

print('writing data')
with open(outputfile, 'w', newline='') as g:
    r = csv.writer(g)
    for entry in entries:
        r.writerow(entry)
