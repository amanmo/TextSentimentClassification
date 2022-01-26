from string import punctuation

STOPWORDS = ['a', 'an', 'the', 'i', 'we', 'us', 'he', 'him', 'his', 'she', 'her', 'hers', 'my', 'they', 'them', 'their', 'theirs', 'you', 'your', 'yours', 'be', 'is', 'was', 'were', 'am', 'this', 'that', 'for', 'it', 'at', 'or', 'in', 'to', 'from', 'on', 'as', 'and']
SUFFIXES = ['ion', 'ions', 'dom', 'ity', 'ities' , 'ism', 'ment', 'ments', 'ist', 'ify', 'ish', 'ive', 'able', 'ible']

def removePunctuation(data):
    'Function to remove punctuation marks from each string'

    for i in range(len(data)):
        string = data[i][0]
        for character in string:
            if character in punctuation:
                if character != "'":
                    string = string.replace(character, ' ')
                else:
                    string = string.replace(character, '')
        data[i][0] = string
    return data

def removeNumbers(data):
    'Function to remove digits from each string'

    for i in range(len(data)):
        string = data[i][0]
        for number in '0123456789':
            string = string.replace(number, ' ')
        data[i][0] = string
    return data

def removeStopwords(data):
    'Function to remove stopwords from each string'

    for i in range(len(data)):
        string = data[i][0]
        split_list = string.split()
        ans = []
        for j in range(len(split_list)):
            if split_list[j] not in STOPWORDS:
                ans += [split_list[j]]
        data[i][0] = " ".join(ans)
    return data

def removeExcessWhitespace(data):
    'Function to remove excess whitespace from each string'

    for i in range(len(data)):
        string = data[i][0]
        data[i][0] = string.replace('  ', ' ')
    return data

def stemText(data):
    'Function to stem each word in a string'

    for i in range(len(data)):
        string = data[i][0]
        ans = []
        for word in string.split():
            flag = True
            for suffix in SUFFIXES:
                if word.endswith(suffix) and len(word) - len(suffix) > 2:
                    ans += [word[:len(word) - len(suffix)]]
                    flag = False
                    break
            if flag:
                ans += [word]
        data[i][0] = " ".join(ans)
    return data

def preprocess(data):
    'Function to supervise the preprocessing of training data'

    data = removePunctuation(data)
    data = removeNumbers(data)
    data = removeExcessWhitespace(data)
    data = removeStopwords(data)
    data = stemText(data)
    return data