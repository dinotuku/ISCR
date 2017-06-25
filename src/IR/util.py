import sys
import os
import operator
# from retmath import *

import math

def cross_entropy(p1, p2):
    return (0 if p2 == 0 else -1 * p1 * math.log(p1 / p2))


def renormalize(distdict):
    Z = sum(distdict.values())
    if Z == 1:
        return distdict
    nol = {}
    for key in distdict.iterkeys():
        nol[key] = distdict[key] / Z
    return nol


def gaussian(mean, var, x):
    return 1 / math.sqrt(2 * math.pi * var) *\
        math.exp(-1 / 2 * math.pow((x - mean) / math.sqrt(var), 2))


def entropy(model):
    ent = 0.0
    for key, p in model.iteritems():
        ent += -1 * p * math.log(p)
    return ent


def cross_entropies(model1, model2):
    cent = 0.0
    for key1, p1 in model1.iteritems():
        if model2.has_key(key1):
            p2 = model2[key1]
            cent += cross_entropy(p1, p2)
    return cent


def IDFscore(model, inv_index):
    x = sorted(model.iteritems(), key=operator.itemgetter(1), reverse=True)
    if len(x) > 20:
        x = x[:20]
    scores = []
    for key, val in x:
        if len(inv_index[key]) > 0:
            scores.append(math.log(5047.0 / float(len(inv_index[key]))))

    maxS = 0.0
    avgS = 0.0
    if len(scores) > 0:
        maxS = max(scores)
        avgS = sum(scores) / float(len(scores))
    return maxS, avgS


def QueryScope(model, inv_index):
    N = 10
    if len(model) < 10:
        N = len(model)
    docs = []
    for wordID, prob in sorted(model.iteritems(),
                               key=operator.itemgetter(1), reverse=True)[:N]:
        docs = list(set(docs) | set(inv_index[wordID].values()))
    return -1 * math.log(float(len(docs) + 1.0) / 5047.0)


def idfDev(model, inv_index):
    N = 10
    if len(model) < 10:
        N = len(model)
    idfs = []
    for wordID, prob in sorted(model.iteritems(),
                               key=operator.itemgetter(1), reverse=True)[:N]:
        N = float(len(inv_index[wordID]))
        idf = math.log(5047.0 + 0.5) / (N + 1) / math.log(5047.0 + 1)
        idfs.append(idf)
    return stdev(idfs)


def stdev(data):
    norm2 = 0.0
    mean = sum(data) / float(len(data))
    for d in data:
        norm2 += math.pow(d, 2) / float(len(data))
    if norm2 - math.pow(mean, 2) < 0:
        return 0
    return math.sqrt(norm2 - math.pow(mean, 2))


def Variability(ret, segments):
    means = []
    vars = []
    D2 = 0.0
    D1 = 0.0
    for i in range(segments[-1] + 1):
        D1 += ret[i][1]
        D2 += math.pow(ret[i][1], 2)
        if i in segments:
            mean = D1 / float(i)
            var = math.sqrt(D2 - math.pow(mean, 2))
            means.append(mean)
            vars.append(var)
    return means, vars


def Exp(lamda, x):
    return lamda * math.exp(-1 * lamda * x)


def FitExpDistribution(ret, lamda):
    expdist = [Exp(lamda, x) for x in range(100)]
    mean = sum(expdist) / float(len(expdist))
    expdist = [x / mean for x in expdist]

    rank = map(operator.itemgetter(1), ret)[:100]
    meanX = sum(rank) / float(len(rank)) + rank[-1]
    if meanX == 0:
        return 0.
    rank = [(r + rank[-1]) / meanX for r in rank]

    err = 0.0
    for i in range(len(rank)):
        err += math.pow(rank[i] - expdist[i], 2)
    return math.sqrt(err)


def FitGaussDistribution(ret, m, v):
    gaussdist = [gaussian(m, v, x) for x in range(100)]
    mean = sum(gaussdist) / float(len(gaussdist))
    gaussdist = [x / mean for x in gaussdist]

    rank = map(operator.itemgetter(1), ret)[:100]
    meanX = sum(rank) / float(len(rank)) + rank[-1]
    if meanX == 0:
        return 0.
    rank = [(r + rank[-1]) / meanX for r in rank]

    err = 0.0
    for i in range(len(rank)):
        err += math.pow(rank[i] - gaussdist[i], 2)
    return math.sqrt(err)


def readLex(fname):
    fin = file(fname)
    lex = {}
    num = 0
    for line in fin.readlines():
        word = line.replace('\n', '')
        num += 1
        lex[word] = num
    return lex

def readList(fname):
    fin = file(fname)
    lst = []
    for line in fin.readlines():
        word = line.replace('\n', '')
        lst.append(word)
    return lst

def docNameToIndex(fname):
    return int(fname[1:])

def IndexToDocName(index):
    name = 'T'
    if index < 10:
        name += '000' + str(index)
    elif index < 100:
        name += '00'  + str(index)
    elif index < 1000:
        name += '0'   + str(index)
    else:
        name += str(index)
    return name

def readQuery(fname, lex):
    fin = file(fname)
    queries = []
    for line in fin.readlines():
        tokens = line.replace('\n', '').split()
        leng = int(tokens[0])

    query = {}
    for i in range(1, len(tokens), 2):
        wordID = lex[tokens[i]]
        val = float(tokens[i+1])
        query[wordID] = val
    queries.append(query)
    fin.close()
    return queries

def readFoldQueries(fname):
    fin = file(fname)
    queries = []
    indexes = []
    for line in fin.readlines():
        pair = line.replace('\n', '').split('\t')
        tokens = pair[1].split()
    query = {}
    for i in range(len(tokens)):
        p = tokens[i].split(':')
        query[int(p[0])] = float(p[1])
    queries.append(query)
    indexes.append(int(pair[0]))
    fin.close()
    return queries, indexes


def writeQueryFold(fname, queries, indexes):
    fout = file(fname, 'w')
    for i in range(len(queries)):
        q = queries[i]
        fout.write(str(indexes[i])+'\t')
    for wordID, val in sorted(q.iteritems(), key=operator.itemgetter(0)):
	    fout.write(str(wordID) + ':' + str(val) + ' ')
    fout.write('\n')
    fout.close()

def readInvIndex(fname):
    fin = file(fname)
    inv_index = {}
    for line in fin.readlines():
        pair = line.replace('\n', '').split('\t')
        wordID = int(pair[0])
        if pair[1] == '':
            inv_index[wordID] = {}
            continue
    docset = {}
    for valpair in pair[1].split():
        key, val = valpair.split(':')
        docset[int(key)] = float(val)
    inv_index[wordID] = docset
    fin.close()
    return inv_index

def readCleanInvIndex(fname):
    fin = file(fname)
    inv_index = {}
    for line in fin.readlines():
        pair = line.replace('\n', '').split('\t')
        wordID = int(pair[0])
        if pair[1] == '':
            continue
    docset = {}
    for valpair in pair[1].split():
        key, val = valpair.split(':')
        docset[int(key)] = float(val)
    inv_index[wordID] = docset
    fin.close()
    return inv_index

def readBackground(fname, lex):
    fin = file(fname)
    background = {}
    for line in fin.readlines():
        pair = line.replace('\n', '').split()
        wordID = lex[pair[0]]
        val = float(pair[1])
        background[wordID] = val
    fin.close()
    return background

def readAnswer(fname, lex):
    fin = file(fname)
    answer = []
    for i in range(163):
        answer.append({})
    for line in fin.readlines():
        tokens = line.replace('\n', '').split()
        index = int(tokens[0])
        docID = docNameToIndex(tokens[2])
        answer[index-1][docID] = 1
    return answer

def readDocLength(fname):
    fin = file(fname)
    docLengs = {}
    for line in fin.readlines():
        pair = line.replace('\n', '').split()
        docID = docNameToIndex(pair[0])
        docleng = float(pair[1])
        docLengs[docID] = docleng
    fin.close()
    return docLengs

def readDocModel(fname):
    fout = file(fname)
    model = {}
    for line in fout.readlines():
        tokens = line.split()
        word = int(tokens[0])
        val = float(tokens[1])
        model[word] = val
    return model

def printRetrievedList(retrieved, fname):
    fout = file(fname, 'w')
    for i in range(len(retrieved)):
        list = retrieved[i]
    for key, val in list:
        fout.write(str(i+1)+' '+str(i+1)+' '+IndexToDocName(key)+\
            ' 0 '+str(val)+ ' EXP\n')
    fout.close()

def readKeytermlist(cpsID, fileIDs):
    keyterms = {}
    if cpsID == 'lattice.tandem':
        cpsID = 'onebest.tandem'
    elif cpsID == 'lattice.CMVN':
        cpsID = 'onebest.CMVN'

    for fileID, prob in fileIDs.iteritems():
        filename = '../data/ISDR-CMDP/keyterm/'+cpsID+'/'+str(fileID)
        if not os.path.isfile(filename):
            continue
        fin = file(filename)
        for i in range(100):
            line = fin.readline()
            if line == '':
                break
            pair = line.replace('\n', '').split('\t')
            if keyterms.has_key(int(pair[0])):
                keyterms[int(pair[0])] += prob*float(pair[1])
            else:
                keyterms[int(pair[0])] = prob*float(pair[1])
        fin.close()
    sortedKeytermlst = sorted(keyterms.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedKeytermlst

def readRequestlist(cpsID, fileIDs):
    requests = {}
    for fileID, prob in fileIDs.iteritems():
        filename = '../data/ISDR-CMDP/request/'+cpsID+'/'+str(fileID)
        if not os.path.isfile(filename):
            continue
        fin = file(filename)
        for line in fin.readlines():
            pair = line.replace('\n', '').split('\t')
        if requests.has_key(int(pair[0])):
            requests[int(pair[0])] += float(pair[1])
        else:
            requests[int(pair[0])] = float(pair[1])
        fin.close()
    return sorted(requests.iteritems(), key=operator.itemgetter(1), reverse=True)

def readTopicWords(cpsID):
    topicWordList = []
    for i in range(128):
        words = {}
        filename = '../data/ISDR-CMDP/lda/' + cpsID + '/'+str(i)
        fin = file(filename)
        for line in fin.readlines():
            if len(line.split('\t')) <= 1:
                continue
            pair = line.split('\t')
            words[int(pair[0])] = float(pair[1])
        fin.close()
        topicWordList.append(words)
    return topicWordList

def sortTopicByKLtoAnswer(docmodeldir, ans, doclengs, topiclst):
    answer = {}
    for docID in ans.iterkeys():
        fname = docmodeldir+IndexToDocName(docID)
        leng = doclengs[docID]
        docmodel = readDocModel(fname)
    for term, prob in docmodel.iteritems():
        if answer.has_key(term):
            answer[term] += leng * prob
        else:
            answer[term] = leng * prob
    ranking = []
    for i in range(len(topiclst)):
        topicWords = topiclst[i]
        score = 0
        sortwords = sorted(topicWords.iteritems(),\
            key=operator.itemgetter(1), reverse=True)
        for term, prob in sortwords:
            if answer.has_key(term):
                score += cross_entropy(answer[term], prob)
        if score == 0:
            score = -99999
        ranking.append((i, score))
    return sorted(ranking, key=operator.itemgetter(1), reverse=True)

def sortTopicByInferenceProb(cpsID):
    rankings = []
    fin = file('../data/ISDR-CMDP/ldadist/'+cpsID+'.dist')
    for line in fin.readlines():
        ranking = []
        tokens = line.split()
        for i in range(len(tokens)):
            ranking.append((i,float(tokens[i])))
            rankings.append(\
                sorted(ranking,key=operator.itemgetter(1),reverse=True))
    return rankings

def readTopicList(cpsID,qID):
    ranking = []
    fin = file('../data/ISDR-CMDP/topicRanking/'+cpsID+'/'+str(qID))
    for line in fin.readlines():
        tokens = line.split('\t')
        ranking.append((int(float(tokens[0])),float(tokens[1])))
    return ranking

def renormalize(dictdist):
    Z = sum(dictdist.values())
    for term in dictdist.iterkeys():
        dictdist[term] /= Z
    return dictdist

def pruneAndNormalize(dictdist,num,leng=50):
    cnt = 0
    newdict = {}
    for term,prob in sorted(dictdist.iteritems(),key=operator.itemgetter(1),reverse=True):
        if cnt <= num:
            newdict[term] = leng*prob
        else:
            break
        cnt+=1
    return renormalize(newdict)

def readStateFeatures(fname):
    fin = file(fname)
    labels = []
    features = []
    for line in fin.readlines(): # for each data
        pair = line.split('\t')
        if len(pair)!=2 or len(pair[1].split())!=94:
            continue
        tokens = pair[1].split()
        feature = []
        try:
            feature = [float(f) for f in tokens[0:-1]]
        except ValueError:
            continue
    jumpthrough = False
    for f in feature[29:]:
        if f>500:
            jumpthrough = True
        if jumpthrough:
            continue

	#del feature[7]
	#del feature[1]
    acts = [0.0,0.0,0.0,0.0]
    actions = feature[1:5]
    for a in actions:
        if a!=0:
            acts[int(a)-1]+=1.0
    processedFeature = []
    processedFeature.extend(feature[:1])
    #processedFeature.extend(acts)
    processedFeature.extend(feature[6:])
    processedFeature.insert(0,1.0)
    labels.append([float(pair[0])])
    features.append(processedFeature)
    return features,labels

def readWeights(fname):
    fin = file(fname)
    tokens = fin.readline().split()
    return [float(w) for w in tokens]
