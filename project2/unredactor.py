#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
import sys
import re
from pathlib import Path
import shutil
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk import WhitespaceTokenizer
import glob
import pathlib
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

# Gets all the text where entity is PERSON inside of text.
# Deletes duplicate names and sorts the list by length of name.
def get_entity(text):
    namesList = []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                namesList.append(' '.join(c[0] for c in chunk.leaves()))
    namesList = list(dict.fromkeys(namesList))
    namesList = sorted(namesList, key=len, reverse=True)
    return namesList

# Usese get_entity() to extract the names in the the files from the given glob.
def do_extraction(glob_text, myN):
    i = 1
    namesDict = {}
    for theFile in glob_text[0:myN]:
        print('do_extraction:  ', i) # counter to show progress
        i += 1
        with io.open(theFile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            text = text.replace('<br /><br />', '\n')
            tempNames = get_entity(text)
            if len(tempNames) > 0:
                namesDict[os.path.basename(theFile)] = tempNames
    return namesDict

# finds a sting in a text using regex and returns the location
def regExSearch(myString, myText):
    if myString[-1] == '.':
        myString = myString[0:(len(myString)-1)] + r'[\.\b]'
    else:
        myString = myString + r'\b'
    myRegEx = r'\b' + myString
    myRe = re.search(myRegEx, myText)
    return myRe

# use location in a string to replace a smaller string with u2588
def strReplace(myStart, myEnd, myText):
    myLen = myEnd - myStart
    if myStart == 0:
        myText = '\u2588'*myLen + myText[myEnd:]
    else:
        myText = myText[0:myStart] + '\u2588'*myLen + myText[myEnd:]
    return myText

# uses the regExSearch and strReplace methods to redact files
def redact(redactNames, myGlob, myN):
    myDict = {}
    m = 1
    for theFile in myGlob[0:myN]:
        print('redact:  ', m) # counter to show progress
        m += 1
        if os.path.basename(theFile) in redactNames.keys():
            with io.open(theFile, 'r', encoding='utf-8') as myFile:
                myText = myFile.read()
            for i in redactNames.keys():
                for j in redactNames[i]:
                    while regExSearch(j, myText):
                        mySpan = regExSearch(j, myText).span()
                        myStart = mySpan[0]
                        myEnd = mySpan[1]
                        myTempText = myText[myStart:myEnd]
                        wST = WhitespaceTokenizer().tokenize(myTempText)
                        #wST = word_tokenize(myTempText)
                        wST = sorted(wST, key=len, reverse=True)
                        for k in wST:
                            while regExSearch(k, myTempText):
                                myTempSpan = regExSearch(k, myTempText).span()
                                myTempStart = myTempSpan[0]
                                myTempEnd = myTempSpan[1]
                                myTempText = strReplace(myTempStart, myTempEnd, myTempText)
                                myText = myText[0:myStart] + myTempText + myText[myEnd:]
            myDict[os.path.basename(theFile)] = myText
    return myDict

#returns dictionary of list of sorted similar documents
#each key in the return is the test file key, and the list is a list of similar document
#sorted from the most similar to the least similar
def similarityFinder(trainRedactFiles, redactFile):

    trainCorpus = []
    for i in trainRedactFiles.keys():
        trainCorpus.append(trainRedactFiles[i])

    vec = TfidfVectorizer(stop_words = 'english')

    myList = trainCorpus
    myList2 = []
    myTupList = []

    myList = [redactFile] + myList
    corpusVec = vec.fit_transform(myList)
    X = cosine_similarity(corpusVec)
    myList2 = list(zip(trainRedactFiles.keys(), list(X[0,1:])))
    myList2Sorted = sorted(myList2, key=itemgetter(1), reverse=True)
    myTupList.append(myList2Sorted)

    myList3 = []
    for i in myTupList[0]:
        myList3.append(i[0])

    return(myList3)

# finds all redacted names
def extRed(myText):

    myList = []
    myRegEx = r'(([\u2588]+\s*)+)'
    myReList = re.findall(myRegEx, myText)
    for i in myReList:
        if i[0][-1] == ' ':
            myList.append(i[0][:-1])
        else:
            myList.append(i[0])

    return myList

# extracts all features of names sent to it
def extFeatures(candidates):

    myList1 = []
    for i in candidates:
        myList2 = []
        charLen = len(i)
        myList2.append(charLen)
        names = word_tokenize(i)
        numNames = len(names)
        myList2.append(numNames)
        nameBegLen = len(names[0])
        myList2.append(nameBegLen)
        nameEndLen = len(names[-1])
        myList2.append(nameEndLen)
        myList1.append(myList2)

    return myList1

# replace redacted names with predicted names
def repRedactions(redFile, candList):
    myText = redFile
    myText2 = redFile
    # sorts the list of dictionary of lists by the lenth of the ext item of the dictionary
    candList = sorted(candList, key=lambda i: len(i['ext']), reverse=True)
    myList = []
    for i in candList:
        myString = i['ext']
        myRegEx = myString
        myRe = re.search(myRegEx, myText)
        myStart = myRe.span()[0]
        myEnd = myRe.span()[1]
        cands = i['cand']
        if len(cands) > 0:
            myText = myText[0:myStart] + cands[0] + myText[myEnd:]
            myText2 = myText2[0:myStart] + cands[0] + myText2[myEnd:]
        else:
            myText = myText[0:myStart] + '\u2580'*(myEnd-myStart) + myText[myEnd:]
    return myText2

# reads a file
def readFile(filePath):
    with io.open(filePath, 'r', encoding='utf-8') as myFile:
        myText = myFile.read()

    return myText

# main method
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--redacted", type = str, required=True, help="The file(s) to be unredacted.", action="append")
    parser.add_argument("--numTrain", type = int, required=True, help="The number of training files to use.")
    parser.add_argument("--numCand", type = int, required=True, help="The number of possible candidate names to populate each redacted space.")
    args = parser.parse_args()

    myN = args.numTrain
    myCandN = args.numCand

    testList = args.redacted
    testGlob = []
    for i in testList:
        testGlob.extend(glob.glob(i))

    unredFilePathParent = 'LMRDS/unredacted/testUnredFiles/' 
    p = Path(unredFilePathParent)
    if p.is_dir() == True:
        shutil.rmtree(p)
        os.mkdir(p)
    else:
        os.mkdir(p)
    totTruePosCounterK1 = 0
    totTruePosCounterK5 = 0
    totTruePosCounterK10 = 0
    totRedactionsCounter = 0
    for filePath in testGlob:
        print('\n\n', filePath)
        myText = readFile(filePath)

        candidates = extRed(myText)
        candFeatures = extFeatures(candidates)

        trainList = ['LMRDS/aclImdb/train/*.txt']
        trainGlob = []
        for i in trainList:
            trainGlob.extend(glob.glob(i))

        trainRedactNames = do_extraction(trainGlob, myN)
        trainRedactFiles = redact(trainRedactNames, trainGlob, myN)

        simList = similarityFinder(trainRedactFiles, myText)

        # finding list of candidate names to be unredacted for each of the redacted names
        candList = []
        m = 0
        n = myCandN
        for i in candFeatures:
            candDict = {}
            myList = []
            for j in simList:
                jFeatures = extFeatures(trainRedactNames[j])
                k = 0
                for l in jFeatures:
                    if i == l:
                        myList.append(trainRedactNames[j][k])
                        myList = list(dict.fromkeys(myList))
                    k += 1
                    if len(myList) >= n:
                        break
                if len(myList) >= n:
                    break
            candDict['ext'] = candidates[m]
            candDict['cand'] = myList
            candList.append(candDict)
            m += 1
        print('\n\nDictionary of Candidate Lists:\n\n', candList)

        unredFilePath = unredFilePathParent + str(os.path.basename(filePath))
        with io.open(unredFilePath, 'w', encoding='utf-8') as myFile:
            myUnredText = repRedactions(myText, candList)
            myFile.write(myUnredText)

        truePath = pathlib.Path(filePath).parent
        truePath = str(pathlib.Path(truePath).parent) + '/testRedNames/' + str(os.path.basename(filePath)) 
        with io.open(truePath, 'r', encoding='utf-8') as myFile:
            trueList = ast.literal_eval(myFile.read())

        print('\nTrue List:\n\n', trueList)

        # calculate the accuracy @k for each test file and the total for all the test files passed
        truePosCounterK1 = 0
        truePosCounterK5 = 0
        truePosCounterK10 = 0
        redactionsCounter = len(trueList)
        totRedactionsCounter = totRedactionsCounter + redactionsCounter
        for i, j in zip(trueList, candList):
            if len(j['cand']) > 0 :
                if i[0] in j['cand'][0]:
                    truePosCounterK1 += 1
                    totTruePosCounterK1 += 1
                n1 = min(len(j['cand']) + 1, 5)
                if i[0] in j['cand'][0:n1]:
                    truePosCounterK5 += 1
                    totTruePosCounterK5 += 1
                n2 = min(len(j['cand']) + 1, 10)
                if i[0] in j['cand'][0:n2]:
                    truePosCounterK10 += 1
                    totTruePosCounterK10 += 1
        print('\n\nBy File:')
        print('\nNumber of Correct Predictions (K=1, 5, 10):  ', truePosCounterK1, truePosCounterK5, truePosCounterK10)
        print('\nNumber of Redactions:  ', redactionsCounter)
        print('\nNumber of Correct Predictions Divided by Number of Redactions (K=1, 5, 10):  ', truePosCounterK1/redactionsCounter, truePosCounterK5/redactionsCounter, truePosCounterK10/redactionsCounter)
        print('\nCumulatively:')
        print('\nNumber of Total Correct Predictions (K=1, 5, 10):  ', totTruePosCounterK1, totTruePosCounterK5, totTruePosCounterK10)
        print('\nTotal Number of Redactions:  ', totRedactionsCounter)
        print('\nNumber of Total Correct Predictions Divided by Total Number of Redactions (K=1, 5, 10):  ', totTruePosCounterK1/totRedactionsCounter, totTruePosCounterK5/totRedactionsCounter, totTruePosCounterK10/totRedactionsCounter)
        print('\n\n###################################################\n\n')

if __name__ == '__main__':
    main()
