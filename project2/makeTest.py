#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
import sys
from pathlib import Path
import shutil
import re
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import glob
from operator import itemgetter
from nltk.tokenize import WhitespaceTokenizer

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
def do_extraction(globText):
    i = 1
    namesDict = {}
    for theFile in globText:
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
def redact(redactNames, myGlob):
    myDictText = {}
    myDictNames = {}
    m = 1
    for theFile in myGlob:
        myList = []
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
                        myTuple = (myTempText, myStart)
                        myList.append(myTuple)
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
            myDictText[os.path.basename(theFile)] = myText
            myList = sorted(myList, key=itemgetter(1))
            myDictNames[os.path.basename(theFile)] = myList
            #print(redactNames[os.path.basename(theFile)])
            #print(myDictText[os.path.basename(theFile)])
    return myDictText, myDictNames

# main method
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--redact", type = str, required=True, help="The file(s) to redact.", action="append")
    parser.add_argument("--num", type = int, required=True, help="The number of files to redact.")
    args = parser.parse_args()

    # number of the available files to redact
    myNum = args.num

    testList = []
    for i in args.redact:
        testList.append(i)

    testGlob = []
    for i in testList:
        testGlob.extend(glob.glob(i))

    testGlob = testGlob[0:myNum]

    redactNames = do_extraction(testGlob)
    redactFiles = redact(redactNames, testGlob)
    redactedFiles = redactFiles[0]
    redactedNames = redactFiles[1]

    # write redacted files to storage
    pStr = '/project/cs5293sp19-project2/LMRDS/redacted/testRedFiles/'
    p = Path(pStr)
    if p.is_dir() == True:
        shutil.rmtree(p)
        os.mkdir(p)
    else:
        os.mkdir(p)
    for i in redactedFiles.keys():
        path = pStr + i
        with open(path, 'w') as myFile:
            myFile.write(redactedFiles[i])

    # write redacted names to storage
    pStr = '/project/cs5293sp19-project2/LMRDS/redacted/testRedNames/'
    p = Path(pStr)
    if p.is_dir() == True:
        shutil.rmtree(p)
        os.mkdir(p)
    else:
        os.mkdir(p)
    for i in redactedNames.keys():
        path = pStr + i
        with open(path, 'w') as myFile:
            myFile.write(str(redactedNames[i]))

if __name__ == '__main__':
    main()
