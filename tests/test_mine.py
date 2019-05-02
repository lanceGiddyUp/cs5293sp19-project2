import pytest
from project2 import unredactor

filePath = 'LMRDS/redacted/testRedFiles/1406_4.txt'
myText = unredactor.readFile(filePath)

filePathList = ['LMRDS/aclImdb/test/1406_4.txt']
myDict = unredactor.do_extraction(filePathList, 1)

redFile = unredactor.redact(myDict, filePathList, 1)
ext = unredactor.extRed(redFile['1406_4.txt'])
extFeat = unredactor.extFeatures(ext)

mySimil = unredactor.similarityFinder(redFile, myText)

def test_readFile_sanity():
    assert len(myText) > 0

def test_do_extraction_sanity():
    assert type(myDict) == dict

def test_do_extraction_length():
    assert len(myDict['1406_4.txt']) == 10

def test_redact_sanity():
    assert type(redFile) == dict

def test_extRed_value():
    assert ext[0] == '\u2588'*4

def test_extRed_length():
    assert len(ext) == 19

def test_extFeatures_value():
    assert extFeat[0] == [4, 1, 4, 4]

def test_extFeatures_length():
    assert len(extFeat) == 19

def test_similarityFinder_trivial():
    assert mySimil == ['1406_4.txt']
