import re
import pickle

def getPermissionAndApi():
    method = []
    with open("D:/androguardDemo/permissionAndApi/mapping_4.1.1.csv",'r') as f:
        l = f.readline()
        l = f.readline()
        while l != '':
            method.append('L'+l.split(',')[0]+';'+l.split(',')[1])
            l = f.readline()
    return method

def getSenstiveApi():
    Apis = []
    perType = []
    with open("D:/androguardDemo/SuSi/SourceSinkLists/Android 4.2/SourcesSinks/Ouput_CatSources_v0_9.txt",'r') as f:
        l = f.readline()
        while l != '':
            if l.startswith('<'):
                ptype = l.replace('\n','').split('>')[-1].split(' ')
                if '' in ptype:
                    ptype.remove('')
                perType.append(ptype)
                l = re.match('<(.*)>',l)[1].split(' ')
                Apis.append('L'+l[0].replace(':','').replace('.','/')+';'+l[2].replace('()',''))
            l = f.readline()

    with open("D:/androguardDemo/SuSi/SourceSinkLists/Android 4.2/SourcesSinks/Ouput_CatSinks_v0_9.txt", 'r') as f:
        l = f.readline()
        while l != '':
            if l.startswith('<'):
                ptype = l.replace('\n', '').split('>')[-1].split(' ')
                if '' in ptype:
                    ptype.remove('')
                perType.append(ptype)
                l = re.match('<(.*)>', l)[1].split(' ')
                Apis.append('L' + l[0].replace(':', '').replace('.','/') + ';' + l[2].replace('()', ''))
            l = f.readline()
    return Apis,perType

def getPermissionTypes():
    with open('permissionTypes.pickle','rb') as f:
        pertypes = pickle.load(f)
    return pertypes

def featOneHot(features,permissions,perType):
    one_hot = [0]*89
    one_hot[88] = 1
    is_sen = False
    if features in permissions[0]:
        is_sen = True
        one_hot[88] = 0
        for item in permissions[1][permissions[0].index(features)]:
            one_hot[perType.index(item)] = 1
    return one_hot,is_sen

