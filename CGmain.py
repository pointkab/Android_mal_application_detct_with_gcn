# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from androguard.cli import androcg_main
import networkx as nx


args = {'APK':'',
        'verbose':False,
        'classname':'.*',
        'methodname':'.*',
        'descriptor':'.*',
        'accessflag':'.*',
        'no_isolated':True,
        'show':False,
        'output':'callgraph.gml'}

def FCGextract(apk,output='default.gml',no_write=True):
    args['APK'] = apk
    args['output'] = output
    return androcg_main(APK=args['APK'],
                 verbose=args['verbose'],
                 classname=args['classname'],
                 methodname=args['methodname'],
                 descriptor=args['descriptor'],
                 accessflag=args['accessflag'],
                 no_isolated=args['no_isolated'],
                 show=args['show'],
                 output=args['output'],
                 no_write=no_write)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    FCGextract('D:\Compressed\Riskware\/0fa1a178a1f6c89801e582b565291ca2.apk', 'test.gml',no_write=False)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
