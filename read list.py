#-*- coding: utf-8 -*-
import os#将各个文件夹中的图像名称及路径写到txt�?
def getFileNames(rootDir,txtpath):
    f=open(txtpath,'w+')
    fileNames = []
    n=0    # 利用os.walk()函数获取根目录下文件夹名称，子文件夹名称及文件名�?
    for dirName, subDirList, fileList in os.walk(rootDir):
        n += 1
        if(n==1):
            continue
        print(dirName)
        file_name = dirName
        f.writelines(file_name+'/'+'\n')
        fileNames.append(dirName)
        # for fname in fileList:
        #
        #     print("n=",n)
        #     file_name= dirName + '/' + fname
        #     f.writelines(file_name+" "+dirName[-6:]+ '\n')
        #     fileNames.append(dirName+'/'+fname)
        #     n=n+1
    return fileNames
txtpath = "/home2/ZiXiangXu/Last_ding/res4_jiu/list.txt"
path = "/home2/ZiXiangXu/Last_ding/res4_jiu/result/"
aa=getFileNames(path,txtpath)
