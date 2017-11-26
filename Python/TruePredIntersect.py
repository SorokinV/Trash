#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import numpy as np
from   scipy.ndimage import label

class TruePredIntersect :
    
    #
    #
    # Функция - главарь
    #
    # В функции используются увеличенные на 1 с каждой стороны массивы для простоты обработки граничных условий.
    # Например, границы изображения не 2048х2048 как должны были бы быть, а 2050Х2050 (2048+2=2050)
    # Само изображение вложено в границы (1:2048)x(1:2048)
    #
    def labelling0 (self, oo, minPoints=False, maxPoints=False) :
        temp  = np.zeros(oo.shape,dtype=np.uint16);

        #print(datetime.datetime.now(),'x0')
        # первоначальная разметка вниз-вправо
        itemp  = 0
        for xx in range(1,oo.shape[0]-1) :
            for yy in range(1,oo.shape[1]-1) :
                 if oo[xx,yy]>0  :
                     ii = max(temp[xx,yy],temp[xx-1,yy],temp[xx+1,yy],temp[xx,yy-1],temp[xx,yy+1])
                     if ii==0 : 
                        itemp +=1; ii = itemp;
                     temp[xx,yy] = ii;

        # поиск коллизий
        atemp = []
        for xx in range(1,oo.shape[0]-1) :
            for yy in range(1,oo.shape[1]-1) :
                 if temp[xx,yy]>0  :
                        ll = [temp[xx+lx,yy+ly] for lx,ly in [(0,-1),(0,1),(-1,0),(1,0)] 
                              if temp[xx+lx,yy+ly]>0 and temp[xx+lx,yy+ly]>temp[xx,yy]]
                        if len(ll)>0 : atemp.append(ll+[temp[xx,yy]]);

        # первоначальная разметка коллизий
        ntemp = [set([k]) for k in range(temp.max()+1)]
        for ll in atemp :
            llll = set([])
            for lll in ll   : llll = llll | ntemp[lll];
            for lll in llll : ntemp[lll] = llll;     

        #print(datetime.datetime.now(),'x1')
        # насыщение
        OK = True
        while OK :
            OK = False;
            for ll in ntemp :
                old  = ll
                new  = set([])
                for lll in old  : new = new | ntemp[lll];
                if (old==new) : continue;
                for lll in new  : ntemp[lll] = new;
                OK = True

        #print(datetime.datetime.now(),'x2')
        # построение перенумерации
        btemp = np.array([min(ss) for ss in ntemp])
        itemp = 0
        for i in range(1,btemp.shape[0]) :
            if btemp[i]==i : itemp+=1; btemp[i]=itemp;
            else :           btemp[i]=btemp[btemp[i]];

        # Перенумерация
        for xx in range(1,oo.shape[0]-1) :
            for yy in range(1,oo.shape[1]-1) :
                temp[xx,yy] = btemp[temp[xx,yy]]

        #print(datetime.datetime.now(),'x3')
        if minPoints or maxPoints :

            # расчет количества точек в множествах и их зануление
            stemp = np.zeros(temp.max()+1);
            for i in temp.ravel() : stemp[i] += 1
            if minPoints : stemp[stemp<minPoints] = 0;
            if maxPoints : stemp[stemp>maxPoints] = 0;

            # построение перенумерации. Расчет новых номеров. 
            btemp = np.zeros(stemp.shape)
            itemp = 0
            for i in range(1,btemp.shape[0]) :
                if stemp[i]>0  : itemp+=1; btemp[i]=itemp;
                else :           btemp[i]=0;

            # Перенумерация на новые номера и зануление.
            for xx in range(1,oo.shape[0]-1) :
                for yy in range(1,oo.shape[1]-1) :
                    temp[xx,yy] = btemp[temp[xx,yy]]


        #print(datetime.datetime.now(),'L')

        return(temp)

    #
    #
    # Процедура для нумерация объектов в двоичной матрице 
    #
    # Нумерация объектов в двоичном наборе (==0, >0). 
    # Нумерация ведется с 1.. в возрастающем порядке без пропусков.
    # Нумерация ведется в порядке вниз-направо.
    # Связность определяется по 4 соседям
    #
    # При наличии minPoints, maxPoints объекты зануляются, как не входящие в диапазон [minPoints,maxPoints] 
    #
    # Непосредственная нумерация происходит в labelling0 
    #
    # Подразумевается, что входной массив имеет размерность (rows,cols)
    #
    #

    def labelling (self, oo, minPoints=False, maxPoints=False) :

        temp = np.zeros ((oo.shape[0]+2, oo.shape[1]+2), dtype=oo.dtype)
        temp[1:1+oo.shape[0],1:1+oo.shape[1]] = oo

        temp = self.labelling0(temp, minPoints=minPoints, maxPoints=maxPoints);

        return temp[1:1+oo.shape[0],1:1+oo.shape[1]]

    #
    # Функция для расчета размера объектов в размеченной матрице (0 - пустота)
    #
    # Возвращает массив с количеством точек в объектах, 
    # объекты нумеруются в матрице. Ноль - пустое место
    #
    #
    
    def sizing (self,oo) :
        rr = np.zeros(int(oo.max()+1));
        for rc in oo.ravel() : rr[int(rc)] +=1;
        return(rr)

    #
    # Функция для поиска top-left точки объекта на размеченной матрице
    #
    # Возвращает массив с координатами top-left точки объекта в стиле: row,column
    #
    
    def topleft (self,oo) :
        res = np.zeros((int(oo.max())+1,2),dtype=np.int16);
        #print(res.shape,oo.max(),int(oo.max()))
        res[:,:] = max(int(oo.shape[0])+1,int(oo.shape[1])+1)
        for cc in range(oo.shape[1]) : 
            for rr in range(oo.shape[0]) : 
                ooo = int(oo[rr,cc])
                if ooo>0 :
                    if (res[ooo,1]>cc)                       : res[ooo,0], res[ooo,1] = rr,cc;
                    if (res[ooo,1]==cc) and (res[ooo,0]==rr) : res[ooo,0], res[ooo,1] = rr,cc;
                        
        res[0,0], res[0,1] = 0, 0 # for empty
        
        return(res)

    #
    # Функция для поиска top-down-left-right окаймления объекта на размеченной матрице
    #
    # Возвращает массив с координатами top-down-left-right  объекта в стиле: top-row,down-row,left-col,right-col
    #
    
    def tdlr (self,oo) :
        res = np.zeros((int(oo.max())+1,4),dtype=np.int16);
        #print(res.shape,oo.max(),int(oo.max()))
        res[:,0], res[:,1], res[:,2], res[:,3] = oo.shape[0]+1, -1, oo.shape[1]+1, -1
        for cc in range(oo.shape[1]) : 
            for rr in range(oo.shape[0]) : 
                ooo = int(oo[rr,cc])
                if ooo>0 :
                    if (res[ooo,0]>rr): res[ooo,0] = rr;
                    if (res[ooo,1]<rr): res[ooo,1] = rr;
                    if (res[ooo,2]>cc): res[ooo,2] = cc;
                    if (res[ooo,3]<cc): res[ooo,3] = cc;
                        
        res[0,0], res[0,1] = 0, 0 # for empty
        
        return(res)

    #
    #
    # Функция для построения массивов множеств пересечения объектов в матрицах trueL и predL 
    #
    # На входе:  две размеченные объектами матрицы
    # На выходе: два массива множеств. Каждое множество содержит номера объектов из другой матрицы.
    #
    #
    
    def intersect0 (self,trueL,predL) :

        trueO = np.array([set([])]*(int(trueL.max())+1))
        predO = np.array([set([])]*(int(predL.max())+1))

        for rr in range(trueL.shape[0]) :
            for cc in range(trueL.shape[1]) :
                tt,pp = trueL[rr,cc], predL[rr,cc]
                trueO[tt] |= set([pp]); 
                predO[pp] |= set([tt]); 

        return (trueO,predO)

    #
    # Функция для построения массивов размеров пересечения/объединения объектов в матрицах true и pred 
    #
    # На входе:  две бинарные матрицы [0,1] - true и pred 
    # На выходе: две размеченные матрицы trueL, predL и матрицы размера пересечения (inter) и объединения (union)
    #
    # onlyNotZeros==True - расчет идет только для ненулевых точек. В противном случае считаются пересечения с пустотами.
    #
    # PS. То есть, если массив true содержит 21 объект, а массив pred содержит 42 объекта, 
    #     то выходные массивы (inter и union) будут с shape = (21+1)x(42+1) = 22х43. 
    #     [0,:] и [:,0] оставлены для расчета пересечения и объединения с пустотами, 
    #     они считаются если параметр onlyNotZeros установлен в False. 
    #
    # PS. Таким образом, для 12 объекта из true (trueL) и 15 объекта из pred (predL) 
    #           размер пересечения = inter[12,15], объединения = union[12,15].
    #     Номера объектов соответствуют номерам в размеченных матрицах (trueL, predL).
    #     TopLeft для объекта можно узнать через функцию topleft, она выдает массив topleft координат 
    #       в формате массива row,columns. 
    #
    # 2017-11-27 Реализация перестроена на ndimage.label. Ускорение существенно.
    #
    #
    
    def intersect (self, true, pred, onlyNotZeros=True) :

        #print(datetime.datetime.now(),'0')
        (trueL,trueS), (predL, predS) = label(true), label(pred) # матрицы с нумерацией объектов 0..
        #print(datetime.datetime.now(),'1')
        
        # массивы для пересечения и объединения для пар объектов по номерам в списках trueO, predO 
        inter, union = np.zeros((trueS+1,predS+1), dtype=np.int32),np.zeros((trueS+1,predS+1), dtype=np.int32)
        
        trueS, predS = np.zeros(trueS+1), np.zeros(predS+1)

        # Считаем размер пересечения
        for rr in range(trueL.shape[0]) :
            for cc in range(trueL.shape[1]) :
                tt = trueL[rr,cc]
                pp = predL[rr,cc]
                if onlyNotZeros : 
                    if (tt>0) and (pp>0) : inter[tt,pp] +=1
                else : inter[tt,pp] +=1

        # Считаем размер объединения
        for tt in range(union.shape[0]) :
            for pp in range(union.shape[1]) :
                if onlyNotZeros : 
                    if (tt>0) and (pp>0) : union[tt,pp]=trueS[tt]+predS[pp]-inter[tt,pp]
                else : union[tt,pp]=trueS[tt]+predS[pp]-inter[tt,pp]
        
        
        return (trueL, predL, inter, union)