
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve, NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols,calibrate_ns_ols
import datetime
from scipy.optimize import lsq_linear
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

#EOD data of all tickers
eodDataFile = 'EOD_20210527.csv'
end = '2021-5-27'

#decomposes entries in matrix into linear composition of eigen vectors, prints eigen vectors
#ie, calculates loading factors
def svd(filename, resultfile = 'yieldsvdnodiff.csv'):
    data = pd.read_csv(filename)
    orig = np.array(data.copy())

    data = data.iloc[:,1:].to_numpy()
    
    AtA = np.matmul(data.transpose(),data)

    evalues, evectors = geteigen(AtA,numvectors = 5, numiterations=10)
    results = np.zeros((data.shape[0],evectors.shape[0]+1))

    for d in range (0,data.shape[0]):
        est = np.zeros(data.shape[1])
        coef = np.zeros(evectors.shape[0])
        for i in range(0,evectors.shape[0]):
            coef[i] = np.dot(data[d],evectors[i])
            est += (np.dot(data[d],evectors[i]))*evectors[i]
        coef = np.append(coef,np.sum((data[d]-est)**2))
        results[d] = coef

    
    df = pd.DataFrame(data=results)
    dates = pd.DataFrame(data=orig[:,0])
    df['date'] = dates

    df.to_csv(resultfile,index=False)

    print(evectors)

#performs power method to find SVD
#A: matrix to perform SVD on
#numvectors: number of eigen vectors to obtain
#numiterations: number of interations for power method to perform
def geteigen(A,numvectors = 3, numiterations=10):
    evalues = np.zeros(numvectors)
    evectors = np.empty((numvectors,A.shape[1]))
    A_next = A

    for v in range(0,numvectors):
        b = np.random.rand(A_next.shape[1])
        b = b/np.linalg.norm(b)

        for i in range(0,numiterations):
            b1 = A_next.dot(b)
            b1norm = np.linalg.norm(A_next.dot(b))
            b = b1 / b1norm
        
        eigenval = (np.matmul(A_next,b)/b)[0]
        evectors[v] = b

        bt = np.array([b])
        A_next = A_next-eigenval*(bt.transpose()@bt)

        evalues[v] = eigenval

    return evalues, evectors

#calculates correlation between 5 day difference of each loading factor and specified tickers on rolling 1 year window
#yieldfile: result of svd()
#indexesfile: EOD file containing data for indexes, defaults to full EOD file
def yieldvsindex(yieldfile,indexfile=eodDataFile):
    tickers = ['SPY','DIA','QQQ','XLF','IAT','IAI']
    yielddata = pd.read_csv(yieldfile)
    yielddata['date'] = pd.to_datetime(yielddata['date'])
    indexdata = pd.read_csv(indexfile)
    indexdata['date'] = pd.to_datetime(indexdata['date'])

    results = pd.DataFrame()
    for ticker in tickers:
        results[ticker+'corr'] = None

    dayoffset = 5

    yielddata.loc[:, yielddata.columns != 'date'] = yielddata.loc[:, yielddata.columns != 'date'].diff(dayoffset)

    #Match dates
    start1y = '2005-5-27'
    yielddata3m = yielddata.loc[((yielddata['date']>start1y)&(yielddata['date']<=end))]
    yielddata3m=yielddata3m.reset_index(drop=True)
    indexdata3m = pd.DataFrame(columns=['ticker','date','close'])

    for index,row in yielddata3m.iterrows():
        currdate = row['date']
        for ticker in tickers:
            match = indexdata.loc[(indexdata['ticker']==ticker)&(indexdata['date']==currdate)]
            if match.empty:
                yielddata3m=yielddata3m.drop(index)
                break
            else:
                indexdata3m=indexdata3m.append({'ticker':ticker,'date':currdate,'close':match['close'].item()},ignore_index=True)

    batchdays = 253
    rowcounter = -1 
    print(yielddata3m.iloc[0].loc['date'])
    print(yielddata3m.iloc[-1].loc['date'])
    for coef in range(0,len(yielddata.columns)-2):
        x = yielddata3m.iloc[:,coef].to_numpy().reshape(-1,1)
        x1 = np.zeros(x.shape[0])
        for i in range(0,x.shape[0]):
            x1[i]=x[i,0]
        corr = pd.DataFrame()
        x1=x1.reshape(-1,1)

        rowcounter = rowcounter+1
        offset = 0
        for ticker in tickers:
            y = indexdata3m.loc[(indexdata3m['ticker']==ticker),'close'].to_numpy().reshape(-1,1)
            y1 = np.zeros(y.shape[0])
            for i in range(0,y.shape[0]):
                y1[i]=y[i,0]
            y1=y1.reshape(-1,1)
            
            rowcounter = rowcounter - offset
            offset = 0
            
            for b in range (0,yielddata3m.shape[0]-batchdays-1):
                try:
                    startday = yielddata3m.iloc[b,:].loc['date'].date()
                    endday = yielddata3m.iloc[b+batchdays,:].loc['date'].date()
                except:
                    print('Error: date out of range')
                    exit()

                dataindex = 'lf'+str(coef)+':'+str(startday)+'-'+str(endday)

                x1curr = x1[b:b+batchdays]
                y1curr = y1[b:b+batchdays]

                corr['x'] = x1curr.flatten()
                corr['y'] = y1curr.flatten()
                corrm = corr.corr(method='pearson')

                results.loc[rowcounter,ticker+'corr'] = corrm.iloc[0,1]
                results.loc[rowcounter,['range']] = dataindex

                rowcounter = rowcounter+1
                offset = offset+1
    
    results.to_csv('yieldvsindexrolling.csv')


#constructs zero curve based off current bond rates
#bondData: WSJ bond data as csv file
#plot: True will show plot of constructed zero curve
#program saves file of resulting data in ratedata.csv
def zeroCurve(bondData = 'bonds.csv',plot = True):
    data = pd.read_csv(bondData)
    data['SPREAD'] = data['ASKED']-data['BID']
    clean = pd.DataFrame()
    rates = pd.DataFrame(columns=['date','rate'])
    
    d=0
    while d<data.shape[0]:
        currRow = data.iloc[d,:]
        currDate = currRow['MATURITY']
        matches = data.loc[data['MATURITY']==currDate]

        matches = matches.sort_values(by=['SPREAD'])
        clean = clean.append(matches.iloc[0,:],ignore_index=True)
        d = d+matches.shape[0]
    
    clean['adjprice'] = None
    for index,row in clean.iterrows():
        splitA = str(row['ASKED']).split('.')
        adjustA = int(splitA[0])+(int(splitA[1])*10**(-len(splitA[1])))/0.32 #convert from fractional to decimal
        splitB = str(row['BID']).split('.')
        adjustB = int(splitB[0])+(int(splitB[1])*10**(-len(splitB[1])))/0.32 #convert from fractional to decimal
        clean.iloc[index,-1] = (adjustA+adjustB)/2

    rateData = pd.DataFrame(columns=['date','rate'])
    rateData=rateData.append({'date':0,'rate':0.001},ignore_index=True)

    for index,row in clean.iterrows():
        done = False
        daystil = (datetime.datetime.strptime(row['MATURITY'],"%m/%d/%y")-datetime.datetime.now()).days
        if (daystil/365)<0.25: 
            continue

        t=daystil/365

        bondPrice = row['adjprice']
        coupon = row['COUPON']/2

        #adjust for dirty price
        numCoupons = math.trunc(t/0.5)
        firstCoupon = t-0.5*numCoupons
        daysPassed = 0.5-firstCoupon
        bondPrice += (daysPassed/0.5)*coupon
        clean.iloc[index,-1] = bondPrice

        if rateData.shape[0] <= 1 or t<0.5: #first entry?
            rateData = rateData.append({'date':t,'rate':(np.log(coupon+100)-np.log(bondPrice))/t},ignore_index=True)
            rateData = rateData.sort_values(by='date') 
            continue
        
        pv = 0
        rate=0
        try:
            interp = CubicSpline(rateData['date'], rateData['rate'])
        except:
            print('error at')
            print(row)
            rateData.to_csv('ratedata.csv')
            exit()
        
        for c in range(0,math.trunc(t/0.5)): #loop through all coupons before expiration
            rate = 0 
            coupont = t-(0.5*(math.trunc(t/0.5)-c)) #time til current coupon

            if coupont <= rateData.iloc[0,0]:
                rate = rateData.iloc[0,1]
            elif coupont >= rateData.iloc[-1,0]:
                if t - rateData.iloc[-1,0] > 2:
                    nelsonsiegel = True
                else:
                    nelsonsiegel=False

                ecoupons = math.floor((t-rateData.iloc[-1,0])/0.5)

                guesspv = pv
                bonddata = pd.read_csv('recentbond.csv')
                bonddata['rate'] = bonddata['rate'] * (10**-2)
                bondinterp = CubicSpline(bonddata['time'],bonddata['rate'])

                initGuess = pd.DataFrame(columns=['time','rate'])
                for r in range(0,ecoupons-1):
                    coupont = (t-0.5*(r+1))
                    initGuess = initGuess.append({'time':coupont,'rate':bondinterp(coupont)},ignore_index=True)
                    guesspv += coupon/(math.exp((bondinterp(coupont))*coupont))

                initGuess = initGuess.sort_values(by='time') 

                for iteration in range(0,1):
                    lastr = (np.log(coupon+100)-np.log(bondPrice-guesspv))/t
                    twoyrrateDate = rateData[rateData['date']>2]
                    guessData = twoyrrateDate.append({'date':t,'rate':lastr},ignore_index=True)
                    i=0
                    while i<guessData.shape[0]-1: #drop negative future rates
                        if guessData.iloc[i,1]-guessData.iloc[i+1,1]>0:
                            guessData=guessData.drop(guessData.index[i+1])
                            i -= 1
                        i += 1
                    
                    guesspv = pv

                    for r in range(0,ecoupons-1): #interpolate coupon rates
                        coupont = (t-0.5*(r+1))
                        couponrate = 0
                        if nelsonsiegel:
                            curve_fit, status = calibrate_nss_ols(guessData['date'].to_numpy(),guessData['rate'].to_numpy())
                            couponrate = NelsonSiegelSvenssonCurve.zero(curve_fit,coupont)
                        else:
                            guessinterp = CubicSpline(guessData['date'], guessData['rate'])
                            couponrate = guessinterp(coupont)
                        guesspv += coupon/(math.exp(couponrate*coupont))
                
                for r in range(0,ecoupons-1): #add new coupons to data
                    coupont = (t-0.5*(r+1))
                    couponrate = 0
                    if nelsonsiegel:
                        curve_fit, status = calibrate_nss_ols(guessData['date'].to_numpy(),guessData['rate'].to_numpy())
                        couponrate = NelsonSiegelSvenssonCurve.zero(curve_fit,coupont)
                    else:
                        guessinterp = CubicSpline(guessData['date'], guessData['rate'])
                        couponrate = guessinterp(coupont)
                    rateData = rateData.append({'date':coupont,'rate':couponrate},ignore_index=True)

                rateData = rateData.append({'date':t,'rate':lastr},ignore_index=True)
                rateData = rateData.sort_values(by='date')  
                done = True
                
                break
            else: 
                rate = interp(coupont) #coupon date is within our interp
                pv += coupon/(math.exp(rate*coupont))

        if not done:
            rateData = rateData.append({'date':t,'rate':(np.log(100+coupon)-np.log(bondPrice-pv))/t},ignore_index=True)
        rateData = rateData.sort_values(by='date')  
        continue

    if plot:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xc = np.arange(rateData.iloc[0,0],rateData.iloc[-1,0],0.1)
        ax.scatter(rateData['date'],rateData['rate'])
        plt.show()
    
    rateData.to_csv('ratedata.csv')


def main():

    zeroCurve()


if __name__ == '__main__':
    main()


