import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
pop_csv = r'C:\Users\US Retail\Documents\camkpaper\camkimaging'
fnames = os.listdir(pop_csv)
#auxdf = pd.read_csv(os.path.join(pop_csv,fnames[0]))
#max_activity = np.percentile(auxdf.loc[:,'W_activity'].values,95)
#print(' MAX =',max_activity)
#auxdf.loc[:,['W_activity','NR_activity','REM_activity']]=auxdf.loc[:,['W_activity','NR_activity','REM_activity']].values/max_activity
#df = auxdf
listNR=[]
listREM=[]
listW=[]
fnames = [f for f in fnames if f.endswith('.csv')]
for i,f in enumerate(fnames):
    auxdf = pd.read_csv(os.path.join(pop_csv,f))
    #normalize by mean nrem acitvity
    meanNR =   auxdf.NR_activity.mean()
    auxdf.loc[:,'W_activity'] = auxdf.loc[:,'W_activity'].values/meanNR
    auxdf.loc[:,'NR_activity'] = auxdf.loc[:,'NR_activity'].values/meanNR
    auxdf.loc[:,'REM_activity'] = auxdf.loc[:,'REM_activity'].values/meanNR

    if auxdf.hasREM.max():
        listREM+=list(auxdf.REM_activity.values)
        listNR+=list(auxdf.NR_activity.values)
        listW+=list(auxdf.W_activity.values)
        meanw,meann,meanr = auxdf.W_activity.mean(), auxdf.NR_activity.mean(),auxdf.REM_activity.mean() #mean activity for every cell
        error = [(0,0,0),[auxdf.W_activity.std()/(len(auxdf)**0.5),auxdf.NR_activity.std()/(len(auxdf)**0.5), auxdf.REM_activity.std()/(len(auxdf)**0.5)]]
    else:
        meanw,meann,meanr = auxdf.W_activity.mean(), auxdf.NR_activity.mean(),0 #mean activity for every cell
        error = [(0,0,0),[auxdf.W_activity.std()/(len(auxdf)**0.5),auxdf.NR_activity.std()/(len(auxdf)**0.5),0]]
        listNR+=list(auxdf.NR_activity.values)
        listW+=list(auxdf.W_activity.values)
        
    # plt.subplot(2,2,i+1)
    # bp=plt.bar([0,1,2], [meanw,meann,meanr],
    # yerr=error,align='center',alpha=1, ecolor='k',capsize=3)
    # plt.xticks([0, 1,2], ('WAKE', 'NREM','REM'))
    # plt.ylabel('Mean Z score')
    # bp[0].set_color((0,1,0))
    # bp[1].set_color('b')
    # bp[2].set_color('r')
#plt.tight_layout()
#     max_activity = np.percentile(auxdf.loc[:,'W_activity'].values,95)
#     print(' MAX =',max_activity)
#     auxdf.loc[:,['W_activity','NR_activity','REM_activity']]=auxdf.loc[:,['W_activity','NR_activity','REM_activity']].values/max_activity
#     df=df.append(auxdf)
# df['REM_activity'].loc[df.REM_activity==0]=np.nan
# print(df.head())
# W = df.W_activity.values
# NR = df.NR_activity.values
# R = df.REM_activity.values

#pb=plt.bar([1,2,3],[np.nanmean(W), np.nanmean(NR), np.nanmean(R)])
#sns.barplot(y='W_activity',x ='Experiment',data=df)
error = [(0,0,0),[np.std(listW)/(len(listW)**0.5),np.std(listNR)/(len(listNR)**0.5), np.std(listREM)/(len(listREM)**0.5)]]
bp=plt.bar([0,1,2], [np.mean(listW),np.mean(listNR),np.mean(listREM)],
yerr=error,align='center',alpha=1, ecolor='k',capsize=3)
plt.xticks([0, 1,2], ('WAKE', 'NREM','REM'),fontsize=15)
plt.ylabel('Mean Z score',fontsize=15)
bp[0].set_color((0,1,0))
bp[1].set_color('b')
bp[2].set_color('r')
plt.title('Normalized activity by state',fontsize=21)

#Statss
dfpop=pd.DataFrame(columns=['Normalized_DF','state'])
dfpop.Normalized_DF=listW+listREM+listNR
dfpop.state = ['W']*len(listW) + ['R']*len(listREM) + ['NR']*len(listNR)

dfpop.to_csv('Normalized_dF.csv',index=False)
plt.show()


#  ax=plt.subplot(grid[0, 2]) #plotting bar plot with the difference in activity for cells who had larger activity during hte first W bout
#         meanw,meann,meanr = self.activityW.mean(axis=1), self.activityN.mean(axis=1),self.activityR.mean(axis=1) #mean activity for every cell
#         error = [(0,0,0),[stats.sem(meanw),stats.sem(meann), stats.sem(meanr)]]
#         bp=plt.bar([0,1,2], [meanw.mean(),meann.mean(),meanr.mean()],
#                    yerr=error,align='center',alpha=1, ecolor='k',capsize=5)
#         plt.xticks([0, 1,2], ('WAKE', 'NREM','REM'))
#         plt.ylabel('Mean Z score')
#         bp[0].set_color((0,1,0))
#         bp[1].set_color('b')
#         bp[2].set_color('r')
# #plt.bar()
