#R code for running the mixed-effects models in Antony et al. (2020, Neuron)
library(Hmisc)
library(data.table)
library(tidyr)
library(lme4)
library(lmerTest)
library(scales)
library(brms)

############ POSSESSION LEVEL MIXED-EFFECT MODEL ANALYSES #####################

#these data include all possessions from all subjects (20*157, or 3140 rows)
#missing data (for instance, from eye tracking) are indicated by NaNs
# behavior and memory
d=data.table(read.csv('LinMemRS.csv'))#
d$Sub_num <- factor(d$Sub_num)
d$P_num <- factor(d$P_num)
#add PAC data (note this is formatted to fit ALL possessions)
dPossMemA=data.table(read.csv('PAMemA.csv'))
dPossMemA$Sub_num <- factor(dPossMemA$Sub_num)
d$PAC=dPossMemA$PAC
d$PAC_pre=dPossMemA$PAC_pre
d$Blinks=dPossMemA$Blinks
d$Saccades=dPossMemA$Saccades
d$Shot=dPossMemA$Shot
d$Lum=dPossMemA$Lum
d$LocLum=dPossMemA$LocLum
d$Mot=dPossMemA$Mot
d$LocMot=dPossMemA$LocMot
d$Pros=dPossMemA$Pros
d$Courtpos=dPossMemA$Courtpos
d$Aud=dPossMemA$Aud
d$PAn8=dPossMemA$PAn8
d$PAn7=dPossMemA$PAn7
d$PAn6=dPossMemA$PAn6
d$PAn5=dPossMemA$PAn5
d$PAn4=dPossMemA$PAn4
d$PAn3=dPossMemA$PAn3
d$PAn2=dPossMemA$PAn2
d$PAn1=dPossMemA$PAn1
d$PA0=dPossMemA$PA0
d$PA1=dPossMemA$PA1
d$PA2=dPossMemA$PA2
d$PA3=dPossMemA$PA3
d$PA4=dPossMemA$PA4
d$PA5=dPossMemA$PA5
d$PA6=dPossMemA$PA6
d$PA7=dPossMemA$PA7
d$PA8=dPossMemA$PA8

#add VTA, similarly configured
dPossHMMInd=data.table(read.csv('Event_HMM_XCond-ROI-Poss-Ind-R.csv'))#
d$HC_late_post=dPossHMMInd$HC_late_post
d$VTA_early_post=dPossHMMInd$VTA_early_post
d$VTA_early_pre=dPossHMMInd$VTA_early_pre

#re-configure to make BC / BIC surprise
d$BeliefIncons=(d$Sur+d$Ent_d)/2
d$BeliefCons=(d$Sur-d$Ent_d)/2
d$BeliefIncons_pre=(d$Sur_pre+d$Ent_d_pre)/2
d$BeliefCons_pre=(d$Sur_pre-d$Ent_d_pre)/2

#for these analyses, it's easier to select only participants w/ eyetracking data
dPAC=d[c(472:628,943:2355,2513:3140),] 

######## pupil area change model - Fig 4D, Table S2 ########################
#model #1
lmPACSur<-lmer(PAC~GR+Lum+LocLum+Mot+LocMot+Aud+Pros+Courtpos+Sur+(1|Sub_num),data=d)
summary(lmPACSur)

#bootstrapping control analysis for reviewer
x<-1:14 # #subs with ET data
nPoss<-157 # #possessions
iters<-1000 # #bootstraps
sur_coeff<-matrix(0,iters) #initialize matrix for coefficients
for (i in 1:iters) {
  resample<-sample(x,replace=TRUE) #resample subjects w/ replacement
  #print(resample)
  for (ii in x) {
    begg=(nPoss*(resample[ii]-1))+1 #index of 1st poss
    endg=(nPoss*resample[ii]) #index of last poss
    table1=dPAC[begg:endg,] #grab table rows
    if (ii==1) table2=table1 #initialize if 1st subject in sample
    #source for rbind code
    #https://stackoverflow.com/questions/23079400/concatenate-merge-tables-in-r
    else table2=do.call(rbind,list(table2,table1)) #grow table sub by sub
  }
  lmPACSur_loop<-lmer(PAC~GR+Lum+LocLum+Mot+LocMot+Aud+Pros+Courtpos+Sur+(1|Sub_num),data=table2) #run model
  test<-summary(lmPACSur_loop) #grab summary data
  names(test)
  sur_coeff[i]<-test$coefficients[[10,1]] #10 b/c intercept + 9th predictor
}
#print(sur_coeff)
hist(sur_coeff) #plot histogram of coefficients
percc<-((iters-sum(sur_coeff>0))/iters)*2 #2-tailed p val % of distribution above 0

###### just to mention some more complicated models that ended up having issues
#random, uncorrelated slopes #(Sur||Sub_num) - doesn't converge
#lmPACSur_RS<-lmer(PAC~GR+Lum+LocLum+Mot+LocMot+Aud+Pros+Courtpos+Sur+(Sur||Sub_num),data=d)

#brms analysis - also doesn't get Rhat down low enough, even with ~ maximized parameters
#same deal goes for random slopes (Sur|Sub_num)
#pr=prior(normal(0,1),class="b")
#bmPACSur<-brm(PAC~GR+Lum+LocLum+Mot+LocMot+Aud+Pros+Courtpos+Sur+(1|Sub_num),
#              data=dPAC,
#              iter=20000,
#              prior=pr,
#              chains=2,
#              cores = 4,
#              control=list(max_treedepth=15,adapt_delta=0.98)
#              ) #no random slopes 
#summary(bmPACSur)

#follow-up model - Table S2
lmPACBCBIC<-lmer(PAC~GR+Lum+LocLum+Mot+LocMot+Aud+Pros+Courtpos+BeliefCons+BeliefIncons+(1|Sub_num),data=d)
summary(lmPACBCBIC)

######## pupil area @ each second from -8 to +8 - Fig 4E, Table S3 ############
lmPACSur<-lmer(PAn8~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn7~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn6~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn5~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn4~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn3~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn2~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PAn1~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA0~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA1~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA2~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA3~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA4~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA5~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA6~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA7~Sur+(1|Sub_num),data=d)
summary(lmPACSur)
lmPACSur<-lmer(PA8~Sur+(1|Sub_num),data=d)
summary(lmPACSur)

######## memory model - Fig 6C, Table S5
loglm<-glmer(Mem~Expert+Odd+PrevMem+Sur_pre+Sur+PAC_pre+PAC+
               VTA_early_pre+VTA_early_post+(1|Sub_num),data=d, family = binomial)
summary(loglm)

#bootstrapping control analysis 
x<-1:20 # #subs 
nPoss<-157 # #poss
iters<-1000 # #bootstraps
sur_coeff<-matrix(0,iters) #initialize matrix for coefficients
for (i in 1:iters) {
  resample<-sample(x,replace=TRUE)
  #print(resample)
  for (ii in x) {
    begg=(nPoss*(resample[ii]-1))+1 #index of 1st possession 
    endg=(nPoss*resample[ii]) #index of last possesion
    table1=d[begg:endg,] #grab table
    if (ii==1) table2=table1 #if first subject in this sample
    else table2=do.call(rbind,list(table2,table1)) #grow table bit by bit
  }
  #run model
  lmPACSur_loop<-glmer(Mem~Expert+Odd+PrevMem+Sur_pre+Sur+PAC_pre+PAC+
                         VTA_early_pre+VTA_early_post+(1|Sub_num),data=table2, family = binomial)
  test<-summary(lmPACSur_loop)
  names(test)
  sur_coeff2[i]<-test$coefficients[[6,1]] #6 b/c intercept
}
print(sur_coeff2)
hist(sur_coeff2,breaks=100)
percc<-((iters-sum(sur_coeff2>0))/iters)*2 #2-tailed p val % of distribution above 0

#uncorrelated, random slopes model - unused, but part of reviewer ask
loglm_RS<-glmer(Mem~Expert+Odd+PrevMem+Sur_pre+Sur+PAC_pre+PAC+VTA_early_pre+VTA_early_post+(Sur||Sub_num),data=d, family = binomial)
summary(loglm_RS)
bm<-brm(Mem~Expert+Odd+PrevMem+Sur_pre+Sur+PAC_pre+PAC+VTA_early_pre+VTA_early_post+(1|Sub_num),data=d)
summary(bm) #still sig
bm_RS<-brm(Mem~Expert+Odd+PrevMem+Sur_pre+Sur+PAC_pre+PAC+VTA_early_pre+VTA_early_post+(Sur|Sub_num),data=d)
summary(bm_RS) #still sig

#follow-up model - Table S5
loglmBCBIC<-glmer(Mem~Expert+Odd+PrevMem+BeliefIncons+BeliefCons+PAC_pre+PAC+
               VTA_early_pre+VTA_early_post+(1|Sub_num),data=d, family = binomial)
summary(loglmBCBIC)
