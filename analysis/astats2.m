function [r] = astats2(g1,g2)
%spits back t-stat, p value, and cohen's d, given two groups
[~,p,~,q]=ttest2(g1,g2);t=q.tstat;df=q.df;
n1=length(g1);n2=length(g2);
pools=sqrt(((n1-1)*nanvar(g1)+(n2-1)*nanvar(g2))/(n1+n2-2));
d=(nanmean(g1)-nanmean(g2))/pools;%Cohen's d
r.p=p;r.t=t;r.d=d;r.df=df;r.g1m=nanmean(g1);r.g2m=nanmean(g2);
r.g1s=nanstd(g1)/sqrt(length(g1));r.g2s=nanstd(g2)/sqrt(length(g2));
end