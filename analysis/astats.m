function [r] = astats(g1,g2)
%spits back t-stat, p value, and cohen's d, given two groups
[~,p,~,q]=ttest(g1,g2);t=q.tstat;df=q.df;
n1=length(g1);n2=length(g2);
g3=g1-g2;dz=nanmean(g3)/nanstd(g3);%Cohen's dz
r.p=p;r.t=t;r.dz=dz;r.df=df;r.g1m=nanmean(g1);r.g2m=nanmean(g2);
r.g1s=nanstd(g1)/sqrt(length(g1));r.g2s=nanstd(g2)/sqrt(length(g2));
corr=corrcoef(g1,g2);r.r_within=corr(2);
end