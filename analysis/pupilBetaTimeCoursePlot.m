%plots pupil beta time course (using pupil area, not pupil area change) 
%featuring correlations with surprise @ each time point, input from R
%produces Fig 4E
%input plin, betas, stes, adj_p (q) into Table S3
clear;close all;[base,root]=load_root();
plotdefs2;set(0,'DefaultFigureWindowStyle','normal');set(0,'DefaultAxesFontSize',25);
betas=[-4.4 -6 -4.3 -5.4 -4.0 ...
    -3.4 2.9 5.0 8.5 7.4 ...
    9.7 8.5 9.9 6.9 4.6 4.8 4.4];%input betas manually from R 
stes=[2.7 2.7 2.7 2.7 2.7 ...
    2.7 2.7 2.8 2.8 2.7 ...
    2.7 2.7 2.7 2.7 2.7 2.7 2.7];%standard errors from the mixed effect estimate
tvals=[1.7 2.3 1.6 2.0 1.5 ...
    1.3 1.1 1.8 3.0 2.8 ...
    3.6 3.1 3.7 2.6 1.7 1.8 1.6];
pvals=[0.10 0.024 0.11 0.044 0.14 ...
    0.2 0.28 0.08 0.0026 0.0057 ...
    0.0003 0.0017 0.00025 0.0096 0.087 0.072 0.1];
[h, crit_p,adj_ci_cvrg,adj_p]=fdr_bh(pvals,0.05,'dep','yes');adj_p(adj_p>1)=1;
betas=betas/1000;stes=stes/1000;
figure;prange=8;plin=linspace(-prange,prange,prange*2+1)+0.5;
errorbar(plin,betas,stes,'k','LineWidth',4);gg=gca;gg.LineWidth=2;
xlabel('Time (relative to possession change)');ylabel('Beta');box off;
sn=[root 'pics/SurpriseVPhasicPupilAreaBETASTimeCourse'];print(sn,'-dpdf','-bestfit');
adj_p=adj_p';betas=betas';plin=plin';
pvals=pvals';stes=stes';tvals=tvals';
save pupilTimePVals plin betas stes adj_p tvals pvals;