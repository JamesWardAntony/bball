function psc(dat,yl,cons,cats,dc,sn,tilt)
%this customizes/ condenses the within-subject plotSpread connector code
plotdefs;connw=0.5;lw=3;aw=4;compco=[0.75 0.75 0.75];
figure;h=plotSpread(dat,'yLabel',yl,'distributionColors',dc,'xNames',cats);if exist('xl','var');xlabel(xl);end
if ~isempty(cons)
    for ii=1:size(cons,1)
        i1=cons(ii,1);i2=cons(ii,2);
        temp = get(h{1}(i1));axVals = temp.XData;temp = get(h{1}(i2));bxVals = temp.XData;
        for i=1:size(axVals,2);plot([axVals(i) bxVals(i)],[dat(i,i1),dat(i,i2)],'Color',compco,'LineWidth',connw);end;hold on;%'k'
    end
end
for pt=1:size(dat,2)
    ec=char(dc(pt));
    bar(pt,mean(dat(~isnan(dat(:,pt)),pt)),'BarWidth',0.6,'LineWidth',lw,'FaceColor','none','EdgeColor',ec);
end
ax=gca;ax.LineWidth=aw;
set(gca,'XTick',1:length(cats),'XTickLabel',cats,'FontName','Helvetica');
gg=gca;gg.XAxis.Visible='off';gg.XTick=[];gg.YLim=[0 1];gg.YTick=[0 0.25 0.50 0.75 1];box off;gg.FontSize=30;
if exist('tilt','var');xtickangle(45);end
ppos;print(sn,'-dpdf','-bestfit');
end