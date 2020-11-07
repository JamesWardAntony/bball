function vplot(sn,dist,true,fs,lw)
%violin plot function that calls 'violin.m', sets position, and saves.
%note: this is for a single distribution vs single true value. if multiple
%columns are desired, will have to re-work this into another function.
msz=400;rg=-0.5:0.5:0.5;
figure;violin(dist','mc',[],'medc',[],'facecolor',[0 0 0],'edgecolor','k');
set(gca,'YTick',rg,'YLim',[rg(1) rg(end)],'FontName','Helvetica',...
    'FontSize',fs,'box','off','LineWidth',lw);
gg=gca;gg.XAxis.Visible='off';gg.XTick=[];ylabel('{\it r}');
hold on;scatter(1,true,msz,'k','filled');ppos2;
% sets position and orientation for plotting
h=gcf;set(h,'Position',[50 50 300 300]);
set(h,'PaperOrientation','landscape');
print(sn,'-dpdf','-bestfit');
end

