function pimc(x,y,z,tn,cl)
%plot imagesc one way, with colorbar
imagesc(x,y,z);c = colorbar;colormap(jet);c.Label.String = cl;
if strcmpi(cl,'Win probability');c.Limits=[0 100];c.Ticks=[0 20 40 60 80 100];
elseif strcmpi(cl,'Error');c.Limits=[0 1];c.Ticks=[0 0.2 0.4 0.6 0.8 1];
elseif strcmpi(cl,'Suspense');title(tn);%c.Limits=[0 1];c.Ticks=[0 0.2 0.4 0.6 0.8 1];
else;if max(max(z))>30;c.Limits=[0 90];c.Ticks=[0 30 60 90];else;c.Limits=[0 30];c.Ticks=[0 10 20 30];end
end
xlabel('Time left (s)');
if size(z,1)==41;ylabel('Home - visitor score');
else;ylabel('Win probability for winning team');end
set(gca,'YDir','normal');set(gca,'XDir','reverse');%title(tn);
end

