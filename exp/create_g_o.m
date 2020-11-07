%create_g_o;
rng('default');reset(RandStream.getGlobalStream,sum(100*clock)); 
%scanner games
gf{1}.gf={'4_8'};gf{1}.home={'(8) Creighton'};
gf{1}.visitor={'(9) Alabama'};gf{1}.hs={'52'};gf{1}.vs={'50'};
gf{2}.gf={'1_7'};gf{2}.home={'(7) Notre Dame'};
gf{2}.visitor={'(10) Xavier'};gf{2}.hs={'57'};gf{2}.vs={'55'};
gf{3}.gf={'2_2'};gf{3}.home={'(2) Missouri'};
gf{3}.visitor={'(15) Norfolk St'};gf{3}.hs={'75'};gf{3}.vs={'74'};
gf{4}.gf={'4_7'};gf{4}.home={'(7) St Marys'};
gf{4}.visitor={'(10) Purdue'};gf{4}.hs={'53'};gf{4}.vs={'63'};
gf{5}.gf={'3_3'};gf{5}.home={'(3) Florida St'};
gf{5}.visitor={'(14) St Bonaventure'};gf{5}.hs={'55'};gf{5}.vs={'52'};
gf{6}.gf={'1_5'};gf{6}.home={'(5) Wichita St'};
gf{6}.visitor={'(12) VCU'};gf{6}.hs={'53'};gf{6}.vs={'54'};
gf{7}.gf={'1_4'};gf{7}.home={'(4) Indiana'};
gf{7}.visitor={'(13) New Mexico St'};gf{7}.hs={'69'};gf{7}.vs={'55'};
gf{8}.gf={'1_6'};gf{8}.home={'(6) UNLV'};
gf{8}.visitor={'(11) Colorado'};gf{8}.hs={'52'};gf{8}.vs={'57'};
gf{9}.gf={'3_7'};gf{9}.home={'(7) Gonzaga'};
gf{9}.visitor={'(10) West Virginia'};gf{9}.hs={'66'};gf{9}.vs={'43'};
sgs=tngs;
for ng=1:length(gf)
    fn=[root 'exp/vids/720res/' char(gf{ng}.gf) 'cut.mov'];
    Phil=VideoReader(fn);
    ml(ng) = Phil.Duration;
end
[mls,mlso]=sort(ml);
mlso1(:,1)=mlso(1:ngs);mlso1(:,1)=mlso1(randperm(ngs),1);
mlso1(:,2)=mlso(ngs+1:ngs*2);mlso1(:,2)=mlso1(randperm(ngs),2);
mlso1(:,3)=mlso(ngs*2+1:ngs*3);mlso1(:,3)=mlso1(randperm(ngs),3);
maxmll=15;
while maxmll<19
    b=randperm(ngs);c=randperm(tngs/ngs);
    g_o=[mlso1(b(1),c(1)) mlso1(b(1),c(2)) mlso1(b(1),c(3)) mlso1(b(2),c(1)) mlso1(b(2),c(2)) mlso1(b(2),c(3)) ...
        mlso1(b(3),c(1)) mlso1(b(3),c(2)) mlso1(b(3),c(3))];
    mll1=sum(ml(g_o(1:ngs)))/60;mll2=sum(ml(g_o(ngs+1:ngs*2)))/60;mll3=sum(ml(g_o(ngs*2+1:ngs*3)))/60;
    maxmll=max([mll1 mll2 mll3]);
    Indiana=find(g_o==7);VCU=find(g_o==6);if Indiana<VCU;maxmll=18;end
    Gonzaga=find(g_o==9);if Gonzaga<VCU;maxmll=18;end
    StMarys=find(g_o==4);Missouri=find(g_o==3);if StMarys<Missouri;maxmll=18;end
end
% for i=1:1000
%     b=randperm(ngs);
%     g_o=[mlso1(b(1)) mlso2(b(2)) mlso3(b(3)) mlso1(b(2)) mlso2(b(3)) mlso3(b(1)) ...
%         mlso1(b(3)) mlso2(b(1)) mlso3(b(2))]
%     mll1=sum(ml(g_o(1:ngs)))/60;mll2=sum(ml(g_o(ngs+1:ngs*2)))/60;mll3=sum(ml(g_o(ngs*2+1:ngs*3)))/60;
%     maxmll(i)=max([mll1 mll2 mll3]);minmll(i)=min([mll1 mll2 mll3]);
%     D(i)=maxmll(i)-minmll(i);
% end
% max(maxmll);max(D);
% these are for checking maximum length and maximum disparity
%   20.1433    0.8583
%g_o=randperm(tngs);
%beliefs games
q=sgs+1;gf{q}.gf={'1_2'};gf{q}.home={'(2) Duke'};
gf{q}.homestring=gf{q}.home{1}(5:end);
gf{q}.visitor={'(15) Lehigh'};gf{q}.hs={'51'};gf{q}.vs={'55'};
q=sgs+2;gf{q}.gf={'2_8'};gf{q}.home={'(8) Memphis'};
gf{q}.homestring=gf{q}.home{1}(5:end);
gf{q}.visitor={'(9) Saint Louis'};gf{q}.hs={'44'};gf{q}.vs={'45'};
q=sgs+3;gf{q}.gf={'3_6'};gf{q}.home={'(6) Cincinnati'};
gf{q}.homestring=gf{q}.home{1}(5:end);
gf{q}.visitor={'(11) Texas'};gf{q}.hs={'51'};gf{q}.vs={'48'};
q=sgs+4;gf{q}.gf={'4_4'};gf{q}.home={'(4) Michigan'};
gf{q}.homestring=gf{q}.home{1}(5:end);
gf{q}.visitor={'(13) Ohio'};gf{q}.hs={'57'};gf{q}.vs={'61'};
q=sgs+5;gf{q}.gf={'2_5'};gf{q}.home={'(5) New Mexico'};
gf{q}.homestring=gf{q}.home{1}(5:end);
gf{q}.visitor={'(12) Long Beach St.'};gf{q}.hs={'59'};gf{q}.vs={'59'};
gb_o=randperm(tnbgs);

q=100+1;gf{q}.home={'(2) Duke'};gf{q}.visitor={'(15) Albany'};gf{q}.hs={'64'};gf{q}.vs={'54'};gf{q}.hwp={'98.9'};
q=q+1;gf{q}.home={'(2) Ohio State'};gf{q}.visitor={'(15) Iona'};gf{q}.hs={'86'};gf{q}.vs={'58'};gf{q}.hwp={'100'};
q=q+1;gf{q}.home={'(2) Georgetown'};gf{q}.visitor={'(15) Florida Gulf Coast'};gf{q}.hs={'47'};gf{q}.vs={'59'};gf{q}.hwp={'6.7'};
q=q+1;gf{q}.home={'(2) Miami (FL)'};gf{q}.visitor={'(15) Pacific'};gf{q}.hs={'71'};gf{q}.vs={'43'};gf{q}.hwp={'100'};
q=q+1;gf{q}.home={'(3) Michigan State'};gf{q}.visitor={'(14) Valparaiso'};gf{q}.hs={'58'};gf{q}.vs={'40'};gf{q}.hwp={'99.8'};
q=q+1;gf{q}.home={'(3) New Mexico'};gf{q}.visitor={'(14) Harvard'};gf{q}.hs={'53'};gf{q}.vs={'57'};gf{q}.hwp={'31.7'};
q=q+1;gf{q}.home={'(3) Florida'};gf{q}.visitor={'(14) Northwestern State'};gf{q}.hs={'68'};gf{q}.vs={'45'};gf{q}.hwp={'100'};
q=q+1;gf{q}.home={'(3) Marquette'};gf{q}.visitor={'(14) Davidson'};gf{q}.hs={'42'};gf{q}.vs={'49'};gf{q}.hwp={'15.9'};
q=q+1;gf{q}.home={'(4) Saint Louis'};gf{q}.visitor={'(13) New Mexico State'};gf{q}.hs={'59'};gf{q}.vs={'39'};gf{q}.hwp={'99.9'};
q=q+1;gf{q}.home={'(4) Kansas State'};gf{q}.visitor={'(13) La Salle'};gf{q}.hs={'60'};gf{q}.vs={'58'};gf{q}.hwp={'67.6'};
q=q+1;gf{q}.home={'(4) Michigan'};gf{q}.visitor={'(13) South Dakota State'};gf{q}.hs={'63'};gf{q}.vs={'50'};gf{q}.hwp={'99.3'};
q=q+1;gf{q}.home={'(4) Syracuse'};gf{q}.visitor={'(13) Montana'};gf{q}.hs={'76'};gf{q}.vs={'29'};gf{q}.hwp={'100'};
q=q+1;gf{q}.home={'(5) Oklahoma State'};gf{q}.visitor={'(12) Oregon'};gf{q}.hs={'48'};gf{q}.vs={'60'};gf{q}.hwp={'1.7'};
q=q+1;gf{q}.home={'(5) Wisconsin'};gf{q}.visitor={'(12) Mississippi'};gf{q}.hs={'41'};gf{q}.vs={'45'};gf{q}.hwp={'24.1'};
q=q+1;gf{q}.home={'(5) VCU'};gf{q}.visitor={'(12) Akron'};gf{q}.hs={'86'};gf{q}.vs={'39'};gf{q}.hwp={'100'};
q=q+1;gf{q}.home={'(5) UNLV'};gf{q}.visitor={'(12) Cal'};gf{q}.hs={'48'};gf{q}.vs={'54'};gf{q}.hwp={'10.2'};
q=q+1;gf{q}.home={'(6) Memphis'};gf{q}.visitor={'(11) St. Mary''s'};gf{q}.hs={'44'};gf{q}.vs={'40'};gf{q}.hwp={'70.9'};
q=q+1;gf{q}.home={'(6) Arizona'};gf{q}.visitor={'(11) Belmont'};gf{q}.hs={'66'};gf{q}.vs={'54'};gf{q}.hwp={'98.6'};
q=q+1;gf{q}.home={'(6) UCLA'};gf{q}.visitor={'(11) Minnesota'};gf{q}.hs={'56'};gf{q}.vs={'77'};gf{q}.hwp={'0.2'};
q=q+1;gf{q}.home={'(6) Butler'};gf{q}.visitor={'(11) Bucknell'};gf{q}.hs={'49'};gf{q}.vs={'42'};gf{q}.hwp={'86.7'};
q=q+1;gf{q}.home={'(7) Creighton'};gf{q}.visitor={'(10) Cincinnati'};gf{q}.hs={'52'};gf{q}.vs={'49'};gf{q}.hwp={'76.5'};
q=q+1;gf{q}.home={'(7) Notre Dame'};gf{q}.visitor={'(10) Iowa State'};gf{q}.hs={'47'};gf{q}.vs={'74'};gf{q}.hwp={'0'};
q=q+1;gf{q}.home={'(7) San Diego State'};gf{q}.visitor={'(10) Oklahoma'};gf{q}.hs={'61'};gf{q}.vs={'51'};gf{q}.hwp={'93.4'};
q=q+1;gf{q}.home={'(7) Illinois'};gf{q}.visitor={'(10) Colorado'};gf{q}.hs={'50'};gf{q}.vs={'46'};gf{q}.hwp={'82.2'};
q=q+1;gf{q}.home={'(8) Colorado State'};gf{q}.visitor={'(9) Missouri'};gf{q}.hs={'74'};gf{q}.vs={'65'};gf{q}.hwp={'92.1'};
q=q+1;gf{q}.home={'(8) Pittsburgh'};gf{q}.visitor={'(9) Wichita State'};gf{q}.hs={'40'};gf{q}.vs={'56'};gf{q}.hwp={'1'};
q=q+1;gf{q}.home={'(8) North Carolina'};gf{q}.visitor={'(9) Villanova'};gf{q}.hs={'63'};gf{q}.vs={'54'};gf{q}.hwp={'93.8'};
q=q+1;gf{q}.home={'(8) North Carolina State'};gf{q}.visitor={'(9) Temple'};gf{q}.hs={'51'};gf{q}.vs={'59'};gf{q}.hwp={'8.2'};
gs_o=randperm(tnsgs);

save(sn,'g_o','gb_o','gs_o','ml','gf');