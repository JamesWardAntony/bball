% runs a simple correlation between two columns of a matrix
figure;scatter(a(:,1),a(:,2));xlabel(xl);ylabel(yl);lsline;
tn =['r = ' num2str(r)];title(tn);