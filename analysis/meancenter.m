function [vect] = meancenter(vect)
vect=vect-nanmean(vect);%mean center data
end

