function [vect] = scale01(vect)
%scales data between 0 and 1
tmin=min(vect);tmax=max(vect);trange=tmax-tmin;vect=(vect-tmin)/(trange);
end

