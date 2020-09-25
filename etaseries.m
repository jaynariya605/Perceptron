function x = etaseries(a,b,d)

if a==b, 
    error ('first and last element of series can not be same'); 
end
if d <= 2,
    error('Difference of sereies should > = 2.'); 
end

% Linear annealing
n = (b - a)/(d-1);
x = [a:n:b];