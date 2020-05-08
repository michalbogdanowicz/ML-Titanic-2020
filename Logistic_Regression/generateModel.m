function X = generateModel(data, max_power)
% rows == the number of rows
% power == the maximum power of the function
% data == the data that needs to be used to creat the model function formula
  n_rows = rows(data);
  X = [ones(n_rows,1)]; %start with only the bias
  for power = 1 : max_power
    for current_position = 1 : columns(data) 
      X = [X data(:,current_position).^ power];
    endfor
 endfor
end
