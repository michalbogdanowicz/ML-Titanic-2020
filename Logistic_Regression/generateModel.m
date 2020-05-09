function X = generateModel(data, max_power)
% max_power == the maximum power of the function
% data == the data that needs to be used to creat the model function formula
  n_rows = rows(data);
  X = [ones(n_rows,1)]; %start with only the bias
  temp = data .* (data != 1); % only consider data that is different from 0
  temp = sum(temp); %sum by column.
  temp = temp != 0; % take the columns that are different than 0
  % count the total columns
  % fprintf('the array of the elments with columns with values differen tform 0 and 1 are %d \n', sum(temp(:)))
  
  % now add the power of the featuers only when it makes sense.
  for power = 1 : max_power
    for current_position = 1 : columns(data) 
      if (temp(current_position) == 1 || power == 1)
        X = [X data(:,current_position).^ power];
      endif
    endfor
 endfor
end
