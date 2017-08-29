%%
% Display Cost history at each iteration
%
%%

function [stop] = displayCost(x, values, state)
  iter = values.iter;
  cost = values.fval;
  
  hold on;
  grid on;
  h = figure(1);
  plot(iter, cost);
  xlabel('No of Iterations');
  ylabel('Error/Cost');
  legend('Training Error');
  % fprintf('Iteration %d, Cost: %f\n', iter, cost);

  % stop if the error is less than 0.4%
  stop = cost < 0.004;
  if stop
	% print(h,'-djpg','-color','cost_history.jpg')
	hold off;

end
