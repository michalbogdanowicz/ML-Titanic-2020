function plotBoundry(theta,X,y,regFlag)
  positive = find(y==1);
  negative = find(y==0);
  if regFlag==0
    figure;hold on;
    % Plot, and adjust axes for better viewing
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    plot(X(positive,2),X(positive,3), 'g+','LineWidth',2,'MarkerSize',7);
    plot(X(negative, 2), X(negative, 3), 'r*', 'MarkerFaceColor', 'y', 'MarkerSize',7);
    plot(plot_x, plot_y);
    hold on;
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    title('Decision Boundary for non regularized model');
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
    hold off;
   else
    u= X(:,2);
    v= X(:,3);
    fh = @(u,v) -0.0824-u.*0.268 - v.*0.41 +u.^2*3.8e-06+ v.^2*0.001233 +u.*(v.*0.0105);
    fprintf('\nProgram paused. Press enter to continue.\n');
    pause;
    figure;hold on;

    plot(X(positive,2),X(positive,3), 'g+','LineWidth',2,'MarkerSize',7);
    plot(X(negative, 2), X(negative, 3), 'r*', 'MarkerFaceColor', 'y', 'MarkerSize',7);

    hold on;
    ezplot(fh,[28 100]);
    axis equal;

    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    title('Decision Boundary for regularized model');
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    hold off;
   endif
end
