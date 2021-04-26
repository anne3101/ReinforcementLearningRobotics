classdef FindExtrema
    
    properties
        x_init
        learning_rate
        max_iteration
        decay
        decay_rate
        min_learning_rate
    end
    
    methods
        
        function objFunc = objectiveFunction(obj,x)
            % Calculate function value of objective function
            objFunc = 0.99 * x.^5 - 5 * x.^4 + 4.98 * x.^3 + 5 * x.^2 - 6*x -1; 
        end
        
        function grad = analyticGradient(obj,x)
            % Calculate analytic gradient of function f(x)
            grad = 4.95 * x.^4 - 20 * x.^3 + 14.94 * x.^2 + 10*x -6;  
        end
        
        function [x_min, objFunc_min, iteration] = gradientDescent(obj)
             iteration = 0;
             step_size = obj.learning_rate;
             beta_1 = 0.9;
             beta_2 = 0.999;
             eps = 1e-8;
             m = 0;
             v = 0;
            
             % Plot objective function   
             init_plot_objectiveFunction(obj)
            
             x = obj.x_init;
             grad = analyticGradient(obj,x);
             plot_iteration(obj, x);
             
             while (abs(grad) > 1e-1)
                iteration = iteration + 1;
                
                % Calculate current gradient
                grad = analyticGradient(obj,x);
                
                
                m = beta_1 * m + (1-beta_1)*grad;
                v = beta_2 *v + (1-beta_2)*grad.^2;
                
                m_tilde = m / (1 - beta_1);
                v_tilde = v / (1 - beta_2);
                
                % Update step
                x = x - step_size * m_tilde/ (sqrt(v_tilde) + eps);
                objFunc = objectiveFunction(obj,x);
   
                x_min = x;
                objFunc_min = objFunc;
                
                plot_iteration(obj, x);
                
                % Learning rate decay
                if obj.decay == true && step_size > obj.min_learning_rate
                    step_size = step_size - obj.decay_rate * step_size;
                end
                
             end 
        end
        
        function [x_max, objFunc_max, iteration] = gradientAscent(obj)
             iteration = 0;
             step_size = obj.learning_rate; 
             beta_1 = 0.9;
             beta_2 = 0.999;
             eps = 1e-8;
             m = 0;
             v = 0;
            
             % Plot objective function   
             init_plot_objectiveFunction(obj)
            
             x = obj.x_init;
             grad = analyticGradient(obj,x);

             plot_iteration(obj, x);
             
             while (abs(grad) > 1e-1)
                iteration = iteration + 1;
                
                % Calculate gradient
                grad = analyticGradient(obj,x);
                
                m = beta_1 * m + (1-beta_1)*grad;
                v = beta_2 *v + (1-beta_2)*grad.^2;
                
                m_tilde = m / (1 - beta_1);
                v_tilde = v / (1 - beta_2);
                
                % Update step
                x = x + step_size * m_tilde/ (sqrt(v_tilde) + eps);
                objFunc = objectiveFunction(obj,x);
                
                x_max = x;
                objFunc_max = objFunc;
                
                plot_iteration(obj, x)
                
                % Learning rate decay
                if obj.decay == true
                    step_size = step_size - obj.decay_rate * step_size;
                end
                
             end 
        end
        
        function plot_iteration(obj, x_current)
            f_current = objectiveFunction(obj,x_current);
            plot(x_current,f_current,'r*')
            hold on
            drawnow
        end
        
        function init_plot_objectiveFunction(obj)
            x_input = -1 : 0.05 : 3;
            objFunc = objectiveFunction(obj,x_input);
            figure
            plot(x_input, objFunc,'b')
            title(sprintf('Objective Function with x_{init} = %.0f',obj.x_init))
            xlabel('x')
            ylabel('f(x)')
            hold on
        end
            
            
    end
    
end