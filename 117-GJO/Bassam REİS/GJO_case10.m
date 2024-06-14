function [bestSolution, bestFitness, iteration]=GJO_case10(fhd, dimension, maxIteration, fNumber)

config;

%% initialize Golden jackal pair
dim = dimension;
Male_Jackal_pos = zeros(1, dim);
Male_Jackal_score = inf; 
Female_Jackal_pos = zeros(1, dim);  
Female_Jackal_score = inf; 
lb = lbArray;
ub = ubArray;
Max_iter = maxIteration;
SearchAgents_no = 50; % Number of search agents

%% Initialize the positions of search agents
Positions = initialization(SearchAgents_no, dim, ub, lb);

Convergence_curve = zeros(1, Max_iter);

% Set threshold for diversity
diversity_threshold = 0.1 * (ub - lb);

% Loop counter
Max_iter = Max_iter / SearchAgents_no;

% Main loop
for l = 1:Max_iter
    for i = 1:size(Positions, 1)  

        % Boundary checking
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;               

        % Calculate objective function for each search agent
        fitness(i) = testFunction(Positions(i, :)', fhd, fNumber);

        % Update Male Jackal 
        if fitness(i) < Male_Jackal_score 
            Male_Jackal_score = fitness(i); 
            Male_Jackal_pos = Positions(i, :);
        end  
        if fitness(i) > Male_Jackal_score && fitness(i) < Female_Jackal_score 
            Female_Jackal_score = fitness(i); 
            Female_Jackal_pos = Positions(i, :);
        end
    end
    
    E1 = 1.5 * (1 - (l / Max_iter));
    RL = 0.05 * levy(SearchAgents_no, dim, 1.5);
     
    % Calculate population diversity
    diversity = sum(std(Positions));
    
    for i = 1:size(Positions, 1)
        for j = 1:size(Positions, 2)
            r1 = rand(); % r1 is a random number in [0,1]
            E0 = 2 * r1 - 1;            
            E = E1 * E0; % Evading energy
            
            if abs(E) < 1
                %% EXPLOITATION
                if diversity < diversity_threshold
                    index = dFDB(Positions, fitness, 10, l, Max_iter);
                else
                    index = fitnessDistanceBalance(Positions, fitness);
                end
                D_male_jackal = abs((RL(i,j) * Male_Jackal_pos(j) - Positions(index,j)));
                Male_Positions(i,j) = Male_Jackal_pos(j) - E * D_male_jackal;
                
                random_index = randi(size(Positions, 1));
                D_female_jackal = abs((RL(i,j) * Female_Jackal_pos(j) - Positions(random_index,j)));
                Female_Positions(i,j) = Female_Jackal_pos(j) - E * D_female_jackal;
            else
                %% EXPLORATION
                if diversity < diversity_threshold
                    index = dFDB(Positions, fitness, 10, l, Max_iter);
                else
                    index = fitnessDistanceBalance(Positions, fitness);
                end
                D_male_jackal = abs((Male_Jackal_pos(j) - RL(i,j) * Positions(index,j)));
                Male_Positions(i,j) = Male_Jackal_pos(j) - E * D_male_jackal;
                
                random_index = randi(size(Positions, 1));
                D_female_jackal = abs((Female_Jackal_pos(j) - RL(i,j) * Positions(random_index,j)));
                Female_Positions(i,j) = Female_Jackal_pos(j) - E * D_female_jackal;
            end
            
            Positions(i,j) = (Male_Positions(i,j) + Female_Positions(i,j)) / 2;
        end
    end
    
    Convergence_curve(l) = Male_Jackal_score;
end

bestSolution = Male_Jackal_pos;
bestFitness = Male_Jackal_score;
iteration = l;

end