import random
from collections.abc import Sequence
from itertools import repeat
from torch.utils.data import DataLoader
import torch
from copy import deepcopy
from typing import Sequence
import warnings
import numpy as np

from models import MutationNetwork, FeatureScaler, MutationDataset

def learned_mutation(population, policy_path, lower, upper, 
                      policy = None, prob_per_variable = None, batch_size = 16, precision = 3, *args, **kwargs):
    """
    AI-driven mutation policy.
    Works only with float genes.
    This is the offline version (model comes from home already trained).

    :param population: Alleles of the population to mutate (lists of floats).
    :param lower: Lower bounds for each gene.
    :param upper: Upper bounds for each gene.    
    :param policy: A neural network producing mutated individuals.         
    :param prob_per_variable: Probability to mutate for each gene (by default, 1 / number of genes).
    :param batch_size: How many individuals are passed together (in batch) to the model.
    :param precision: Determines to which decimal mutated alleles are rounded.    
    :returns: Mutated individual as a list.
    """
    # Verify inputs      
    size = len(population[0])
    if any(len(ind) != size for ind in population):
        raise ValueError("Individuals have inconsistent lengths")
    
    if len(population) > batch_size:        
        batch_size = len(population)
    
    if not isinstance(lower, Sequence):
        lower = [lower] * size
    elif len(lower) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(lower), size))
    if not isinstance(upper, Sequence):
        upper = [upper] * size
    elif len(upper) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(upper), size))
    
    # Define gene mutation probability
    if prob_per_variable is None:
        prob_per_variable = 1.0 / size

    # Load model
    if policy is None:
        policy = MutationNetwork(input_dim=size) 
        policy.load_state_dict(torch.load(policy_path)) # Offline
    
    # Normalize alleles and pass them to the correct formatting
    scaler = FeatureScaler(lower_bounds=lower, upper_bounds=upper)
    dataset = MutationDataset(np.array(population), transform=scaler)     
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Select GPU/CPU   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform inference
    policy.eval()    
    policy.to(device)
    children = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            outputs = policy(x_batch)    
            children.append(outputs.cpu())
    children = torch.cat(children, dim=0).numpy()  # Flatten        

    # Round and mutate individuals
    for idx, mut_ind in enumerate(children):
        mut_ind = scaler.inverse(mut_ind) # De-normalize individuals

        # Clamp the values
        clamped_ind = [] 
        for x, xl, xu in zip(mut_ind, lower, upper):
            if x != min(max(x, xl), xu):
                warnings.warn('A gene of an individual was clamped. If this happens a lot, the deep learning model may not be working correctly')
            clamped_ind.append(min(max(x, xl), xu))
        mut_ind = clamped_ind
        
        mut_ind = [round(i, precision) for i in mut_ind]  # Round the values

        for i in range(size): # Mutate genes
            if random.random() <= prob_per_variable:
                population[idx][i] = mut_ind[i]

    return population 

def polynomial_dynamic_mutation(ind, lower, upper, mut_param=5, prob_per_variable=None, precision=3, 
                                prev_fitness=None, current_fitness=None, improvement_threshold=2, *args, **kwargs):
    """
    Polynomial mutation with adaptive mutation parameter and mutation probability.

    :param ind: List of float genes.
    :param lower: Lower bounds (scalar or list).
    :param upper: Upper bounds (scalar or list).
    :param mut_param: Initial mutation parameter.
    :param prob_per_variable: Initial mutation probability.
    :param precision: Decimal rounding.
    :param prev_fitness: Fitness before mutation.
    :param current_fitness: Fitness after mutation.
    :param improvement_threshold: Minimum required improvement to consider as 'better'.
    """
    size = len(ind)
    if prob_per_variable is None:
        prob_per_variable = 1.0 / size

    # Dynamic adjustment based on performance
    if prev_fitness is not None and current_fitness is not None:        
        improved = current_fitness > prev_fitness + improvement_threshold
        relative_improvement = max(0.0, current_fitness - prev_fitness) / (prev_fitness + 1e-8)
        if not improved:            
            # Increase mutation strength & probability
            decay = 0.5 * (improvement_threshold - relative_improvement)
            mut_param = max(mut_param * (1.0 - decay), 0.5)
            prob_per_variable = min(prob_per_variable * (1.0 + decay), 1.0)
    
    # This calls the original polynomial mutation function, that cannot be shown
    return polynomial_mutation(ind, lower, upper, mut_param=mut_param, prob_per_variable=prob_per_variable, precision=precision, *args, **kwargs)

def guide_mutation(population, fitnesses, elite_ratio=0.1, poor_ratio=0.3, alpha=0.3):
    """
    Moves the worst individuals towards the best ones.

    :param population: List of individuals (list of lists)
    :param fitnesses: List of fitness values
    :param elite_ratio: Fraction of top-performing individuals to be elites
    :param poor_ratio: Fraction of worst-performing individuals to be guided
    :param alpha: Step size towards the elite (0 < alpha <= 1)
    :return: Updated population
    """
    pop_size = len(population)
    num_elites = max(1, int(elite_ratio * pop_size))
    num_poor = max(1, int(poor_ratio * pop_size))

    # Sort population by fitness
    sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
    elites = [population[i] for i in sorted_indices[:num_elites]]
    poors = sorted_indices[-num_poor:]

    # Move each poor individual toward a random elite
    for idx in poors:
        elite = random.choice(elites)
        poor = population[idx]
        new_ind = [
            p + alpha * (e - p) for p, e in zip(poor, elite)
        ]
        population[idx] = new_ind 

    return population

def multi_parent_sbx(parents, lower, upper, cx_param=2, prob_per_variable=0.5, precision=3):
    """
    Performs multi-parent simulated binary crossover (SBX).
    
    :param parents: List of parent individuals (each is a list of floats).
    :param lower: Lower bounds for each gene (list or scalar).
    :param upper: Upper bounds for each gene (list or scalar).
    :param cx_param: Crossover distribution index (higher = less variation).
    :param prob_per_variable: Probability to perform crossover on a gene.
    :param precision: Decimal rounding precision for resulting genes.
    :return: List of offspring individuals (same number as parents).
    """
    num_parents = len(parents)
    size = min(len(p) for p in parents)
    
    if not isinstance(lower, Sequence):
        lower = [lower] * size
    elif len(lower) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(lower), size))
    if not isinstance(upper, Sequence):
        upper = [upper] * size
    elif len(upper) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(upper), size))

    # Deepcopy to preserve original parents
    offspring = [deepcopy(p) for p in parents]

    for i in range(size):  # For each allele
        if random.random() <= prob_per_variable:
            # Extract allele values from all parents at this position
            gene_values = [p[i] for p in parents]
            x_min = min(gene_values)
            x_max = max(gene_values)

            if abs(x_max - x_min) < 1e-14:
                continue  # No meaningful variation

            for j in range(num_parents):
                xl, xu = lower[i], upper[i] # Bounds
                rand = random.random()

                beta = 1.0 + (2.0 * (gene_values[j] - xl) / (x_max - x_min)) # Calculate beta
                alpha = 2.0 - beta ** -(cx_param + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (cx_param + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (cx_param + 1))

                c = 0.5 * (x_min + x_max - ((-1) ** j) * beta_q * (x_max - x_min)) # Create offspring
                c = min(max(c, xl), xu)
                offspring[j][i] = round(c, precision)

    return offspring