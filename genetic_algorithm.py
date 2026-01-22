"""
Genetic Algorithm for Portfolio Optimization
Research: Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs
Author: Kholood Alsager, Nada Almuairi

This module implements the Genetic Algorithm used to optimize portfolio weights
subject to risk minimization and return constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class GeneticAlgorithmPortfolio:
    """
    Genetic Algorithm for portfolio weight optimization
    
    Objective: Minimize portfolio risk (variance)
    Subject to: Minimum return constraint (≥ 0.92%)
    
    Based on Section 4.5.2 of the research paper
    """
    
    def __init__(self, 
                 population_size=100,
                 n_generations=200,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 elite_size=5,
                 random_seed=42):
        """
        Initialize Genetic Algorithm
        
        Parameters:
        -----------
        population_size : int
            Number of individuals in population
        n_generations : int
            Number of generations to evolve
        crossover_rate : float
            Probability of crossover (0-1)
        mutation_rate : float
            Probability of mutation (0-1)
        elite_size : int
            Number of best individuals to preserve
        random_seed : int
            Random seed for reproducibility
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Storage for results
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.convergence_history = []
    
    def initialize_population(self, n_assets):
        """
        Initialize random population of portfolio weights
        
        Parameters:
        -----------
        n_assets : int
            Number of assets in portfolio
        
        Returns:
        --------
        population : np.array
            Population of weight vectors, shape (population_size, n_assets)
        """
        population = np.random.random((self.population_size, n_assets))
        
        # Normalize weights to sum to 1
        population = population / population.sum(axis=1, keepdims=True)
        
        return population
    
    def fitness_function(self, weights, returns, cov_matrix, 
                        fuzzy_risk_adjustments, min_return=0.92):
        """
        Fitness function: portfolio risk (with penalty for constraint violation)
        
        Objective: Minimize σ²ₚ = Σᵢ Σⱼ wᵢwⱼσᵢⱼ
        
        Parameters:
        -----------
        weights : np.array
            Portfolio weights
        returns : np.array
            Expected returns for each asset
        cov_matrix : np.array
            Covariance matrix
        fuzzy_risk_adjustments : np.array
            Fuzzy-adjusted risk factors from M-PFFG
        min_return : float
            Minimum required return (%)
        
        Returns:
        --------
        fitness : float
            Fitness value (lower is better)
        """
        # Calculate portfolio variance (risk)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        # Apply fuzzy risk adjustments
        fuzzy_adjusted_risk = portfolio_variance * np.dot(weights, fuzzy_risk_adjustments)
        
        # Calculate portfolio return
        portfolio_return = np.dot(weights, returns)
        
        # Penalty for violating minimum return constraint
        return_penalty = 0
        if portfolio_return < min_return:
            return_penalty = 1000 * (min_return - portfolio_return) ** 2
        
        # Penalty for weights not summing to 1 (should be near 0 due to normalization)
        weight_sum_penalty = 1000 * (abs(weights.sum() - 1.0)) ** 2
        
        # Total fitness (lower is better)
        fitness = fuzzy_adjusted_risk + return_penalty + weight_sum_penalty
        
        return fitness
    
    def selection(self, population, fitness_values):
        """
        Tournament selection
        
        Parameters:
        -----------
        population : np.array
            Current population
        fitness_values : np.array
            Fitness values for population
        
        Returns:
        --------
        selected : np.array
            Selected individuals for reproduction
        """
        tournament_size = 3
        selected_indices = []
        
        for _ in range(self.population_size - self.elite_size):
            # Randomly select tournament_size individuals
            tournament_indices = np.random.choice(
                len(population), 
                size=tournament_size, 
                replace=False
            )
            
            # Select the best from tournament
            tournament_fitness = fitness_values[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        return population[selected_indices]
    
    def crossover(self, parent1, parent2):
        """
        Uniform crossover
        
        Parameters:
        -----------
        parent1, parent2 : np.array
            Parent weight vectors
        
        Returns:
        --------
        child1, child2 : np.array
            Offspring weight vectors
        """
        if np.random.random() < self.crossover_rate:
            # Uniform crossover mask
            mask = np.random.random(len(parent1)) < 0.5
            
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            
            # Normalize to ensure sum = 1
            child1 = child1 / child1.sum()
            child2 = child2 / child2.sum()
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Gaussian mutation
        
        Parameters:
        -----------
        individual : np.array
            Weight vector to mutate
        
        Returns:
        --------
        mutated : np.array
            Mutated weight vector
        """
        if np.random.random() < self.mutation_rate:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05, len(individual))
            mutated = individual + noise
            
            # Ensure non-negative
            mutated = np.maximum(mutated, 0)
            
            # Normalize
            mutated = mutated / mutated.sum()
        else:
            mutated = individual.copy()
        
        return mutated
    
    def optimize(self, returns, cov_matrix, fuzzy_risk_adjustments, 
                 stock_names=None, min_return=0.92, verbose=True):
        """
        Run genetic algorithm optimization
        
        Parameters:
        -----------
        returns : np.array
            Expected returns for each asset
        cov_matrix : np.array
            Covariance matrix
        fuzzy_risk_adjustments : np.array
            Fuzzy-adjusted risk factors
        stock_names : list, optional
            Names of stocks
        min_return : float
            Minimum required return (%)
        verbose : bool
            Print progress
        
        Returns:
        --------
        best_weights : np.array
            Optimal portfolio weights
        best_fitness : float
            Optimal fitness value
        """
        n_assets = len(returns)
        
        if stock_names is None:
            stock_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Initialize population
        population = self.initialize_population(n_assets)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Genetic Algorithm Portfolio Optimization")
            print(f"{'='*70}")
            print(f"Population Size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Crossover Rate: {self.crossover_rate}")
            print(f"Mutation Rate: {self.mutation_rate}")
            print(f"Minimum Return Constraint: {min_return}%")
            print(f"{'='*70}\n")
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness for all individuals
            fitness_values = np.array([
                self.fitness_function(ind, returns, cov_matrix, 
                                    fuzzy_risk_adjustments, min_return)
                for ind in population
            ])
            
            # Track best solution
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_solution = population[best_idx].copy()
            
            # Store history
            self.fitness_history.append(fitness_values)
            self.convergence_history.append(self.best_fitness)
            
            # Print progress
            if verbose and (generation % 20 == 0 or generation == self.n_generations - 1):
                avg_fitness = np.mean(fitness_values)
                portfolio_return = np.dot(self.best_solution, returns)
                portfolio_risk = np.dot(self.best_solution, 
                                      np.dot(cov_matrix, self.best_solution))
                
                print(f"Generation {generation:3d} | "
                      f"Best Fitness: {self.best_fitness:.6f} | "
                      f"Avg Fitness: {avg_fitness:.6f} | "
                      f"Return: {portfolio_return:.2f}% | "
                      f"Risk: {portfolio_risk:.4f}")
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_values)[:self.elite_size]
            elite = population[elite_indices]
            
            # Selection
            selected = self.selection(population, fitness_values)
            
            # Create new population
            offspring = []
            
            for i in range(0, len(selected) - 1, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Combine elite and offspring
            population = np.vstack([elite, offspring[:self.population_size - self.elite_size]])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Optimization Complete!")
            print(f"{'='*70}")
        
        return self.best_solution, self.best_fitness
    
    def plot_convergence(self, save_path='ga_convergence.png'):
        """
        Plot convergence history
        
        Parameters:
        -----------
        save_path : str
            Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Best fitness over generations
        axes[0].plot(self.convergence_history, linewidth=2, color='#2E86AB')
        axes[0].set_xlabel('Generation', fontsize=12)
        axes[0].set_ylabel('Best Fitness (Risk)', fontsize=12)
        axes[0].set_title('GA Convergence - Best Fitness', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Population fitness distribution (last generation)
        last_gen_fitness = self.fitness_history[-1]
        axes[1].hist(last_gen_fitness, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[1].axvline(self.best_fitness, color='red', linestyle='--', 
                       linewidth=2, label=f'Best: {self.best_fitness:.6f}')
        axes[1].set_xlabel('Fitness Value', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Final Population Fitness Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConvergence plot saved to {save_path}")
        plt.close()
    
    def plot_weights(self, weights, stock_names, save_path='portfolio_weights.png'):
        """
        Plot portfolio weights distribution
        
        Parameters:
        -----------
        weights : np.array
            Portfolio weights
        stock_names : list
            Names of stocks
        save_path : str
            Path to save plot
        """
        # Create pie chart (reproduces Figure 11 from paper)
        colors = ['#0066CC', '#FF6600', '#CC0000', '#999999', '#00CC66']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            weights,
            labels=stock_names,
            autopct='%1.0f%%',
            colors=colors[:len(stock_names)],
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title('Portfolio Weights Distribution (M-PFFG-Risk)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio weights plot saved to {save_path}")
        plt.close()


def load_optimization_data():
    """
    Load data required for portfolio optimization
    Based on Tables 22, 23, 24, 25, 26 from the research paper
    
    Returns:
    --------
    data : dict
        Dictionary containing all required data
    """
    # Stock names
    stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    # Predicted returns from LSTM (Table 24)
    predicted_returns = np.array([0.85, 0.72, 1.10, 0.65, 1.95])
    
    # Volatility (Table 22)
    volatilities = np.array([1.25, 1.18, 1.45, 1.10, 2.10]) / 100  # Convert to decimal
    
    # Create covariance matrix (simplified - in practice use historical data)
    # Using volatilities and assuming some correlations
    correlation_matrix = np.array([
        [1.00, 0.65, 0.55, 0.70, 0.40],
        [0.65, 1.00, 0.60, 0.75, 0.35],
        [0.55, 0.60, 1.00, 0.58, 0.50],
        [0.70, 0.75, 0.58, 1.00, 0.38],
        [0.40, 0.35, 0.50, 0.38, 1.00]
    ])
    
    # Covariance matrix = D * R * D where D is diagonal matrix of volatilities
    D = np.diag(volatilities)
    cov_matrix = D @ correlation_matrix @ D
    
    # Fuzzy-adjusted risk from Table 26
    fuzzy_risk_adjustments = np.array([0.12, 0.11, 0.14, 0.10, 0.19])
    
    return {
        'stocks': stocks,
        'returns': predicted_returns,
        'volatilities': volatilities,
        'cov_matrix': cov_matrix,
        'fuzzy_risk_adjustments': fuzzy_risk_adjustments
    }


def run_portfolio_optimization():
    """
    Main function to run portfolio optimization using Genetic Algorithm
    Reproduces Table 26 from the research paper
    """
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║    Genetic Algorithm Portfolio Optimization               ║
    ║    Research: M-Polar Fermatean Fuzzy Graphs              ║
    ║    Objective: Minimize Risk                               ║
    ║    Constraint: Return ≥ 0.92%                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    data = load_optimization_data()
    
    # Initialize GA
    ga = GeneticAlgorithmPortfolio(
        population_size=100,
        n_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=5,
        random_seed=42
    )
    
    # Run optimization
    optimal_weights, optimal_fitness = ga.optimize(
        returns=data['returns'],
        cov_matrix=data['cov_matrix'],
        fuzzy_risk_adjustments=data['fuzzy_risk_adjustments'],
        stock_names=data['stocks'],
        min_return=0.92,
        verbose=True
    )
    
    # Calculate portfolio metrics
    portfolio_return = np.dot(optimal_weights, data['returns'])
    portfolio_variance = np.dot(optimal_weights, np.dot(data['cov_matrix'], optimal_weights))
    portfolio_volatility = np.sqrt(portfolio_variance) * 100  # Convert to percentage
    
    # Display results (Table 26 format)
    print(f"\n{'='*70}")
    print(f"Optimal Portfolio Weights (October 2024, Risk-Focused, Corrected)")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame({
        'Stock': data['stocks'],
        'Predicted_Return_%': data['returns'],
        'Volatility_%': data['volatilities'] * 100,
        'Fuzzy_Adjusted_Risk': data['fuzzy_risk_adjustments'],
        'Weight': optimal_weights
    })
    
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"Portfolio Performance Metrics:")
    print(f"{'='*70}")
    print(f"Expected Return:        {portfolio_return:.2f}%")
    print(f"Portfolio Volatility:   {portfolio_volatility:.2f}%")
    print(f"Portfolio Variance:     {portfolio_variance:.6f}")
    print(f"Fuzzy-Adjusted Risk:    {optimal_fitness:.6f}")
    print(f"Weights Sum:            {optimal_weights.sum():.6f}")
    print(f"{'='*70}\n")
    
    # Save results
    results_df.to_csv('optimal_portfolio_weights_ga.csv', index=False)
    print("✓ Results saved to: optimal_portfolio_weights_ga.csv")
    
    # Plot convergence
    ga.plot_convergence('ga_convergence_history.png')
    
    # Plot portfolio weights (Figure 11)
    ga.plot_weights(optimal_weights, data['stocks'], 'portfolio_weights_distribution.png')
    
    return optimal_weights, results_df


if __name__ == "__main__":
    """
    Main execution: Run Genetic Algorithm for portfolio optimization
    """
    optimal_weights, results = run_portfolio_optimization()
    
    print("\n✓ Genetic Algorithm optimization complete!")
    print("\nGenerated files:")
    print("  - optimal_portfolio_weights_ga.csv")
    print("  - ga_convergence_history.png")
    print("  - portfolio_weights_distribution.png")
